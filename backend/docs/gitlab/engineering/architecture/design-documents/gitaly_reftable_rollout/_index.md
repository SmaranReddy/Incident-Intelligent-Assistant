---
# This is the title of your design document. Keep it short, simple, and descriptive. A
# good title can help communicate what the design document is and should be considered
# as part of any review.
title: Reftable rollout in Gitaly
status: ongoing
creation-date: "2024-09-19"
authors: [ "@knayakgl" ]
owning-stage: "~devops::systems"
participating-stages: []
# Hides this page in the left sidebar. Recommended so we don't pollute it.
toc_hide: true
---

<!-- Design Doucments often contain forward-looking statements -->
<!-- vale gitlab.FutureTense = NO -->

<!-- This renders the design document header on the detail page, so don't remove it-->
{{< engineering/design-document-header >}}

<!--
Don't add a h1 headline. It'll be added automatically from the title front matter attribute.

For long pages, consider creating a table of contents.
-->

## Summary

Git repositories in Gitaly use the 'files' backend for storing references. This blueprint covers the migration of all existing and new repositories to the new reftable backend. The reftable backend stores references in a binary format, which overcomes some shortcomings of the 'files' backend. More details about the reftable backend can be found in [our blog post](https://about.gitlab.com/blog/2024/05/30/a-beginners-guide-to-the-git-reftable-format/) on the topic.

The reftable backend is generally more performant (with the exception of successive writes when each write uses a separate transaction), supports transactions and atomic writes, and has more efficient storage. While the end result is highly beneficial, we need to ensure a safe migration path to avoid corrupting any data.

This blueprint specifically targets the rollout for GitLab SaaS. Once transactions in Gitaly is fully rolled out, we can follow suit.

Since we won't be exposing the reference backend to clients, there is no involvement expected from admins.

Note: Converting repositories from the 'files' backend to the 'reftable' backend does not affect clients in any way. Clients can continue using the 'files' backend on their systems. The only changes are server-side, and clients should not notice any difference.

## Issue Tracking

The rollout will be tracked in its [own epic](https://gitlab.com/groups/gitlab-org/-/epics/14946). Which is further split into [staging](https://gitlab.com/groups/gitlab-org/-/epics/12503) and [production](https://gitlab.com/groups/gitlab-org/-/epics/14942) rollouts to facilitate easier tracking and linking to individual OKRs.

__Current Status: Reftables have been successfully rolled out to staging.__

This is part of a bigger epic to [enable reftables in GitLab](https://gitlab.com/groups/gitlab-org/-/epics/4220), which also contains some non-blocking improvements.

## Motivation

Traditionally, Git shipped with only a single reference backend called the 'files' backend. The 'files' backend stores references in loose files, where the reference name is the path of the file and its content is the reference value. During housekeeping, several of the loose references are combined into a single 'packed-refs' file.

The 'files' backend has several shortcomings:

- Lookups require checking for loose reference files in addition to reading the references from the packed-refs file. A repository usually contains a mix of both.
- Reference transactions are not committed atomically.
- Since it is stored in a human-readable text format, storage is not efficiently utilized.
- Deleting references from 'packed-refs' requires rewriting the entire file. For repositories with many references, this can be a significantly slow operation.
- Compaction of loose refs into packed-refs is an expensive operation.

[Since Git v2.45.0](https://about.gitlab.com/blog/2024/04/30/whats-new-in-git-2-45-0/#reftables-a-new-backend-for-storing-references), we now have a new reference backend called 'reftable'. The reftable backend addresses all the shortcomings of the 'files' backend, namely:

- It uses a binary format with prefix skips to ensure that we optimize for storage.
- Faster access times with fewer files to read.
- Supports atomic changes using a global reference store lock.
- Deleting references is a O(1) operation.
- Performs compaction on the go.

One of the pain points for repositories hosted in Gitaly has always been the growing number of references in large repositories and the accompanying scalability issues.

For e.g. we can see the how deleting refs in Gitaly is a very slow operation since it require re-packing compared to writing refs, which only requires creation of a new loose ref file.

__DeleteRefs Latency ([Ref](https://dashboards.gitlab.net/d/gitaly-feature-status/gitaly3a-gitaly-feature-status?from=1726030947986&to=1726052547986&var-PROMETHEUS_DS=mimir-gitlab-gprd&var-environment=gprd&var-stage=main&orgId=1&var-method=DeleteRefs&viewPanel=8))__

![DeleteRefs Latency](/images/handbook/engineering/architecture/design-documents/gitaly_reftable_rollout/gitaly_reftable_rollout_deleterefs.png)

<br/>

__WriteRef Latency ([Ref](https://dashboards.gitlab.net/d/gitaly-feature-status/gitaly3a-gitaly-feature-status?from=1726030800000&to=1726052459999&var-PROMETHEUS_DS=mimir-gitlab-gprd&var-environment=gprd&var-stage=main&orgId=1&var-method=WriteRef&viewPanel=8))__

![WriteRef Latency](/images/handbook/engineering/architecture/design-documents/gitaly_reftable_rollout/gitaly_reftable_rollout_writerefs.png)

<br/>

You can see how deleting refs has such a high average latency, ~20x of WriteRefs. This difference is much higher for larger repositories with bigger pack-files. To solve for such problems and the ones stated above, it makes sense to migrate existing repositories from the 'files' backend to the newer reftable backend. This migration would include making the reftable backend the default for new repositories and eventually deprecating the 'files' backend in Gitaly.

### Goals

- Existing repositories will be migrated to the reftable backend from the files backend.
- New repositories will use the reftable backend by default.
- If unexpected issues arise during or after migration, the recovery strategy will be to address and resolve them.
- Metrics will track changes in performance and storage related to reading and writing references before and after migration.
- A plan will be developed for self-hosted customers to migrate to using reftable. The rest of the document unless specified applies to GitLab SaaS.

### Non Goals

- Reftables will only be supported with transactions in Gitaly. This means Praefect without transactions will not be supported. This is due to reftables obtaining a global lock over the ref database. This can cause a deadlock when used with Praefect, if two or more concurrent requests try to do updates at the same time. However, with transactions this is no longer an issue since each write operation will operate on a snapshot.

## Proposal

With the rollout to Staging completed. This is the current plan to rollout to Production.

- Prod Canary
  - Migrate targeted repositories using the MigrateReferenceBackend RPC (details below).
  - Evaluate metrics post migrations and watch out for issues/errors.
  - Enable the opportunistic migrator to migrate existing repositories.
  - Enable reftable for new repositories.
- Rest of Production
  - Post rollout of the WAL, enable the opportunistic migrator to migrate existing repositories and enable reftables for new repositories.

We single out the canary instance since we've had the WAL running on canary for a while and there are no plans to rollback the same.

### Production Canary Repositories to target

For the initial migration, we plan to target the following repositories:

- gitlab-renovate-forks/gitlab
- gitlab-community/gitlab-org/gitlab
- gitlab-org/gitlab

These are the top 3 active repositories in Canary. But we don't want to target `gitlab-org/gitlab` first, since it is critical in nature and contains millions of references. We will start the migration with `gitlab-renovate-forks/gitlab`, since this is an internal repository. Once this has been stable for 1 week, we can then target `gitlab-community/gitlab-org/gitlab` followed by `gitlab-org/gitlab`.

## Design and implementation details

We need to split the rollout into two streams:

1. New repositories
2. Existing repositories

If we implement the rollout only for existing repositories, we'll always be catching up to new repositories. Conversely, if we only implement the rollout for new repositories, we'll never migrate the existing ones.

Therefore, it's essential to plan for both. The easiest to target among the two would be the new repositories. New repositories generally tend to be smaller and this allows us to test out the reftable library in a smaller scale. This also ensures we draw a line where only the existing repositories would need to be migrated. The exception to this would be when repositories are imported from URLs. This value is much smaller (around 2% of all new repositories) and since we rollout incrementally and the repository itself is being created from scratch and doesn't involve an in-place migration, we shouldn't face any issues.

Another option would be to migrate a few existing repositories using feature-flags. Validating the correctness and efficacy of the migration and the reftable library . Then enabling reftables for new repositories and then migrating the remaining repositories. This approach gives us a safety latch to migrate new repositories back to the files backend. This is because we'd have already tested the migration tooling. But then again, we wouldn't really test migration from reftables to files backend, since for existing repositories the best way to revert would be to restore from a backup.

In conclusion, we will first dogfood by migrating few GitLab repositories. Then we will slowly enable reftables for all new repositories. Finally, we'll migrate the remaining existing repositories.

### New repositories

All new repositories created in Gitaly are either done via `git-clone(1)` or via `git-init(1)`. Both of these commands take a `--ref-format=<backend>` argument which can dictate the reference backend used.

To use reftable for new repositories we have to enable the `gitaly_new_repo_reftable_backend` featureflag.

### Existing repositories

Git provides the necessary tooling for migration via the `migrate` subcommand of `git-refs(1)`. While this command provides the means for migrating to the reftable backend, it doesn't lock the entire repository during the migration, which can result in an inconsistent state if concurrent writes occur. So we rely on the transaction manager as a dependency for the reftable migration. Since the transaction manager provides isolated snapshots

#### MigrateReferenceBackend RPC

To migrate a specific repository, we provide the `MigrateReferenceBackend` RPC. This RPC migrates a specified repository. This is beneficial to migrate single repositories for testing and to revert them back in case of any performance degradation observed.

We can find and migrate repositories from the rails console as follows:

```ruby
project = Project.find_by_full_path('gitlab-org/gitlab-test')
gitaly_repository_client = Gitlab::GitalyClient::RepositoryService.new(project.repository)

# Will print the current reference backend used in Git
gitaly_repository_client.repository_info

# To migrate to the Files backend.
gitaly_repository_client.migrate_reference_backend

# To migrate to the reftable backend.
gitaly_repository_client.migrate_reference_backend(to_reftable: true)
```

#### Opportunistic Migrator

While migration of single repositories is a quick way to test the reftable backend, we require an automated process for a wider rollout. The opportunistic migrator adds a middleware to all incoming RPCs and initiates a migration when:

1. There is an incoming read request
2. There is a finishing write request

A background goroutine picks up new migrations to perform one at a time and performs them in a new/separate transaction. This ensures that we do not interfere with the RPC itself. The migrator also cancels any ongoing migrations for a repository if there is a new write request for the repository. This ensures we try and avoid as much writes conflict as possible.

If a migration fails (e.g. another incoming write request, causes a conflict), we log the reason and use an exponential backoff mechanism to not overload the system with high activity repositories.

To enable the opportunistic migrator, use the `gitaly_reftable_migration` featureflag.

#### Forced migration

The opportunistic migrator _attempts_ to migrate repositories but doesn't guarantee migration of all repositories. Once we gain enough confidence with reftables, we will want to force the remaining repositories to be migrated. This can be done by adding a blocking migrator.

The WAL provides a framework for adding such migrations, which will be run before any other request for the repository is processed. We can utilize this framework and plug in our existing migration code to ensure that all repositories will be converted to using reftables.

#### Checking for correctness

When we migrate existing repositories from the files backend to the reftable backend, we must ensure that the migration upholds the integrity of the repository. While the filesystem state will change, the logical state must remain the same. We can verify this by comparing the hash of running `git for-each-ref --include-root-refs` before and after the migration.

Since the migration is performed in transaction, we only commit the changes if the verification is successful. Any failure, would simply abort the transaction and keep the repository as is before migration.

### Evaluating the efficacy

The expectation of using the 'reftables' backend, is that it would eventually outperform the 'files' backend. This means we need to we capture sufficient metrics to showcase the differences. These metrics also should provide information for improvements to make to the 'reftable' backend.

Currently we [capture metrics](https://gitlab.com/gitlab-org/gitaly/-/blob/master/internal/command/command.go#L29) for all the Git commands which run in Gitaly. The simplest way would be to add a 'ref-backend' label to these metrics so we can then build higher level graphs from these commands.

Some of the details we would really like to check are:

- CRUD latency for single reference
- CRUD latency for multiple references in a single transaction
- Time taken for compaction
- Average latency for RPCs which deals with refs. Namely the RPCs in [ref service](https://gitlab.com/gitlab-org/gitaly/-/blob/master/proto/ref.proto).

We will track this information in this [epic](https://gitlab.com/groups/gitlab-org/-/epics/15072).

We also will have E2E running atop reftables possibly also with [GPT](/handbook/support/workflows/gpt_quick_start/), [tracked here](https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks/-/issues/2432).

## Metrics and Logging

- [Grafana Dashboard](https://dashboards.gitlab.net/d/ce1mnfe77u9s0f/reftable-rollout?from=now-30m&orgId=1&timezone=browser&to=now&var-PROMETHEUS_DS=mimir-gitlab-gstg&var-stage=main).
- [Logs for Opportunisitc Migrator](https://nonprod-log.gitlab.net/app/r/s/iu5RU)

## Reverting

Since we take a multi-step approach of dry-running the migration before rolling out, the chance for errors is highly reduced. Nonetheless there are still unknown unknown's which could occur, which we potentially have to deal with.

There are two broad scenarios which we can categorize the issues into:

### Unsuccessful migration

A migration is unsuccessful when there is an error during the migration process. This could occur due to number of issues. But since we use Gitaly's transaction manager, any unsuccessful migration is simply rolled back. Since the migration also includes validating before committing, the system rolls back any unsuccessful attempt. There is no intervention required apart from assessing why it was unsuccessful.

### Successful migration

A successful migration is one where the repository is finally using reftables and the validation step of the migration was successful. The migration is then committed. In such scenarios ideally, there shouldn't be a need for rollback, however we should consider the following scenarios:

1. The repository's performance with reftables is not satisfactory.
2. The reftable backend contains bugs, but repository state is okay. Since the repository state is okay, we consider such bugs as Git bugs, and the Git team will address them. They will either patch the bug in the Git version used by Gitaly or apply the fix directly upstream.
3. The repository has become corrupted over time. If the reftable backend corrupts the repository over time, the only solution is to restore a disk backup. There is some discussions on how effective this would be [here](https://gitlab.com/gitlab-com/gl-infra/data-access/durability/team/-/issues/16#note_2230262996) and the what the future for backups restoration in Gitaly would look like.
4. Gitaly transaction mode (WAL) is disabled on a node where some repositories have already migrated to reftables. For `.com` this is fine since we don't use Praefect. Reftables would continue to work, if transactions are reverted. However we need to consider this possibility for self-hosted customers using Praefect.

#### Revert single repository

For reverting a single repository we can use the `MigrateReferenceBackend` RPC to simply target a given repository and migrate to the files backend.

#### Reverting Multiple repositories

We can use still use the same approach as the single repository for multiple repository migration. If there is a repository corruption on a large scale, we'll have to restore from backup or tend to a more case-by-case fix. A generic revert solution will not apply here.

With the precursor that reftables rollout will follow the WAL. We should ensure that the WAL is already stable on a given node, before migration of repositories on that node to reftables. In such a case that we forsee a rollback of the WAL, we would need to add custom logic to revert repositories to the files backend. This can follow the recovery logic of the WAL in `internal/gitaly/storage/storagemgr/middleware_recovery.go`.

## Dedicated

TBA

## Self-Hosted

The reftable rollout is dependent on the transaction manager. In non-praefect environments, reftables can work without the WAL. To avoid any confusion, let’s focus solely on a future where we enable the WAL.

However, there is an edge case we must consider. A self-hosted customer running praefect, might enable the WAL, migrate all their repositories to reftable and then eventually decide to rollback to using praefect. This wouldn't work since reftables aren't compatible with praefect. We need to plan and device for such a scenario.

### Reverting the reftable backend when disabling transactions

Gitaly nodes do not know if they're sitting behind a praefect node. So targetting environments where the customers re-enable praefect is hard, but we always know when the customer enables/disables the transaction manager.

When customers disable transactions, we can detect the presense of repositories using reftables
