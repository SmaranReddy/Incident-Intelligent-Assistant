---
title: 'Cells: Organization migration'
status: proposed
creation-date: "2024-05-01"
authors: [ "@dbalexandre", "@mkozono" ]
coaches: [ "@ayufan", "@sxuereb" ]
dris: [ "@sranasinghe", "@luciezhao" ]
approvers: [ "@sranasinghe", "@luciezhao" ]
owning-stage: "~devops::tenant scale"
participating-stages: ["~devops::data stores", "~devops::systems"]
toc_hide: true
aliases:
  - /handbook/engineering/architecture/design-documents/cells/migration/
---

{{< engineering/design-document-header >}}

## Summary

All user data will be wrapped in an [Organization](../cells/goals.md#organizations)
which provides isolation and enables moving an organization from one Cell to
another, especially from the [Legacy Cell](../cells/goals.md#legacy-cell).

In [Protocells](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/1616), we
will define cohorts consisting of top-level groups that can be moved
to an organization and then migrated to a Cell.

[Defining](https://gitlab.com/gitlab-com/gl-infra/mstaff/-/issues/474) the cohorts
is the first part of work, but we also need to build tooling to move organizations
from the Legacy Cell to a Cell.

This design document focuses on the migration tooling that moves an organization
from source to destination. It only mentions Cohorts and Top Level Group
migration and doesn't go into implementation detail of those.

## Motivation

Cells is only successful if we meet its primary goal to horizontally scale
GitLab.com. For us to scale we need to move the existing data we have on Legacy
Cell to new Cells to permanently remove load before we hit database scaling
limits. This migration capability is essential for future-proofing our GitLab.com
services as GitLab grows.

### Goals

1. _Interruptible_: If a migration is interrupted like a compute failure or stopped
  by an operator it should start where it left off.
1. _Hands Off_: The migration should run in the background, and we shouldn't have
  a team member laptop running the migration.
1. _Code Reuse_: Geo was built to replicate data from one GitLab instance to
  another, we are doing the same but it's on an organization level.
1. _No Data Loss_: All data that lives on the source Cell should be available on the
  destination Cell. This means that we have account for all data types such as
  Object Storage, Postgres, Advanced Search, Exact Code Search, Git, and Container Registry.
1. _No Cell Downtime_: When migrating an organization the source Cell and
  destination Cell shouldn't incur any downtime except for the organization
  being transferred.
1. _No Visible Downtime_: The organization should not realize that we are migrating their
  data. We will never get zero downtime and we will start with some
  downtime/read-only but will continuously improve this the higher profile
  customers we migrate.
1. _Large Organizations Support_: Able to migrate terabytes of data in a
  timely fashion. This means we have to make our tooling scalable to the data.
1. _Concurrency_: Able to migrate multiple organizations at the same time without
  affecting one another.
1. _Cell Local_: A migration should happen on the destination Cell to prevent a
  single point of failure for all migrations.
1. _Minimal Throwaway Work_: We should iterate on the migration tooling instead
  of re-writing it multiple times.
1. _Observability_: At any point in time we need to know where the migration is
  at, and if there are any problems.
1. _Cell Aware_: The migration tooling needs to also update information in
  Topology Service to start routing requests to the correct Cell.
1. _No User Visible Performance Impact_: Migration should not degrade performance for
  neither the source or destination Cell.
1. _Rollback Capability_: If we need to migrate an organization back to the
  source destination this should be possible.
1. _Dry Run Support_: Operators should be able to test migrations with validation
  and time estimates without actually moving data.
1. _Security_: All data in transit should be encrypted, and cross-cell communication
  must use proper authentication and authorization.

### Non-Goals

- The decision of which organization lives in which cell.
- Support for self-hosted installations.
- Be a replacement to any disaster recovery tooling.

## Cohort Definitions

To satisfy the [Protocells exit criteria](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/1616),
it is expected that we will need to migrate a substantial portion of the top 1,000 active namespaces,
which consumes about [67% of database time](https://gitlab.com/groups/gitlab-com/gl-infra/-/work_items/1616#note_2990334245).

A cohort is a set of GitLab root namespaces and their data,
selected as a single collection to incrementally transfer/migrate to other cells.

Cohort Naming Convention: We use 0 for the test cohort because it must complete successfully before we proceed to production cohorts. Subsequent cohorts (A, B, C, etc.) use letters to indicate they can be executed in parallel without sequential dependencies.

| Cohort ID | Cohort Name | Cohort size indication | Purpose | Simplified eligibility criteria | Impact on Exit criteria | Details |
|-----------|-------------|------------------------|---------|--------------------------------|------------------------|----|
| Cohort 0 | Test cohort | Up to 100 orgs | Use test namespaces to test the transfer & migration process from end-to-end | | None | [Migration Plan](cohort0.md) |
| Cohort A | Subset of Inactive Free users | Up to 5,000 orgs | To establish Protocells as part of real, production use, and refine the migration process. | - Inactive root namespaces<br/><br/>- Free plan<br/><br/>- Private only | Tiny impact on database size | |
| Cohort B | Active opt-in Beta | Up to 1000 orgs | Gain experience with real daily active users. | - Opt-in / guided<br/><br/>- Active root namespaces<br/><br/>- Free, or paid<br/><br/>- Private only | Tiny impact on WAL, LWLocks and database size | [ADR-001: Criteria](decisions/001_cohort_b_criteria.md) |
| Cohort C | Top 1000 opt-in | Up to 300 orgs | Relieve the legacy cell | - Opt-in / guided<br/><br/>- Top 1000 root namespaces by database time<br/><br/>- Private only<br/><br/>- Prerequisite: Feature parity | At least 20% `[1]` decrease in WAL saturation, and Database size | |
| Cohort D | Active long tail opt-in | Approximately 10,000 orgs | Relieve the legacy cell | - Opt-in / self-service<br/><br/>- Active root namespaces<br/><br/>- Private only<br/><br/>- Prerequisite: Feature parity<br/><br/>- Free, or paid | At least 10% `[2]` decrease in WAL saturation, and Database size | |

- `[1]`: The 20% target is derived from 1/3 × 67% [database time](https://gitlab.com/groups/gitlab-com/gl-infra/-/work_items/1616#note_2990334245) consumed by the top 1000 namespaces.
- `[2]`: The 10% target comes from potentially moving 1/3 of 33% of long tail database time.

## Migration Design Documentation

### Database Migration Service (DMS)

1. [DMS Blueprint](dms-blueprint.md) - Strategy for migrating PostgreSQL data from GCP to AWS using AWS DMS
1. [DMS Integration with Dedicated Tooling](dms-instrumentor-integration.md) - Integrating DMS into Instrumentor, AMP, and Tenant Model Schema

### Cohort Migration Plans

1. [Cohort 0: Test Migration](cohort0.md) - Initial migration of 100 test TLGs to validate the end-to-end process

## Decisions

- [ADR-001: Cohort B Criteria](decisions/001_cohort_b_criteria.md) - Eligibility and enrollment criteria for the Active Opt-In Beta cohort
