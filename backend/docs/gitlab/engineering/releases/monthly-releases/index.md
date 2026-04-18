---
title: "Monthly releases"
---

## Monthly Release Policy

Monthly releases are the vehicle for delivering new features, bug fixes, and performance improvements to self-managed customers. Unlike [patch releases](/handbook/engineering/releases/patch-releases/) and [internal releases](/handbook/engineering/releases/internal-releases/), monthly releases can include new functionality.

For the general release policy framework including ownership, exception process, and escalation paths, see the [Release Policy](/handbook/engineering/releases/#release-policy) section.

{{< alert color="info" >}}
**For Product & Marketing Teams:** Events, announcements, and feature launches must align with the [monthly release schedule](/handbook/engineering/releases/monthly-releases/#monthly-release-schedule). Features ship in monthly releases only — patch and internal releases do not include new functionality. If a feature misses the monthly deadline, communicate the delay to stakeholders; the next monthly release is the appropriate vehicle to release it. See [Ownership When Deadlines Are Missed](/handbook/engineering/releases/#ownership-when-deadlines-are-missed).
{{< /alert >}}

## Monthly release overview

The monthly release is a [semver](https://semver.org) versioned package containing changes from many successful [deployments on GitLab.com](/handbook/engineering/deployments-and-releases/deployments/). Users on GitLab.com, therefore, receive features and bug fixes earlier than users of self-managed installations.

The [deployments and releases page](/handbook/engineering/deployments-and-releases/) details how the two processes work together.

### Relationship to Other Release Types

Monthly releases create the foundation for all other release types:

* **Stable branches** are created when the release candidate is tagged, forming the basis for patch and internal releases
* **[Patch releases](/handbook/engineering/releases/patch-releases/)** are cut from these stable branches to deliver bug and security fixes to self-managed customers
* **[Internal releases](/handbook/engineering/releases/internal-releases/)** use the same stable branches to deliver critical fixes to GitLab Dedicated before public disclosure

## Monthly release schedule

{{< release-schedule >}}

### Monthly release process

The monthly release timelines are concentrated around the [release date](/handbook/engineering/releases/).

Overview of the steps involved on the monthly release process:

![Monthly release overview](/images/engineering/releases/monthly-releases/monthly-release-overview.jpg)

* [Diagram source](https://docs.google.com/presentation/d/1YRjA1dYCXNXp06VltDYlik1MdFyzUvaeXKk69mMPcA4/edit#slide=id.g2951f7d5d31_1_0)

The end-to-end process consists on the following stages:

1. **First steps** - Initial steps to setup the monthly release, including setting up the release schedule and the deployment cadence.
1. **GitLab.com deployments** - From the start of the milestone up to one week before the release date, GitLab.com receives multiple
    deployments per day. For application changes to be included in a monthly release they need to be successfully deployed to GitLab.com.
1. **Release candidate** - A test [release candidate (RC)](#what-is-a-release-candidate-and-when-are-they-created) is created, along with a stable branch for the targeted [semver](https://semver.org) version. The release candidate package is built,
    tested and deployed to the [pre environment](/handbook/engineering/infrastructure-platforms/environments/#pre). A successful outcome indicates this package can be used as the final version. At this point [Release Managers](https://about.gitlab.com/community/release-managers/) will
    announce the final commit to be included in the release.
1. **Tag** - Release Managers tag the final version of the release based on the release candidate. The release is built and deployed to the [Release environment](/handbook/engineering/infrastructure-platforms/environments/#release).
1. **Release** - On the release day, the release packages are published.

### Release Steps

#### Release Preparation (Up-to and including Tuesday of Release Week)

This provides a window to validate changes, address bugs, and ensure stability in what is destined to be released.
Cutting our first RC is when the stable branch of that version is created.
Many RC's can be created to address bugs that arise.
It should be noted that once the first RC is created, any bugs need to be directly ported to the stable branch.
No new features will be included at this stage.

| Step | Purpose |
| --- | --- |
| Auto-Deploy | Runs multiple times daily up until the Tuesday of the week of the release. Provides verification and chance to remediate problems prior to release. |
| RC Creation | Build a package similar to that for which customers will receive.  Validation of build procedures, similar to what would occur when tagging builds. QA runs on the receiving instance for further verification. |

#### Tag Day (Third Wednesday)

This day is our point of no return.
Once a tag is created it is immutable.
All changes destined for the release are set in stone.

| Step | Purpose |
| --- | --- |
| Version tagging | Create official version tag (e.g., 18.8.0) |
| Package building | Build Omnibus packages for all platforms |
| Container image builds | Build CE, EE, UBI, and FIPS container images |
| QA update scenario | Test upgrade path from previous version |
| Pre-release validation | Final verification of all packages |

#### Release Day (Third Thursday)

| Step | Purpose |
|------|---------|
| Release publication | Publish packages and announce release (13:00 UTC) |
| Post-release verification | Confirm packages available in public repositories |
| Release finalization | Update tracking systems and close release |

### Timelines

The only guaranteed date throughout the release cycle is the [release date](/handbook/engineering/releases/). On this date,
the `monthly release` will be published together with the release announcement.

**All other dates** are **a guideline only** and cannot be considered a deadline
when it comes to what will be included into any type of release. This includes the
[development month](/handbook/engineering/workflow/#product-development-timeline) and the dates defined there as well as any promises given to
customers. This is strictly because there are a lot of moving parts to be considered
when preparing a release which includes working on highest priority and severity
issues as well as security related issues.

If it is absolutely necessary to get a certain feature
ready for a specific version, merge the feature early in the development cycle.
Merges closer to the release date are absolutely not guaranteed to be included
in that specific `monthly release`.

## How can I determine if my merge request will make it into the monthly release?

Multiple tools are available to help determine if the merge request will be included in the upcoming monthly release. This section consolidates all available tooling for easy reference.

### Available tooling

1. [Labels on merge requests](#labels-indicating-inclusion-in-upcoming-monthly-release)
1. [Merge request widget](#merge-request-widget)
1. [ChatOps commands](#chatops-commands)
1. [Release dashboard](#monthly-release-information-dashboard)
1. [Slack announcements on #releases channel](#slack-announcements-on-releases-channel)

### Labels indicating inclusion in upcoming monthly release

Merge requests included in a release candidate (RC) will be part of the monthly release. See ["What is a release candidate and when are they created?"](#what-is-a-release-candidate-and-when-are-they-created) for more information on release candidates. Merge requests will receive the `released::candidate` label to indicate their inclusion into the RC.

A merge request will receive the `released::published` label (which replaces the `released::candidate` label)
when included in the final release, such as `13.6.0` or `13.5.2`, and deployed to
[`release.gitlab.net`](/handbook/engineering/infrastructure-platforms/environments/#release).

### Merge request widget

The merge request widget shows the environment and the time of deployment in every merge request.
This provides information to understand where in the release process any merge request is:

* `release` is for the final version that is going out for self-managed users. When an MR is deployed to [`release.gitlab.net`](/handbook/engineering/infrastructure-platforms/environments/#release) instance, the `released::published` label is applied to the MR to indicate that it will be included in the upcoming monthly release.
* `pre` is for release candidates and versions that are used as a preparation for a final release for self-managed users. When an MR is included in a release candidate, the `released::candidate` label is applied to the MR to indicate that it will be included in the upcoming monthly release.

The merge request widget can also be used to reference where in the deployment process any merge request is:

* `gstg-cny` is the [canary stage](https://about.gitlab.com/handbook/engineering/infrastructure-platforms/environments/canary-stage/) of the GitLab SaaS staging environment
* `gprd-cny` is the [canary stage](https://about.gitlab.com/handbook/engineering/infrastructure-platforms/environments/canary-stage/) of the GitLab SaaS production environment
* `gstg` is the staging environment for GitLab SaaS - [staging.gitlab.com](https://staging.gitlab.com/)
* `gprd` is the production environment for GitLab SaaS - [GitLab.com](https://gitlab.com/)
* `db/gstg` indicates post migrations included in the merge request have been executed in the staging environment.
* `db/gprd` indicates post migrations included in the merge request have been executed in the production environment.

No environment in the widget means that the MR is not deployed to any environment yet

### ChatOps commands

GitLab team members can use ChatOps commands in the Slack `#releases` channel to check the status of merge requests.

#### Check the MR status with respect to monthly releases

There is a ChatOps command that can be used to check the status of an MR with respect to monthly releases:

```bash
/chatops run release check <MR URL> <upcoming release version (optional)>
```

For example: `/chatops run release check https://gitlab.com/gitlab-org/gitlab/-/merge_requests/12345 14.4`.

MRs from the `gitlab-org/gitlab`, `gitlab-org/security/gitlab`, `gitlab-org/omnibus-gitlab` and `gitlab-org/security/omnibus-gitlab` projects are supported by this command.

Two scenarios where this command can be useful:

1. **Check which version an MR was first released in**: `/chatops run release check <MR URL>` (when the version is omitted, it will only check if the MR has already been released)

1. **Check the likelihood of an MR being included in the upcoming monthly release**: `/chatops run release check <MR URL> <upcoming release version>`

### Monthly release information dashboard

GitLab team members can view the [internal Grafana dashboard "Release Information"](https://dashboards.gitlab.net/d/delivery-release_info/delivery3a-release-information?orgId=1) for the following information:

* Upcoming monthly release version
* Upcoming monthly release date
* Upcoming monthly release due date
* Current status of the upcoming monthly release

The due date for merge requests to be included in the upcoming monthly release is set to the date when the initial release candidate and the stable branches are created. Past that point, no more features will be included in the monthly release.

The monthly release dashboard also contains a panel that describes the current status whether it is open and accepting deployed merge requests or if it's already closed and no new features can be added.

The metrics used to display this information are updated automatically throughout the [monthly releases process](#monthly-release-process).

#### Slack announcements on #releases channel

In the runup to the [release date](/handbook/engineering/releases/), Release Managers will announce the final commit to make it
into the release. Such notifications are shared in Slack [#releases](https://gitlab.slack.com/archives/C0XM5UU6B) channel and will look something like this (format is defined in the [release-tools monthly template](https://gitlab.com/gitlab-org/release-tools/-/blob/master/templates/monthly.md.erb)):

> :mega: The stable branch has been created and the release candidate is tagged. Barring any show-stopping issues, this is the final commit to be released on the 21st. https://gitlab.com/gitlab-org/security/gitlab/-/commits/18-3-stable-ee

Merge Requests that have been included in the monthly release will receive [a label indicating inclusion](#labels-indicating-inclusion-in-upcoming-monthly-release).

## Frequently Asked Questions

### Who are the Release Managers for release X?

You can find this out by taking a look at the [GitLab Release Managers](https://about.gitlab.com/community/release-managers/) schedule.

### What is a release candidate and when are they created?

A release candidate (RC) is a GitLab package that includes the changes that will make it into the final monthly release, except for the rare case where a change may need to be reverted. RCs are only created for the monthly release, not patch releases. Usually, one RC is created per release, additional RCs can be created based on the state of Gitlab.com.

Release candidates are created in the run-up to the final release, two days before the release day. If necessary, release candidates can be created earlier if required. This will depend on factors such as:

* Any incidents on GitLab.com that are or have been going on in the run-up to
  the release.
* Any (critical) [patch releases](/handbook/engineering/releases/patch-releases) that require the attention of release
  managers.
* Any issues with our auto-deployment pipelines.
* Other release-related work that may delay or prevent the creation of a release
  candidate until said work is dealt with.

In other words, if you want to know when a release candidate is created your
best option is to join one of the following Slack channels:

* [#releases](https://gitlab.slack.com/archives/C0XM5UU6B)
* [#f_upcoming_release](https://gitlab.slack.com/archives/f_upcoming_release)

Release candidates are deployed to [`pre.gitlab.com`](/handbook/engineering/infrastructure-platforms/environments/#pre) for both automated and
manual testing.

### Will Release Managers create a release candidate earlier if I ask them to?

It is up to a Release Manager to decide when to create a release candidate,
taking into account the state of deployments and GitLab.com.

Please do not message a Release Manager in private about release-related
questions or requests. Instead, post your request/question in the [#releases](https://gitlab.slack.com/archives/C0XM5UU6B)
channel.

### When do I need to have my MR merged in order for it to be included in the monthly release?

> [!WARNING] :warning:
> The earlier in the monthly cycle your MR is merged, the higher the chances are for it to be included in that month's release.

**Target due date for including MRs in the monthly release: 16 UTC on the day before release procedures begin** (typically 2 days before the [release date](/handbook/engineering/releases/monthly-releases/#monthly-release-schedule)).

The quality and stability of what is delivered by everyone define the final list of MRs that will be included in the monthly release. When GitLab.com is stable, MRs merged 2 days before release can be included. When stability is lower due to any team's changes, even MRs merged 1 week before release may not make it.

Even if your MR deploys successfully before the due date, it may not be included in the release if:

* **Any** change on GitLab.com causes an incident or rollback
* GitLab.com experiences stability issues from other MRs
* A rollback occurs that reverts your changes along with problematic ones

For your MR to be included, it must be:

1. **Merged** to the default branch
2. **Deployed** successfully to GitLab.com production
3. **Remain deployed** without being rolled back

For real-time status of your MR, use the [available tooling](#how-can-i-determine-if-my-merge-request-will-make-it-into-the-monthly-release).

### How can I get a high severity bug fix released?

Any high severity issue should start with an issue labeled with the appropriate bug and severity labels.

Depending on the bug details, follow one of the following processes:

* For [high severity security bugs](/handbook/engineering/releases/patch-releases/#patch-release-types)
* For [high severity bugs affecting self-managed users](/handbook/engineering/releases/patch-releases/#patch-release-types). If the bug has been found close to the [release date](/handbook/engineering/releases/) of the month, please also alert the Release Managers in [#releases](https://gitlab.slack.com/archives/C0XM5UU6B).
* For [high severity bugs affecting GitLab.com](/handbook/engineering/deployments-and-releases/deployments/#gitlabcom-pick-label)
* For [high severity bugs affecting security merge requests](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/bugs_introduced_by_security_merge_request.md)
* For [high severity bugs affecting Dedicated](/handbook/engineering/releases/internal-releases/)
