---
title: "Stable branches"
---

## Overview

Stable branches are the source of the GitLab release packages delivered to GitLab customers. They serve
as a foundation for versioned releases, and it is fundamental that they remain reliable for any GitLab
release process (monthly, patch or internal).

## Principles

- At GitLab, it is everyone's responsibility to keep stable branches green and reliable. Similar to master branch failures, addressing stable branch failures takes priority over everything else development-related.
- Stable branches receive bug fixes and security patches according to the [maintenance policy](https://docs.gitlab.com/policy/maintenance/).
- Stable branches can also receive upkeeping efforts & catch-up corrective improvements (performance improvements, documentation updates, spec fixes, etc).

## Workflow

1. Stable branches are created when the initial release candidate of a monthly release is tagged.
1. After creation, stable branches are automatically propagated to [security](https://gitlab.com/gitlab-org/security) and [dev](https://dev.gitlab.org/) repositories via repository mirroring.
1. Bug and security fixes are [backported](/handbook/engineering/releases/backports) to stable branches based on the [maintenance policy](https://docs.gitlab.com/policy/maintenance/).
1. Changes backported to stable branches are automatically deployed to [release environments](https://gitlab.com/gitlab-com/gl-infra/release-environments) to guarantee they're compatible with the GitLab version.
1. Patch and internal releases are created from the content of stable branches.

## Characteristics

- Stable branch permissions are based on the [maintenance policy](https://docs.gitlab.com/policy/maintenance/). Stable branches associated with
the [maintained versions](https://docs.gitlab.com/policy/maintenance/#maintained-versions) are opened to GitLab maintainers for merging while older stable branches are limited to Release Managers.
- To account for security fixes and [release environments](https://gitlab.com/gitlab-com/gl-infra/release-environments), security repositories represent the SSOT for stable branches.
- Tests on stable branches are the same on canonical, security and dev repositories. With the exception
of release environments that are only available on the GitLab security repository.

## Broken Stable Branches

When stable branch pipelines fail, they follow the same escalation process as [broken master incidents](/handbook/engineering/workflow/#broken-master-escalation), with one key difference: **incidents are created in the [gitlab-org/release/tasks](https://gitlab.com/gitlab-org/release/tasks) project** instead of the master-broken-incidents project.

**Process Overview:**

- Automation detects stable branch failures and creates incidents with appropriate group/category labels
- Slack notifications are sent to attributed group channels
- Same triage, escalation, and resolution procedures apply as master-broken incidents
- Incidents follow the same 4-hour escalation timeline if unaddressed

Engineers should treat stable branch incidents with the same priority as master-broken incidents, as they can block critical releases and security patches.

## Stable branches over threshold

- When stable branch pipelines run over [thresholds](https://gitlab.com/gitlab-org/quality/analytics/ci-alerts/-/blob/c2cf86eea7f0d566de0f9d8f9d9f3d904f073528/lib/shared/constants.rb#L22-26)(for past 7 days), **issues are created in [canonical](https://gitlab.com/gitlab-org/gitlab/-/issues?sort=created_date&state=opened&label_name%5B%5D=automation%3Apipeline-duration-threshold&label_name%5B%5D=automation%3Abot-authored&first_page_size=20) project** and tagged with label `automation:pipeline-duration-threshold`.
- The [daily_pipeline_thresholds](https://gitlab.com/gitlab-org/quality/analytics/ci-alerts/-/pipeline_schedules) is responsible for creating the issues and also updates any further threshold violations for last 24hrs as a comment on existing issues.
- Issues are created uniquely based on stable branch and pipeline tier combination for the 3 latest stable branches.
