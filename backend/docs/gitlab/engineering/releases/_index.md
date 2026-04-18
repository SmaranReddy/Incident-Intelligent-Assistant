---
title: "GitLab Releases"
---

Looking for product release information? See [release posts](https://about.gitlab.com/releases/categories/releases/), [releases page](https://gitlab.com/gitlab-org/gitlab/-/releases), [What's new](https://docs.gitlab.com/ee/administration/whats-new.html), [changelog](https://gitlab.com/gitlab-org/gitlab/blob/master/CHANGELOG.md), or [feature and deprecations overview](https://gitlab-com.gitlab.io/cs-tools/gitlab-cs-tools/what-is-new-since/).

## Overview

GitLab has three different release types:

1. **[Monthly release](/handbook/engineering/releases/monthly-releases/)**: A GitLab version (XX.YY.0) published on the 3rd Thursday of every month, containing new features, bug fixes and performance improvements from successful [deployments on GitLab.com](/handbook/engineering/deployments-and-releases/deployments/).
1. **[Patch release](/handbook/engineering/releases/patch-releases/)**: Bug fixes and security updates for [maintained versions](https://docs.gitlab.com/policy/maintenance/#maintained-versions), published twice monthly.
1. **[Internal release](/handbook/engineering/releases/internal-releases/)**: Private releases for delivering high-severity fixes to GitLab Dedicated within [remediation SLAs](/handbook/security/product-security/vulnerability-management/sla/).

## Release Policy

### Why Release Policies Matter

GitLab follows [semantic versioning](https://semver.org/), which establishes a contract with our customers:

* **MAJOR** versions may contain breaking changes
* **MINOR** versions contain new features (backward compatible)
* **PATCH** versions contain bug fixes only (safe to auto-upgrade)

Customers rely on this contract to make upgrade decisions. Many customers auto-upgrade patch releases without extensive testing because they trust that patches contain only bug fixes. When we violate this contract, we:

* **Erode customer trust** in our entire release process
* **Put customer environments at risk** with unexpected changes
* **Undermine release predictability** that customers depend on for planning
* **Create cascading problems** when rushed releases introduce new bugs requiring additional patches

### What Each Release Type Contains

| Release Type | Contains | Does NOT Contain |
|--------------|----------|------------------|
| **Monthly** | New features, bug fixes, performance improvements | N/A - this is the vehicle for all new work |
| **Patch** | Bug fixes, security fixes, performance regressions | New features, feature flag enables, incomplete work |
| **Internal** | Critical bug fixes, security fixes for Dedicated instances | New features, feature flag enables, incomplete work |

For SLO/SLA commitments on bug and security fix timelines, see [Patch release SLO/SLA Commitments](/handbook/engineering/releases/patch-releases/#slo-commitments).

### Ownership When Deadlines Are Missed

When a feature misses the monthly release deadline, the appropriate path is the **next monthly release**, not an exception to release policy.

| Team | Owns |
|------|------|
| Development | Shipping code in the next policy-compliant release |
| Product | Communicating the delay to stakeholders; adjusting roadmap expectations |
| Customer Success | Managing customer expectations; negotiating timeline changes |
| Release | Executing policy-compliant releases that maintain customer trust in GitLab releases |

Release Managers do not own accommodating missed deadlines.

### Exception Process

Exceptions to release policy are rare and require explicit approval outside the Release team.

**Release Managers are authorized to decline requests that violate release policies.** Pressure from Customer Success, Sales, or other teams does not constitute approval for policy exceptions.

If an exception is believed necessary:

1. **Requestor** documents the business justification and customer impact assessment
2. **Requestor** escalates to their Director/VP for sponsorship
3. **Sponsoring Director/VP** requests exception from Engineering leadership within the Delivery/Platforms organization:
   * **Backport exceptions (N-3+)**: Senior Engineering Manager or above
   * **Out-of-band releases**: Director of Engineering or above
   * **Policy violations (features in patches, monthly redos)**: VP of Engineering or above
4. **Approving leader** provides written approval with acknowledgment of policy violation
5. **Release Manager** executes only after receiving written approval
6. **Post-release:** Retrospective scheduled to address root cause of the exception request

Exceptions approved without this process set precedent that undermines all release policies.

### Escalation Path

| Situation | Contact |
|-----------|---------|
| Process questions | #releases Slack channel |
| Exception requests | Your Director/VP (not Release Managers) |
| Policy clarification | [Delivery/Platforms leadership](/handbook/engineering/infrastructure-platforms/gitlab-delivery/#teams) |

## Release Coordination

### Teams Involved

| Team | Responsibility |
|------|----------------|
| [Release Team](/handbook/engineering/infrastructure-platforms/gitlab-delivery/delivery/) | Deployments, Process coordination; executing policy-compliant releases |
| Infrastructure | Environment updates |
| Security | Vulnerability fixes and disclosure coordination |
| Product | Feature readiness; owning delays when features miss deadlines |
| Marketing | Release communications |

### Communication

* **#releases** (Slack): Release status and questions
* **[Release Post](https://about.gitlab.com/releases/categories/releases/)**: Public announcements

## Maintenance Policy

See the [GitLab maintenance policy](https://docs.gitlab.com/policy/maintenance/) for details

## Terminology

* **[Maintenance policy](https://docs.gitlab.com/policy/maintenance/)**: Describes the release pace of our major, minor and patch releases for self-managed users.
* **Upcoming version**: [New GitLab release](/handbook/engineering/releases/monthly-releases/#monthly-release-schedule) (XX.YY.0) being developed.
* **Maintained versions**: GitLab versions covered by the [maintenance policy](https://docs.gitlab.com/policy/maintenance/#maintained-versions).
* **[Backports](/handbook/engineering/releases/backports/)**: Bug or security fixes from a recent version applied to an older version.
* **[Stable branches](/handbook/engineering/releases/stable_branches/)**: Source of the GitLab release packages delivered to customers.
* **[Auto-deploy](/handbook/engineering/deployments-and-releases/deployments/)**: GitLab process to deploy application changes to GitLab.com.
* **[Release managers](https://about.gitlab.com/community/release-managers/)**: DRIs for GitLab releases and deployments to GitLab.com.

## Resources

### Monthly release

* [Monthly release schedule and process](/handbook/engineering/releases/monthly-releases/)
* [When do I need to have my MR merged?](/handbook/engineering/releases/monthly-releases#when-do-i-need-to-have-my-mr-merged-in-order-for-it-to-be-included-in-the-monthly-release)
* [How can I determine if my MR will be included?](/handbook/engineering/releases/monthly-releases#how-can-i-determine-if-my-merge-request-will-make-it-into-the-monthly-release)

### Patch release

* [Patch release process and policy](/handbook/engineering/releases/patch-releases/)
* [Patch release runbook for engineers](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/patch/engineers.md)
* [Security runbook for engineers](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/engineer.md)

### Internal release

* [Internal release process](/handbook/engineering/releases/internal-releases/)
* [Internal release runbook for engineers](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/internal-releases/engineers.md)

### Links

* [Release Manager schedule](https://about.gitlab.com/community/release-managers/)
* [How to reach the Release and Deploy team](/handbook/engineering/infrastructure-platforms/gitlab-delivery/delivery/#reaching-our-team)
