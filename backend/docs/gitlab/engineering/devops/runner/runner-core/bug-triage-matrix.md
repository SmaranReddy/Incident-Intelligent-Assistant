---
title: "Runner Core Bug Triage Decision Matrix"
description: "This establishes a transparent, data-driven framework for bug prioritization and autonomous closure decisions. The goal is to align with Product and Customer Support and also empower the team to make consistent, defensible decisions about which bugs to fix or close. This doesn't remove the cases where Product or Engineering makes exceptions for clearly articulated business cases."
---

## The RICE-Adapted Priority Score

### Formula

```text
Priority Score = Reach + Customer Impact + Community Signal + Workaround Factor
```

**Range: 0-10 points**

### Scoring Components

#### 1. Reach (0-3 points) - Executor Impact

Based on Strategic investment:

| Tier | Executors | Score |
|------|-----------|-------|
| Supported | kubernetes, docker, shell, instance, docker autoscaler | 3 |
| Phase-out | custom, docker+windows, ssh | 1 |
| Deprecated | docker+machine | 0 |

#### 2. Customer Impact (0-3 points)

| Score | Criteria |
|-------|----------|
| 3 | High-value customer blocker / ARR impact |
| 2 | Customer-reported with support ticket / Customer Impact |
| 1 | Community-reported and validated/reproduced |
| 0 | No customer or community validation |

#### 3. Community Signal (0-2 points)

| Score | Criteria |
|-------|----------|
| 2 | 20+ upvotes |
| 1 | 10-19 upvotes |
| 0 | \<10 upvotes |

#### 4. Workaround Availability (0-2 points)

| Score | Criteria |
|-------|----------|
| 2 | No workaround possible |
| 1 | Complex workaround exists |
| 0 | Simple, documented workaround |

---

## Decision Actions by Score

| Score | Decision |
|-------|----------|
| **5-10** | Fix - Schedule for appropriate milestone based on score (higher = sooner) |
| **0-4** | Close - Close as "Won't Do" with explanation |

---

## Scenario

Bug Wranglers may close bugs when a combination of these conditions are met:

- Bug affects `docker+machine` or other deprecated feature
- Low Impact - Priority Score \<= 4
  - No customer escalation
  - Bug age \> 1 Year (\~60% of our current backlog are bugs over a year old)
- Behaviour is documented
- Cannot reproduce after 2 attempts
- Reporter pinged for more info and no response within 7 days

### Immediate Closure Conditions

| Scenario | Criteria | Action |
|----------|----------|--------|
| **Outside support window** | Bug reported on version \< 16.x | Close immediately, unsupported |
| **Fixed in supported version** | Bug cannot be reproduced on 18.6 (current) AND reporter is on 17.x or 16.x | Close with upgrade guidance |

---

## Applying to Current Backlog

Here are examples based on our current backlog:

### Likely Schedule

- [Error with recursive submodule checkout (fatal: transport 'file' not allowed)](https://gitlab.com/gitlab-org/gitlab-runner/-/work_items/38908) - (24 upvotes, Tier 1 executor)
- [GitLab Runner build failures for Docker deployments (Docker 29)](https://gitlab.com/gitlab-org/gitlab-runner/-/work_items/39100) - (19 upvotes, Tier 1 executor)
- [Jobs fail despite successful task completion when pods are evicted (since Runner 17.9)](https://gitlab.com/gitlab-org/gitlab-runner/-/work_items/38678) - (18 upvotes, Tier 1 executor)

### Close

- [Docker-machine Preparation Failed](https://gitlab.com/gitlab-org/gitlab-runner/-/work_items/26564)
  - I closed this and there's been minimal pushback (1 comment) despite it having 22 upvotes and 8 disappointed emojis
- [Gitlab Runner with docker+machine keeps spawning ec2 instances but is unable to use them](https://gitlab.com/gitlab-org/gitlab-runner/-/work_items/4193)
  - I closed and there's been no pushback despite it having 23 upvotes
- [Fails to stop spot instance](https://gitlab.com/gitlab-org/gitlab-runner/-/work_items/3687)
  - I closed this and there's been no pushback despite it having 10 upvotes
- [UTF-8 and GBK encoding bug in gitlab-runner + virtualbox + pwsh](https://gitlab.com/gitlab-org/gitlab-runner/-/work_items/29145)
  - Closed: Reported in Runner v15.1.0 with no evidence of preproduction in a recent version
- [403 forbidden during helper image pull](https://gitlab.com/gitlab-org/gitlab-runner/-/work_items/29337)
  - Close! First reported on v15.4.0 with no significant traction since then.

#### Sample Copies for Closing

- General low priority/score

```markdown
Thank you for reporting this issue and for your patience.

After evaluating this bug against our backlog, we've determined it falls below our current prioritization threshold. This means the impact is limited relative to other issues in our backlog, a workaround exists or low related reports. We're also constantly working on broad architecture improvements to Runners which could make this issue obsolete in the future.

This is not a judgment on the validity of your report. We simply have finite capacity and must focus on issues with the broadest impact.

If circumstances change (wider impact or condition changes making this more urgent), we're more than happy to revisit.

/status "Won't do"
```

- Can't reproduce

```markdown
Thank you for reporting this issue.

We attempted to reproduce this on Runner {{current_version}} but were unable to confirm the issue. This may indicate the bug was resolved in a subsequent release, is environment-specific or additional context is required to reproduce. See [maintained versions](https://docs.gitlab.com/policy/maintenance/#maintained-versions).

If you're still experiencing this issue, please let us know by confirming the issue occurs in our current Runner version with additional information to help us reproduce this.

We're happy to revisit once we can reliably reproduce the behavior.

/status "Won't do"
```

- Deprecated docker+machine

```markdown
Thank you for reporting this issue.

This bug affects the `docker+machine` executor, which was deprecated in GitLab 17.5 and is scheduled for removal.

We are no longer investing engineering effort in fixes for deprecated executors. We recognize this may be frustrating if you're still relying on this functionality. Please see our [documentation](https://docs.gitlab.com/runner/executors/) for other executor options that may work for you.

We appreciate your understanding as we focus resources on supported executors.

/status "Won't do"
```

---

## Research

This draws from industry best practices like:

- **GitLab's Severity/Priority Matrix**:
  - https://handbook.gitlab.com/handbook/product-development/how-we-work/issue-triage/#outdated-issues
  - "We simply can't satisfy everyone. We need to balance pleasing users as much as possible with keeping the project maintainable" - https://handbook.gitlab.com/handbook/product-development/how-we-work/issue-triage/#lean-toward-closing
  - "After 14 days, if no response has been made by anyone on the issue, the issue should be closed" - https://handbook.gitlab.com/handbook/product-development/how-we-work/issue-triage/#outdated-issues
  - "Bug fix backports are maintained for the current (first) version..." - https://docs.gitlab.com/policy/maintenance/#maintained-versions
- **RICE Scoring Model** (Intercom): Reach, Impact, Confidence, Effort
  - https://www.intercom.com/blog/rice-simple-prioritization-for-product-managers/
- **Kubernetes Triage Lifecycle**:
  - https://www.kubernetes.dev/docs/guide/issue-triage/#bugs
  - Example: https://github.com/kubernetes/test-infra/issues/25967#issuecomment-1285025238
- **Chromium Won't Fix Categories**:
  - https://www.chromium.org/chromium-os/developer-library/reference/bugs/life-of-a-bug/#no-action-issues
