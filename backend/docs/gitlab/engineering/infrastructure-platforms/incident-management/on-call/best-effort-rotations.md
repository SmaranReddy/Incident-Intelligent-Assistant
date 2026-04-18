---
title: On-Call Processes and Policies - Best Effort Rotations
---

Some systems use a best-effort rotation rather than an on-call system. For example, the Release Managers rotation is best effort where a pair of engineers are on duty for a week at a time.

Best effort rotations do not have an SLA.

The decision to use a best effort rotation should center around the on-call requirements rather than the size or nature of the team. However, it is recognized that best effort rotations might be the best option due to the size of your team.

With a Best Effort Rotation:

1. There is no set on-call rotation (team members are not on-call).
2. When there is an escalation to this group, every team member in that region is paged.
3. If you are paged, there is an expectation that you will make your best effort to respond to the page.
4. Conversely, Tier 1 rotations will make their best effort to only issue a page in dire circumstances.

If you want to use a Best Effort Rotation, you need to [create an issue for tracking](https://gitlab.com/gitlab-com/gl-infra/production-engineering/-/issues/new?description_template=onboarding-best-effort-rotation). Please include:

1. The participants for this rotation
2. The location of the participants
3. Why you are choosing a Best Effort Rotation
4. How you are preparing to elevate your rotation to 24x5 and what timeframe you expect for this

## Active Best Effort Rotations

### Release Managers

- Rotation Leader: Michele Bursi
- Coverage: best-effort
- Schedule: [schedule](https://gitlab.pagerduty.com/schedules#PY12KEX)

### Database Operations

- Rotation Leader:
- Coverage: best-effort
- Schedule: schedule

#### More information

Database Operations - ([details](/handbook/engineering/data-engineering/database-excellence/database-operations/dbre-escalation-process))
