---
title: "Team processes"
description: "Team Decision-Making & Transparency Guidelines"
---

## Decision Authority Framework

The decision authority framework exists to accelerate the decision making process while maintaining the minimum amount of noise and distraction for engineers. It also exists to empower the engineer to make [two-way-door decisions](/handbook/values/#make-two-way-door-decisions) early on and iterate on the solution.

The [DRI (Directly Responsible Individual)](/handbook/people-group/directly-responsible-individuals/) for an epic has primary decision-making authority within their scope. For more details on DRI responsibilities, see the [DRI & Supporting contributors section](_index.md#dri--supporting-contributors) in our team overview.

When making decisions, we may not always reach consensus. In such cases, it's important to follow the principle of [disagree and commit](/handbook/values/#disagree-and-commit). The DRI should make data-driven decisions based on available information and reasoning, rather than arbitrary choices. When multiple voices are involved, it's up to the DRI to use the authority level table below to escalate and commit to a particular decision.

Use this matrix to determine when to escalate:

| Authority Level | Who Decides | When to Use |
|----------------|-------------|-------------|
| **No Escalation, No FYI** | DRI or assigned engineer | Easily reversible decisions within epic scope |
| **FYI Only** | DRI (notify team after) | Easily reversible decisions with learning value for the team |
| **Team Escalation** | DRI after team discussion | Not easily reversible and/or impacts beyond the epic |
| **Management Escalation** | DRI with Engineering Manager support | Strategic impact or unresolvable conflicts that impact delivery |

## Decision Making Principles

### 1. Say Why, Not Just What

Document the reasoning behind every decision. Consider the following when writing a summary:

- What problem we're solving
- What alternatives we considered
- Why we chose this approach
- How it affects other teams
- How we'll measure success

- [Example issue](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/519)
- [Example issue (Includes ADR)](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/364)

**In practice**: Document decisions in GitLab issues with clear titles, proper labels, and links to related work. For detailed guidance on issue tracking and labeling requirements, see our [Issue Tracking section](_index.md#issue-tracking). When making a decision that will impact the project in general we make an [ADR](/handbook/engineering/architecture/design-documents/cells/#decision-log)

### 2. Document Synchronous Decisions

Discussions happen in meetings, but decisions must be written down:

- Meeting summary and optionally a recording link
- Action items with owners
- The rationale (why we decided)

**In practice**: Create or update a GitLab issue after any meeting where decisions are made.

### 3. Keep Information Accessible

All decisions are searchable in GitLab. No barriers, no gatekeeping.

**In practice**:

- Use consistent labels and clear titles
- Link related issues and dependencies
- Update the [handbook](https://handbook.gitlab.com) when decisions establish new practices or architectural decisions

## Staying Informed (Pull Model)

You're empowered to seek out information rather than waiting to be told everything.

**How to stay informed**:

- Subscribe to labels for epics relevant to your work
- Search GitLab issues to find decisions and context
- Engage proactively when you have expertise or concerns

**When to engage across epics**:

- You have expertise that could help avoid problems
- You see potential integration issues
- A decision might impact your work
- You have questions about timeline or rationale

**How to engage**:

1. Comment on issues or contact the epic DRI
2. Raise concerns in team syncs if they affect timelines
3. Escalate to Management or Senior Tenant Scale staff for strategic issues

## Interrupt Rotation Process

Every week, a Cells Infrastructure engineer is assigned to be the DRI for handling incoming requests from other teams and working on keep-the-lights-on (KTLO) epic work in priority order.

Process summary:

- Every week, a Slack reminder will let the group know that a new interrupt rotation shift is starting.
- Every Cells Infrastructure engineer is expected to be aware of their upcoming rotation (as per the schedule below) and take action as per the slack reminder.
- The DRI currently assigned to the rotation should then dedicate their week to:
  - Work on KTLO epic work in priority order
  - Monitor and respond to incoming cells related requests from other teams in [#g_cells_infrastructure](https://gitlab.enterprise.slack.com/archives/C06V0EQD05M), [#f_protocells](https://gitlab.enterprise.slack.com/archives/C07QTTAMXBY) and [#s_tenant_scale](https://gitlab.enterprise.slack.com/archives/C07TWC3QX47)
  - Respond to any alerts that come into [#g_cells_infrastructure](https://gitlab.enterprise.slack.com/archives/C06V0EQD05M)
- If the DRI is unable to perform an upcoming interrupt rotation shift due to any reason (e.g. PTO, sick leave, other responsibilities taking precedence), they are expected to swap their rotation with another team member or notify the EM to facilitate. Once the swap is identified, the schedule should be updated by using [creating an override](/handbook/engineering/infrastructure-platforms/incident-management/on-call/#swapping-on-call-duty).
- The DRI needs to update the KTLO epic for the current quarter with their rotation activities and provide a rough estimate of time spent on interrupt duties.

At the end of the rotation, each engineer should provide handover notes inside the KTLO epic:

- Using our standardized Duo/AI Chat prompt for handing over below
- Proofread and correct Duo's output if needed
- Post it directly on the issue, with a ping to the new DRI, in a new root comment, not on a thread
- If needed, the new DRI should ask clarifying questions on a reply to the comment or in Slack. Setting up a meeting to go over more difficult context would work if timezones aligned

### Duo/AI Prompt for Interrupt Rotation Handover

```markdown
Please concisely summarize the current status of this epic in order to hand it off to the next engineer on interrupt rotation duty for the Cells Infrastructure team.
Use this template:

A handover comment generated by Duo and reviewed by me:

### Current status

<!-- Content of current status -->

### Action items

- <!-- GitLab handle of DRI -->: <!-- Action item -->
```

- Be sure, or be conservative. When making assertions, be wary of hallucinations. Use qualifying language as appropriate.
- If you reference a resource with a URL, such as a GitLab issue, then make it a link.
- Read linked URLs if the added context might improve your answer.
- Answer with only Markdown code. I will review it and post it.

### Schedules

Schedules are tracked in [incident.io](https://app.incident.io/gitlab/on-call/schedules/01KJ45GBKRS9764MJ7WNPH6DNF?startTime=2026-02-23T00%3A00%3A00&timePeriodOption=two_weeks&calendarToggle=timeline).
