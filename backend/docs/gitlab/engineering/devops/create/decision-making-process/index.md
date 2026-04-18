---
title: "Decision-making process: Clarifying roles and responsibilities"
description: This page documents relevant information to help our team members follow a consistent and dependable process when making decisions
---

In order to increase visibility and consistency of the decision making process within the groups within Create, this section aims at clarifying expectations and best practices.

## Goals

- Respect ownership & process
- Transparent communication to others
- Encourage opinionated collaboration for efficiency

## Values Reminder

These are values extracted from our current Values framework that have assisted in evolving and documenting the practice.

<details>
<summary><strong>Expand to see a reminder of the sub-values</strong></summary>

### Collaboration: Kindness

> We value caring for others. Demonstrating we care for people provides an effective framework for challenging directly and delivering feedback. **Kindness doesn't mean holding back on feedback or avoiding disagreements, these are crucial to professional growth and getting results for customers.** Kindness means you make a separation between the work and the person, you can criticize someone's work but still be respectful to the person. Give as much positive feedback as you can, and do it in a public way.

[/handbook/values/#kindness](/handbook/values/#kindness)

### Results for Customers: Operate with a bias for action

> It's important that we **keep our focus on action, and don't fall into the trap of analysis paralysis** or sticking to a slow, quiet path without risk. Decisions should be thoughtful, but delivering fast results requires the fearless acceptance of occasionally making mistakes; our bias for action also allows us to course correct quickly. Try to get results as fast as possible, but without compromising our other values and ways of working.

[/handbook/values/#operate-with-a-bias-for-action](/handbook/values/#operate-with-a-bias-for-action)

### Results for Customers: Sense of urgency

> Time gained or lost has compounding effects. Try to get the results as fast as possible, but without compromising our other values and ways we communicate, so the compounding of results can begin and we can focus on the next improvement.

[/handbook/values/#sense-of-urgency](/handbook/values/#sense-of-urgency)

### Efficiency: Freedom and responsibility over rigidity

> When possible, we give people the responsibility to make a decision and hold them accountable for that, instead of imposing rules and approval processes. You should have clear objectives and the freedom to work on them as you see fit. Freedom and responsibility are more efficient than rigidly following a process, or creating interdependencies, because they enable faster decision velocity and higher rates of iteration.
>
> When team members have freedom and responsibility over rigidity, they have more room to help others.

[/handbook/values/#freedom-and-responsibility-over-rigidity](/handbook/values/#freedom-and-responsibility-over-rigidity)

</details>

## DRI table

| Type of issue                                   | DRI for Severity                                              | DRI for Priority        |
|-------------------------------------------------|---------------------------------------------------------------|-------------------------|
| `type::feature`                                 | n/a                                                           | Product Manager         |
| `type::maintenance`                             | n/a                                                           | Engineering Manager     |
| `type::bug` + `UX` or `bug::UX`                 | Product Designer                                              | Engineering Manager     |
| `type::bug` + `bug::performance`                | Performance Enablement group `@gl-dx/performance-enablement`  | Engineering Manager     |
| `type::bug` + `bug::vulnerability`              | AppSec `@gitlab-com/gl-security/appsec`                       | Engineering Manager     |
| `type::bug` + `bug::availability`               | Engineering Manager                                           | Engineering Manager     |
| `type::bug` + `test`                            | Developer Experience stage                                    | Engineering Manager     |
| `type::bug` + `GitLab.com Resource Saturation`  | Scalability Engineering Manager                               | Engineering Manager     |

**Sources:**

- [Cross Functional Prioritization - PM:Features, EM:Maintenance and Bugs](/handbook/product/product-processes/cross-functional-prioritization/#prioritization-and-dri-by-component)
- [Severity DRI per bug type](/handbook/product-development/how-we-work/issue-triage/#severity)
- [Priority DRI](/handbook/product-development/how-we-work/issue-triage/#priority)

## Practice

### Issues/Triage

Each decision taken around an issue will have a formal DRI (see [DRI Table](#dri-table)) that will always have the last word on said decision. Team members must always push the decision to the right DRI whether about Severity or Priority.

When someone other than the strict DRI has high confidence or strong suggestion on an outcome, can and should apply the changes to the issue, write a comment (outcome needs to be in a public comment, discussions may happen in internal notes) documenting the reasoning and ensuring the DRI is pinged at the same moment via @mention (and optionally via Slack, too). The DRI needs to give an explicit reaction through a comment or a clearly positive emoji reaction.

✅ **Do:** Leave a comment in an issue pinging the DRIs while at the same time triaging said issue by setting `severity::*`, `priority::*` and milestone.

❌ **Don't:** Triage an issue by setting `severity::*`, `priority::*` and milestone without pinging the DRIs for each task.

### Data as primary piece

On all forms of decision-making, from triaging to assessing rollouts, everyone on the team should strive to base the outcomes on data. This means gathering relevant metrics or research insights, analyzing trends, and using evidence to support our choices rather than relying solely on intuition or assumptions. By grounding our decisions in concrete information, we can make more objective assessments, reduce bias, and create a shared foundation for discussion that helps the team move forward with confidence and alignment.

### Collaborating Across Teams on Issues

Issues may require input from another team or, upon investigation, be identified as another team's responsibility to resolve.

**When seeking input from another team:** The DRI or assigned team member should @mention the relevant DRI from the other team in a comment and, if necessary, follow up via Slack to ensure visibility. Discussions and negotiations may happen within an internal note, guaranteeing that the outcome lands in a public note for visibility of customers.

**When transferring responsibility to another team:** The DRI should:

1. Update the issue labels to reflect the appropriate team
2. @mention the receiving team's DRI in a comment explaining the rationale for reassignment
3. Notify the team via their Slack channel (optional)

**For cross-team dependencies:** If an issue impacts your team's area of responsibility but requires implementation by another team, maintain periodic follow-up to track progress and understand the timeline for resolution.

### Merge Requests

[Code Review Guidelines](https://docs.gitlab.com/development/code_review/) need to be followed and any overrides need to be explicitly approved in the Merge Request by the Engineering Manager of the team of the author.

### Launches of Feature Flag rollouts

Any decisions about launches or Feature Flag Rollouts need to be approved by both the Product Manager, Product Designer and Engineering Manager. Approvals should be captured in a thread of the Rollout issue with explicit messages or emoji reactions of approval from the DRIs.

**Exception:** For [`gitlab_com_derisk` type feature flags](https://docs.gitlab.com/development/feature_flags/#gitlab_com-derisk-type), the assigned engineer may use their judgment to prioritize GitLab.com stability. Inform the Engineering Manager and Product Manager, but explicit approval is not required.

**Applies to:** internal team members, [gitlab.com](https://gitlab.com) and defaulting (Self Managed/Dedicated).

**Doesn't apply to:** turning on for specific team members for the purposes of testing and developing the feature.

### Any other type of decision

Domain of expertise should apply by default.

## Avoiding single-points-of-failure

Whenever a DRI is out-of-office, if the decision is time-sensitive or critical to customers, the team should do their best to contact the next person in the chain of command.

For non-critical or time-sensitive decisions, team members are encouraged to be proactive, make the decision tagging the DRI and ensuring they review it upon their return.
