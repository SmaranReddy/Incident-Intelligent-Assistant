---
title: Analytics Instrumentation Group
description: "The Analytics Instrumentation group work on feature enhancements and implementing privacy focused product analytics across GitLab projects"
---

## Vision

The Analytics Instrumentation Group is part of the Analytics section.
Our group focuses on:
    1. Empowering internal teams within GitLab with data-driven product insights to build a better GitLab.
    2. Helping evolve GitLab's product offerings to include usage based services that align costs with customer value.

To do this, we build data collection and analytics tools within the GitLab product in a privacy-focused manner.

Insights generated from Analytics Instrumentation enable us to identify the best places to invest people and resources, what product categories mature faster, where our user experience can be improved, and how product changes impact the business.

The billing events collected by the Analytics Instrumentation tooling enable GitLab to provide robust usage metering capabilities across GitLab's product portfolio.

You can learn more about the overall direction of our team from the [Analytics Instrumentation Direction page](https://about.gitlab.com/direction/monitor/analytics-instrumentation/).
You can learn about our team's roadmap from the [Analytics Instrumentation wiki page](https://gitlab.com/groups/gitlab-org/analytics-section/analytics-instrumentation/-/wikis/Analytics-Instrumentation-Direction-and-Roadmap)

## How we work

- We work in accordance with our [GitLab values](/handbook/values/).
- We work [transparently](/handbook/values/#transparency) with nearly everything public.
- We get a chance to work on the things we want to work on.
- We have a bias for action.
- We make data-driven decisions.
- Everyone can contribute to our work.

## How to reach us

If you have any questions start by @ mentioning the product manager for the Analytics Instrumentation Group or by creating an issue in our issue board.

## Responsibilities

### Internal Events

We provide the tooling to allow for instrumentation of product usage events across GitLab's services through a system we call internal events which uses the [Snowplow event collection infrastructure](https://snowplow.io/) under the hood.
A list of all instrumented events can be found in our [metrics dictionary](https://metrics.gitlab.com/snowplow).
It is the responsibility of each engineering group to create and maintain their own events as those are domain specific, but Analytics Instrumentation is always willing to help on those issues be it via pair programming or code reviews.

We are actively monitoring the overall health of internal events via [Monte Carlo alerts and Tableau dashboards](/handbook/engineering/data-engineering/analytics/analytics-instrumentation/monitoring_troubleshooting/#snowplow).

### Billing Events

We provide tooling to allow for emitting billing events across GitLab's services, also via the internal events system. You can find more details about the Usage Billing documentation [here](https://docs.google.com/document/d/1uJXz4PaRysMPS6yAo8V9W_gOFp4qb27a-q0VYGz1-04/edit?tab=t.0#heading=h.wgt8cuxlxai7).

### Service Ping Metrics

We're responsible to deliver a reliable [Service Ping](https://docs.gitlab.com/ee/development/internal_analytics/service_ping/) that runs every week on SaaS and Self Managed instances. Our responsiblity is tooling and automations for metric collections to set the company up for success to deliver Service Ping data to our data warehouse.
It is the responsibility of each engineering group to create and maintain their own metrics as those are domain specific, but Analytics Instrumentation is always willing to help on those issues be it via pair programming or our [office hours](https://docs.google.com/document/d/13GHTIfaPTHKh_eYXAhhCyYHHisZQvKlVNqhlo6EyqbE).
A list of all metrics can be found on [metrics.gitlab.com](https://metrics.gitlab.com/).

For questions related to a specific metric, its definition and/or implementation, please contact the Product Manager of the group which owns the metric. You can find information about the metric including its [data category](/handbook/legal/privacy/customer-product-usage-information/), whether it is considered an xMAU metric, its group designation and more in the [metric dictionary](/handbook/product/product-processes/analytics-instrumentation-guide/).
We are actively monitoring the overall health of Service Ping via [Monte Carlo alerts and Tableau dashboards](/handbook/engineering/data-engineering/analytics/analytics-instrumentation/monitoring_troubleshooting/#monitoring-1).

### Projects

These are important projects that we maintain besides our involvement in the main GitLab codebase.

1. [Version App](https://gitlab.com/gitlab-org/gitlab-services/version.gitlab.com)
1. [Metrics Dictionary](https://gitlab.com/gitlab-org/analytics-section/analytics-instrumentation/metric-dictionary) and [Metrics Dictionary from Growth group](https://gitlab.com/gitlab-org/growth/product-intelligence/metric-dictionary).
1. [Snowplow Pseudonymization](https://gitlab.com/gitlab-org/analytics-section/analytics-instrumentation/snowplow-pseudonymization)
1. [Iglu Schema repository for GitLab](https://gitlab.com/gitlab-org/iglu)

## Team members

The following people are permanent members of the Analytics Instrumentation Group:

{{< engineering/stable-counterparts role="Analytics.+Analytics.Instrumentation" >}}

## Project management process

Our team uses a hybrid of Scrum for our project management process. This process follows GitLab's [monthly milestone release cycle](/handbook/marketing/blog/release-posts/#schedule).

- We only work off of issue boards which act as our single source of truth.
- We continuously progress issues to the next workflow stage.
- We work on both product and engineering initiatives.
- We prioritize and estimate all issues we work on.
- We do monthly milestone planning to prepare for our upcoming milestone.
- We do weekly check-ins to share updates within the team

### Check-In

We do a weekly automated check-in within a [separate Slack channel](https://gitlab.slack.com/archives/C04SHHKTQFM).
We feel that a weekly cadence is enough to keep everyone up to date about the most important developments within the team.
A bot asks every team member autoamtically the following questions at the beginning of the week and posts them to the channel

- **What Victories did you have in the last week?**: This should be the most important achievements from the last week. They can also be personal achievements.
- **What Impediments did you have in the last week? Which impediments do you anticipate this week?**: This is important information for other stakeholders, to understand where a team member might need support.
- **Which Priorities do you have for the current week?**: This should be the most important things you want to achieve during the week.

### Retrospective

We believe that continuously evolving and improving the way we work is crucial to delivering good outcomes. Traditionally, retrospectives are used to facilitate this improvement.
Due to our geographical distribution we cannot run typical sync retrospectives that include everyone.
We've also found that completely async retrospectives only see limited participation and discussion.

We're currently trialing a mixed retrospective format, we're team members pair on filling in the retro issue. The process is as follows:

1. Every milestone the [async-retrospective project](https://gitlab.com/gitlab-org/async-retrospectives) automatically creates an
   issue in the [analytics-instrumentation retro project](https://gitlab.com/gl-retrospectives/analytics-instrumentation/-/issues) which contains a list of retro questions.
1. Every 4th week a [donut Slackbot](https://www.donut.com/) will pair up all team members in random groups of two.
1. Every group of two should pair on filling out the retro issue for the current milestone together within that week:
   1. This should happen ideally through a sync meeting or alternatively via Slack.
   1. Focus on the what happened the last two weeks, since the last pairing.
   1. Pairing should involve taking the time to think through the questions, talking about the answers with your pair and coming up with action items based on the identified problems.
   1. Expect an overall time commitment of 30 minutes to an hour for the bi-weekly pairing.
1. Every 2nd week we have a 30-minute time blocker event, that is meant for everyone to take the time to answer Retro discussions.

### Workflow

Our team use the following workflow stages defined in the [Product Development Flow](/handbook/product-development/how-we-work/product-development-flow/#workflow-summary):

#### Validation stage

| Label | Usage |
| -- | -- |
| `~"workflow::validation backlog"` | Applied by the Product Manager for incoming issues that have not been refined or prioritized. |
| `~"workflow::problem validation"` | Applied by the Product Manager for issues where the PM is developing a thorough understanding of the problem |
| `~"workflow::design"` | Applied by the Product Manager or Designer (or Analytics Instrumentation Engineer) to ideate and propose solutions. The proposed solutions should be reviewed by engineering to ensure technical feasibility. |
| `~"workflow::solution validation"` | Applied by the Product Manager or Designer (or Analytics Instrumentation Engineer) to validate a proposed solution through user interviews or usability testing. |

#### Build stage

| Label | Usage |
|--|--|
| `~"workflow::planning breakdown"` | Applied by the Product Manager for Engineers to begin breaking down issues and [adding estimates](#estimation). |
| `~"workflow::ready for development"` |  Applied by either Engineering or Product Manager after an issue has been broken down and scheduled for development. |
| `~"workflow::in dev"` | Applied by the Engineer after work (including documentation) has begun on the issue. An MR is typically linked to the issue at some point throughout this stage. |
| `~"workflow::in review"` | Applied by the Engineer indicating that all MRs required to close an issue are in review. |
| `~"workflow::verification"` | Applied by the Engineer after the MRs in the issue have been merged, this label is applied signaling the issue needs to be verified in staging or production. |
| `~"workflow::complete"` | Applied by the Engineer after all MRs have merged and the issue has been verified. At this step, the issue should also be closed. |
| `~"workflow::blocked"` | Applied by any team member if at any time during development the issue is blocked. For example: technical issue, open question to PM or PD, cross-group dependency. |

### Issue boards

We use issue boards to track issue progress on a daily basis. Issue boards are our single source of truth for the status of our work. Issue boards should be viewed at the highest group level for visibility into all nested projects in a group.

- [**Analytics Instrumentation Issue Board - Current Milestone**](https://gitlab.com/groups/gitlab-org/-/boards/5071664?milestone_title=Started)
- [**Analytics Instrumentation Issue Board _- by milestone_**](https://gitlab.com/groups/gitlab-org/-/boards/2774881?scope=all&not[label_name][]=product%20work&not[label_name][]=Technical%20Writing&not[label_name][]=UX)

### Issue Labels

#### Stage and Section labels

We use the following labels to track work done by the Analytics Instrumentation team

- Group: `~group::analytics instrumentation`

Usage of above label automatically triggers the addition of the section label via the GitLab Bot

- Section: `~section::analytics`

#### Product Direction labels

We use the following labels to track epics and issues under each of our goals from our product direction. Every Issue or Epic we work on shall have at least one of the following labels to indicate how it contributes to our overall goals.

| Goals from Product Direction | Label |
| -- | -- |
| Easy Instrumentation |  `~"Analytics Instrumentation::Easy Instrumentation"` |
| Broaden Instrumentation Coverage |  `~"Analytics Instrumentation::Broaden Instrumentation Coverage"` |
| Deepen Instrumentation Adoption |  `~"Analytics Instrumentation::Deepen Instrumentation Adoption"` |
| Data Quality |  `~"Analytics Instrumentation::Data Quality"` |
| Engineering & Tech Debt |  `~"Analytics Instrumentation::Engineering & Tech Debt"` |

#### Label for Unscheduled issues

It happens that sometimes we need to work on an issue or task that wasn't planned or scheduled in the current milestone.
In that case apply the `~Unscheduled` label and assign it to the current milestone so that we can track it on our [milestone board](https://gitlab.com/groups/gitlab-org/-/boards/5071664?milestone_title=Started).
This helps the team track inefficiencies in the planning process.

#### Label for Incidents

We use the following labels for incidents

| Incident labels |
|---|
| `~"Analytics Instrumentation::Incident-High Severity"` |
| `~"Analytics Instrumentation::Incident-Medium Severity"` |
| `~"Analytics Instrumentation::Incident-Low Severity"` |

### Picking something to work on

Engineers can find and open [the board for the current milestone](https://gitlab.com/groups/gitlab-org/-/boards/5071664?milestone_title=Started).
Engineers should start at the top of the "workflow::ready for development" column and pick the first available, non-assigned issue.
When picking an issue, the engineer should assign themselves as a signal that they are taking ownership of the issue and move them to "workflow::in development" to signal the start of development.

If the next available issue is not a viable candidate (due to amount of capacity vs. issue weight, complexity, knowledge domain, etc.) the engineer may choose to skip an issue and pick the next issue in order of priority.

If all work within a milestone is picked, engineers are free to choose what to work on. Acceptable options include:

- Post in Slack channel to see if any engineers would like help/pair on something they are working on
- Reach out to EM to pull something from the next milestone into the current one
- Create/work on tech debt issue
- Work on a passion issue
- Other (study, research, learning)

### Prioritization

We prioritize our product roadmap in the [Issue Board by Milestone](https://gitlab.com/groups/gitlab-org/-/boards/2774881). Issues appear on each list in order of priority and prioritization of our product roadmap is determined by our product managers.

60% of our development time is spent on issues priotized by product management and the remaining 40% on issues prioritized by engineering as described in our [engineering initiatives](/handbook/engineering/).

### Quarterly OKRs planning

The Analytics Instrumentation team uses OKRs to provide clear alignment between company objectives and team execution. The OKRs are defined by the EM and the PM with close involvement from the team members. The goal of the OKR process is to define measurable outcomes that track progress against company-wide objectives.

**Timeline: 1 month before the [finacial quarter](/handbook/finance/#fiscal-year) begins**

- **Week 1:** Identify broad themes we should work on.
  - PM/EM identify 3-4 major themes for the quarter based on:
    - [Company-wide objectives](https://levelup.edcast.com/pathways/ECL-4c48518c-565b-4817-a15c-62a98d80c79f)
    - Product direction and roadmap priorities
    - Technical debt and platform improvement epics labelled [Analytics Instrumentation: Engineering & Tech Debt](https://gitlab.com/gitlab-org/gitlab/-/issues?sort=updated_desc&state=opened&label_name%5B%5D=Analytics%20Instrumentation%3A%3AEngineering%20%26%20Tech%20Debt&first_page_size=100)

- **Week 2:** Get team input on the themes
  - PM/EM present themes to engineering team.
  - Gather feedback on feasibility, dependencies, and technical considerations.
  - Refine themes based on team input.

- **Week 3:** OKR Drafting
  - PM/EM draft Objectives and Key Results
  - Each Objective should clearly state how it contributes to company goals
  - Key Results should be measurable and time-bound

- **Week 4:** Finalization
  - PM/EM present draft OKRs to team
  - Incorporate feedback and refine
  - Finalize OKRs before quarter begins
  - Add OKR's to wiki roadmap for transparency

### Milestone Planning and Timeline

Our team mostly follows the [Product Development Timeline](/handbook/engineering/workflow/#product-development-timeline) as our group is dependent on the [GitLab self-managed release cycle](https://about.gitlab.com/upcoming-releases/).

The specific application of this timeline to the Analytics Instrumentation Milestone planning process is summarized below.

#### Overview

We orient our planning around the milestone start and end days.
The Planning & Breakdown Phase for next milestone starts 2 weeks prior to the end of current milestone, and is expected to complete before the end of the current milestone.
The Development Phase begins at the beginning of the next milestone.

#### 1. Planning & Breakdown Phase

- **Timeline**: 2 weeks prior to the end of current milestone

##### Tasks

1. Initial Planning
    1. PM: Milestone planning issue gets created ([example](https://gitlab.com/gitlab-org/analytics-instrumentation/-/issues/623)).
    1. PM: Adds overall objective and theme for the milestone to the planning issue
    1. PM: Adds and prioritizes issues from product perspective in the planning issue.
    1. EM: Adds and prioritizes issues from engineering perspective as well as high priority bugs in the planning issue.
    1. EM & PM: Ensures all issues in the planning issue are assigned to the correct milestone and all other issues, not in the planning issue are removed from the milestone.

2. Breakdown and weighing
    1. EM: Removes weight from issues that have been weighed more than 6 months ago to make sure weights consider recent changes.
    1. EM: Moves all unweighed issues to ~"workflow::planning breakdown" stage and tags engineers for estimation.
    1. Engineers: Add a solution proposal if none is present yet.
    1. Engineers: Add their [estimation](#estimation), ask clarifying questions, link potential blockers, and break down the issues further if needed.
    1. EM: Moves estimated and broken down issues into ~"workflow::ready for development" stage.
    1. Engineers: Indicate their availability for the next milestone in our capacity planning sheet.

3. Final Planning
    1. EM: Calculates estimated [capacity](#milestone-capacity).
    1. EM: Adds issues that probably need to be carried over from the current milestone to the planning issue.
    1. EM & PM: Add or remove issues based on weights and capacity.
    1. EM: Prioritizes issues in the [Issue Board by Milestone](https://gitlab.com/groups/gitlab-org/-/boards/2774881) based on the planning issue.
    1. EM & PM: Present the plan for the milestone in the last Analytics Instrumentation sync of the previous milestone focusing on the overall objective and themes.

#### 2. Development Phase

- **Timeline**: Start of the new milestone

##### Tasks

1. Engineers: Work on the issues in the milestone based on the outlined priority:
    1. Engineers assign themselves to issues based on interest/experience.
    2. If no more issues are available in the milestone, they first check if they can take over or help with problems in the milestone assigned to another engineer. Otherwise, they inform the EM, who pulls in issues from the next milestone.

#### Milestone Capacity

Our milestone capacity tells us how many issue weights we can expect to complete in a given milestone. To estimate this we calculate the average daily weight completed by an engineer per day across the previous three milestones. This is multiplied with the actual working days available to us in a given milestone.

**Previous Three Milestones:**

- **Total weights completed:** 140 weights
- **Available work days:** 60 days * 5 engineers = 300 days
- **Actual work days:** 300 days - 20 days off = 280 days
- **Average weight per engineer/day:** 140 weights / 280 days = 0,5 weights/day

**Next Milestone:**

- **Available work days:** 20 days * 5 engineers = 100 days
- **Actual work days:** 100 days - 10 days off = 90 actual days
- **Maximum capacity:** 90 days * 0,5 weights/day = 45 weights

In this example, the next milestone's capacity is 0,5 weights for the whole team. Keep in mind that neither estimations nor this calculation are an exact science. The capacity planning is supposed to help the EM and PM set realistic expectations around deliverables inside and outside time. We do not expect to hit the exact amount of predicted weights.

#### Milestone Commitment

A milestone commitment is a list of issues our team aims to complete in the milestone. The product team follows our GitLab principle of [planning ambitiously](/handbook/product/product-principles/#how-this-impacts-planning) and therefore expect that we won't always be able to deliver everything that we wanted in every milestone. After issues are broken down, estimated, and prioritized, the product manager will apply the `~Deliverable` label to applicable issues. Issues marked with the `~Deliverable` label represent the commitment we are intending to ship in that milestone.

#### Due dates

To properly set expectations for product managers and other stakeholders, our team may decide to add a due date onto an issue. Due dates are not meant to pressure our team but are instead used to communicate an expected delivery date.

We may also use due dates as a way to timebox our iterations. Instead of spending a month on shipping a feature, we may set a due date of a week to force ourselves to come up with a smaller iteration.

### Estimation

We estimate issues async and aim to provide an initial estimate (weight) for all issues scheduled for an upcoming milestone.

We require a minimum of two estimations for weighing an issue. Exceptions can be made for issues that come up during the milestone or if only a single engineer with the required specialty is available. We consider reacting with a ➕ emoji to the estimation as agreeing with it (and thus contributing to the minimal count of estimations).
If both estimations agree, the engineer who did the second estimation should add the agreed-upon weight to the issue. If there is disagreement, the second engineer should @-mention the first one to resolve the conflict.

Estimating includes adding a "Proposed Solution" to the issue if none is documented yet or the estimation brings up a different one than originally documented.
Spikes are exempted from this as discovering solutions is their main point and we default spike issues to a weight of 8.

In planning and estimation, we value velocity over predictability. The main goal of our planning and estimation is to focus on the [MVC](/handbook/values/#minimal-valuable-change-mvc), uncover blind spots, and help us achieve a baseline level of predictability without over-optimizing. We aim for 70% predictability instead of 90%.

If an issue has many unknowns where it's unclear if it's a 1 or a 5, we will be cautious and estimate high (5).

If an initial estimate needs to be adjusted, we revise the estimate immediately and inform the Product Manager. The Product Manager and team will decide if a milestone commitment needs to be changed.

- [Unweighed, upcoming issues in gitlab-org](https://gitlab.com/groups/gitlab-org/-/issues?sort=created_date&state=opened&label_name[]=group::analytics+instrumentation&weight=None&milestone_title=Upcoming&not[label_name][]=product+work)
- [Unweighed, upcoming issues in gitlab-services](https://gitlab.com/groups/gitlab-services/-/issues?sort=created_date&state=opened&label_name[]=group::analytics+instrumentation&weight=None&milestone_title=Upcoming&not[label_name][]=product+work)

#### Issues estimation examples

| Weight | Definition | Example (Engineering) |
| ------ | ---------- | ------------------------- |
| 1 | The simplest possible change. We are confident there will be no side effects. | [Add missing metric definition for "counts_monthly.promoted_issues"](https://gitlab.com/gitlab-org/gitlab/-/issues/340940),<br />[Add instrumentation classes for license standard metrics](https://gitlab.com/gitlab-org/gitlab/-/issues/336340),<br />[Update Registration Features text](https://gitlab.com/gitlab-org/gitlab/-/issues/335051) |
| 2 | A simple change (minimal code changes), where we understand all of the requirements. | [VersionApp: Add indexed on other tables that are exported](https://gitlab.com/gitlab-org/gitlab/-/issues/352019),<br />[Set values for StandardContext in Frontend](https://gitlab.com/gitlab-org/gitlab/-/issues/342993) |
| 3 | A simple change, but the code footprint is bigger (e.g. lots of different files, or tests effected). The requirements are clear. | [Update Registration Features CTA for repository size limit](https://gitlab.com/gitlab-org/gitlab/-/issues/349307),<br />[More paid features available to free users](https://gitlab.com/gitlab-org/gitlab/-/issues/341442) |
| 5 | A more complex change that will impact multiple areas of the codebase, there may also be some refactoring involved. Requirements are understood but you feel there are likely to be some gaps along the way. | [Spike Service Ping health dashboard](https://gitlab.com/gitlab-org/gitlab/-/issues/346431),<br />[Remove `deprecated` metric status](https://gitlab.com/gitlab-org/gitlab/-/issues/340847) |
| 8 | A complex change, that will involve much of the codebase or will require lots of input from others to determine the requirements. | [Dispatch Snowplow events from their event definitions](https://gitlab.com/gitlab-org/gitlab/-/issues/346751),<br />[Add metrics yml files for usage data metrics definition](https://gitlab.com/gitlab-org/gitlab/-/issues/270107) |
| 13| A significant change that may have dependencies (other teams or third-parties) and we likely still don't understand all of the requirements. It's unlikely we would commit to this in a milestone, and the preference would be to further clarify requirements and/or break in to smaller Issues. | [Create Snowplow monitoring framework](https://gitlab.com/gitlab-org/gitlab/-/issues/331103),<br />[Enable batch counting for some individual queries](https://gitlab.com/gitlab-org/gitlab/-/issues/208923) |
| ? | For issues where don't know how to estimate | |

#### Estimation template

The following is a guiding mental framework for engineers to consider when contributing to estimates on issues.

```markdown
### Refinement / Weighing

**Ready for Development**: Yes/No

<!--
Yes/No

Is this issue sufficiently small enough, or could it be broken into smaller issues? If so, recommend how the issue could be broken up.

Is the issue clear and easy to understand?
-->

**Weight**: X

**Reasoning**:

<!--
Add some initial thoughts on how you might break down this issue. A bulleted list is fine.

This will likely require the code changes similar to the following:

- replace the hexdriver with a sonic screwdriver
- rewrite backups to magnetic tape
- send up semaphore flags to warn others

Links to previous example. Discussions on prior art. Notice examples of the simplicity/complexity in the proposed designs.
-->

**Iteration MR/Issues Count**: Y
<!--

Are there any opportunities to split the issue into smaller issues?

- 1 MR to update the driver worker
- 1 MR to update docs regarding mag tape backups

Let me draw your attention to potential caveats.
-->

**Documentation required**: Y/N
<!--
- Do we need to add or change documentation for the issue?
-->
```

## Epic, issue and MR creation

We aim to create issues in the same project as where the future merge request will live and we aim to create epics at the topmost-level group that makes the most sense for its collection of child epics and issues.
For example, if an experiment is being run in the CustomersDot, the epic should be created in the `gitlab-org` group, and the issue should be created in the `gitlab-org/customers-gitlab-com` project.

We focus on creating the issue in the right project to avoid having to close and move it later in the development process.

When creating an issue, use the [linked template](https://gitlab.com/gitlab-org/gitlab/-/issues/new?issuable_template=Analytics%20Instrumentation%20Issue) and follow its instructions.

In case the issue is not created for the [GitLab project](https://gitlab.com/gitlab-org/gitlab), copy the template's content into the appropriate project.

### Ratio of issues to MRs

The ratio of issues to MRs is at the responsible engineer's discretion. MRs should follow the [MVC principle](/handbook/values/#minimal-valuable-change-mvc).
If it is evident in advance that an issue will require more than 2 MRs we should evaluate whether we can split the issue further to document the split of the work more clearly.

### Merge request labels

MR labels should mirror issue labels (which is automatically done when created from an issue):

**Required labels**

- Section: `~section::analytics`
- Group: `~group::analytics instrumentation`
- Type: `~"type::bug"`, `~"type::feature"`, `~"type::tooling"`, `~"type::maintenance"`

### Milestones

We tag each issue and MR with the planned milestone or the milestone at time of completion.

## Team Meetings

Our group holds synchronous meetings to gain additional clarity and alignment on our async discussions. We aim to record all of our meetings as our team members are spread across several timezones and often cannot attend at the scheduled time.

### Meeting rules

- Agenda items should be filled in 6 hours before meetings otherwise it's possible to cancel the meeting.
- It's fine to add agenda items during the meeting as things come up in sync meetings we might not have thought about beforehand.
- Meetings start :30 seconds after start time
- Whoever has the first agenda item starts the meeting.
- Whoever has the last agenda item ends the meeting.
- Meetings end early or on time.
- Any missed agenda items are bumped to the next meeting.

### Our meetings

- **Analytics Instrumentation Sync:** an optional weekly meeting for the Analytics Instrumentation team to discuss any topics they please. At the end of the sync we shut off the recording and continue the call as a social chat.

## Error budget

We maintain [UsageData API endpoints](https://docs.gitlab.com/ee/administration/settings/usage_statistics.html) under the `service_ping` feature to track events, and because of this we must monitor our [budget spend](/handbook/engineering/error-budgets/).

To investigate budget spend, see the [overview](https://dashboards.gitlab.net/d/stage-groups-analytics_instrumentation?orgId=1) and [details](https://dashboards.gitlab.net/d/stage-groups-detail-analytics_instrument?orgId=1) Grafana dashboards for Analytics Instrumentation. You can also check requests contributing to spending the budget in Kibana by filtering by the `service_ping` feature. An example Kibana view can be found [here](https://log.gprd.gitlab.net/goto/8e82ff10-ecb8-11ec-8656-f5f2137823ba).

Note that the budget spend is calculated proportionally by requests failing apdex or failing with an error, and not by how much the target is exceeded. For example, if we had an endpoint with a set goal of 1s request duration, then bringing the request duration from 10s to 5s would not improve the budget.

## Incidents

Within Analytics Instrumentation we have an incident process that is consistent with the Incident management framework defined by the [Data Governance team](/handbook/enterprise-data/data-governance/incident-management/) (Related discussion [here](https://gitlab.com/gitlab-data/analytics/-/issues/24468)).

The process below outlines the different stages of the Analytics Instrumentation incident detection and resolution process and the steps to be taken by the corresponding Directly Responsible Individuals (DRIs).
Please reach out to the [Analytics Instrumentation Group EM/PM](/handbook/engineering/data-engineering/analytics/analytics-instrumentation/#team-members) for any recommendations to changes in the process.

### Incident Definition

We define incidents as a deviation from the intended process that significantly disrupts the reporting of metrics to the point that immediate action is required.

Incidents may manifest through:

- **Availability Issues:** Data, dashboards, or analytics tools becoming inaccessible or unavailable to users
- **Quality Degradation:** Data inaccuracy, corruption, validation failures, or unexpected schema changes
- **Timeliness Violations:** Data not refreshing within expected timeframes, stale metrics, or delayed reporting
- **Processing Failures:** Pipeline breakages, data loading, transformation or extraction job failures, model errors, or instrumentation logic failures that impact downstream processes
- **Security Concerns:** Unauthorized data exposure, access control breaches, or data leakage
- **Collection Disruptions:** Interruption to event tracking, data capture mechanisms, or source system failures

Incident Criteria - not all issues qualify as incidents. An incident must meet one or more of these criteria:

- Whether there is immediate impact on downstream models or dependencies, business operations or data consumers
- Whether there is an SLO breach
- Whether there is potential for permanent data loss or corruption if not addressed immediately
- Whether there is an immediate action required to prevent impact on customer SLAs SLOs of other engineering teams.

 These reasons should lead to the creation of an incident:

1. Any SEV-1 Monte Carlo alert posted into #g_analytics_instrumentation_alerts that's not directly associated with an exisiting incident.
1. Any disruption of event collection on our main Snowplow event collection infrastructure.
1. Any other suspected loss or delay of analytics data that might affect metrics with a `performance_indicator_type` and could be originating in the Analytics Instrumentation domain.

Examples of incidents:

- [High severity incidents](https://gitlab.com/gitlab-org/gitlab/-/issues/442875)
- [Medium severity incidents](https://gitlab.com/gitlab-org/gitlab/-/issues/443639)

When choosing whether to declare an incident:

1. Rather err on the side of declaring an incident even if you run the risk of it not being one. We'd rather close an incident as false positive than miss out on one.
1. If there's an existing incident issue with the data team still follow our process and link the data team's issue in ours.

### Incident Severity

Incident severity is determined by evaluating three key dimensions:

- Business disruption and impact
- Data criticality and impact
- Downstream dependencies

Based on the factors above, we formulate the severity levels below:

- `~"Analytics Instrumentation::Incident-High Severity"`: Production data pipeline failures, blockers for customer-facing teams, inability to make critical business decisions, data breach/exposure, impending or actual data loss affecting multiple systems/metrics, or moderate to severe degradation in business-critical metrics.

- `~"Analytics Instrumentation::Incident-Medium Severity"`: Significant workflow disruption requiring manual workarounds, data pipeline delays impacting downstream consumers, potential security vulnerability/compliance issues, or partial system/service degradation.

- `~"Analytics Instrumentation::Incident-Low Severity"`: Inconveniences with minimal impact, non-critical data pipeline delays, inappropriate access to internal-only data, performance degradation in development/staging environments, or minor data quality issues with known workarounds.

- **Not an incident:** Nice-to-have features/improvements missing, cosmetic issues, documentation updates needed, or technical debt items with no immediate operational impact.

When in doubt between two severity levels, choose the higher one initially. Severity can be downgraded or re-classified as more information becomes available and understanding improves.

### Incident Creation

_(DRI: The team/individual detecting the issue or first team member to see the
alert)_

1. Create an issue and fill all necessary information using the [Analytics Instrumentation Incident Template](https://gitlab.com/gitlab-org/gitlab/-/issues/new?issuable_template=Analytics+Instrumentation+Incident).
1. Add appropriate label using the severity guideline described above.
1. Assign the issue to [Analytics Instrumentation Group PM and EM](/handbook/engineering/data-engineering/analytics/analytics-instrumentation/#team-members).
1. Post in the [#g_analyze_analytics_instrumentation](https://gitlab.slack.com/archives/CL3A7GFPF) slack channel and tag [Analytics Instrumentation Group PM and EM](/handbook/engineering/data-engineering/analytics/analytics-instrumentation/#team-members).
1. Notify these slack channels [#analytics-section](https://gitlab.slack.com/archives/C03GRURTGM9), [#data-rd-analytics](https://gitlab.slack.com/archives/C02C82WDP0U), [#data](https://gitlab.slack.com/archives/C8D1LGC23) with link to the issue.
1. Depending on your own experience either take on the role of resolution DRI, or actively tag EM and engineers in slack to find DRI for incident.

### Incident Resolution

_(DRI: To be identified by EM of the Analytics Instrumentation group)_

1. DRI to work on resolving the issue as quickly as possible. The first priority is to find a fix, even if that is a temporary one, before working on a long term resolution. Our [monitoring and troubleshooting guide](./monitoring_troubleshooting.html) can be helpful here.
1. EM to review severities assigned by detection DRI.

#### Operating Procedure for different severity levels

1. In case of a ~"Analytics Instrumentation::Incident-High Severity" incident:

- DRI to create a temporary channel for the incident in Slack and invite the whole group including PM and relevant stakeholders based on the incident.
- EM (or DRI if EM not available) to announce a feature change lock specific to the analytics instrumentation group.
- All group members concentrate on finding a fix for the issue.
- DRI to post an update in the channel about the current status at least twice per day.

1. In case of a ~"Analytics Instrumentation::Incident-Medium Severity"" incident:

- DRI to create a slack thread within [#g_analyze_analytics_instrumentation](https://gitlab.slack.com/archives/CL3A7GFPF) for coordination around the incident.
- DRI to post daily updates in the channel and issue about the current status of the incident.

1. In case of a ~"Analytics Instrumentation::Incident-Low Severity"" incident:

- DRI to post daily updates in the issue about the current status of the incident.

For all incidents

- DRI is expected to pause the work on their existing commitments until the issue is mitigated.
- DRI is expected to update the incident issue with an expected resolution time as soon as one is clear.
- EM, in coordination with the PM and the individual/team reporting the incident, to close the issue after verifying that the fix is working.
- If a patch release is necessary:
- DRI to create a merge request for a patch release if required and link the merge request to the main issue
- DRI to announce in the main issue when the Patch release is completed

### Incident SLOs

The expected timeline for us to address incidents.

| Severity | Time to mitigate (TTM)(1) | Time to resolve (TTR)(2) |
|-|-|-|
| `~"Analytics Instrumentation::Incident-High Severity"` | Within 24 hrs | Within 7 days |
| `~"Analytics Instrumentation::Incident-Medium Severity"` | Within 72 hrs | Within 30 days |
| `~"Analytics Instrumentation::Incident-Low Severity"` | Within 7 days | Within 60 days |

(1) - Mitigation aims to reduce further impact by investigating and addressing the cause as quickly as possible.

(2) - Resolution uses standard work processes, eg. code review, to completely fix the cause and recover lost data if possible.

### Incident Notification

_(DRI: The PM of the Analytics Instrumentation group)_

1. Notify these slack channels [#g_analyze_product_analytics](https://gitlab.slack.com/archives/C03M4R74NDU), [#data_rd_fusion](https://gitlab.slack.com/archives/C02C82WDP0U), [#data](https://gitlab.slack.com/archives/C8D1LGC23) with link to the issue.
1. Inform Analytics Stage Engineering & Product GPM.
1. Update aforementioned slack channels and individuals on resolution time, changes to resolution times, and when incident is resolved.
1. Ensure the incident and status is reflected in the next [monthly state of data issue](https://gitlab.com/groups/gitlab-com/-/epics/1608 "Monthly State of Data").

## Out of office coverage process

An OOO coverage process helps reduce the mental load of "remembering all the things" while preparing for being away from work. This process allows us to organize the tasks we need to complete before time off and make the team successful.

Open a new issue in the [Analytics Instrumentation project](https://gitlab.com/gitlab-org/analytics-section/analytics-instrumentation/internal/-/issues/new?issuable_template=out_of_office_coverage_template) with the [`out_of_office_coverage_template`](https://gitlab.com/gitlab-org/analytics-section/analytics-instrumentation/internal/-/blob/master/.gitlab/issue_templates/out_of_office_coverage_template.md).

## Onboarding

All new team members to the Analytics Instrumentation teams are provided an onboarding issue to help ramp up on our analytics tooling. New team member members should create their own onboarding issue in the [gitlab-org/analytics-section/analytics-instrumentation/internal project](https://gitlab.com/gitlab-org/analytics-section/analytics-instrumentation/internal/-/issues) using the `engineer_onboarding` template.

## Quick Links

| Resource                                                                                                                          | Description                                               |
|-----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------|
| [Internal Analytics Docs](https://docs.gitlab.com/ee/development/internal_analytics/) | Docs for instrumenting internal analytics at GitLab |
| [Analytics Instrumentation Monitoring and Troubleshooting](monitoring_troubleshooting.html) | Information around Troubleshooting Analytics Instrumentation infrastructure|
| [Analytics Instrumentation Infrastructure](infrastructure.html) | Information about the infrastructure we run |
| [Service Ping Guide](https://docs.gitlab.com/ee/development/internal_analytics/service_ping/)     | An implementation guide for Service Ping      |
| [Privacy Policy](https://about.gitlab.com/privacy/)        | Our privacy policy outlining what data we collect and how we handle it     |
| [Analytics Instrumentation Direction](https://about.gitlab.com/direction/monitor/analytics-instrumentation/)  | The roadmap for Analytics Instrumentation at GitLab  |
