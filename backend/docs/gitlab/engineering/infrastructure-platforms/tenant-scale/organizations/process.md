---
title: "Team processes"
description: "How the Organizations team operates"
---

## Work

The Product Manager (PM) compiles the list of issues following
the [product prioritization process](/handbook/product/product-processes/#prioritization),
with input from the team, Engineering Manager (EM), and other stakeholders.
The iteration cycle lasts until the 2nd Friday of a month, and starts new on the following Monday.
Each milestone is identified by the GitLab version set to be released.

### Milestone Planning

Before starting a milestone, the group coordinates using [planning issues](https://gitlab.com/gitlab-org/tenant-scale-group/group-tasks/-/issues/?label_name%5B%5D=Planning%20Issue).
We follow this process:

- The PM defines the goals of the milestone.
- The team members comment about the issues they consider relevant for the milestone.
- The PM and EM work together to decide the final list of issues.
- The whole team reviews the items lined up before the milestone begins.

### What To Work On

The primary source for things to work on is the [planning board](https://gitlab.com/groups/gitlab-org/-/boards/7487616?label_name[]=group%3A%3Aorganizations&milestone_title=Started),
which lists all issues scheduled for the current cycle.
When you assign yourself to an issue, you indicate that you are working on it.

If anything is blocking you from getting started with an issue immediately,
like unanswered questions or unclear requirements, you can skip it, as long as
you put your findings and questions in the issue.
This helps the next engineer who picks up the issue.

Usually issues are not directly assigned to people, except when
a person has clearly the most knowledge or context to work on an issue.
However, we encourage engineers to have a sense of ownership of specific
projects or epics to make significantly more impact in the company.

### Product Development Workflow

We follow the GitLab [product development workflow](/handbook/product-development/how-we-work/product-development-flow/)
guidelines. To get a high-level overview of the status of all issues in the
current milestone, check the [development workflow board](https://gitlab.com/groups/gitlab-org/-/boards/2594854).

The process primarily follows this:

- `workflow::ready for design` to signal an issue is ready to begin the design work
- `workflow::design` to indicate that a designer is actively working on an issue
- `workflow::planning breakdown` to signal that the design is complete and is ready to be broken down into sub-issues for implementation. To preserve context and decisions during the design process, where possible, we re-use the design issue by promoting it to an epic, and add the implementation issues to it. That way, the epic can be used as the [SSOT](/teamops/shared-reality/#single-source-of-truth-ssot) for design, all discussions are in one place, and we don't end up with inconsistencies between the original design issue and the corresponding implementation issue.
- `workflow::refinement` to signal that an issue needs to be refined by engineering. In this step, an implementation guide and a weight should be added to the issue.
- `workflow::ready for development` to signal an item is ready to be worked on by engineering

### Development Workflow

We follow the GitLab [engineering workflow](/handbook/engineering/workflow/)
guidelines. To get a high-level overview of the status of all issues in the
current milestone, check the [development workflow board](https://gitlab.com/groups/gitlab-org/-/boards/2594854).

As owners of the issues assigned to them, engineers are expected to keep the
workflow labels on their issues up to date. When an engineer starts working an
issue, they mark it with the `workflow::in dev` label as the starting point
and continue [updating the issue throughout development](/handbook/engineering/workflow/#updating-workflow-labels-throughout-development).
Before closing an issue, it's important to add the `workflow::complete` label, because this is one
of the requirements for the completed items to appear in the Improvements and Bugs
overview of each month's release post. The process primarily follows this diagram:

``` mermaid
graph LR

  classDef workflowLabel fill:#428BCA,color:#fff;

  A(workflow::in dev):::workflowLabel
  B(workflow::in review):::workflowLabel
  C(workflow::verification):::workflowLabel
  F(workflow::complete):::workflowLabel

  A -- Push an MR --> B
  B -- Merged --> C
  C --> D{Works on production?}
  D -- YES --> F
  F --> CLOSE
  D -- NO --> E[New MR]
  E --> A
```

### Issue Boards

We track our work on the following issue boards:

- [Group::Organizations milestone prioritization](https://gitlab.com/groups/gitlab-org/-/boards/5548886?label_name[]=group%3A%3Aorganizations)
- [Group::Organizations cross-functional prioritization](https://gitlab.com/groups/gitlab-org/-/boards/4424394?label_name[]=group%3A%3Aorganizations)
- [Group::Organizations planning](https://gitlab.com/groups/gitlab-org/-/boards/7487616?label_name[]=group%3A%3Aorganizations)
- [Group::Organizations validation](https://gitlab.com/groups/gitlab-org/-/boards/7487708?not[label_name][]=workflow%3A%3Ain%20dev&not[label_name][]=workflow%3A%3Ain%20review&label_name[]=group%3A%3Aorganizations)
- [Group::Organizations development workflow](https://gitlab.com/groups/gitlab-org/-/boards/2594854?label_name[]=group%3A%3Aorganizations)
- [Group::Organizations bugs](https://gitlab.com/groups/gitlab-org/-/boards/7487700?label_name[]=type%3A%3Abug&label_name[]=group%3A%3Aorganizations)
- [Group::Organizations release posts](https://gitlab.com/groups/gitlab-org/-/boards/7487687?label_name[]=group%3A%3Aorganizations&label_name[]=type%3A%3Afeature)
- [Group::Organizations milestones](https://gitlab.com/groups/gitlab-org/-/boards/5549104?label_name[]=group%3A%3Aorganizations)
- [Group::Organizations team members](https://gitlab.com/groups/gitlab-org/-/boards/5549106?label_name[]=group%3A%3Aorganizations)
- [Group::Organizations important](https://gitlab.com/groups/gitlab-org/-/boards/1438588?label_name[]=group%3A%3Aorganizations)
- [Group::Organizations community contributions](https://gitlab.com/groups/gitlab-org/-/boards/7487739?label_name[]=Community%20contribution&label_name[]=group%3A%3Aorganizations)

### Tracking Dashboards

In addition to issue boards, we also track progress on major initiatives with bespoke dashboards such as:

- [Schema migration](https://cells-progress-tracker-gitlab-org-tenant-scale-g-f4ad96bf01d25f.gitlab.io/schema_migration)
- [Sharding key migration](https://cells-progress-tracker-gitlab-org-tenant-scale-g-f4ad96bf01d25f.gitlab.io/sharding_keys)

The dashboards are part of the [Cells Progress Tracker](https://gitlab.com/gitlab-org/tenant-scale-group/cells-progress-tracker) project.
The team has also spun off [Epic Dashboards](https://gitlab.com/gitlab-org/tenant-scale-group/epic-dashboard) into a project that other teams can use to create their own epic-based tracking dashboards.

### Capacity Planning

We use a simple issue weighting system for capacity planning, ensuring a
manageable amount of work for each milestone. We consider both the team's
throughput and each engineer's upcoming availability from Time Off by Deel using a
[Google Apps Script](https://script.google.com/home/projects/1cH4Hrv03Kf_dlqPyxPdoyxWcV_x2d2u2PKNnGP_YwNjGifjcD4c29GKJ/edit).

The weights are intended to be used in aggregate, and what takes one person a
certain amount of time may be different for another, depending on their level of
knowledge of the issue. We should strive to be accurate, but understand that
they are estimates. Change the weight if it is not accurate or if the issue
becomes more difficult than originally expected. Leave a comment indicating why the
weight was changed and tag the EM and PM so we can better understand the scope
and continue to improve.

#### Weights

To weigh an issue, consider the following important factors:

- Volume of work: expected size of the change to the code base.
- Complexity:
  - Problem understanding: how well the problem is understood.
  - Problem-solving difficulty: the level of difficulty we expect to encounter.

When estimating development work, please assign an issue the appropriate weight:

| Weight     | Description                                                                                                                                                                                                                                                                                      | Examples                                                                                                                                                                                                                                           |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1: Trivial | The simplest possible change. We are confident there will be no side effects. Negligible complexity.                                                                                                                                                                                             | Documentation updates, simple regressions, and other bugs that have already been investigated and discussed and can be fixed with a few lines of code, or technical debt that we know exactly how to address, but just haven't found time for yet. |
| 2: Small   | A simple change (minimal code changes), where we understand all of the requirements. Some small uncertainties exist but we are confident of a solution.                                                                                                                                          | Simple features, like a new API endpoint to expose existing data, or regular bugs or performance issues where all investigation has already taken place.                                                                                           |
| 3: Medium  | A change with a bigger code footprint (e.g. lots of different files, or tests affected). There are uncertainties that we will need to work through.                                                                                                                                              | Regular features, potentially with a backend and frontend component, or most bugs or performance issues.                                                                                                                                           |
| 5: Large   | A more complex change that will impact multiple areas of the codebase. There may also be some refactoring involved. Requirements are poorly understood and you feel there are multiple important gaps. We will need to break this issue into smaller pieces before we can begin a merge request. | Large features with a backend and frontend component, or bugs or performance issues that have seen some initial investigation but have not yet been reproduced or understood.                                                                      |

Anything with a weight of 5 or larger should be broken down if possible.

### Backlog Refinement

Every week the engineering team completes a backlog refinement process
to review upcoming issues. The goal of this effort is for all issues to have a
weight so we can more accurately plan each milestone and also improve our
knowledge sharing.

In addition to the backlog refinement process, engineers can estimate any
issues without following this backlog refinement process.

#### Step 1: Identifying Issues for Refinement

The team will identify issues that need to be refined using the
`workflow::refinement` label. If there are issues that are good
candidates for the backlog refinement process (without weight,
unclear requirements, etc.), please use the label. We will refine
a maximum of 5 issues per week.

[Refinement issues](https://gitlab.com/gitlab-org/tenant-scale-group/group-tasks/-/blob/main/scripts/refinement) are auto-generated at the beginning of each week.
The scripts can be adjusted in our [stage project](https://gitlab.com/gitlab-org/tenant-scale-group/group-tasks).

#### Step 2: Refining Issues

Over the week, each engineer on the team will look at the list of issues
selected for backlog refinement. [Current backlog refinement issues](https://gitlab.com/gitlab-org/tenant-scale-group/group-tasks/-/issues/?label_name%5B%5D=workflow%3A%3Arefinement).

For each issue, team members will review the issues and provide:

- An estimated weight
- A breakdown of issues, if needed
- An implementation guide

When refining issues, consider the following:

- Keep the conversation on the original issues or provide links to relevant discussions in the issue to preserve context
- Update the issue description, implementation plan, and labels as more information is gathered
- For efficiency, engineers can skip the refinement of already refined issues that are marked as completed
- Where the fix is clear and easy, engineers can assign the issue to themselves, give it a weight of 1, push the fix, and close the issue

#### Step 3: Finalizing Refinement

After engineers have had a chance to provide input, the EM or PM will:

- Assign a weight
- Inform stable counterparts if there are any concerns
- Remove the `workflow::refinement` label
- Add the `workflow::ready for development` label

For any issues that were not discussed and given a weight, we will work with
the engineers to see if we need to get more information from PM or UX.

### Retrospectives

We hold scheduled "per milestone" retrospectives, and can have ad-hoc "per project"
retrospectives.

#### Per Milestone

We have [milestone retrospectives issues](https://gitlab.com/gl-retrospectives/enablement-section/tenant-scale/-/issues).
These include the EM, PM, engineers, UX, and all stable counterparts.
Participation is highly encouraged for every milestone. For more information, see [group retrospectives](/handbook/engineering/careers/management/group-retrospectives/) created on the 26th of each
month, for the currently running milestone.

#### Per Project

If an issue, a feature, or other sort of project turns into a
particularly useful learning experience, we may hold a synchronous or
asynchronous retrospective to learn from it. If you think something you're
working on deserves a retrospective:

1. [Create an issue](https://gitlab.com/gitlab-org/tenant-scale-group/group-tasks/-/issues) explaining why you want to have a retrospective and indicate whether this should be synchronous or asynchronous.
1. Include your EM and anyone else who should be involved (like the PM and counterparts).
1. Coordinate a synchronous meeting if applicable. Add all feedback from the retrospective to the issue for future reference.

## Error Budgets

GitLab uses [error budgets](/handbook/engineering/error-budgets/) to measure the
availability and performance of our features. Each engineering group has its own
budget spend. The current 28-day spend for the Tenant Scale group can be found
in this [Grafana dashboard](https://dashboards.gitlab.net/d/product-tenant_scale_error_budget/product3a-error-budgets-tenant-scale?orgId=1&from=now-28d&to=now%2Fm&timezone=utc&var-PROMETHEUS_DS=mimir-gitlab-gprd&var-environment=gprd&var-stage=main).

An error budget exception of 99.79% was [approved](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/16897)
to allow the group to focus on long-term scalability work.

## Engineering Customer/Support Rotation Process

{{< alert type="note" >}}
**Note**: this process is currently unique to the Groups & Projects half of the team, while Organizations is still in development. We expect to roll out something similar for Organizations when the product is on general release.
{{< /alert >}}

Every two weeks, a Groups & Projects engineer is assigned to be the DRI for the technical assessment of customer support tickets,  monitoring [#g_organizations](https://gitlab.slack.com/archives/g_organizations) channel for support issues or `master` pipeline failures, and addressing bugs and support requests which have been triaged and prioritised.

### Process summary

- Every two weeks, a Slack reminder in [#g_organizations_standup](https://gitlab.enterprise.slack.com/archives/C054LN3G0CE) channel will let the group know that a new support shift is starting for technical assessment triage.
- Every engineer is expected to be aware of their upcoming rotation (as per the schedule below) and take action as per the Slack reminder.
- The DRI currently on rotation should then dedicate their two weeks to the following work (in priority order):
    1. Responding to any [alerts](https://gitlab.com/gitlab-org/gitlab/-/pipelines/1858249063) for broken `master` pipeline failures triggered in [#g_organizations](https://gitlab.slack.com/archives/g_organizations)
    1. Working on any scheduled support issues in the current milestone
    1. Working on any scheduled bug issues in the current milestone
    1. Triaging any new issues from the customer support [backlog](https://gitlab.com/gitlab-com/request-for-help/-/issues/?sort=created_date&state=opened&label_name%5B%5D=Help%20group%3A%3AOrganizations&first_page_size=20) and asking for followup information where needed
- There is a live-updating issue here with the above items listed: https://gitlab.com/gitlab-com/gl-infra/tenant-scale/organizations/groups-and-projects/discussions/-/work_items/2
- If the DRI is unable to perform an upcoming triage rotation shift due to any reason (e.g. PTO, sick leave, other responsibilities taking precedence), they are expected to swap their rotation with another team member or notify the EM to facilitate. Once the swap is identified, the schedule should be updated.
- The DRI needs to update this [issue](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/organizations/groups-and-projects/discussions/-/work_items/8) for the 2026 rotation when handing over to the next participant.

At the end of the rotation, each engineer should provide handover notes inside the [team support issue](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/organizations/groups-and-projects/discussions/-/work_items/8):

- Use the example format in the issue description to summarise what was worked on during the rotation
- Post this directly on the issue, with a ping to the new DRI, in a new root comment (not on a thread)
- If needed, the new DRI should ask clarifying questions on a reply to the comment or in Slack.
- Set up a meeting, if required, to go over more difficult context for handover

### Rotation schedule

Schedules are tracked in a [Google Sheet (internal link)](https://docs.google.com/spreadsheets/d/1Y0DI8rNG9hMC21fsVTAQlKfu1F883zUn2SsjFLvckYM/edit?usp=sharing). Please ask before changing this.
