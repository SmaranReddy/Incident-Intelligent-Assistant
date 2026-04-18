---
title: "Create:Code Review BE Engineering Manager Responsibilities"
description: A list of responsibilities and resources for the Create:Code Review BE Engineering Manager
---

The [Backend Engineering Manager role at GitLab](/job-description-library/engineering/development/management/engineering-manager/) is an exciting and dynamic position that encompasses a wide range of responsibilities. In addition to leading and managing a team of talented engineers, the role involves overseeing projects from start to finish, making sure everything runs smoothly and meets high standards. As an engineering manager, you'll also have the opportunity to shape the team by actively participating in recruiting and hiring. One of the key aspects of the role is setting ambitious goals and measuring progress through OKRs (Objectives and Key Results), providing a clear roadmap for success. Moreover, fostering collaboration between different departments and teams is vital to ensure a seamless and efficient workflow. It's a multifaceted role that combines the best of leadership, project management, and team building, making each day full of diverse and engaging challenges.

This page will capture some of the responsibilities and resources for the Create:Code Review Backend Engineering Manager.

## People Management

### Talent Assessments

Talent assessments happen [on a regular basis]https://theloop.gitlab.com/home.

## Milestone Planning

### Milestone Schedule

Here are the activities done by the engineering manager throughout the course of a milestone.

The release is on the [3rd Thursday of every month](/handbook/engineering/releases/monthly-releases/#monthly-release-schedule). We use a cutoff date of the Friday before that to get any work completed.
You can see more about the [release process here](/handbook/engineering/releases/).

On the first of the month a new [planning issue](https://gitlab.com/gitlab-org/create-stage/-/issues/?sort=created_date&state=opened&label_name%5B%5D=group%3A%3Acode%20review&label_name%5B%5D=Planning%20Issue) is automatically created for the next milestone. Create threads for the various projects / topics (e.g. Rapid Diffs, CLI, Bugs) and tag the backend team to get ideas for issues to include in the upcoming milestone planning.

### Planning week activities

The last week of the milestone is planning week for the next milestone. Here are the tasks to do during this week and the following one:

| Day | Action |
|-----|--------|
| Mon | Make sure we have a capacity estimate for the next milestone. Check with team members to make sure all PTO has been entered. Use the [capacity planning spreadsheet](https://docs.google.com/spreadsheets/d/1YMnFy-kQySXs7GCQ-hnnM_oB0Ur9us5mXh8GkdjNgaM) and add the capacity estimate to the planning issue. |
| Mon | Estimate slippage by asking team members which issues are likely/sure to slip, and the total weight expected to complete the work in the next milestone. |
| Mon | Have an initial planning meeting with the PM and frontend EM for Code Review to align on priorities for the milestone and set initial expectations on slippage. |
| Thu | Have a final planning meeting with the PM and frontend EM for Code Review to pick issues for the milestone that fit within the available capacity. |
| Fri | Assign issues to team members based on capacity. |
| Fri | Label all issues with the `Deliverable` and `workflow::ready for development` labels. |
| Fri | Post a comment with an overview of all assigned issues in the planning issue. |
| - | Milestone ends |
| Mon | Calculate the percentage breakdown of features, bugs, and maintenance and add it to the current planning issue. |
| Tue | Review all open issues from the previous milestone, move them to the appropriate milestone. |

### Capacity planning

We use a lightweight system of issue weighting to help with capacity planning.
These weights help us ensure that the amount of scheduled work in a cycle is
reasonable, both for the team as a whole and for each individual. The "weight
budget" for a given cycle is determined based on the team's recent output, as
well as the upcoming availability of each engineer.

Since [things take longer than you think](https://erikbern.com/2019/04/15/why-software-projects-take-longer-than-you-think-a-statistical-model.html), it's OK if an issue takes longer
than the weight indicates. The weights are intended to be used in aggregate, and
what takes one person a day might take another person a week, depending on their
level of background knowledge about the issue. **That's explicitly OK and expected.**
We should strive to be accurate, but understand that they are estimates!
Change the weight if it is not accurate or if the issue becomes harder than
originally expected. Leave a comment indicating why the weight was changed and tag
your EM so that we can better understand weighting and continue to improve.

#### Weights

The weights we use are:

{{% include "includes/engineering/create/weight_table.md" %}}

Anything larger than 5 should be broken down if possible.

Security issues are typically weighted one level higher than they would normally
appear from the table above. This is to account for the extra rigor of the
[patch release process](https://gitlab.com/gitlab-org/release/docs/blob/master/general/security/patch.md).
In particular, the fix usually needs more-careful consideration, and must also
be backported across several releases.

### Availability

Approximately 5-10 business days before the start of a new release, the EM will begin determining how "available" the team will be. This is recorded in the [capacity planning spreadsheet](https://docs.google.com/spreadsheets/d/1YMnFy-kQySXs7GCQ-hnnM_oB0Ur9us5mXh8GkdjNgaM) for transparency. Some of the things that will be taken into account when determining availability are:

- Upcoming training
- Upcoming time off / holidays
- Upcoming on-call slots
- Potential time spent on another teams deliverables

Availability is a percentage calculated by _(work days available / work days in release) * 100_.

All individual contributors start with a "weight budget" of 7, meaning they are capable (based on historical data) of completing a maximum number of issues worth 7 weight points total. Then, based on their availability percentage, weight budgets are reduced individually. For example, if you are 70% available, your weight budget becomes 5.

Product will prioritize issues based on the teams total weight budget.

## Project Management

### Service Level Objectives Monitoring

The Engineering Manager will take proactive steps to avoid having reported bugs to go past the target duration defined in [Severity SLOs](/handbook/product-development/how-we-work/issue-triage/#severity-slos).

The [Bug Triage Board](https://gitlab.com/groups/gitlab-org/-/boards/2546239) and [Triage Reports](https://gitlab.com/gitlab-org/quality/triage-reports/-/issues/?sort=created_date&state=opened&label_name%5B%5D=triage%20report&label_name%5B%5D=group%3A%3Acode%20review) can be used to review all open bugs.

### Priority Issues

There are some classifications of issues that need immediate attention. These could be reliability, security, or high-priority issues. The Engineering Manager needs to watch for these issues on a regular basis, typically multiple times per week.

The [Code Review Priority Issue Board](https://gitlab.com/groups/gitlab-org/-/boards/4778535) is a way to see all these issues in one place.
