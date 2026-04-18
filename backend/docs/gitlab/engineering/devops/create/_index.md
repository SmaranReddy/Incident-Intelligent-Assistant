---
title: Create
description: >-
  The Create Stage is a group of teams including
  Code Review, Remote Development, and Source Code.
---

## Hello

[We're the Create Stage](/handbook/engineering/devops/create/engineering-managers/) and we are a group of teams within the DevOps Department. We're comprised of three areas [within the GitLab product](/handbook/product/categories/#create-stage).

| Team | Engineering Managers |
| ---- | -------------------- |
| [Create Stage](/handbook/engineering/devops/create/) | [André Luís](/handbook/company/team/#andr3) (Frontend) |
| [Create:Code Review](/handbook/engineering/devops/create/code-review/) | [François Rosé](/handbook/company/team/#francoisrose) (Backend) [André Luís](/handbook/company/team/#andr3) (Frontend) |
| [Create:Source Code](/handbook/engineering/devops/create/source-code/) | [Andre Richards](/handbook/company/team/#andrevr) (Backend) [André Luís](/handbook/company/team/#andr3) (Frontend) |
| [Create:Import](/handbook/engineering/devops/create/import/) | [Thiago Figueiró](/handbook/company/team/#thiagocsf) (Fullstack) — Reports to {{< manager-by-report-name "Thiago Figueiró" >}} |
| [Create:Remote Development](/handbook/engineering/devops/create/remote-development/) | — |

## Mission

### What do we do?

The Create Stage supports software development teams accelerate their delivery and reduce cycle times, by improving efficiency and collaboration. Our stage provides tools that support the beginning of the DevOps Life Cycle.

### Whom do we serve?

We build tools for:

- Software Developers
- DevOps Engineers
- Development Team leads
- Product Managers
- Product Designers

### How do we serve them?

The [Product Vision Handbook Page](https://about.gitlab.com/direction/create/#categories-in-create) provides specific examples for how we serve each type of GitLab user.

## Vision

The following areas have been defined as our direction for the year:

- [Fast, reliable, and easily managed Git storage](https://about.gitlab.com/direction/create/#fast-reliable-and-easily-managed-git-storage)
- [Binary file workflows](https://about.gitlab.com/direction/create/#binary-file-workflows)
- [Delightful code review](https://about.gitlab.com/direction/create/#delightful-code-review)
- [Better collaboration in OSS and within large organizations across many teams](https://about.gitlab.com/direction/create/#better-collaboration-in-oss-and-within-large-organizations-across-many-teams)
- [Unifying our editing experiences](https://about.gitlab.com/direction/create/#unifying-our-editing-experiences)
- [Better integrations](https://about.gitlab.com/direction/create/#better-integrations)

## Professional Development

Each team member is encouraged to take positive steps towards **improving their skills and knowledge**.

**Growing your skillset** will lead to more insightful contributions to the GitLab Product. As your expertise grows, your impact on the product and the company also grows. Here at GitLab, it is important to us to invest in the Professional Development of our team members.

If there is an area you are interested in learning more about, please reach out to your manager so they can provide an environment that will allow you to grow in your areas of preference. Some recommended resources are below:

**Engineering Managers**

- [Recommended Books](/handbook/engineering/devops/create/engineering-managers/books/)
- [Training Materials](/handbook/engineering/devops/create/engineering-managers/training/)

**Individual Contributors**

- [Professional Development](/handbook/engineering/devops/create/engineers/professional-development/)
- [Recommended Books](/handbook/engineering/devops/create/engineers/books/)
- [Training Materials](/handbook/engineering/careers/training/)
- [Transitioning from an Individual Contributor to a Manager](/handbook/engineering/careers/training/ic-to-manager)
- [Skip-level Meetings](/handbook/engineering/devops/create/engineers/skip-level/)
- [Iteration](/handbook/engineering/devops/create/engineers/iteration/)

## How we work

Each team work in the manner that best meets the needs of their product and their team.

- [Create Engineering Managers](/handbook/engineering/devops/create/engineering-managers/)
- [Remote Development Team](/handbook/engineering/devops/create/remote-development/)
- [Code Review Team](/handbook/engineering/devops/create/code-review/#work)
- Source Code Team [Backend](/handbook/engineering/devops/create/source-code/backend/#workflow), [Frontend](/handbook/engineering/devops/create/code-review/frontend/#work)

### Working with product and design

We follow the [Product Development Flow](/handbook/product-development/how-we-work/product-development-flow) when possible.

Weekly calls between the product manager and engineering managers are listed in the respective group's shared calendar. Everyone is welcome to join and these calls are used to discuss any roadblocks, concerns, status updates, deliverables, or other thoughts that impact the group.

The monthly milestone planning issue is triggered to be created the week before the next milestone starts. The PM (Product Manager), EM (Engineering Manager), and PD (Product Designer) will work together to plan the milestone. Product Designers are responsible for:

- **Triaging UX bugs on a bi-weekly basis:** The Product Design Manager for Create will ping the PD fo each team to look through the unscheduled list of bugs every 2 weeks. We recommend you timebox this exercise to 30 minutes max and start by scanning through the list and see if anything jumps out rather than opening each issue 1 by 1. The Designer will assign a `severity` and `bug::ux` labels to triaged issues. Any additional work, like adding design proposals, is out of scope of this process.
- **Recommending at least 5 UX bugs and/or UX enhancements for each milestone:** In addition to reviewing the regular issue backlog, consider using this GLQL [Wiki page](https://gitlab.com/groups/gitlab-com/create-stage/-/wikis/Create-UX-Bugs-Dashboards) to review UX bugs and issue enhancements. During the week leading up to the milestone, the PD should add a list of at least 5 bugs and/or UX enhancements to the planning issue that are most impactful and important to prioritize. The PD should advocate for these items and collaborate on getting them prioritized, but the final decision is still subject to the team's capacity, feasibility/blocked status and other priorities (High severity bugs and Interlock items).

### Defining the Definition of Done with Product

During the [Build track](/handbook/product-development/how-we-work/product-development-flow/#build-track), we document what the measure of success is within the issue or epic description with quantitative or qualitative success metrics. The Product Manager and Product Designer are responsible for proposing these metrics and collaborating with Engineering to instrument and validate them.

### Cross-Team Planning and Refinement

These guidelines are intended to facilitate identification of cross-team dependencies, and early
collaboration. This process is done as part of the Product Development Flow, [Build Phase 1: Plan](/handbook/product-development/how-we-work/product-development-flow/#build-phase-1-plan) and should be done before you schedule the epic and/or issues for execution.

#### Planning breakdown

The main questions to be answered in the `Planning Breakdown` stage are:

- Are requirements clear enough to understand the intent of the request?
- Do we know the boundaries of work to be accomplished? (e.g. code maintained by another team)

If either answer is "No", discussion continues to improve the DRI's understanding of the request.

Engineering output:

- Identify and resolve outstanding questions or discussions.
- Engage other teams if the issue is relevant to them; e.g.: touches shared code, impacts their
  APIs, affects their roadmap.
- Explain any dependencies on other teams in the description, and engage your EM, PM, stable
  counterparts to facilitate early collaboration.
- For epics: create draft implementation issues within the epic(s).
- Apply `Refinement` status to all issues.

#### Refinement

Engineers assigned to refine issues are encouraged to ask questions and push back if issues lack
the information required for successful refinement and execution.

##### Refinement guidelines

1. Check the issue for completeness:
   - It has the necessary designs, including UI, if applicable.
   - The functionality is clearly articulated.
   - The technical details are outlined, and discussions are resolved.
   - Dependencies have been called-out.
1. If the issue is not complete:
   - Tag the relevant people that can help complete the issue and outline what is needed.
   - Tag the EM and PM, so they are aware of the blocker.
1. Ensure the issue is fully understood.
   - Update the issue description with the final description of what will be implemented.
   - Update the issue description with an implementation plan.

In order for someone to understand the issue and its implementation, they should not have to read
through all the comments. The necessary information must be in the description, as the single
source of truth.

##### Implementation Plan

A list of the steps and the parts of the code that need to be updated to address this request.
The implementation plan should also call-out any responsibilities for other team members
or teams, and confirm that the relevant Engineering Manager is in agreement with the plan.

The goal of the implementation plan is to spur critical analysis of the issue and have the DRI
think through what parts of the application will get touched. The implementation plan also permits
other engineers to review the issue and highlight areas of the application that might have
dependencies or have been overlooked.

For complex changes, request a review from another engineer and consider looping in
a subject-matter expert.

## Decision-making process

See [Decision-making process](/handbook/engineering/devops/create/decision-making-process) for details on the workflows the teams in Create follow to ensure a clear and dependable process for decision-making.

## Templates

We use templates in order to make our processes more transparent and efficient.  Documenting practices once and reusing them often provides guidance and support throughout the stage.

## Create Stage Operational Dashboard

This [dashboard](https://gitlab.com/groups/gitlab-org/-/wikis/CXO-Operational-Dashboards/-Core-DevOps-Executive-Dashboard/Create-Stage-Operational-Dashboard) serves as our central command center for maintaining quality and reliability across the Create stage. In alignment with our Quality First mindset, we use this wiki to track, monitor, and act on the operational health of our systems and customer experience.

### Our Commitment

Quality and reliability are not negotiable. By tracking key metrics on this wiki and setting clear targets, we ensure that quality and reliability concerns take priority over non-interlock feature work when trends move in the wrong direction.

### How It Works

When we see infradev issues, bugs, error budgets, incidents, or customer escalations trending negatively or missing targets, we immediately shift team capacity away from non-interlock features to address quality and reliability issues. This creates an operational discipline where quality and reliability problems are prioritized.

- Wiki shows weekly trends for infradev issues and bugs (opened vs. closed)
- We monitor error budget status (🟢🔴) as a reliability indicator
- We track past due S1-S4 infradev issues and bugs against targets
- We track active incidents and customer escalations as measures of system reliability and customer impact
- When metrics decline or targets are missed, we pause or slow non-essential feature work and redirect team member capacity to quality and reliability remediation

### The Outcome

- Quality and reliability are maintained as the first priority because the team responds proactively to declining metrics rather than reactively to outages or customer pain. (per [Product process](/handbook/product/product-processes/#prioritization))
- Interlock commitments (including dedicated reliability initiatives) continue, but discretionary features yield to quality and reliability needs, ensuring we never accumulate technical debt that compromises system stability, product integrity, or customer experience.

## Bug Resolution Strategy

We follow a sustainable bug management process that ensures timely resolution, prevents backlog accumulation, and maintains predictable team capacity through defined bug burn-down cycles.

### Bug Burn-down Cycle Overview

A **bug burn-down cycle** is an intensive, focused period where the team prioritizes resolving existing bugs to restore system health and meet quality standards. Teams enter this cycle when bug debt reaches critical thresholds.

### Exit Criteria: Leaving the Bug Burn-down Cycle

#### Requirements to Exit (Applies to All Severity Levels)

1. **SLO Compliance**

   - Achieve zero bugs with missed Service Level Objectives (SLOs)
   - Successfully complete one full milestone without any missed SLO bugs

2. **Milestone Assignment**

   - Assign 100% of bugs to specific release milestones
   - Eliminate all bugs in "Backlog" status
   - Distribute bugs across current and future milestones for proper planning and capacity management

3. **Consistent Weekly Throughput**

   - Close minimum X bugs per milestone (X determined by Engineering Manager based on team capacity and historical throughput)
   - Maintain this threshold consistently to demonstrate sustainable pace

4. **Post-Exit Review**

   - Reassess weekly commitment target after exiting the burn-down cycle
   - Adjust commitment based on incoming bug rate and team capacity

#### Re-entry Triggers: Entering a Bug Burn-down Cycle

The team will **enter a bug burn-down cycle** if any of the following conditions occur:

##### Trigger Conditions

- **2 or more bugs** with missed SLOs exist within a single milestone
- Bugs begin accumulating in **"Backlog"** status (unassigned to milestones)
- Team falls **below minimum number of issues** threshold for bug closure per milestone

### Ongoing Sustainability Practice

Assign all new bugs to specific milestones immediately upon discovery.  

## How we work across teams

The Create Stage teams work together and play together. We are fortunate to be able to rely on each other to support and complement our features. Some examples of Cross Team Collaboration:

- The Source Code and Gitaly Teams often collaborate together on issues
- Gitaly Team Members have mentored Source Code team members on the Go Programming language

Every quarter we participate in a cross team bonding activity, Create Team Day.

## How we live our values

Engineering Managers live our [values](/handbook/values/) every day.

[Read More about how Engineering Managers live GitLab Values](/handbook/engineering/devops/create/engineering-managers/live/)

## How we measure Results

- [OKRs](/handbook/engineering/devops/create/engineering-managers/okrs/)
- Dashboards
- Issue Boards
- [Monitoring](/handbook/engineering/devops/create/engineering-managers/monitoring/)

## How we measure Iteration

- Logs
- MR Rate

## How we track productivity

In an effort to drive further clarity and align expectations on what is expected for each role, the Create stage has expanded on Engineering performance indicators such as the [MR Rate](/handbook/engineering/development/performance-indicators/#mr-rate) (above 12 per month) and will further specify expectations for each seniority level.

### Why?

During talent assessments, Engineering Managers take a holistic approach to evaluate team members' performance. They consider a diverse set of interconnected metrics across our ecosystem, ensuring fair assessment against the expectations for each [Job Grade](/handbook/total-rewards/compensation/compensation-calculator/#gitlab-job-grades). These metrics include:

- Merge requests merged
- Code reviews conducted
- Maintainer and interviewer status
- Number of interviews participated in
- Mentoring activities
- Merge Request Impact
- Multiple Others

This comprehensive approach allows for a well-rounded evaluation of an engineer's contributions and growth. For more details, refer to our [Talent Assessments](/handbook/engineering/devops/create/talent-assessments/) page.

We believe it's important to communicate expectations clearly and transparently so we can all know what is expected and what to expect. This section looks to add clarity to this topic and promote higher awareness among the entire Create stage.

### What metrics do we focus?

The goal of any team performance metric is to contribute partially to the understanding of a reality. This is because each metric might be imperfect on its own but alongside others can offer a more complete picture and better insights into the contributions given by the team and/or individual.

Currently we're setting clear monthly expectations for the following metrics:

- **Merge Request Rate**:
  - When applied to a **group**: The numerator is the number of merge requests merged into a set of projects (see [notes below](#metrics_notes)).  The denominator is the number of people in the group.
  - When applied to an **individual**: It's the number of merge requests merged into a set of projects (see [notes below](#metrics_notes)).
- **Reviews Rate**:
  - When applied to a **group**: The numerator is the number of code reviews given to merge requests that merged into a set of projects (see [notes below](#metrics_notes)) in a given period (usually a month). The denominator is the number of people in the group.
  - When applied to an **individual**: It's the number of code reviews given to merge requests that merged into a set of projects (see [notes below](#metrics_notes)) in a given period (usually a month).

#### Dashboards

- Merge Request Rate: [tableau dashboard](https://10az.online.tableau.com/#/site/gitlab/views/MTTMAllMRs/MTTMDashboard)
- Reviews Rate: [tableau dashboard](https://10az.online.tableau.com/#/site/gitlab/views/AverageReviewTime/ReviewStatsbyUser?:iid=4)

Note: **If you don't have access to Tableau,** reach out to your direct manager to either provide you with a screenshot for the desired period or if applicable they can open an access request to provide you with permanent access.

#### Notes: {#metrics_notes}

1. These metrics include all MRs affecting the product. The specific projects included in the dataset are listed in [this seed file](https://gitlab.com/gitlab-data/analytics/-/blob/master/transform/snowflake-dbt/seeds/seed_engineering/projects_part_of_product.csv?ref_type=heads).
1. We will iterate on this process over time by expanding our metric set and refining them to ensure alignment with team contributions and evolving role expectations. Any changes will be clearly communicated to all team members.

### Baseline targets for each job grade

In the table below, we outline the baseline numbers for each of the metric related to the Seniority level:

| Metric      | Associate | Intermediate | Senior | Staff |
|-------------|-----------|--------------|--------|-------|
| MR Rate     | 5         | 5            | 8      | 13    |
| Review Rate | 3         | 10           | 16     | 16    |

### How were the targets calculated?

By analysing the existing teams' metrics and collaborating with all Engineering Managers in the stage, targets were defined to accurately reflect the expectations set during a normal calibration session for talent assessments at the [Performing](/handbook/people-group/talent-assessment/#performing) level of each role.

These targets were set having adherence to our CREDIT values in mind and are both ambitious and realistic.

### What do the targets mean?

We expect team members to be **on or above target for at least 6 out of the 12 months of a given year**.

**For Individual Contributors,** these targets can provide a health status signal in alignment with expectations for the role and seniority they're in.

**For Engineering Managers,** these targets can provide a signal to review team processes, planning, or individual cases to better support their team members.

For any teams that have specific contexts that justify a deviation, that extra context should be documented in their handbook team page as much as possible.
