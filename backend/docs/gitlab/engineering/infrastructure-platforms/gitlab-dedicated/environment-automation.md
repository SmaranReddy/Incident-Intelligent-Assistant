---
title: Environment Automation Team
---

## Summary

Environment Automation is a team within the [Dedicated Group](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/). Our mission is to develop and operate the automated plumbing of the GitLab Dedicated solution.

We follow the same processes as listed on the [Dedicated Group](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/), unless a difference exists which is explicitly noted on this page.

## Team Members

{{< team-by-manager-slug "o-lluch" >}}

{{< team-by-manager-slug "denhams" >}}

{{< team-by-manager-slug "nitinduttsharma" >}}

## Working with us

To engage with the Environment Automation team:

- [Create an issue](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/issues/new) in the GitLab Dedicated team issue tracker
- Label the issue with:
  - `workflow-infra::Triage`
  - `group::environment automation`
- When creating an issue, it is not necessary to `@`mention anyone
- In case you want to get attention, use a specific team handle (Ex: @gitlab-dedicated/environment-automation ) as defined in [Dedicated group hierarchy](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/#gitlab-group-hierarchy)
- Slack channels
  - For _Environment Automation_ specific questions, you can find us in [#g_dedicated-environment-automation-team](https://gitlab.slack.com/archives/C074L0W77V0)
    - Our Slack group handle is `@dedicated-envauto-team`
  - The Dedicated Group as a whole leverages: [#g_dedicated-team](https://gitlab.slack.com/archives/C025LECQY0M)
  - Other teams in Dedicated group have their own work channels for team work discussions:
    - [#g_dedicated-us-pubsec](https://gitlab.slack.com/archives/C03R5837WCV)
    - [#g_dedicated-switchboard-team](https://gitlab.slack.com/archives/C04DG7DR1LG)
- When you need help within hours or days:
  - We follow the [Request For Help (RFH)](/handbook/support/workflows/how-to-get-help/#the-request-for-help-landing-page) process. Look for our specific templates [here](https://gitlab.com/gitlab-com/request-for-help#infrastucture-section).
  - For customer impact, please follow our incident management process for [Dedicated Commercial](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/#urgent-availability-or-security-events)
 and page us.
  - For Geo migrations, EA and PS have a [specific engagement model](https://gitlab-com.gitlab.io/gl-infra/gitlab-dedicated/team/runbooks/geo_migration_engagement_model/geo_migration_engagement_model.html), according to migration phase.

### How We Work

Our preference is to work asynchronously, within our project issue tracker as described in [the project management section](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/#project-management).

The team also has a set of regular synchronous calls:

1. Environment Automation Team Sync (alternate weeks):
    1. EMEA/AMER: Tue 15:00 UTC (Good for EMEA and US East)
    1. PST/APAC: Wed 00:00 UTC (Good for APAC and US West)
1. Dedicated on GCP - Weekly Demo: Wed 07:30 UTC
1. [Dedicated Group Demo](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/#meetings-and-scheduled-calls)

### Quarterly Planning Process {#quarterly-planning-process}

<details>
<summary>Quarterly Planning Process</summary>

Quarterly planning is used to balance **strategic priorities** with **team capacity**. It ensures consistent execution, async alignment, and visibility across stakeholders. Planning starts 4 weeks before the new quarter and includes integrated Epic backlog refinement to ensure execution-ready work items.

#### Step 1 – Planning Issue Creation (Week –4)

- **Owner:** Engineering Manager (EM)
- **Actions:**
  - Create a Quarterly Planning Issue ([example: Q4 FY26 Planning](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/issues/9938)).
  - Use this issue for async discussion, links to strategic docs, and final decision logging.
  - Share issue link with PMs, team, and stakeholders.
- **Deliverables:** Planning issue opened and shared with all stakeholders.

#### Step 2 – Strategic Input (Weeks –4 to –3)

- **Owners:** PMs, Company Interlock process
- **Actions:**
  - PMs **regularly update** the [FY26 Dedicated Stage Roadmap Planning spreadsheet](https://docs.google.com/spreadsheets/d/1LLxUPQ5nEaC7J9OcO3TbG1txVOtAQdtqKbkJYa4Jcvk/edit?gid=0#gid=0).
  - Company Interlock process provides top-tier initiatives ([Interlock Tracker](https://docs.google.com/spreadsheets/d/10ZMNCQVhRTbELC1WV4Zme5uEOrmOH3WMqYC1sHdJouw/edit?gid=2120796429#gid=2120796429)).
  - EA EMs and PM hold a **weekly Tuesday conversation** to discuss:
    1. Issues-of-the-day
    2. Flags for team awareness
    3. Quarterly planning inputs and updates
- **Deliverables:** Strategic priorities documented in the planning issue.

#### Step 3 – Team Planning (Weeks –3 to –1)

- **Owners:** EMs and Team Members
- **Actions:**
  - EMs update the [EA Continuous Planning Spreadsheet](https://docs.google.com/spreadsheets/d/10ZMNCQVhRTbELC1WV4Zme5uEOrmOH3WMqYC1sHdJouw/edit?gid=0#gid=0) with projects, timeline estimates, and staffing needs.
  - Team members review estimates, flag risks, and surface dependencies.
  - EMs and team **use comments in the planning issue to gauge capacity and connect their inputs**.
- **Deliverables:** Draft quarterly execution plan linked in the planning issue.

#### Step 4 – Epic Backlog Refinement (Weeks –2 to –1)

- **Owners:** EM + PM with Team Members
- **Actions:**
  - Take Epics identified for the upcoming quarter from the roadmap planning spreadsheet
  - For each Epic targeted for execution, ensure it contains:
    - **MVC Scope** clearly defined
    - **Business Case / Rationale** documented
    - **Link to high-level design** included
    - **Estimated level of complexity** assessed with engineering input
  - Move Epics through status progression: **Triage → Proposal → Ready**
  - Use planning issue comments for Epic refinement discussions and decisions
  - Team members provide technical input on complexity estimates and design feasibility
  - Pull in different stakeholders as necessary to complete Epic information
- **Deliverables:** Refined Epic backlog with all targeted Epics in "Ready" status for immediate quarter execution.

#### Step 5 – Finalization (Week 0)

- **Owners:** EMs with PM + stakeholders
- **Actions:**
  - Async review in the planning issue.
  - Confirm scope, capacity allocation, and commitments.
  - Verify Epic backlog refinement completion and readiness status.
  - Post summary comment with final decisions, artifacts, and links.
  - EA EMs and PM use the **weekly Tuesday conversation** to review outstanding issues, flags, and confirm quarterly decisions.
- **Deliverables:** Locked-in quarterly plan with refined Epic backlog; planning issue closed.

#### Step 6 – Quarterly Kickoff (Week 1)

- **Owners:** EMs with PM + stakeholders
- **Actions:**
  - PM and EMs collaborate on Dedicated Stage Kickoff Presentation (example: [FY26Q3 Dedicated Kickoff](https://docs.google.com/presentation/d/1LBtzebORZ57CGZ6nqJzpF642WPm8iyDmIUU5OV_xjPs/edit?slide=id.g3160fa3b0e2_0_3#slide=id.g3160fa3b0e2_0_3))
  - PM and EMs fill in previous quarter achievements and next quarter initiatives
  - PMs and EMs record kickoff video and share with the team
  - Engineers can immediately begin work on refined, "Ready" status Epics
- **Deliverables:** Quarterly kickoff presentation and recorded video for team alignment

#### Planning Inputs & Sources

- **Strategic:** Company Interlock Process, PM Roadmap, Customer Commitments
- **Operational:** KTLO Work, On-call Capacity, Unplanned Work, Engineering Backlog
- **Epic-Level:** Technical designs, complexity assessments, MVC definitions

#### Planning Artifacts & Templates

- [FY26 Dedicated Stage Roadmap Planning](https://docs.google.com/spreadsheets/d/1LLxUPQ5nEaC7J9OcO3TbG1txVOtAQdtqKbkJYa4Jcvk/edit?gid=0#gid=0)
- Customer Roadmap Presentation
- [Company Interlock Tracker](https://docs.google.com/spreadsheets/d/10ZMNCQVhRTbELC1WV4Zme5uEOrmOH3WMqYC1sHdJouw/edit?gid=2120796429#gid=2120796429)
- [EA Continuous Planning Spreadsheet](https://docs.google.com/spreadsheets/d/10ZMNCQVhRTbELC1WV4Zme5uEOrmOH3WMqYC1sHdJouw/edit?gid=0#gid=0)
- Epic Refinement Checklist (MVC, Business Case, Design, Complexity)
- Epic Status Tracking (Triage/Proposal/Ready)

#### Roles & Responsibilities

| Role | Responsibilities |
|------|-----------------|
| Product Managers | Maintain strategic roadmap, update planning spreadsheet, provide Epic business cases |
| Engineering Managers | Lead capacity assessment, update EA Continuous Planning, create/manage planning issue, coordinate Epic refinement |
| Team Members | Review estimates, flag risks, surface dependencies, provide Epic complexity input, async discussion |
| Stakeholders | Provide feedback, validate commitments during Finalization, contribute to Epic refinement as needed |

#### Success Metrics

- Planning issue closed by Week 0
- Team acknowledges capacity commitments
- Strategic priorities mapped to execution capacity
- Dependencies and risks clearly identified
- All targeted Epics in "Ready" status with complete refinement information
- Balanced allocation: Feature Dev 40–60%, KTLO 20–30%, On-call 15–25%, Engineering 10–15%

#### Risks & Best Practices

- Assume 10–20% buffer for reactive/unplanned work
- Document discussions inside the planning issue
- Link spreadsheets and roadmaps for traceability
- Review past quarter's unplanned work
- Balance strategic work with operational requirements
- Complete Epic refinement before quarter start to enable immediate engineering execution
- Ensure Epic complexity estimates include engineering perspective

#### Backlog Refinement Integration

This quarterly planning process includes integrated Epic backlog refinement during **Step 4 (Weeks -2 to -1)**. The refinement ensures that engineers can quickly get started on an Epic once it's ready to be picked up during the quarter execution phase.

Having this set of refined epics helps us plan for the upcoming quarter with accurate capacity estimates and allows engineers to immediately begin productive work when the quarter begins, rather than spending time on discovery and scoping activities.

</details>

#### Reviewer roulette

Reviewer roulette is an internal tool for use on GitLab.com projects that randomly picks a maintainer + reviewer.  Environment Automation uses it to spread the MR review workload.  To do so:

1. Go to the [reviewer roulette](https://gitlab-org.gitlab.io/gitlab-roulette/?currentProject=environment-automation&sortKey=stats.avg30&mode=show&order=-1) page.
1. Click on `Spin the wheel`.

See the [full MR process](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/#merge-requests).

#### Epic Status Updates

Each week, automation will prompt DRIs for progress updates on their epics; these are collated and collectively reviewed by the business.  DRIs should [ensure their Epics are configured correctly](https://gitlab.com/gitlab-com/gl-infra/epic-issue-summaries#child-epics) and updates written or delegated in case of PTO.

#### Example responses

Here are some concrete examples of responses to capacity planning alerts.

- Removing a metric from capacity planning - [Advanced search memory pressure](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/instrumentor/-/merge_requests/3322)
  does not follow long-term trends and was not a useful prediction.
  It remains a metric that is _alerted_ on if it exceeds practical limits.
- Remove saturation metric entirely - [kube_pool_cpu](https://gitlab.com/gitlab-com/runbooks/-/merge_requests/7412)
  was incorrect in many cases, and difficult to get right.
  It needed to be replaced with a different saturation metric (node-based CPU).
- Add Saturation metrics - [Kubernetes PVCs](https://gitlab.com/gitlab-com/runbooks/-/merge_requests/7411)
  were not being monitored at all, leading to near-miss incidents
- Fix the saturation metric - [Advanced search disk](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/instrumentor/-/merge_requests/3331)
  was inaccurate and needed to be replaced with better promql expressions
