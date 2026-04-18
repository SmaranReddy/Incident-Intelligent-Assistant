---
title: "Test Governance Group's Workflows"
description: "This page contains group Test Governance's official workflows"
---

## Test Governance Workflows

![workflows.png](/images/engineering/infrastructure-platforms/developer-experience/test-governance/workflows.png)

### Workflow labels definitions**

These definitions are based on guidelines of [The Infrastructure Platforms Project Management](/handbook/engineering/infrastructure-platforms/project-management/#labels)

#### workflow-infra::Triage

- **Purpose:** Entry point for all new issues.
- **What goes here:**
  - All new [Request for Helps (RFH) issues](https://gitlab.com/gitlab-org/quality/test-governance/request-for-help/-/issues) created by team members from other groups and departments
  - All new [team-task issues](https://gitlab.com/gitlab-org/quality/test-governance/request-for-help/-/issues) created by Test Governance team members or anyone in DevEx
- **Organization:**
  - Issues with assigned priority should be moved to **workflow-infra::Proposal**
  - No issue should have more than one workflow label
  - No issue with a priority label should remain in Triage column
  - Currently, no mechanism exists to auto-assign this label at issue creation.
- See #processes-and-their-frequencies section for detailed process information.

#### workflow-infra::Proposal

- **Purpose:** Where the team begins the refinement process.
- **Organization:**
  - Issues are organized by priority in descending order
  - Issues with the same priority are ordered by creation date (oldest first)
  - Refined issues should be moved to **workflow-infra::Ready**
- See #processes-and-their-frequencies section for detailed refinement process.

#### workflow-infra::Ready

- **Purpose:** Contains only refined issues that are already refined.
- **Organization and rules:**
  - If an issue needs to be broken down into multiple child items during refinement, it is promoted to an epic. Only child items appear in this column.
  - Issues with the same parent epic should be grouped together
  - Issues with assignees that won't be worked on soon stay at the top and organized by priority
  - Issues without assignees follow below, also organized by priority
  - Manager and project lead are responsible for reorganizing issue order as necessary
- **When picking up an issue:** Team members should:
  - Assign themselves (if not already assigned)
  - Set start date and due date based on time estimation (adjustable as work progresses)
  - Move issue to **workflow-infra::In Progress**
- **Note:** Issues with assignees always stay on top because they've been planned for the current quarter. When team members complete all assigned issues, they should pick the next unassigned issue from the list. If that issue has a future quarter label, update it to the correct label and notify the manager or project lead to update the quarter planning issue.

#### workflow-infra::In Progress

- **Purpose:** Contains only issues with assignees that are actively being worked on.
- **Capacity guidelines:**
  - Total issues in this column should never exceed 3 times team size
  - No team member should have more than 3 open In Progress items simultaneously
  - These limits ensure team productivity, morale, and wellness
- **Assignee responsibilities:**
  - Once linked MR is ready for review with an assigned reviewer, move corresponding issue to **workflow-infra::Under Review**
  - Once MR is merged, move to either:
    - **workflow-infra::Verify** if monitoring/verification is required
    - **workflow-infra::Done** if implementation is confirmed working (issue can be closed)
- **Additional actions:**
  - If work can be demoed: Consider scheduling for the next demo session
  - If follow-up work is needed: Create separate issues with appropriate priority labels and move to the appropriate column
  - If blocked: Move to **workflow-infra::Blocked** and communicate with project lead or manager to determine next steps.
  - If additional MRs are needed: Create new linked issues with appropriate labels
  - If reassigned to higher priority work (likely P1): Move current issue to **workflow-infra::Stalled** and return to it after completing the higher priority issue, or work with manager to re-prioritize/re-assign
- **Tip:** Consider picking up a new issue from **workflow-infra::Ready** once your current issue moves to Under Review.

### Priority labels definitions

- **Gitlab's Production P1**: Absolutely critical. These are normally production incidents that affect customers or deployments in real time. They take precedence over all other P1. Therefore, must be assigned and worked on immediately.
- **Team’s priority::1 (P1)**: Very important. The completion of these issues directly affects the outcome of or unblock another high priority project that directly impacts the business. Must be assigned, refined and worked on as soon as possible. Expected completion within the same quarter or by project deadline, whichever comes first.
- **priority::2 (P2)**: Important to the team’s mission and roadmap. To be refined and assigned quarterly after all P1 are assigned and actively worked on. Can be completed within 1 \- 2 quarters.
- **priority::3 (P3)**: Helpful to have. Can be considered within the next 3 \- 4 quarters or sooner if time and resources allowed or promoted to P2.
- **priority::4 (P4)**: Nice to have. No plan to get to within a fiscal year but can be promoted to higher priority if time and resources allowed.

### Processes and their frequencies

#### Triage

- **Why:** To evaluate and sort issues/requests by importance or urgency, determining which require immediate attention by adding appropriate priority labels.
- **When:** To be decided by EM and/or Staff Engineer
- **Who:** EM and/or Staff Engineer
- **What:** Newly created issues and/or RFHs
- **How:**
  - Identify issues to reject or redirect to different teams. Apply **workflow-infa::Cancelled** label and close rejected issues
  - Assign priority to accepted issues and move to **workflow-infra::Proposal** in descending order (P2, then P3, then P4)
  - Move P1 issue directly to **workflow-infra::Proposal** for immediate refinement (no triage needed). P1 refinement should happen ASAP, ideally within the same week. P1 issue should always be at the top of any column it stays in.
  - [Assign DRI](/handbook/engineering/infrastructure-platforms/developer-experience/#starting-a-new-project) who owns the success of their project. DRI can ping other members of the team or other teams for consultations in order to make this decision.

#### Refinement

- **Why:** To maintain a prioritized, clear, and detailed product backlog that ensures the team works on valuable items, reduces ambiguity, and fosters continuous collaboration and improvement. Also, to ensure a healthy pool of work that is ready to be worked on when a team member becomes available for new tasks.
- **When:** every week alternating between sync (2 sessions to cover all timezones) and async
- **Who:** Test Governance team
- **What:** Epics, Issues
- **How:**
  - Meeting agendas will be created and populated with issues and epics to be refined each week. During refinement sessions, the team selects issues to work on and breaks down tasks for any issue sized Small or larger. The goal is for one issue to require only one MR to close. More than one MR indicates the need for multiple issues. This is to honor the [iteration](/handbook/engineering/workflow/iteration/) value at Gitlab.
  - Definition of Ready (DoR) **\-** A refined issue must have:
    - One clear task with a definition of done
    - Weight assigned using Fibonacci sequence: 1, 2, 3, 5, 8, 13 (to be refined by the team over time)
    - Alignment among all team members involved in implementation on the solution and outcome.
  - During refinement, if the team determines that an issue has dependency that is a blocker, move it to **workflow-infra::Blocked**. Clearly state what is needed to unblock it or collaborate with DRI, EM or Staff Engineer to resolve or unblock the issue.
  - Once a project has assignees, it does not need to be on the refinement agenda anymore. The project’s DRI(s) can decide on how to carry out and communicate throughout the project on their own.

#### Quarter planning

- **Why:** Quarterly planning is a strategic event during which the team members identify what they will be working on the next 3 months and in what order. It is intended to provide clarity and transparency, fostering collaboration, improving predictability and preventing the team from getting lost in the short-term.
- **When:** Quarterly
- **Who:** Test Governance team
- **What:** Epics, Issues
- **How:**
  - Ideally, there should be planning issues well in advance, but this is for the future once we have fully refined and adopted these new workflows.
  - The team will finalize the planning issue for the upcoming quarter during the last week of the current quarter.
  - Issues from the same epic should be considered in the same quarter. Likely, all child items of an epic inherit the epic’s priority.
  - During the last week of the current quarter, we will have a sync call to kick off planning for next quarter as well as reflecting on what has happened during this quarter (going over retro issue). Close the current quarter planning issue after this call.
  - Example planning issue: [https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks/-/issues/3930](https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks/-/issues/3930)

#### Retrospective

- **Why:** Retrospective is an event dedicated to foster continuous improvement by providing dedicated space for team members to reflect on their work, identify what went well, what didn’t and create actionable plans to enhance process and outcomes in future work cycles.
- **When:** Monthly but grouped by quarter
- **Who:** Test Governance team
- **What:** Epics, Issues, feedback, improvements, suggestions
- **How:**
  - Each quarter will have 3 corresponding retro issues. Each month in a quarter, we will start and conclude a retro issue for that month.
  - Team members will be assigned to the retro issue when the corresponding month starts.
  - Retro issue serves as a place to record thoughts, feelings, feedback, and complaints while they're fresh
  - At the end of each month, the corresponding retrospective issue will be summarized. The first 2 will be closed out and their summaries will be carried over to the last retrospective of the quarter. At the end of the quarter, we will have sync calls to go over all summaries, lessons learned and any actionable items to wrap up the quarter.
  - Note: Ad-hoc retro issues can be created as necessary.
  - Example retro issue: [https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks/-/issues/3931](https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks/-/issues/3931)

#### Team sync, knowledge sharing

- **Why:** To facilitate knowledge sharing, provide async updates, and build team connections.
- **When:** Bi-weekly
- **Who:** Test Governance team
- **What:** Knowledge sharing, team attention items, socialization
- **How:**
  - Use this meeting to sync on topics including (but not limited to):
    - How the team operates
    - Team commitments
    - Knowledge sharing
    - Additional topics team members want to bring to the team's attention.
  - Updates and comments are added asynchronously so live conversation can focus on items requiring real-time discussion.

#### Standup

- **Why:** To discuss progress and identify blockers.
- **When:** Daily
- **Who:** Test Governance team members with updates
- **What:** Important updates, asking for help
- **How:**
  - Team members receive a Slack notification to share updates in the dedicated slack channel.
  - It's okay to skip updates if nothing significant happened.
  - It's important to share if you're blocked or stuck and ask for help.

#### Pipeline DRI

- **Why:** A form of on-call duty for SETs that ensures high availability and prompt resolution of incidents and various issues to prevent significant financial or reputational damage.
- **When:** Full week rotation for SETs, depending on time zone.
- **Who:** Test Governance team member(s) on Pipeline DRI duty
- **What:** Scheduled pipelines that run on multiple environments
- **How:** See [Pipeline DRI responsibilities](/handbook/engineering/testing/oncall-rotation/#responsibility)
