---
title: "AI Coding Group Milestone Planning"
description: "Describes the milestone planning process for the AI Coding Group."
---

At GitLab and on the AI Coding team, we believe in a structured yet flexible approach to milestone planning to ensure that our engineering and product teams are aligned, focused, and able to deliver high-quality work efficiently. This page outlines our process for planning and preparing for milestones.

## Issue Creation

When creating a new issue, add as many details as possible during initial creation when knowledge is fresh. [Assume low context](/handbook/communication/#low-context) for anyone reading or picking up the issue, and write down any information you think would be relevant.

Label new issues with “workflow::refinement” if they need further discussion or estimation.

Follow this template in the description:

```markdown
## Context

_Summary or background of the feature, feature change, bug, or investigation. If applicable, include subsections such as "Further Details", "Scope", etc._

## References and Resources

_Include any links to documentations, other issues, or MRs that you think may be useful to the assignee. If applicable, indicate a domain expert or resource person/group._

## Proposal

_Proposed/suggested solution or any guidance related to it. For investigations or spikes, this may describe an expected outcome or expected output._

## Definition of Done / Acceptance Criteria

_List the specific conditions that must be met for this issue to be considered complete._
```

## Issue Refinement

The goal of issue refinement is to have engineers review issues to ensure they are ready for development and provide lightweight estimates.
Use the [template checklist](/handbook/engineering/ai/ai-coding/how-we-work/milestone_planning/#refinement-template) in the next section as a guideline.

1. Initial Labeling:
    - When a new issue is created or identified as a potential priority, it should be labeled with "workflow::refinement"
    - This label indicates that the issue needs refinement/triaging before it's ready for development
2. Refinement Timing and Capacity:
    - Refinement work is considered equal priority to development work
    - Engineers should plan to spend approximately 20% of their time on refinement activities
    - Focus on refining issues for the next 1-2 milestones to balance advance planning with changing requirements
    - Avoid leaving all refinement work to the end of a milestone
3. Asynchronous Refinement Process:
    - Team members, regardless of their timezone, can review issues with the "workflow::refinement" label at their convenience
    - Use the issue comments for discussion and clarification. You may loop in other team members or domain experts from outside the team to provide further context and/or give feedback on a proposed solution.
    - Individual team members are encouraged to schedule regular, asynchronous backlog refinement sessions on their calendars
    - Use the [AI Coding Refinement Board](https://gitlab.com/groups/gitlab-org/-/boards/8994789) to make issues needing refinement easy to find
    - Make sure to timebox your refinement efforts. Remember that the goal of refinement is not to provide a detailed solution, but to provide clarity on the problem and to provide a starting point on the solution. See [Where to Draw the Line Between Refinement and Implementation](#where-to-draw-the-line-between-refinement-and-implementation).
4. Refinement Tasks:
    - Team members should focus on the following when refining an issue:
        - Validate the problem is clear and aligned with roadmap priorities
        - Ensure issue labels are accurate
        - Clarify the issue title, description and requirements (including the `Context`, `Proposal`, and `Acceptance Criteria` sections)
        - Break down the issue into smaller, manageable tasks if needed
        - Identify and document any dependencies or blockers
        - Estimate the effort required and assign a corresponding weight to the issue. Refer to [issues estimation](#issues-estimation) section for guidance.
5. Refinement Completion:
    - Once a team member believes the issue has been sufficiently refined, they should:
        - Post the completed refinement checklist as a comment on the issue
        - Ensure that the issue has the appropriate weight
        - Add the "workflow::ready for development" label and remove the "workflow::refinement" label

### Refinement Template

While refining an issue, copy the below template into a comment and work through each todo item. Once you've completed refinement, post the comment to document your refinement work:

```markdown
## Refinement Checklist

- [ ] Validated the problem is clear and aligns with team priorities, engaged in any discussions with the PM or other stakeholders as needed
- [ ] Updated title for clarity (if needed)
- [ ] Updated labels for proper categorization (if needed)
- [ ] Added or improved the "Context" section with necessary background information
- [ ] Added or improved the "Proposal" section with potential implementation approach or suggested ideas to further investigate
- [ ] Added or improved the "Acceptance Criteria" or "Definition of Done" section
- [ ] Identified dependencies or blockers (if any)
- [ ] Considered if the issue should be broken down into smaller tasks
  - [ ] Created child issues if needed (link them here)
- [ ] Assigned a weight using [issue estimation](/handbook/engineering/ai/ai-coding/milestone_planning/#issues-estimation)

/weight X
/label ~"workflow::ready for development"
```

### Where to Draw the Line Between Refinement and Implementation

During refinement, focus on providing enough context and direction for someone else to implement the issue efficiently. While some investigation is often necessary during refinement, consider these guidelines:

- **Refinement**: Identifying the problem, proposing a general approach, providing helpful context, estimating effort, and setting success criteria.
- **Implementation**: Writing specific code, determining exact file locations, making detailed technical decisions, and executing the solution.

If your refinement work leads you to start implementing the solution, consider either:

1. Assigning yourself to the issue and transitioning to implementation
2. Documenting your findings without implementing, so someone else can benefit from your research

Remember that the goal of refinement is to make issues clear enough that any team member could pick them up and know what to do, without prescribing every implementation detail.

## Milestone Planning

Once we have a sufficient pool of refined issues, we can proceed with milestone planning:

1. Capacity Planning:
    - The Engineering Manager calculates available capacity for the upcoming milestone using:
        - Planned time off
        - Historic velocity
        - Allocated refinement time
        - Other non-development activities (meetings, reviews, etc.)
2. Carryover Work:
    - Identify any issues that will not be completed in the current milestone
    - Adjust weights to reflect remaining work
    - Consider impact on next milestone's capacity
3. Refinement Prioritization:
    - The Engineering Manager and Product Manager identify and prioritize issues that need refinement for the upcoming 1-2 milestones
4. Issue Selection:
    - The Product Manager defines strategic priorities
    - Creates a prioritized list of issues for the upcoming milestone
    - Focus on issues that have already been refined when possible. These are issues with label "workflow::ready for development" and a weight.
    - If the PM identifies an issue that must go in the next milestone but is not yet labelled as "workflow::ready for development", they must flag that as early as possible and loop in a team member to refine the issue.
5. Issue Labels
    - The Product manager adds the correct `code-creation-priority` labels to indiciate the importance of each issue
    - The Engineering Manager adds the ["Deliverable" label](/handbook/product-development/how-we-work/product-development-flow/#required-labels) to items we commit to focusing on in this milestone
    - Mark issues that exceed predicted capacity as "Stretch"

## Role of Team Members

Engineers

- Actively participate in issue refinement activities
- Provide insights and recommendations on technical feasibility and implementation approach
- Share domain knowledge and expertise during refinement discussions
- Self-manage refinement work to meet the 20% time allocation target

Engineering Manager:

- Calculates team capacity and manages milestone planning
- Works with PM to prioritize upcoming issues
- Ensures refinement activities are properly distributed across the team

Product Manager:

- Defines strategic priorities for upcoming milestones
- Works with EM to prioritize upcoming issues
- Ensures issue requirements and acceptance criteria are clear
- Works with EM to identify future refinement needs for upcoming work

## Issues Estimation

Here are the weights we use for issue estimation:

{{% include "includes/engineering/create/weight_table.md" %}}

A weight of 5 generally indicates the problem is not clear or a solution should be instead converted to an Epic with sub-issues. In these cases we may want to create a spike issue, which is an issue that is used to investigate and clarify the scope of the original issue. That spike issue can be used to help break down the original issue.

## What To Work On

### Work Prioritization

Issues are prioritized using a combination of two label types:

- The `Deliverable` label indicates work we commit to completing in the current milestone
- The `code-creation-priority` labels (high, medium, low) indicate the strategic importance of the work

The priority order for work is:

1. Issues labeled as both `Deliverable` and `code-creation-priority::high`
2. Other `Deliverable` issues (regardless of their priority level)
3. Non-deliverable issues based on their priority:
    - `code-creation-priority::high`
    - `code-creation-priority::medium`
    - `code-creation-priority::low`

### What to Work on First: Deliverables

- **Priority**: Deliverables are considered the highest priority and should be completed before the end of the milestone, aligning with our iteration cycle and monthly release schedule.
- **Self-Assignment**: Engineers can pick any issue marked with the ["Deliverable" label](/handbook/product-development/how-we-work/product-development-flow/#required-statuses-1) and assign it to themselves, making these issues the top priority for the milestone. Issues for the current milestone can be found on the [AI Coding workflow board](https://gitlab.com/groups/gitlab-org/-/boards/10081212).
- **Importance**: It's crucial for engineers to make a concerted effort to complete these deliverables within the cycle, and to promptly communicate any obstacles to their engineering manager.
- **Communication and Flexibility**: Unexpected challenges can arise, potentially impacting the completion of a deliverable. Early communication of these challenges allows for possible solutions, such as scope reduction or reassignment. If a deliverable cannot be completed within the cycle, it is moved to the next cycle for completion. This scenario is an opportunity for both the engineer and the engineering manager to reflect and learn from the experience.

### What to Work on Next

- **Stretch Issues**: After fulfilling deliverable responsibilities and other activities, engineers can use any remaining time to work on Stretch issues. These are lower priority and not expected to be completed within the current iteration cycle but are potential deliverables for the next cycle. Early progress on these issues is considered a bonus.
- **Other Ideas**: Engineers are encouraged to use spare time to contribute to any work they believe will significantly benefit the product or company. As the [general guidelines state](/handbook/values/#dont-wait), "we recognize that inspiration is perishable, so if you're enthusiastic about something that generates great results in relatively little time feel free to work on that." While pursuing such initiatives, engineers should ensure proper issue assignment and consider sharing their work in relevant channels for visibility and collaboration.
- **Responsibility over Rigidity**: We expect people to be [managers of one](/handbook/values/#efficiency) and prefer [responsibility over rigidity](/handbook/values/#efficiency), so there's no need to ask for permission if you decide to work on something that's not on the issue board, but please keep your other responsibilities in mind, and make sure that there is an issue, you are assigned to it, and you share it with the team.

## Workflow and Label Management

- **Label Updates**: It's crucial for engineers to update the ["workflow" labels](/handbook/product-development/how-we-work/product-development-flow/#build-track) accurately as issues progress from "ready for development" to "in dev" to "in review."
- **Responsibility**: Engineers are responsible for keeping the workflow labels up to date on all issues they are working on during the milestone.

This milestone planning process is designed to maximize our team's effectiveness by ensuring that everyone is aligned on priorities and responsibilities. It fosters a collaborative environment where every team member contributes to the planning process, leading to a more organized and productive development cycle.

## Planning Metrics and Continuous Improvement

In order to measure the efficiency of this process we should track and regularly review the metrics

We track metrics such as:

- Average time an issue spends in refinement
- Number of issues refined per week
- Time spent on refinement activities vs. development

We also gather feedback from team members during retrospectives on the process.
