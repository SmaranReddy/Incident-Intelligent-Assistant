---
title: Work Refinement and Estimation
description: "Continuous refinement process and estimation guidelines for Growth teams"
---

[Epics](https://docs.gitlab.com/ee/user/group/epics/) organize issues and sub-epics that share a strategic theme, enabling higher-level planning and roadmap building. Epics are defined at the group level and can contain multiple issues and child epics.

[Issues](https://docs.gitlab.com/ee/user/project/issues/) exist within projects and represent individual units of work with clear deliverables. An issue can only belong to a single epic, creating a clear hierarchy from strategic themes down to executable tasks.

## Workflow

Growth follows the GitLab [Product Development Flow](/handbook/product-development/how-we-work/_index.md), supported by a series of automations that we’ve fine-tuned to improve clarity during refinement, streamline prioritization, and speed up experiment delivery.

The refinement process reduces chaos by creating natural quality gates. Epics must be refined and broken down before issues enter the pipeline. Issues must be estimated and clarified before they're ready for development. This structured flow prevents half-baked ideas from consuming engineering time and ensures that when someone picks up work, they have everything needed to succeed. For a team running experiments with uncertain outcomes, this upfront investment in clarity pays dividends by allowing engineers to move fast on execution rather than spending cycles figuring out what to build.

The refinement process is driven by [triage bot automations and policies](https://gitlab.com/gitlab-org/quality/triage-ops/-/blob/master/policies/groups/gitlab-org/growth/refine-automations.yml) to ensure that it's smooth and consistent.

Note: The Growth team currently operates as a single, unified team following a kanban-style workflow. As a result, any process / automated workflows or nudges are designed to target the broader Growth Engineering stage rather than individual groups.

## Epic-Level Workflow and Refinement

**Problem Validation** (`~"workflow::problem validation"`)

- Product Manager validates that the problem is clearly defined, worth solving, and free of blockers
- Product Manager maintains Epic ownership during this phase

**Design** (`~"workflow::ready for design"`) - **Skip for Epics without UX/UI requirements**

- Product Designer creates a UX issue under the Epic
- Designer iterates on designs and gathers early feedback
- Designer refines designs based on discussion
- When the scope is clear and the discussions have been addressed:
  - Designer documents final design in UX issue body
  - Designer links the UX issue in the parent epic description

**Refinement** (`~"workflow::refinement"`)

- PM moves the Epic to the refinement phase labelling the Epic with `~"workflow::refinement"`
- Triage-bot generates a refinement thread titled "🚧 Epic Refinement" daily at 00:00 UTC. Will switch to reactive generation after labelling when [Epic webhook payload issue](https://gitlab.com/gitlab-org/gitlab/-/issues/584285) is resolved. To see the automations in action, check out this [demo](https://youtu.be/EfjPzFI2bT0?si=GiuJ7mbYxy3yuosq).
- Team evaluates technical approach, identifies risks, and surfaces dependencies in the thread
- Engineers may volunteer to become the Engineering DRI at any time during refinement by self-assigning the Epic
- If refinement uncovers significant complexity, technical constraints, or timeline conflicts with existing commitments, engineers must surface these immediately and collaborate with PM/Design to adjust scope, approach, or timeline before proceeding. Product Management and Engineering must acknowledge and address these concerns before the Epic can advance
- Refinement concludes when the thread receives at least 3 ✅ reactions in the refinement thread

**Planning Breakdown** (`~"workflow::planning breakdown"`)

- Epic transitions automatically to this phase by triage-bot after refinement completion
- Designer closes UX issue
- If no engineer volunteered as Engineering DRI during refinement, the Epic awaits for a volunteer for 3 business days
- After 3 business days without an Engineering DRI volunteer, triage-bot notifies Engineering Manager for assignment
- Engineering DRI decomposes Epic into implementation issues using [Experiment Implementation](https://gitlab.com/gitlab-org/gitlab/-/issues/new?description_template=Experiment%20Implementation) or [Implementation](https://gitlab.com/gitlab-org/gitlab/-/issues/new?description_template=Implementation) templates
- Issues with a clearly defined scope do not require issue-level refinement. These will be labeled `~"workflow::scheduling"` and include weight estimates. This helps reduce refinement noise when the scope is already established at the epic level.
- Issues requiring clarification labeled with `~"workflow::refinement"` for [issue refinement process](#issue-level-refinement)
- Apply `~"workflow::blocked"` labels with "blocking/blocked by" relationships to indicate sequential dependencies

**Ready for Development** (`~"workflow::ready for development"`)

- Once all issues are created, labeled, and refined, Engineering DRI advances Epic to `~"workflow::ready for development"`
- This indicates breakdown completion and readiness to commence work

**In Development** (`~"workflow::in dev"`)

- Engineering DRI transitions Epic when implementation begins on constituent issues
- Establish a due date to communicate expected delivery timeframe
- Engineering DRI maintains Epic ownership throughout implementation
- DRI coordinates cross-issue work, manages dependencies, and provides weekly status updates
- DRI ensures consistent progress until all issues reach completion

**Verification** (`~"workflow::verification"`)

- Engineering DRI transitions Epic to this phase upon implementation completion
- the Engineering DRI determines the appropriate verification level - either mentioning the PM and closing for straightforward completions, or requesting formal PM verification for complex changes.

{{% alert title="⚠️ Note" color="warning" %}}
Being the engineering DRI for an Epic doesn't mean you have to implement every issue in it. The DRI is a coordinator, not a sole implementer. Other engineers pick up issues from the kanban board and contribute to the Epic - the DRI ensures the work fits together and the Epic reaches completion.
{{% /alert %}}

## Issue-Level Refinement

1. Issues are moved from `~"workflow::planning breakdown"` to `~"workflow::refinement"` automatically by the triage bot in order of priority (from top to bottom). The bot will only move issues to refinement if there is room in refinement column, meaning there is less issues than maximum limit for this column. This is first chance for PMs to prioritize issues by moving them higher in the `planning breakdown` column. After the issue is moved to refinement, a dedicated `refinement thread` is created, which acts as a place for discussion and weight estimation.
   - 💡 Hint: In rare case when an issue has to be expedited, it's possible to move it to refinement manually by adding the `~"workflow::refinement"` label. This will invoke a reaction from triage bot, which will add `refinement thread` for such issue instantly so the refinement can proceed the same way as with automated path.
2. During refinement the team ensures that the issue is well described and requirements are clear. They can use the `refinement thread` to discuss but they should make sure that any changes and decisions made there are also reflected in issue's description. Once each engineer is comfortable with the way the issue is described, they can vote their estimation of weight based on our [guidelines](#estimation-guidelines). The voting happens by reacting to the thread with one of few possible weight estimates: 1️⃣ 2️⃣ 3️⃣ 5️⃣ or 🚀 (indicates 5+).
3. Each day the triage bot checks all issues in `~"workflow::refinement"` column and if an issue has required minimum number of estimation votes (see `MIN_REACTIONS` constant [here](https://gitlab.com/gitlab-org/quality/triage-ops/-/blob/master/lib/growth_refine_automation_helper.rb?ref_type=heads#L16) for the current setting) it will be moved to `~"workflow::scheduling"`.
   - 💡 Hint: If there is some problem with the issue and it shouldn't be moved forward even if enough engineers estimate it, ❌ reaction can be added to the thread which will stop the bot from transitioning the issue to `~"workflow::scheduling"` as long as this reaction sticks to the thread. This means that whoever put it is also responsible for removing it once the problem is gone.
4. Once the issue is in `~"workflow::scheduling"`, it is awaiting final prioritization by PMs - it has to be manually moved to `~"workflow::ready for development"` depending on the current priorities. This part of the process is PMs responsibility. This allows for additional fine-tuning of priorities and acts as a buffer for our ready for development column.

## Estimation Guidelines

[The development estimation is the time during `~"workflow::in dev"` until moved to `~"workflow::in review"`]

| Weight | LoE (Business Days) | Description |
| ------ | ------ | ------ |
| 1 | 1-3 days | The simplest possible change. We are confident there will be no side effects. |
| 2 | 4-6 days | A simple change (minimal code changes), where we understand all of the requirements. |
| 3 | 7-9 days | A simple change, but the code footprint is bigger (e.g. lots of different files, or tests affected). The requirements are clear. |
| 5 | 10-12 days | A more complex change that will impact multiple areas of the codebase, there may also be some refactoring involved. Requirements are understood but you feel there are likely to be some gaps along the way. |
| 5+ | 2 weeks+ | A significant change that may have dependencies (other teams or third-parties) and we likely still don't understand all of the requirements. It's unlikely we would commit to this in a milestone, and the preference would be to further clarify requirements and/or break into smaller issues. |

- LoE => Level of Effort represents the total number of business days spent across both `workflow::in dev` and `workflow::review` phases.

In planning and estimation, we value [velocity over predictability](/handbook/engineering/development/principles/#velocity). The main goal of our planning and estimation is to focus on the [MVC](/handbook/values/#minimal-valuable-change-mvc), uncover blind spots, and help us achieve a baseline level of predictability without over optimizing. We aim for 70% predictability instead of 90%. We believe that optimizing for velocity (merge request rate) enables our Growth teams to achieve a [weekly experimentation cadence](/handbook/product/groups/growth/#weekly-growth-meeting).

- If an issue has many unknowns where it's unclear if it's a 1 or a 5, we will be cautious and estimate high (5).
- If an issue has many unknowns, we can break it into two issues. The first issue is for research, also referred to as a [Spike](https://en.wikipedia.org/wiki/Spike_(software_development)), where we de-risk the unknowns and explore potential solutions. The second issue is for the implementation.
- If an initial estimate is incorrect and needs to be adjusted, we revise the estimate immediately and inform the Product Manager. The Product Manager and team will decide if a milestone commitment needs to be adjusted.

## Team Participation in Refinement

Operating asynchronously means refinement can't rely on scheduled meetings where everyone shows up at the same time. Instead, the team should adopt a continuous refinement mindset where engineers regularly check the [growth Epic board](https://gitlab.com/groups/gitlab-org/-/epic_boards/2079888) and issue kanban board for items in refinement status. When an Epic appears in ~"workflow::refinement", engineers should review the refinement thread, ask clarifying questions, evaluate technical feasibility, and provide feedback on the proposed direction. Engineers who develop interest and context during refinement are encouraged to volunteer as the Engineering DRI by self-assigning the Epic. Once satisfied with the refinement, engineers add a ✅ reaction to the refinement thread to signal completion. This isn't a passive activity - the goal is to surface concerns, suggest alternatives, ensure the Epic is well-understood, and ideally volunteer to own the breakdown and implementation.

Similarly, the issue kanban board should be monitored for issues in ~"workflow::refinement", ~"workflow::scheduling", and ~"workflow::ready for development". Issues in refinement need estimation votes and technical feedback. Issues in scheduling are waiting for final prioritization but are already well-defined and could be moved to ready for development if priorities shift. Issues in ready for development are immediately available for pickup. By regularly scanning these columns, engineers maintain awareness of upcoming work, can identify issues that align with their expertise or interests, and keep the pipeline moving.

Currently, the team should prioritize any work labeled with ~"Growth::Driving First Orders". These represent high-priority items that need immediate attention.

## Issue sequencing

In order to convey Issue implementation order and blocking concepts, we leverage the [blocking issue linking feature](https://docs.gitlab.com/ee/user/project/issues/related_issues.html#blocking-issues).

More on the discussion can be seen in https://gitlab.com/gitlab-org/growth/team-tasks/-/issues/752.

## Labelling Issues & Epics

We use workflow boards to track issue progress throughout a milestone. Workflow boards should be viewed at the highest group level for visibility into all nested projects in a group.

The Growth stage uses the `~"devops::growth"` label and the following groups for tracking merge request rate and ownership of issues and merge requests.

| Name          | Label                   | gitlab-org | All Groups |
| ----------    | -----------             | ------     | ------     |
| Growth        | `~"devops::growth"`     | [Growth Workflow](https://gitlab.com/groups/gitlab-org/-/boards/4152639) | [-](https://gitlab.com/dashboard/issues?scope=all&utf8=%E2%9C%93&state=opened&label_name[]=devops%3A%3Agrowth) |
| Acquisition   | `~"group::acquisition"` | [Acquisition Workflow](https://gitlab.com/groups/gitlab-org/-/boards/4152639) | [-](https://gitlab.com/dashboard/issues?scope=all&utf8=%E2%9C%93&state=opened&label_name[]=devops%3A%3Agrowth&label_name[]=group%3A%3Aacquisition) |
| Activation    | `~"group::activation"`  | [Activation Workflow](https://gitlab.com/groups/gitlab-org/-/boards/4152639) | [-](https://gitlab.com/dashboard/issues?scope=all&utf8=%E2%9C%93&state=opened&label_name[]=devops%3A%3Agrowth&label_name[]=group%3A%3Aactivation) |
| Engagement    | `~"group::engagement"`  | [Engagement Workflow](https://gitlab.com/groups/gitlab-org/-/boards/4152639) | [-](https://gitlab.com/dashboard/issues?scope=all&utf8=%E2%9C%93&state=opened&label_name[]=devops%3A%3Agrowth&label_name[]=group%3A%3Aengagement) |
| Experiments   | `~"experiment-rollout"` | [Experiment tracking](https://gitlab.com/groups/gitlab-org/-/boards/1352542) | [-](https://gitlab.com/dashboard/issues?scope=all&utf8=%E2%9C%93&state=opened&label_name[]=experiment-rollout) |
| Feature Flags | `~"feature flag"`       | [Feature flags](https://gitlab.com/groups/gitlab-org/-/boards/1725470?&label_name[]=devops%3A%3Agrowth&label_name[]=feature%20flag) |  |

Growth teams work across the GitLab codebase on multiple groups and projects including:

- The [gitlab.com/gitlab-org](https://gitlab.com/gitlab-org/) group
- [gitlab](https://gitlab.com/gitlab-org/gitlab)
- [GLEX](https://gitlab.com/gitlab-org/ruby/gems/gitlab-experiment)
- [customers-gitlab-com](https://gitlab.com/gitlab-org/customers-gitlab-com)
- The [gitlab.com/gitlab-com](https://gitlab.com/gitlab-com/) group
- [about.gitlab.com](https://gitlab.com/gitlab-com/marketing/digital-experience/about-gitlab-com)
