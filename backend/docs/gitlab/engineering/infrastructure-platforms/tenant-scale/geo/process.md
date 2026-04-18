---
title: "Geo and Disaster Recovery - Planning"
description: "How the Geo Team Operates"
---

## Discussions

Discussions are documented in this [GitLab project](https://gitlab.com/gitlab-org/geo-team/discussions/-/issues/?sort=created_date&state=opened&first_page_size=100).

## Planning

### Hierarchy of Work Items

One aspect of planning that provides clarity for development execution is the breakdown of roadmap into a hierarchy of informational pieces that are organized in GitLab's own [work items](https://docs.gitlab.com/ee/development/work_items.html) as part of product management.

This section describes the approach that the Geo team takes to break down the requirements into work-in-progress items that engineers move through the workflow of implementation.

First, we describe general agile work item terminology to create alignment and use common industry vocabulary. We then map these general terms to the specific work item terminology used by GitLab.

Finally we provide some guidelines and rules of thumb when creating new ites and about choosing the right level of granularity for them.

The details of this are documented in our [Geo's Hierarchy of Agile Work Items](../agile-work-items.html) page.

### Kanban

We work in a continuous Kanban manner while still aligning with Milestones.

Epics and issues have the following lifecycle:

```mermaid
graph LR;
  Plan --> Schedule;
  Schedule --> Build;
  Build --> Release;
  Release --> Post-Release;
```

Three boards are used to monitor this process

- [Plan](https://gitlab.com/groups/gitlab-org/-/boards/1181258)
- [Schedule](https://gitlab.com/groups/gitlab-org/-/boards/981066)
- [Build](https://gitlab.com/groups/gitlab-org/-/boards/1181257)

#### Plan

We use the [Kanban Plan board](https://gitlab.com/groups/gitlab-org/-/boards/1181258)
to triage issues that are not yet planned for a milestone. Issues in this board are labeled "group::geo" and use the Statuses "Validation Backlog", "Problem Validation", "Solution Validation",  "Planning Breakdown", and "Refinement".

The Engineering Manager owns this board and uses it to advance issues through to becoming feasible Epics. When work leaves this
board, it is in a fit state for the engineering team to implement the issue.

In this stage, the EM establishes new work that they would like to see in the Geo feature. The planning process is as follows:

##### Problem validation

The EM tries to establish that the issue describes a problem with the Geo feature that is customer-relevant (internal and external),
or has some other clear merit (e.g. a technical improvement in the backend). The EM will draw on other group members, customers and the UX organisation where necessary.
If the problem is validated, the issue is moved forward to the next stage, otherwise it is closed with a short explanation on *why* the problem will not be considered. This explanation should also contain
a link [to the category strategy](/handbook/product/product-processes/#category-direction).

##### Solution validation

Once the problem is validated, the PM will consider creating a work item as per our guidelines for [hierarchy of work](../agile-work-items.html).
They will then reach out to the Engineering Manager(EM) for a technical contact in the team. The EM assigns [ownership](#work-ownership) of the work item to an engineer. The engineer will work with the PM
to determine a technical solution for the problem.

Occasionally, a proof-of-concept (POC) is necessary to determine a feasible technical path. When one is required, the PM
will create a POC issue that contains the context of the research to be conducted along with the goals of the POC. This
issue will be scheduled for work before any further breakdown of tasks is performed.

POC issues are also time-boxed with a due date applied to these items and should be labeled ~"POC". On the due date, the engineer is expected to
provide a comment on the issue regarding the outcome of the POC. Time-boxing these issues is meant to constrain the
scope of the work.

It is also important to note that not all POCs will be successful, and that is OK! Some avenues of research may not be
successful, and the POC will have saved us from investing significant time in a solution that will not meet our needs. The goal is to fail fast!

The PM and the engineer will work together to create all of the [work items](../agile-work-items.html) required to build the solution. These items account for

- Documentation
- Testing
- Post-release items such as rake tasks, migrations or monitoring issues

Ideally, [epics, features and user stories](../agile-work-items.html) should be broken down by external functionality rather than by implementation details. Even refactors and performance improvements should highlight the customer value and be described with a user-centric approach.

Post-release tasks that need to be performed by SRE's are created in the Infrastructure team project and cannot be
added to work items as related issues. For these, it's useful to list them in the highest level work item description.

When they are satisfied that they have a reasonable solution including an estimation of effort for each work item, this breakdown should be shared with the rest of the team
for discussion and feedback. Planning is transparent anyway, but this serves to notify the team that the new work items are ready
for development.

For issues that include a feature that uses a Feature Flag (FF), we follow the feature flag [lifecycle doc](/handbook/product-development/how-we-work/product-development-flow/feature-flag-lifecycle/#development).
The original issue should remain open and blocked on the FF rollout issue until the rollout issue is closed, in case the rollout does not go as planned.

#### Schedule

On an ongoing basis, the EM will queue prepared items for development.

In this process, prepared items will be given the milestone label along with the Status `Ready For Development`, which will pull items into the Build board.

Part of the ongoing prioritization and scheduling effort includes providing answers to these questions

1. What is active now?
1. What needs to be active next?
1. Are there any deliverables?
1. Is the active list in priority order?
1. Bugs list
1. Technical debt items arising

The outcome of these scheduling activities are reflected in an [Outlook Issue](https://gitlab.com/gitlab-org/geo-team/discussions/-/issues/?search=Geo%20Outlook&sort=due_date&state=all&first_page_size=100) that the EM prepares and which encapsulates the direction and scope of the upcoming iteration.

#### Build

We use the [Kanban Build board](https://gitlab.com/groups/gitlab-org/-/boards/1181257)
to look at issues that have the `%Started` milestone in combination with the `group::geo` label.

The Engineering Manager(EM) owns this board and uses it to facilitate building out the Epics and issues that the team have decided
are ready for development.

Issues are added in the "Ready for Development" column in priority order. When an engineer is open, they can pick
unassigned issues from the top of this list. As their work progresses, they advance the issue through the "In Dev" and
"In Review" columns. Engineers need to keep the issues aligned with the status of the corresponding merge requests.

When an issue is "In Review", the MR should be assigned to both the original engineer and the reviewer so that it is
clear that the merge request has an active reviewer. This helps the EM to see how much work is allocated to each person.

The column for "Verification" is where the owner of the Epic, or the EM will ensure that the issue fits into the context
of the solution. This is also where we confirm if a release note is required for this issue, and assign the correct
milestone.

#### Release and Post-Release

For some issues, there are tasks that need to be performed by SRE's after the release is completed. Because these tasks
are often in the infrastructure project, they cannot be added to Epics. We still need to track these and follow them
through to completion.

Finally, we ensure that any communication that needs to go out for this issue is delivered. These can take the form of
release posts, blog posts, video tutorials, or demos.

### Work Ownership

The Geo team uses a variety of [work items](../agile-work-items.html) to describe features or capabilities that will increase the maturity of the Geo categories over time.

Each high-level work item (i.e. Epics and/or Features) should be owned by an engineer who is responsible for all technical aspects of that work item. If at any point, the
owner needs to take leave that is more than a few days, they should assign another engineer to act as owner until they return.

**In the planning phase**, the engineering owner will work closely with the Product Manager to understand what the requirements
are and why they are important to customers. The engineer will decide on how best to deliver the solution and create [Tasks](../agile-work-items.html#task)
that encapsulate the technical work required. They may need to consult with other team members and stable counterparts
to come up with the right approach to delivering the requirements.

They should include Tasks for documentation changes and additional testing requirements that are required. A task should be created to inform Geo experts in the Support group about the change if needed. The engineer
should also consider if there is any technical debt that is appropriate to address at the same time. Additional Tasks are also
needed for any rollout or post-release todos.

The engineer that is the owner does not need to be the only person implementing the Tasks.
They should keep watch of the work that is done on the tasks so that they can verify that the work is progressing correctly.
Ideally, they should also be an approver on each MR for the work items. If there are problems with the work, or lengthy delays,
they need to make sure the Product Manager and Engineering Manager are aware.

**When work is nearing completion**, the owner engineer should check the release note and work with the PM on any changes.
They should also make sure that any additional issues that may have come up during the build process are either
addressed, or scheduled for work. This will help to make sure that we do not build up technical debt while building.

**Finally**, owners should also monitor any work that needs to occur while rolling out the Epic or Feature in production. If there are
rake tasks, database migrations, or other tasks that need to be run, they need to see those through to being
run on the production systems with the help of the Site Reliability counterpart. They may also need to assist the
Product Manager with creating the release note for the work item.

This places a lot of responsibility with the owner, but the PM and EM are always there to support them. This ownership
removes bottlenecks and situations where only the PM or EM is able to advance a work item. In addition, the best people
to decide on how to implement a task are often the people who will actually perform the work.

The responsibility of the Product Manager is to make sure that there is enough information for the engineer to develop the
correct solution that meets the requirements. They are also available to answer clarifying questions or consider how to
approach edge cases. At the end of a work item they also communicate this out to customers and other interested parties.

The Engineering Manager is responsible for clearing the path. They need to make sure the engineers performing the work
have the access to the right information, people, tools, and other resources to get the work done. They try to foresee
problems and clear any blockers that may arise while the work is in progress.

### Opening an Issue

Within the Geo Team we utilize Issue Templates to create consistency in our Backlog to help us be more Efficient and deliver more Results.
Using Issue Templates has been found to help the team in the following ways:

1. Ensures an Issue has all the required information so that any contributor can pick up and start or join a conversation without any outside context.
2. Empowers our Refinement Process to work more efficiently which results in higher Community Contributions and less SME reliance.
3. Helps Issues stay relevant and not unintentionally sink to the bottom of our Backlog.

We use the following Issue Templates:

- [Bug (template)](https://gitlab.com/gitlab-org/gitlab/-/blob/master/.gitlab/issue_templates/Bug.md)
  - Used as the template for Bugs and their investigation.
- [Feature - lean (template)](https://gitlab.com/gitlab-org/gitlab/-/blob/master/.gitlab/issue_templates/Feature%20Proposal%20-%20lean.md)
  - Used as the template for larger Feature requests. These will often drive a conversation and ultimately be promoted to Epics with Implementation Issues being broken off into MVC changes.
- [Implementation (template)](https://gitlab.com/gitlab-org/gitlab/-/blob/master/.gitlab/issue_templates/Implementation.md)
  - Used as the template to break down large Epics, organize MVC level changes, and assist in our Refinement Process.

*Note: Many sections in a template should be treated as optional if there is no relevant information to be added.*

### Issue Labels and Metadata

The Geo team uses a consistent labeling system to organize, prioritize, and track work across the project. Understanding these labels helps ensure issues are properly categorized and can be efficiently triaged and scheduled.

#### Group and Category Labels

- **group::geo**: Applied to all Geo team work items. This is the primary label that identifies an issue as belonging to the Geo team and is used to filter issues across all Kanban boards.
- **Category:Geo**: The product category label that indicates the issue relates to the Geo feature area. This label is used for product management and roadmap tracking.
- **Category:Backup/Restore of GitLab instances**: The product category label that indicates issues related to Backup and Restore feature area, including backup creation, restoration procedures, and backup management.
- **Category:Geo Replication**: Applied to issues related to Geo replication functionality, including data synchronization, consistency, and replication lag improvements.
- **Category:Disaster Recovery**: Applied to issues related to disaster recovery capabilities, failover mechanisms, and recovery procedures.

#### Priority and Severity Labels

We follow GitLab's standards on Priority and Severity labels.

#### Weight

Weight represents the estimated effort required to complete an issue, measured in "best work days" (days with no interruptions or distractions). We use the set [1, 2, 3, 4, 5, 10]:

- **1-3**: Small issues suitable for community contributions or quick wins.
- **4-5**: Medium-sized issues requiring focused effort.
- **10**: Issues that are too large and should be broken down into smaller pieces or promoted to epics.

#### Additional Labels

- **Seeking Community contributions**: Applied to refined issues with weights 1-3 that have implementation guides and are suitable for community contributors.

### Addressing new issues quickly

When new issues arise (through testing, customer support issues, or other means) we still want to be able to address them
quickly without being delayed by process.
If you think a new issue needs to be worked on immediately:

1. Make sure there is enough detail on the work item description for someone else to understand the issue and for someone else to have
enough context to review your work
2. Make sure there is a weight
3. Assign it into the current milestone
4. Add the `group::geo` label and set the Status.
5. Assign it to yourself

### Weights

We use the weights from the set [1, 2, 3, 4, 5, 10] where the value represents a "best work day". A "best work day" means
a day where there are no interruptions, emails or other demands on your time.
For example, if an issue is assigned a weight of 5 then an engineer should be able to complete the work in 5 days if
this were the only thing that they needed to do for those five days.

If an issue is assigned a weight of 10, then the issue is too big and will need to be broken down further. This usually means promoting the issue to an epic and raising individual issues that have a lower weight.

If any issue is weighted above a 3, we should ask ourselves if it can be further broken down. We should do this even if the issue was already broken out from an even larger issue.

### Backlog Refinement Process

Backlog refinement equals moving issues from Status `New` and `workflow::validation backlog` through each stage toward Status `Ready for Development`. Engineers will follow instructions provided in a weekly assigned "refinement issue", which generally aligns with [Product Development Flow](/handbook/product-development/how-we-work/product-development-flow/).

Issues in the [GitLab.org group](https://gitlab.com/groups/gitlab-org/-/issues) labeled `~"group::geo`, `workflow::validation backlog` will be refined.
Each week 3 issues will be randomly selected by a bot and refined by the team. Bugs will be prioritized over feature requests and a go/no go will be given.

1. Refinement issue is created and assigned to engineers. It contains instructions for what to do with each issue and is the source of truth for this part of the process. The rest of the steps below are an overview.
2. Phase 1: Engineers will select 1-3 issues each and put the Status `Problem Validation` on when they start taking a look.
   1. If the issue doesn't have the right issue template/enough details, it will be sent back to the author/PM for clarification
   2. If the issue is a go, move to Status `Refinement` for Phase 2. See below how to decide if an issue is a "go" or "no-go".
3. Phase 2: Engineers will add an implementation guide, the right labels and weights for the issue. Once that is ready, move it to the Status `Ready for Development` for the PM/EM to schedule.
   1. For bug issues, the bug first needs to be reproduced as a part of the refinement process. If the bug issue isn't reproducible, the issue can be closed off. Bug issues also need to be assigned a priority/severity in addition to the regular labels.
   2. For features/maintenance issues, a timebox of 1 hour or so is expected per issue and for bugs this could take 2-3 hours as we need to reproduce it before adding implementation details
   3. Please also label as `~"Seeking Community contributions"` if an implementation guide has been added and weights are 1-3.
   4. If this issue has a weight of 1-3 and could be easy enough for a community contributor, it should also be relatively easy to write up the implementation guide. Timebox to 30 mins - 1 hour.
   5. If the issue is very complex, stop investigating as soon as you’ve realized that it’s going to take > 1 hour to reproduce/write the implementation guide.
   6. If an issue takes too long or you think is not aligned with our current roadmap, be vocal about pushing back on it and discuss it on the refinement issue.

#### Phase 1: How to decide if an issue is a "go" or "no-go"

In phase 1, we decide whether or not an issue is a "go" or "no-go" for refinement in phase 2. An issue that is a "go" will always be moved to phase 2 for refinement.

An issue is a "no-go", if,

1. the feature is not aligned with our current roadmap (i.e. in the next 6 - 12 months)
1. the issue's context is outdated or irrelevant now
1. it is a duplicate (mark it as a duplicate)
1. it is already resolved, completely, or enough to not be worth prioritizing any more
1. the solution is complex and the demand is low

Those issues that are a "no-go" should be closed with the reason mentioned or de-prioritized to priority 4, so that phase 2 engineers do not pick them up for refinement. Please tag the EM and PM when closing an issue if you're not 100% confident what to do with it.

The refinement script can be found [here](https://gitlab.com/gitlab-org/geo-team/bots/-/tree/main/cmd/refinement?ref_type=heads)

#### Engineering Customer/Support Rotation Process

Every two weeks, a Geo engineer is assigned to be the DRI for doing the technical assessment of customer support tickets + monitoring [#g_geo](https://gitlab.enterprise.slack.com/archives/C32LCGC1H) channel for support issues.

Process summary:

- Every week, a slack reminder in [#geo-lounge](https://gitlab.enterprise.slack.com/archives/C7U95P909) channel will let the group know that a new support shift is starting for technical assessment triage.
- Every Geo engineer is expected to be aware of their upcoming rotation (as per the schedule below) and take action as per the slack reminder.
- The DRI currently assigned to the rotation should then dedicate their two weeks to review issues from the customer support [backlog](https://gitlab.com/gitlab-com/request-for-help/-/issues/?sort=created_date&state=opened&label_name%5B%5D=Help%20group%3A%3AGeo&first_page_size=100) and support in the [#g_geo](https://gitlab.enterprise.slack.com/archives/C32LCGC1H) channel.
  - Be first on point to triage questions in our [#g_geo](https://gitlab.enterprise.slack.com/archives/C32LCGC1H) channel. Now that we have a slack workflow -> this should be taught/triggered per more complex questions and quickly answer anything that's short and sweet.
  - Expected duties are: triaging, creating issues, documenting initial research, adding priority labels etc. so that a customer support issue can enter our usual workflow.
- If the DRI is unable to perform an upcoming triage rotation shift due to any reason (e.g. PTO, sick leave, other responsibilities taking precedence), they are expected to swap their rotation with another team member or notify the EM to facilitate. Once the swap is identified, the schedule should be updated with via a MR.
- The DRI needs to update this [issue](https://gitlab.com/gitlab-org/geo-team/discussions/-/issues/5142) for the 2025 rotation.
- The DRI should estimate the amount of time spent on support and track this via `Time Tracking` in each issue approximating to the hour how much they have spent on tickets and triaging. This is a new part of the process which is in a trial phase.

At the end of the rotation, each engineer should provide handover notes inside the [issue](https://gitlab.com/gitlab-org/geo-team/discussions/-/issues/5142) or directly on the RFH:

- Using our standardized Duo/AI Chat prompt for handing over an RFH below
- Proofread and correct Duo's output if needed.
- Post it directly on the issue, with a ping to the new DRI, in a new root comment, not on a thread
- If needed, the new DRI should ask clarifying questions on a reply to the comment or in Slack.  Setting up a meeting to go over more difficult context would work if timezones aligned.

##### Duo/AI Prompt for RFH Handover

```markdown
Please concisely summarize the current status of this issue in order to hand it off to the next backend engineer on customer support duty for the Geo team.
Use this template:

A handover comment generated by Duo and reviewed by me:

### Current status

<!-- Content of current status -->

*Confidence: <!-- Percentage confidence level in the accuracy of this status -->*

### Action items

- <!-- GitLab handle of DRI --> (*Confidence: <!-- Percentage confidence in who should be DRI for this item -->*): <!-- Action item -->. (*Confidence: <!-- Percentage confidence in the validity of this action item -->*)
```

- Be sure, or be conservative. When making assertions, be wary of hallucinations. Use qualifying language as appropriate.
- If you reference a resource with a URL, such as a GitLab issue or Zendesk ticket number, then make it a link.
- Read linked URLs if the added context might improve your answer.
- Answer with only Markdown code. I will review it and post it.

##### Schedules

Schedules are tracked in a [Google Sheet(internal link)](https://docs.google.com/spreadsheets/d/1-kvptYZfhCSV9WTVEs4epfUSVvxIMSf5wHxnT6BmjSw/edit?gid=1214041762#gid=1214041762).

## Retrospectives

Retrospectives are a [crucial component](https://www.retrium.com/blog/i-was-wrong-retrospectives-are-not-the-most-critical-part-of-agile-but-they-are-close) of the agile methodology. However, having a retro should not be about checking-off a mark in our agile todo list. The purpose of the retro is to learn and then take action that leads to a better place. We learn from our past actions and results and we use that knowledge to improve our future execution.

At GitLab there is a general guideline to conduct [group retrospectives](/handbook/engineering/careers/management/group-retrospectives/) and a process automation to run retrospectives [asynchronously](https://gitlab.com/gitlab-org/async-retrospectives). The Geo team however is working with a new process that combines elements of fully asynchronous work with the benefits of active conversation and the increased level of engagement that these types of interactions bring.

The details of this process is documented in our [Geo Team Retrospectives](./retrospectives.md) page.

## FAQ

### What work should I pick up next?

"Pull from the right". That means, start at the right hand side of the build board, and move towards the left.

In order, that means:

- Can you help anyone with reviews?
- Can you unblock anyone in development?
- Pick from the top of the "ready for development" column on the [Kanban Build board](https://gitlab.com/groups/gitlab-org/-/boards/1181257?milestone_title=%23started&&label_name[]=Geo).

## Feature Flag Management

### Retention Guidelines

- **Development Flags**: 5 milestones maximum, aiming for 4 milestones as a general guidance
- **Operational Flags**: 16 months maximum, aiming for 12 milestones as a general guidance.

### Process

1. **DRI Assignment**: Each feature flag must have a designated owner responsible for cleanup
2. **Regular Review**: Monthly check of flags approaching retention limits during team planning
3. **Cleanup**: Remove flags within retention period or request extension with justification

### Flag Lifecycle

- **Creation**: Use `geo_` prefix, set clear description and expected timeline
- **Monitoring**: Track usage and plan removal during development
- **Removal**: Clean up code paths and update documentation

### Extensions

Flags exceeding retention limits require team lead approval with documented justification.

### Resources

- [Team Feature Flag Analysis](https://docs.google.com/spreadsheets/d/1pbPwUQC30gkaueqy_f4m2KRK0Smta3YCHCvh-kEJbi0/edit?gid=1801934973#gid=1801934973)
- [GitLab Feature Flag Documentation](https://docs.gitlab.com/development/feature_flags/)

### AI Prompts used by the team

#### Code reviews

```markdown
You are a maintainer at GitLab and a staff level software engineer.

You review code in Merge requests based on the guidelines given here https://docs.gitlab.com/development/code_review/ and using conventional comments.

The developer guidelines are here https://docs.gitlab.com/operator/developer/guide/
The contributor guidelines are here https://docs.gitlab.com/development/contributing/

Follow the links from all these pages and inform yourself of all the guidelines for contributing to GitLab as a developer, tester and reviewer.

To each of your suggestions/comments please add the technical category of review, e,g, backend, frontend, database, security.

Base your code reviews on all this information.
```

#### Weekly Status Updates on Epics

Each week we have to generate a status report of what happened during the week as a changelog on the epic per DRI.  The following is an AI prompt that can be used to generate this status report

```markdown
Generate a status update using the following format with markdown and achievements should highlight any closed issues within the last 7 days since the last status update with a short(1 sentence to 2 sentences) summary of what the issue accomplishes for customers.  Blockers can be either left empty or use any issues in a status blocked.  Next would be any issues that have an assignee and have the status in dev or in review.

<!-- Create a high level summary (optional) -->

:tada: **achievements**:

-

:issue-blocked: **blockers**:

-

:arrow_forward: **next**:


```
