---
title: Plan:Project Management Team
---

## Plan:Project Management Team

The Plan:Project Management team works on
GitLab's [Project Management category](/handbook/product/categories/#project-management-group) in the [Plan stage](/handbook/engineering/devops/plan/).

For more details about the vision for this area of the product, see the
[Plan stage](/handbook/engineering/devops/plan/) page.

### Team members

{{< team-by-manager-slug manager="midhir" team="Plan:Project Management">}}

### Stable counterparts

{{% engineering/stable-counterparts manager="midhir" role="(.*)Plan:Project Management$|Product Manager(.*)Plan Stage|Security(.*)Plan|Principal(.*)Plan$" %}}

### Hiring chart

Check out our [jobs page](https://about.gitlab.com/jobs/) for current openings.

## Work

You can see how we work as a stage at the [Plan stage page](/handbook/product/categories/#plan-stage).

For the backend team specifically, we use the standard GitLab
[engineering workflow](/handbook/engineering/workflow/). To get in touch with the Plan:Project Management
backend team, it's best to create an issue in the relevant project
(typically [GitLab](https://gitlab.com/gitlab-org/gitlab)) and add the ~"group::project management" label, along
with any other appropriate labels. Then, feel free to ping the relevant
Product Manager and/or Engineering Manager as listed above.

For more urgent items, feel free to use [#s_plan](https://gitlab.slack.com/archives/s_plan) on Slack.

### Capacity planning

#### Estimating effort

When estimating the effort involved in upcoming work, we use the same approach and numerical scale as other groups in the Plan stage.

{{% include "includes/engineering/plan/estimating-effort.md" %}}

#### Weighing bugs

{{% include "includes/engineering/plan/weighing-bugs.md" %}}

#### Refining and organizing feature work

To help drive alignment with our stable counterparts, provide visibility into progress, and breakdown our vision into a series of [MVCs](/handbook/product/product-principles/#the-minimal-valuable-change-mvc), we collaborate with Product and UX during [`~workflow::planning breakdown`](/handbook/product-development/how-we-work/product-development-flow/#description-4) to refine and organize `~type::feature` deliverables into the following structure:

- Feature (Epic) - Contains all the necessary vertical feature slices to default the corresponding feature flag to "on". The feature epic will also serve as the location to generate a corresponding Release Post item MR. The feature epic should be scoped to the [minimal amount of functionality that still provides customer value](/handbook/product/product-principles/#the-minimal-valuable-change-mvc). Additional scope planned for future enhancements should be stored in follow-on epics.
  - Spike (Issue) - If we are unable to accurately estimate the effort necessary to implement the feature, we first conduct a spike
  - UX (Issue) - For larger initiatives, UX creates a separate UX issue that serves as the SSOT for design goals, design drafts, design conversation and critique, and the chosen design direction that will be implemented. [Learn more about UX issues](/handbook/product/ux/product-design/ux-roadmaps/).
  - Vertical Feature Slice (Issue) - A subset of the feature that can be completed within a single milestone, tested, and verified within the `plan-stage` group on production.
    - Engineering Tasks (Task - *Optional*) - One or more engineering tasks that need to be completed in order to deliver the vertical feature slice. The scope of a task should generally correlate to a single MR.

During the `~workflow::planning breakdown` phase, all issues need to be weighted so we can efficiently and effectively collaborate with Product and UX on "right sizing" the feature epic. It's advisable that all issues are connected to a parent epic that describes the broader set of improvements we are proposing within a specific area of the product. The desired outcome is to ensure it's as small as possible, maximizes our ability to iterate, and makes it easy to track overall progress on delivery, while providing meaningful value and avoiding an undue amount of "change fatigue" for our customers.

### Engineering work rotation

To ensure we consistently address technical debt, security issues, and the bug
backlog alongside our product commitments, we're implementing a milestone-based
rotation system. Each milestone, one backend and one frontend engineer will
dedicate their time to engineering-focused work instead of feature development.

#### How it works

**Rotation Schedule**

- One milestone at a time
- 100% dedication during rotation period
- One backend engineer and one frontend engineer per milestone
- Engineers will not be in the rotation pool while serving as a DRI of a long-term initiative

**Focus Areas** Non-exhaustive list of work that can be picked up during
rotations:

- Security issues nearing or past SLAs
- Bug backlog reduction
- Database scalability and performance
- General technical debt and refactoring

#### Why this approach

1. **Consistent Progress**: Engineering health work won't be perpetually
   deferred
1. **Knowledge Sharing**: Different team members gain experience with different
   parts of our codebase and different processes
1. **Deep Focus**: 100% dedication allows engineers the focus to work on complex
   problems without context switching
1. **Predictable Planning**: We know exactly when engineers will be unavailable
   for feature work

#### Implementation guidelines

- Engineers on rotation are protected from feature work pulls (except critical
escalations)
- Each rotation should include handoff notes for continuity
- We'll adjust focus areas based on Core DevOps and company priorities

#### Success Metrics

We'll track:

- Reduction in overdue and overall security issues
- Bug backlog velocity
- Performance improvements
- Availability/error budget improvements
- Team satisfaction with the rotation system

### Collaboration between backend and frontend

#### Using the ~"backend complete" label

The ~"backend complete" label is added to issues with multiple specializations (usually backend and
frontend) to indicate that the backend component is complete. Add this label when the backend work is
functionally complete, merged and verified but frontend, or other, work is ongoing.

### Picking something to work on

The [Plan:Project Management Build board](https://gitlab.com/groups/gitlab-org/-/boards/1285239?label_name[]=backend) shows prioritized work with [workflow columns](/handbook/product-development/how-we-work/product-development-flow/) relevant to implementation. There is an
additional column to show in-progress community contributions. Filtering it by
~backend shows issues for backend engineers to work on.

It's OK to not take the top item if you are not confident you can solve
it, but please post in [#s_plan](https://gitlab.slack.com/archives/s_plan) if that's the case, as this probably
means the issue should be better specified.

### Queueing Experimentation

As part of our ongoing efforts to improve efficiency, we will experiment with [queuing techniques](https://www.brightball.com/articles/story-points-are-pointless-measure-queues) instead of traditional weight/story points for the [migration of legacy issues to work items project](https://gitlab.com/gitlab-org/gitlab/-/issues/461855). Starting in milestone 17.3, we will:

1. **Refine Required Issues**: Review the [list of required issues for the first MVC](https://gitlab.com/gitlab-org/gitlab/-/issues/?sort=created_date&state=opened&label_name%5B%5D=work%20items::ga-issues&label_name%5B%5D=work%20items&first_page_size=100).

    - Determine the best way to group bundles of tasks, using issues or epics.
    - Break down large issues into tasks, aiming for tasks that are small enough to be moved from ~"workflow::in dev" to ~"workflow::in review" within 5 business days.

2. **Prioritize Tasks**: Prioritize the tasks from step 1.
3. **Update Workflow**: Update the #picking-something-to-work-on section to include the list of tasks, monitoring work in progress to keep it as low as possible.

This will help us manage and report progress more effectively using queue management principles.

#### High Severity Issues

{{% include "includes/engineering/plan/high-severity-items.md" %}}

### Working on unscheduled issues

Everyone at GitLab has the freedom to manage their work as they see fit,
because [we measure impact, not activity](/handbook/values/#results). Part of this is the
opportunity to work on items that aren't scheduled as part of the
regular monthly release. This is mostly a reiteration of items elsewhere
in the handbook, and it is here to make those explicit:

1. We expect people to be [managers of one](/handbook/values/#efficiency), and we [use GitLab ourselves](/handbook/values/#collaboration). If you see something that you think
   is important, you can [request for it to be scheduled](/handbook/engineering/workflow/#requesting-something-to-be-scheduled), or you can
   [work on a proposal yourself](/handbook/values/#iteration), as long as you keep your
   other tasks in mind.
1. If you feel like you want to have some specific time set aside, but
   aren't interested in the topics of an existing event, feel free to
   label issues with "For Scheduling" and copy your manager for visibility.

When you pick something to work on, please:

1. Follow the standard workflow and assign it to yourself.
1. Share it in [#s_plan](https://gitlab.slack.com/archives/s_plan) - if not even more widely (like in #development
   or #backend).

### Retrospectives

GitLab milestones end on the **3rd Thursday of each month**, and we run async retrospectives tied to that schedule. The retrospectives consist of three phases:

#### Phase 1: Collection

**When:**
Starts: **3 Mondays before milestone ends**
Ends: **End of release week (milestone Thursday)**

**What happens:**

- Team submits retro topics via the Slack shortcut ([`⚡Project Management Retro Ideas`](https://slack.com/shortcuts/Ft08T8L66DSA/9b29cd637b654c4904e33e9199b88149))
- Weekly reminders go out
- Responses are sent to a google sheet that the facilitator has access to

#### Phase 2: Synthesis

**When:**
Starts: **Friday of release week**
Ends: **Following Tuesday**

**What happens:**

- Facilitator reviews and groups entries into themes
- Duplicates removed
- Poll generated to vote for the top two or three topics to discuss

#### Phase 3: Discussion & Action

**When:**
Starts: **1 week post-release (Tuesday–Friday)**

**What happens:**

- Facilitator adds topics to generated retrospective issue
- Team reviews points async and comments on the items
- Team selects 2–3 action items to carry forward
- Owners are assigned and new issues are created to track follow-up

#### Example Timeline (Milestone 18.3 Ends 2025-07-17)

| Phase             | Dates                    |
|------------------|--------------------------|
| Collection        | June 30 – July 17        |
| Synthesis         | July 18 – July 22        |
| Discussion/Action | July 22 – July 25        |

#### Reminders

- Use the Slack shortcut to submit retro ideas anytime during the collection window.
- Weekly reminder messages will be posted to encourage participation.
- Keep it short — even one sentence helps.

## Useful links

- [Plan:Project Management Build board](https://gitlab.com/groups/gitlab-org/-/boards/1285239?label_name[]=backend) - this shows work in the current release
- [#s_plan](https://gitlab.slack.com/archives/s_plan) in Slack
- [Recorded meetings](https://www.youtube.com/playlist?list=PL05JrBw4t0KoceqcTneOVmAzhEp6NinY0)
- [Retrospectives](https://gitlab.com/gl-retrospectives/plan/issues?scope=all&utf8=%E2%9C%93&state=all&label_name[]=retrospective)
