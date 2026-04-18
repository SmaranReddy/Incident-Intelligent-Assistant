---
title: "The Infrastructure Platforms Project Management"
---

## Project Management in Platforms

We use GitLab epics and issues to communicate the progress and status of our work.
All teams in Infrastructure Platforms follow these guidelines so that it is easy for team-members to contribute to different projects if needed.

### Projects are reviewed weekly in the Grand Review

#### Epic Update Timeline

Infrastructure Platforms epics are aligned with the [GitLab Operating Model (GOM)](https://internal.gitlab.com/handbook/company/gitlab-operating-model/). Many epics are either:

1. Children of [Grand Review Epics](https://gitlab.com/groups/gitlab-com/-/work_items/2115)
2. Linked to [Operating Model Functional Lead Epics](https://gitlab.com/groups/gitlab-operating-model/-/work_items?label_name%5B%5D=Owner%3A%3AInfrastructure%20Platforms&type%5B%5D=epic)

The update schedule is designed to cascade information up through these levels:

- **Thursday EoD (Friday 02:00 UTC)**: DRIs update project epics (automation prompts at Thursday 00:00 UTC)
- **Friday**: Leads update Grand Review and GOM Functional Lead epics using the information from Thursday
- **Monday**: Leadership team discusses updates
- **Tuesday**: Platforms Grand Reviews occur
- **Wednesday**: Company-wide Operating Model reviews where Infrastructure VP and PM use the cascaded information

This timeline ensures alignment with [E-Group Objectives](https://gitlab.com/groups/gitlab-operating-model/-/work_items?label_name%5B%5D=Operating%20Model%3A%3A%202%20-%20Egroup%20Metric&type%5B%5D=epic) and [Company Initiatives](https://gitlab.com/groups/gitlab-operating-model/-/work_items?label_name%5B%5D=Operating%20Model%3A%3A%201%20-%20Company%20Objective&type%5B%5D=epic), providing timely and well-supported data for strategic decision-making at all levels up to the executive team.

Completed epics should remain "Open" with the ~"workflow-infra::In Progress" label.
Update the status block in the epic description to summarize the project and share the completed status.

We use [automation](https://gitlab.com/gitlab-com/gl-infra/epic-issue-summaries/) to collect the status information into top-level epics for each group, and to gather status updates from epic notes.
This automation runs several times each day, and can be triggered by [running the pipeline listed on the project page](https://gitlab.com/gitlab-com/gl-infra/epic-issue-summaries/-/pipeline_schedules).

#### Status updates on project epics

Once weekly status updates are enabled for a project epic, there will be a comment with instructions on how to provide a status update.

**Every Thursday**, the DRI for a project must ensure that the status block in the epic description is updated to:

1. Log the number of hours spent on the project by all contributors.
1. Briefly highlight achievements since the last update.
1. Indicate any project blockers.
1. Indicate planned next steps, or mitigations required to progress.

Comments are automatically generated on project epics to report status.
To enable this functionality, please ensure the epic follows [these steps](https://gitlab.com/gitlab-com/gl-infra/epic-issue-summaries/-/blob/42365bbd7df9757f49d6dd4eb3d2625c399fac59/README.md#child-epics)

The assignee of the epic will be mentioned in a note that looks like this:

```markdown
@<user> Reply to this note to update the status of your epic for YYYY-MM-DD
...
```

GitLab Duo Agentic Chat can help generate a starting point for the update:

1. Load the epic and go to the status note where you are mentioned (after it is enabled).
2. Click "GitLab Duo Chat" in the sidebar, and enable Agentic Mode.
3. Use the following prompt to draft a status note update

```markdown
Please provide a terse high level summary of the last 7 days of activity using the child issues of this epic, highlighting new and issues that were closed in the last 7 days using the following format:

<overview>

:tada: **achievements**:
- ...
:issue-blocked: **blockers**:
- ...
:arrow_forward: **next**:
- ...

Give the status update in raw markdown (preformatted text) and do not use any markdown headings.
```

#### When a project is finished

1. The epic should be left "Open" and retain the ~"workflow-infra::In Progress" label so that the project will still be listed for the next Platforms Grand Review.
1. The DRI is responsible for updating the epic description to ensure each section is up to date including the `Participants` section.
1. The DRI creates the closing status (final status update) which includes:
   1. The original problem
   1. A brief description of the changes made
   1. Information about the impact of the project now that it is completed
   1. A link to the demo / walkthrough video
   1. An instruction to direct the Grand Reviewers to close the epic
   1. Emojis and / or shout-outs are also welcome but not required
   1. Note: remember to update the hours tracking field in the status update template with any hours worked during the final week of the project (not cumulative hours for the entire epic)

Completed epics will be reviewed, celebrated, updated to done, and closed in the next [`Platforms Grand Review`](https://www.youtube.com/playlist?list=PL05JrBw4t0KqDXSHdlUvPWHOj_Hw8JdQ1) (Playlist accessible using GitLab Unfiltered account).

Note that closing statuses are helpful when producing the end-of-year team summaries.

## Epics

Epics must always have the following sections:

- A Problem Statement
- Directly responsible individuals (DRI) responsible for the project completion
- Defined exit criteria
- Development Log, containing the previous Status Updates and any other relevant information to log
- Status Update, with the latest status update of the Epic
- Start date and estimated due date

### Sections format

The DRI of the epic should be the assignee, and there should only be one assignee for an epic.

Use the following headings and structure in the Epic description:

#### Participants Section

This section is dynamic and includes people who are currently contributing to the Epic work.
The DRI and the EM are responsible for updating the list of participants accordingly.

```markdown
### Participants

- @participant1
- @participant2
- @participantN
```

#### Status Section

The status section is generated automatically on sub-epics.
See the section on [status updates on project epics](#status-updates-on-project-epics) for more info.

### Should this be an issue or an epic?

We use epics to show that a set of work can be combined under one theme to meet a specific goal or lead to a specific impact.
Epics can cover any set of work that is likely to take longer than two weeks, or has already been running for longer than two weeks.
We can then report on our progress towards that goal regularly and clearly demonstrate the impact of our work when we are finished.

## Specific Issue Types

### Corrective Actions and Security issues

We also work on corrective actions and security issues labeled as `~"corrective action"` or `~"security"`.
For these issues, `severity::*` labels are set to meet specific SLOs.

- For security issue see the [Time to resolve table by severity](/handbook/security/engaging-with-security/#severity-and-priority-labels-on-security-issues)
- Corrective Actions SLOs are currently based on definitions from [Quality](/handbook/product-development/how-we-work/issue-triage/#severity-slos):

## Labels

The Infrastructure Platform teams uses the following set of labels:

| Description | Labels |
|-------------|--------|
| The section label | `section::developer experience`<br/>`section::gitlab delivery`<br/>`section::gitlab dedicated`<br/>`section::production engineering`<br/>`section::tenant scale` |
| The stage label | `devops::developer experience`<br/>`devops::gitlab delivery`<br/>`devops::gitlab dedicated`<br/>`devops::production engineering`<br/>`devops::data access`<br/>`devops::runtime` |
| The group label | `group::build`<br/>`group::Cloud Cost Utilization`<br/>`group::development analytics`<br/>`group::development tooling`<br/>`group::environment automation`<br/>`group::Foundations`<br/>`group::geo`<br/>`group::git`<br/>`group::gitaly`<br/>`group::Observability`<br/>`group::operate`<br/>`group::Ops`<br/>`group::organizations`<br/>`group::performance enablement`<br/>`group::release-and-deploy`<br/>`group::Runners Platform`<br/>`group::runway`<br/>`group::switchboard`<br/>`group::tenant services`<br/>`group::test governance`<br/>`group::US PubSec` |
| Scoped `workflow-infra::*` labels | (see below) |
| Scoped `infra-category::*` labels | (see below) |
| Optional Scoped `Service` labels | `Service::*` |

### Workflow labels

We leverage scoped workflow labels to track different stages of work.
They show the progression of work for each issue and allow us to remove blockers or change focus more easily.

While issues may not need to go through every state, the standard progression of workflow is from top to bottom in the table below:

| State Label | Description |
| ----------- | ----------- |
| ![Triage](/images/engineering/infrastructure-platforms/project-management/label-triage.png) | Default label. Task is raised and effort is needed to determine the correct action, work required or team ownership |
| ![Proposal](/images/engineering/infrastructure-platforms/project-management/label-proposal.png) | Proposal is created following triage and put forward for a review. <br/>If there are no further questions or blockers, the issue can be moved into "Ready". |
| ![Ready](/images/engineering/infrastructure-platforms/project-management/label-ready.png) | Proposal is complete and the issue is waiting to be picked up for work. |
| ![In Progress](/images/engineering/infrastructure-platforms/project-management/label-in_progress.png) | Issue is assigned and work has started. <br/>While in progress, the issue should be updated to include steps for verification that will be followed at a later stage.|
| ![Under Review](/images/engineering/infrastructure-platforms/project-management/label-under_review.png) | Issue has an MR in review. |
| ![Verify](/images/engineering/infrastructure-platforms/project-management/label-verify.png) | MR was merged and we are waiting to see the impact of the change to confirm that the initial problem is resolved. |
| ![Done](/images/engineering/infrastructure-platforms/project-management/label-done.png) | Issue is updated with the latest graphs and measurements, this label is applied and issue can be closed. |

There are three other workflow labels of importance:

| State Label | Description |
| ----------- | ----------- |
| ![Cancelled](/images/engineering/infrastructure-platforms/project-management/label-cancelled.png) | Work in the issue is being abandoned due to external factors or decision to not resolve the issue. After applying this label, issue will be closed. |
| ![Stalled](/images/engineering/infrastructure-platforms/project-management/label-stalled.png) | Work is not abandoned but other work has higher priority. After applying this label, team Engineering Manager is mentioned in the issue to either change the priority or find more help. |
| ![Blocked](/images/engineering/infrastructure-platforms/project-management/label-blocked.png) | Work is blocked due external dependencies or other external factors. Where possible, a [blocking issue](https://docs.gitlab.com/ee/user/project/issues/related_issues.html) should also be set. After applying this label, issue will be regularly triaged by the team until the label can be removed. |

### Category labels

We use scoped category labels to classify the type or area of work being performed.
These labels help organize and filter issues by their functional category.

| Category Label | Description |
| -------------- | ----------- |
| `infra-category::KTLO` | Keep the lights on. Operational work required to maintain system stability, reliability, and performance. Includes monitoring, alerting, incident response, and routine maintenance. |
| `infra-category::Tech Debt` | Technical debt and refactoring work. Bug fixes or improvements to code quality, architecture, testing, documentation, and infrastructure that reduce future maintenance burden. |
| `infra-category::Customer innovation` | Work directly driven by customer needs and feedback. Features, improvements, and solutions that enhance customer experience and address customer-reported issues. |
| `infra-category::Internal innovation` | Internal improvements and explorations. Work that improves developer experience, team productivity, or explores new technologies and approaches for future adoption. Can be foundational/architectural work if it addresses multiple classes of technical debt. |

### Labels in gitlab-org group

Stage groups use [type labels](/handbook/product/groups/product-analysis/engineering/metrics/#work-type-classification) to label merge requests in projects in the `gitlab-org` group.
If you need a stage group to perform work, it is best to apply the relevant stage group label when the issue is created.

## Triage Ops

For labeling and bot notifications under [gitlab-com/gl-infra](https://gitlab.com/gitlab-com/gl-infra) we use [gitlab-triage](https://gitlab.com/gitlab-org/ruby/gems/gitlab-triage) and the [triage-ops](https://gitlab.com/gitlab-com/gl-infra/triage-ops/) project.
For labeling, enforcing SLOs, and managing workflow labeling there are common policies that are set uniformly for projects in Infrastructure, Platform.
For more information and how to add additional policies see [the project README.md](https://gitlab.com/gitlab-com/gl-infra/triage-ops/-/blob/master/README.md?ref_type=heads).

## Retrospectives

At the end of the quarter, or the completion of a large deliverable, teams should perform a retrospective to capture learnings.
There is no set format for the retrospective though Engineering Managers should be aware of the [GitLab Retrospective Guidelines](/handbook/engineering/careers/management/group-retrospectives/).
The retrospective DRI identifies a list of actions which they consolidate in the Summary of Actions section in the issue description.

Process to identify actions:

1. Add a comment on each thread with Actions as the title of the comment (H3 level)
1. Some threads may not require an action, you may want to state this explicitly at the end of the thread for transparency
1. Below the Actions comment, add a suggestion on an action the team can take
1. Ping the contributors to get a round of validation on the actions and potential refinement
1. Create an issue for each action and list them in the Summary of Actions section
