---
title: AI Framework Group
description: "The AI Framework group is focused on how to support other product groups at GitLab with the AI Abstraction Layer, and GitLab AI feature development."
---

## Vision

The AI Framework group is focused on how to support other product groups at GitLab with the AI Abstraction Layer, and GitLab AI feature development.

### 👌 Team OKRs

If you're interested in the team's Objectives and Key Results (OKRs), you can find them on [GitLab](https://gitlab.com/gitlab-com/gitlab-OKRs/-/issues/?sort=title_asc&state=opened&label_name%5B%5D=group%3A%3Aai%20framework&first_page_size=20).

### 🚀 Team Members

The following team members are permanent members of the AI Framework Group:

{{< team-by-manager-slug manager="martin-w" role="AI Framework" >}}

### Stable Counterparts

The following members of other functional teams are our stable counterparts:

| **Name**                                                 | **Role** |
| ---------------------------------------------------------| ------------------------------------------------------------------------------- |
| [Ashraf Khamis](/handbook/company/team#ashrafkhamis)     | [Senior Technical Writer](/job-description-library/product/technical-writer/) |
| [Amanda Rueda](/handbook/company/team#amandarueda)       | [Senior Product Manager](/job-description-library/product/product-manager/) MCP Server |

## Shared Responsibilities

### ☕ Team Responsibilities

**Team responsibilities include**

* Facilitating the integration of AI capabilities throughout GitLab by leveraging the AI Abstraction Layer and AI Gateway.
* Guaranteeing the presence of comprehensive global observability, monitoring, documentation, and enhancements in latency.
* Providing essential support for our AI chat system framework.
* Incorporating support for new AI providers when required.
* Assisting with the production support and ensuring the readiness of the AI Gateway.
* LLM inference support, including prompt enginering, response evaluation, fine tuning, evaluation, and more.

### ☎️ How to reach us

Depending on the context here are the most appropriate ways to reach out to the IDE Group:

* Slack Channel: `#g_ai_framework`
* Slack Groups: `@ai-framework` (entire team) and `@ai-framework-engs` (just engineers)

## 📦 Team Processes

### 📆 Regular team meetings

**❗️Important**: For every meeting, the [AI Framework team's meeting document](https://docs.google.com/document/d/1rSJNmRZJ0q8hd9S6W_AXlCMvtNTC6cfWr6VU0e2fJCQ/edit#heading=h.3xbjtjtrp0e9) should be used, and filled with the meeting notes, as well as references to any other sync meeting agendas/notes/recordings which have recently occurred. This will make it easier for people to find any meeting notes.

#### Team Meetings

1. **Weekly Refinement Assignment Meeting**
   * **When:** Every Monday, alternating between 14:00 AM GMT+1 and 19:00 PM GMT+1
   * **What:** This meeting replaces the previous Work Assignment Meeting and focuses on refining issues. The Engineering Manager and Product Manager ensure all issues are properly refined.

### Shared calendars

* AI Framework PTO (Calendar ID: `c_eca9440729dba2cbd88b3117fa70839836fb5811cb072132b94c52f912a31bf5@group.calendar.google.com`)
* AI-Powered Stage Calendar (Calendar ID: `c_n5pdr2i2i5bjhs8aopahcjtn84@group.calendar.google.com`)

AI Framework team members should [sync your PTO events](https://theloop.gitlab.com/site/4455aa7f-24d9-41f2-b940-467b54962e4d/page/0fa19bf4-fd6a-41b9-9316-c2dcf3add854) with AI Framework PTO calendar.

### 🖖 Weekly EM Updates

Each week the team EM provides a Weekly Status update issue which aims to capture the most important items for the team to be aware of. These can be found [here](https://gitlab.com/gitlab-org/ai-powered/ai-weekly/-/issues/?sort=created_date&state=all&label_name%5B%5D=AI%20Powered%20Weekly%20Updates&first_page_size=20).

### 📚 AI Framework Board Outline

Our workflow process involves a weekly board walk where we review issues with the **Deliverable** label. Here's how we organize our [board](https://gitlab.com/groups/gitlab-org/-/boards/7346017?label_name[]=group%3A%3Aai%20framework):

#### How do people know what to work on?

An issue is ready to be taken by an assignee when it has all of the following:

* **Deliverable** label
* The current Milestone
* Either **workflow::ready for development** or **workflow::refinement** label

#### Board Lists

1. **Open** 📝: This list includes all identified issues. An engineering manager will be assigned if either the Milestone or the "workflow::ready for development" label is missing.
2. **workflow::design** ✏️: During this phase, issues undergo design refinement. After design considerations are integrated, the "ready for development" label is applied.
3. **workflow::refinement** 🧪: Issues in this stage are undergoing engineering refinement to ensure the proposed solution meets all requirements. Once refined, the "ready for development" label is applied.
4. **workflow::ready for development** 🎯: Issues that are prioritized and assigned to a specific milestone are moved to this list, and the "ready for development" label is applied.
5. **workflow::in dev** 👩‍💻: When a developer begins work on an issue, they should move it to this list and apply the "in dev" label.
6. **workflow::in review** 👀: After development is complete, the issue moves to this list, and the "in review" label is applied.
7. **workflow::verification** ✅: Following a successful code and UX review, the issue should be moved to this list and the "verification" label should be applied.
8. **workflow::complete** 🎉: Once the issue is verified and confirmed to be working properly, it should be moved to this list, the "complete" label should be applied, and the issue should be closed.

### 🔄 Processes

#### 🗓️ Weekly Process

1. **Open 🆕**: (DRI: PM) Capture and list new issues as they arrive. This stage serves as the initial holding area for issues awaiting classification and prioritization.
2. **Workflow Design ✍️**: (DRI: Designer) Move issues into this lane for **design refinement**, focusing on user experience and interface design.
3. **Workflow Refinement 🔧**: (DRI: Engineering Lead) Conduct **technical refinement** to ensure feasibility and adherence to standards.
4. **Workflow Ready for Development 🎯**: (DRI: PM/EM) Transition fully refined and specced issues here, ready for developers to start coding.
5. **Workflow in Development 💻**: (DRI: Assigned Developer) Implement and build the specified features and functionalities.
6. **Workflow in Review 🔍**: (DRI: QA/Reviewer) Perform thorough reviews and testing, including code reviews and usability testing.

### 📝 Issue Guidelines

These guidelines apply to all issues we use for planning and scheduling work within our group. Our Engineers can define specific implementation issues when needed, but the overall goal for our issues are as follows:

* Provide a meaningful **title** that describes a deliverable result.
  * ✅ `Add the new GitLab Duo chat package as a Vue2 extension`
  * ✅ `Chat: move away from using OpenAI embeddings`
  * ❌ `Make Chat better`
* Provide a meaningful description that clearly explains the goal of the issue, and provide some technical details if necessary.
* Should there be critical implementation steps or other useful ways to create small tasks as part of the issue, please use a checklist as part of the issue descriptions.
* The issue should have the Deliverable label, milestone and workflow label assigned.
* Design and frontend engineering use one issue. The same issues moves from workflow::design to workflow::refinement to workflow::ready for development. This ensures that there is a single source of truth for customer-facing issues. If a design issue is too large to be implemented, it may be promoted to an epic.

It's okay to create specific engineering-driven implementation issues for more complex features. These would be called **Child Issues** and they should always link back to their parent. If one issue would spawn many child issues, consider creating an Epic.

### 🏋 Weighting and Estimation Process

#### Weight Guidelines

Issues are weighted using the Fibonacci sequence (0, 1, 2, 3, 5, 8, 13+):

* **Weight 0:** Smallest issues (typos, minor formatting, simple code changes without tests)
* **Weight 1:** Simple issues with minimal uncertainty (good for new contributors)
* **Weight 2:** Straightforward issues requiring multiple code/test updates
* **Weight 3:** Larger issues with some complexity but manageable scope
* **Weight 5:** Should typically be broken down; acceptable for large manual updates with low risk
* **Weight 8/13+:** Placeholder weights indicating need for breakdown; too large or uncertain for immediate implementation

#### Weight Update Process

Every issue assigned to the upcoming milestone needs to be weighed before applying the Deliverable label by Engineering Manager. Engineering Manager needs to check whether weight is assigned and, in case of the weight being equal or above 5, works on breaking issues down into smaller ones.

Engineering manager and Product Manager are responsible for asking to weight issues assigned for the upcoming milestone during weekly team meetings. They should ask engineers to read issue descriptions before the meeting so they are ready to weight them and ask questions if needed. They can split this process between more than one meeting.

## 📝 AI Feature Evaluations Guidelines - Evaluate like I am 5

See the [Evaluate like I am 5](https://gitlab.com/gitlab-org/ai-powered/eli5) Project and read the docs [here](https://gitlab.com/gitlab-org/ai-powered/eli5/-/tree/main/doc?ref_type=heads).

## 👏 Communication

The AI Framework Team communicates based on the following guidelines:

1. Always prefer async communication over sync meetings.
1. Don't shy away from arranging a [sync call](#-ad-hoc-sync-calls) when async is proving inefficient, however always record it to share with team members.
1. By default communicate in the open.
1. All work-related communication in Slack happens in the `#g_ai_framework` channel.

### ⏲ Time Off

Team members should add any [planned time off](/handbook/people-group/time-off-and-absence/time-off-types/) in the "Workday" slack app, so that the Engineering Manager can use the proper number of days off during capacity planning.

### 🤙 Ad-hoc sync calls

We operate using async communication by default. There are times when a sync discussion can be beneficial and we encourage team members to schedule sync calls with the required team members as needed.

## 🔗 Other Useful Links

### 🌍 AIGW Region Deployments

* 🚀 [Staging](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/.runway/runway-staging.yml?ref_type=heads#L12)
* 🌐 [Production](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/.runway/runway.yml?ref_type=heads#L12)

### 📝 Dashboards (internal only)

* [Requests per provider](https://dashboards.gitlab.net/goto/Ta-BL_-NR?orgId=1)
* [Error budgets](https://dashboards.gitlab.net/d/product-ai-powered_error_budget/product-error-budgets-ai-powered?orgId=1)
* [AI Gateway SLIs](https://dashboards.gitlab.net/d/ai-gateway-main/ai-gateway-overview?orgId=1)
* [GCP quota limits](https://dashboards.gitlab.net/d/ai-gateway-main/ai-gateway-overview?orgId=1&viewPanel=1515902021)
* [LLM Sidekiq completions](https://dashboards.gitlab.net/d/sidekiq-main/sidekiq-overview?orgId=1)
* [Duo Chat Question Categorization](https://app.periscopedata.com/app/gitlab/1173299/Duo-Chat-Question-Categorization)
* [Security Issues](https://gitlab.com/groups/gitlab-org/-/issues/?sort=due_date&state=opened&label_name%5B%5D=security&label_name%5B%5D=group%3A%3Aai%20framework&first_page_size=20)
* [Reliability Issues](https://gitlab.com/gitlab-org/gitlab/-/boards/4227439?not[label_name][]=type%3A%3Afeature&label_name[]=section%3A%3Adev&label_name[]=group%3A%3Aai%20framework)
* [Sentry via CompletionWorker](https://new-sentry.gitlab.net/organizations/gitlab/issues/?query=is%3Aunresolved++CompletionWorker&referrer=issue-list&statsPeriod=14d)
* [Sentry via Feature Category](https://new-sentry.gitlab.net/organizations/gitlab/issues/?query=is%3Aunresolved+feature_category%3Aai_abstraction_layer&referrer=issue-list&statsPeriod=24h)
* [Monthly Retros](https://gitlab.com/gl-retrospectives/data-science/ai-powered/ai-framework-retros)
* [Chat QA Evaluation](https://gitlab.com/gitlab-org/ai-powered/ai-framework/qa-evaluation)
* [Chat REST API Error Ratio](https://log.gprd.gitlab.net/app/r/s/lDEwi)
* [ITPM per model](https://dashboards.gitlab.net/goto/-O0w_rsHg?orgId=1)
* [Internal handbook page](https://internal.gitlab.com/handbook/engineering/ai/ai-framework)

### 📹 GitLab Unfiltered Playlist

The AI Framework Group collates all video recordings related to the group and its team members in [a playlist](https://www.youtube.com/playlist?list=PL05JrBw4t0Kpt0DsmS5WSZbeiMgrBeZXv) in the [GitLab Unfiltered](https://www.youtube.com/channel/UCMtZ0sc1HHNtGGWZFDRTh5A) YouTube channel.
