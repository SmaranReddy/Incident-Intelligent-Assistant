---
title: "Editor Extensions: Multi-Platform Group"
description: We own and maintain code editor extensions for JetBrains, Visual Studio, Neovim & Eclipse IDEs, as well as contributing to the Language Server bringing GitLab's core features and AI capabilities directly into developer workflows.
---

## 🚀 Vision

We bring GitLab's core features and AI capabilities directly into developer workflows, unlocking productivity by making GitLab accessible in the tools developers use every day.

**Note:** Along with the [Editor Extensions: VS Code](/handbook/engineering/ai/editor-extensions-vscode/) team, we are responsible for all aspects of the product categories that fall under the [Editor Extensions group](/handbook/product/categories/#editor-extensions-group) of the [AI-Powered stage](/handbook/product/categories/#ai-powered-stage) of the [DevOps lifecycle](/handbook/product/categories/#devops-stages).

---

## 👨‍💻 Team Members

{{< team-by-manager-slug "aelhusseiny" >}}

---

## 🤝 Stable counterparts

Below are our [stable counterparts](/handbook/leadership/#stable-counterparts)

{{< group-by-slugs dashaadu mcorren sam_reiss matthewpetersen>}}

---

## 💬 Where to find us

### Slack

- Team:
  - [#g_editor-extensions](https://gitlab.slack.com/archives/C058YCHP17C) -
  - [#g_editor-extensions_social](https://gitlab.slack.com/archives/C062W19B8NR)
  - [#g_editor-extensions-standup](https://gitlab.slack.com/archives/C058YCWPKMG)
- Projects
  - Language Server: [#f_language_server](https://gitlab.slack.com/archives/C05B1PFHRPU)
  - JetBrains extension: [#f_jetbrains_plugin](https://gitlab.slack.com/archives/C02UY9XKABH)
  - Visual Studio extension: [#f_visual_studio_extension](https://gitlab.slack.com/archives/C0581SE363C)
  - Eclipse extension: [##f_eclipse_plugin](https://gitlab.enterprise.slack.com/archives/C07MKHCFGHG)
  - Neovim extension: [#f_neovim_plugin](https://gitlab.slack.com/archives/C05BF7L6PEX)

### Shared Calendar

Editor Extensions Shared Calendar (Calendar ID: c_673d889354d021f7fa9f20a003b5867185a9bf12989b5eaacbc8b537cc9ef27c@group.calendar.google.com)

---

## 💻 Scope

### Our team owns the following products

1. **GitLab Extension for JetBrains**
   1. [Repo](https://gitlab.com/gitlab-org/editor-extensions/gitlab-jetbrains-plugin)
   2. [Docs](https://docs.gitlab.com/editor_extensions/jetbrains_ide/)
   3. [Backlog](https://gitlab.com/groups/gitlab-org/-/issues/?label_name%5B%5D=Editor%20Extensions%3A%3AJetBrains)
   4. Slack Channel: [#f_jetbrains_plugin](https://gitlab.enterprise.slack.com/archives/C02UY9XKABH)
2. **GitLab Extension for Visual Studio**
   1. [Repo](https://gitlab.com/gitlab-org/editor-extensions/gitlab-visual-studio-extension)
   2. [Docs](https://docs.gitlab.com/editor_extensions/visual_studio/)
   3. [Backlog](https://gitlab.com/groups/gitlab-org/-/issues/?label_name%5B%5D=Editor%20Extensions%3A%3AVisual%20Studio)
   4. Slack Channel: [#f_visual_studio_extension](https://gitlab.enterprise.slack.com/archives/C0581SE363C)
3. **GitLab Extension for Eclipse**
   1. [Repo](https://gitlab.com/gitlab-org/editor-extensions/gitlab-eclipse-plugin)
   2. [Docs](https://docs.gitlab.com/editor_extensions/eclipse/)
   3. [Backlog](https://gitlab.com/groups/gitlab-org/-/issues/?label_name%5B%5D=Editor%20Extensions%3A%3AEclipse)
   4. Slack Channel: [#f_eclipse_plugin](https://gitlab.enterprise.slack.com/archives/C07MKHCFGHG)
4. **GitLab Extension for Neovim**
   1. [Repo](https://gitlab.com/gitlab-org/editor-extensions/gitlab.vim)
   2. [Docs](https://docs.gitlab.com/editor_extensions/neovim/)
   3. [Backlog](https://gitlab.com/groups/gitlab-org/-/issues/?label_name%5B%5D=Editor%20Extensions%3A%3ANeovim)
   4. Slack Channel: [#f_neovim_plugin](https://gitlab.enterprise.slack.com/archives/C05BF7L6PEX)

### As well as co-own with the Editor Extensions: VS Code group

1. **GitLab Language Server**
   1. [Repo](https://gitlab.com/gitlab-org/editor-extensions/gitlab-lsp)
   2. [Backlog](https://gitlab.com/groups/gitlab-org/-/issues/?label_name%5B%5D=Editor%20Extensions%3A%3ALanguage%20Server)
   3. Slack Channel: [#f_language_server](https://gitlab.enterprise.slack.com/archives/C05B1PFHRPU)

---

## 📚 How we work

Something about how we take this as an iterative approach

### Issues' state

We use the issue `Status` field to indicate state, following the [Product Development Flow](/handbook/product-development/how-we-work/product-development-flow/)

{{% details summary="Expand for more details"%}}
. To keep things simple, we focus on the main states below and optionally use others when appropriate.

- **New →** Not yet prioritized or refined.

- **Planning breakdown →** Needs team attention soon (within ~1–2 months); gather scope, risks/deps, acceptance criteria.

- **Ready for development →** Immediate priority; clear scope; should be picked up next and ideally completed within ~2 weeks.

- **In dev →** Assigned DRI(s), milestone set, work in progress.

- **In review →** Implementation complete; MR opened and under review/verification.

- **Blocked →** Cannot proceed due to a dependency or external constraint; comment the blocker and next check-in date.

- **Closed →** Done (or closed as duplicate/won't fix) with outcome noted.

**Note:** We chose `Planning breakdown` & `Ready for development` as these statues exist on both Issues and Tasks, letting us build one unified set of boards and embedded tables with a single status filter easily.

{{% /details %}}

### Issues Boards

- [By assignee](https://gitlab.com/groups/gitlab-org/-/boards/9154815?label_name[]=group%3A%3Aeditor%20extensions)
- [By Status](https://gitlab.com/groups/gitlab-org/-/boards/9651444?label_name%5B%5D=group%3A%3Aeditor%20extensions)

### Weekly Issues' refinement

- **Purpose:** Keep our backlog focused and up-to-date
  - We re-prioritize and adapt to the changing requirements
  - Create clarity about what customer support issues we need to implement
  - Give everyone a chance to surface and champion issues (new/urgent bugs, tech improvements or debt)
- **Format:** 3 stage process described in the "Flow" section below
- **Cadence:** Every week
- **Output:** Up-to-date [Rolling Backlog](#rolling-backlog-wiki-page) + possible weekly updates to the [milestone planning issue](#milestone-planning-issues)

{{% details summary="Expand for more details"%}}

#### Rolling Backlog Wiki Page

This [rolling backlog page](https://gitlab.com/gitlab-org/editor-extensions/meta/-/wikis/Editor-Extensions:-Multi-Platform-Rolling-Backlog) is the output of this refinement process, it serves as a visible, prioritized, and team-reviewed top-of-backlog list for every scope we own (could be the next 1-2 months of work).

**Note:** We also take into account most thumbed up issues to make sure we also capture community feedback.

---

#### Flow

1. **EM & PM async weekly pre-pass**: prepare for the next 2 steps
   1. Some issues sources to consider
      - [Rolling Backlog](#rolling-backlog-wiki-page)
      - Slack pings, team discussions
      - Recent [triage-reports](https://gitlab.com/gitlab-org/quality/triage-reports) issues, for example, [Bugs Prioritization](https://gitlab.com/gitlab-org/quality/triage-reports/-/issues/26385), [Triage Report](https://gitlab.com/gitlab-org/quality/triage-reports/-/issues/26506), [Feature flags requiring attention](https://gitlab.com/gitlab-org/quality/triage-reports/-/issues/26361)
   2. Add the label [workflow::scheduling](https://gitlab.com/groups/gitlab-org/-/issues?sort=updated_asc&state=opened&first_page_size=100&label_name%5B%5D=workflow%3A%3Ascheduling&label_name%5B%5D=group%3A%3Aeditor+extensions) to the issues to discuss in step #3 (EM + PM Sync call)
   3. Update issues as needed to make sure [Rolling Backlog](#rolling-backlog-wiki-page) is up-to-date.

2. **Team async bi-weekly review (timebox ~1 hour)**:
   - EM will create an **issue to trigger** this review and assign all engineers, [example issue](https://gitlab.com/gitlab-org/editor-extensions/meta/-/issues/257), please unassign yourself when you finish review
   - Occurs **every 2 weeks**, to avoid taking too much of the team’s focus each week.
   - **Goals**:
      - Ensure everyone on the team weighs in on the priorities.
      - Advance tech improvements or tech debt issues.
      - Flag issues that should be given higher importance.
      - Identify issues that are no longer needed or should be de-prioritized.
   - **How to Participate:**
     1. Review the current priorities in [Rolling Backlog](#rolling-backlog-wiki-page)
     2. Weigh in: Add/Change the following as needed (including for new issues you want to push):
        1. Status: `Planning breakdown`: Should be on our radar, within the next 2-3 milestones or so
        2. Status: `Ready for development`: Ideally should done this milestone
        3. Label: [workflow::schedulingt](https://gitlab.com/groups/gitlab-org/-/issues?sort=updated_asc&state=opened&first_page_size=100&label_name%5B%5D=workflow%3A%3Ascheduling&label_name%5B%5D=group%3A%3Aeditor+extensions): Immediate priority, should be picked up next (or at least should be discussed this week)
        4. ~"workflow::scheduling"
     3. Comment & tag EM + PM if you change any of the above with reasoning
   - **Notes**:
     1. Don't spend time deep investigating any issue at this point, just high level overview of the priorities is enough
     2. Comment in the relevant issue itself or in the rolling backlog issue if your comment is more relevant there
     3. Link to [references](https://docs.gitlab.com/user/markdown/#gitlab-specific-references) to specific GitLab resources to improve discoverability.
3. **EM + PM weekly sync call:**  Focus on the label [workflow::scheduling](https://gitlab.com/groups/gitlab-org/-/issues?sort=updated_asc&state=opened&first_page_size=100&label_name%5B%5D=workflow%3A%3Ascheduling&label_name%5B%5D=group%3A%3Aeditor+extensions), read team comments, confirm ranking, make trade-offs, and pick target deliverables for the next 1-2 weeks.
      - Meeting should be scheduled on the [shared team calendar](#shared-calendar) and recorded, we will also use zoom transcript so team can have transparency on the reasoning and arguments behind changing an issue priority or order and add it to the planning issue.
      - Once every month use this call will be used to create a milestone planning issue that can be subject to change with the next refinement)
      - EM is responsible for assigning the agreed on & updated deliverables
      - **Output:** Update the milestone planning issue weekly + EM to assign updated priority issues
        - Once a month, before the start of the milestone, the discussion will span a bigger number of tickets in preparation for creating a new [milestone planning issue](#milestone-planning-issues)

---

#### How to Participate (for non-team members)

If you want to bring an issue to the attention of the team, please create an issue, if no issue exist yet, then reach out on [#g_editor-extensions](https://gitlab.enterprise.slack.com/archives/C058YCHP17C)

{{% /details %}}

### Monthly Planning

- **Purpose:** Define and scope what we’ll deliver in the current milestone
  - Plan monthly to shape product priorities, but revisit weekly to stay adaptive to fast-changing AI requirements, quality focus, and urgent user needs
- **Format:** 2 stage process described in the "Flow" section below
- **Cadence:** Every month + possible weekly updates
- **Output:**
  - Up-to-date [milestone planning issue](https://gitlab.com/gitlab-org/editor-extensions/meta/-/issues/?sort=created_date&state=all&label_name%5B%5D=Planning%20Issue)
  - Updated issues fields: `status`, `milestone`, and labels to include `deliverable`

{{% details summary="Expand for more details"%}}

#### Milestone Planning Issues

We use [Milestone Planning Issues](https://gitlab.com/gitlab-org/editor-extensions/meta/-/issues/?sort=created_date&state=all&label_name%5B%5D=Planning%20Issue)
to define our goals for the current/upcoming milestone.
The PM and EM are responsible for aligning on the goals.
The planning issues are [created automatically](https://gitlab.com/gitlab-org/editor-extensions/meta/-/tree/main#issue-creation-process) every month.

---

#### Flow

1. **Fill Initial milestone planning issue**
   - PM drives the goals of the milestone, in alignment with EM
   - All the output of the previous [issues refinement](#weekly-issues-refinement) is taken into consideration
   - Should be done before the [milestone starts](https://mnohr.gitlab.io/milestone-dates/)
2. **Weekly updates** after the weekly [issues refinement](#weekly-issues-refinement)
   - EM + PM update the [milestone planning issue](https://gitlab.com/gitlab-org/editor-extensions/meta/-/issues/?sort=created_date&state=all&label_name%5B%5D=Planning%20Issue), if needed, reflecting latest changes, that could include for example:
     - Urgent bugs that came up
     - Needing to adopt another AI engineering team work into the IDEs
     - Flagged tech debt
   - EM aligns with the team to assign [Deliverable](https://gitlab.com/groups/gitlab-org/-/issues/?label_name%5B%5D=Deliverable) & [Stretch](https://gitlab.com/groups/gitlab-org/-/issues/?label_name%5B%5D=Stretch) labels and ensure issues status, [weight](#issues-weight) & milestone are assigned reflecting the latest priorities
     - Possibly in weekly 1:1 or async

{{% /details %}}

### Difference between Milestone Planning Issue & Rolling Backlog Wiki Page

- [Rolling backlog](#rolling-backlog-wiki-page): a list of all prioritized issues, providing visibility into the top of our backlog, which can span several months of work.
- [Milestone planning issue](#milestone-planning-issues): a subset from the rolling backlog backlog, the issues we prioritized from the backlog to implement in the current milestone.

### Team sync meetings

We have a sync meeting once per week. This is collaborative meeting between this team and the [Editor Extensions: VS Code](/handbook/engineering/ai/editor-extensions-vscode/) team

- Recordings are uploaded to the [Editor Extensions Category](https://www.youtube.com/playlist?list=PL05JrBw4t0KoC0pFfuNOAQjKxe4_ypFKc) playlist on GitLab Unfiltered.
- The timing of the call alternate every week between APAC/AMER & EMEA/AMER so everyone in both team in different timezones can join sync conveniently at least every other call, and everyone can contribute async as well every week
- [Weekly Sync Meeting Agenda](https://docs.google.com/document/d/1UJg-Prf5qGjiGImvaYl5HNjMcJddoeE4u33Ri6SxQ6g), agenda is open, everyone in the team is invited to bring relevant topics to align on

### Weekly async updates

Team members post weekly async updates on the in progress issues using the [Dev Check-in (editor-extensions)](https://gitlab.com/groups/gitlab-org/editor-extensions/-/comment_templates) comment template

### Issues' labels

Some of the labels we use to track our work across different projects

| Label | Description |
|---|---|
| [`Editor Extensions::JetBrains`](https://gitlab.com/groups/gitlab-org/-/issues/?label_name%5B%5D=Editor%20Extensions%3A%3AJetBrains) | JetBrains IDE plugin features, and maintenance. |
| [`Editor Extensions::Visual Studio`](https://gitlab.com/groups/gitlab-org/-/issues/?label_name%5B%5D=Editor%20Extensions%3A%3AVisual%20Studio) | Visual Studio IDE plugin features, and maintenance. |
| [`Editor Extensions::Neovim`](https://gitlab.com/groups/gitlab-org/-/issues/?label_name%5B%5D=Editor%20Extensions%3A%3ANeovim) | Neovim IDE plugin features, and maintenance. |
| [`Editor Extensions::Eclipse`](https://gitlab.com/groups/gitlab-org/-/issues/?label_name%5B%5D=Editor%20Extensions%3A%3AEclipse) | Eclipse  IDE plugin features, and maintenance. |
| [`Editor Extensions::Language Server`](https://gitlab.com/groups/gitlab-org/-/issues/?label_name%5B%5D=Editor%20Extensions%3A%3ALanguage%20Server) | Shared LSP features, maintenance, and IDE-parity efforts. |
| [`Editor Extensions::All`](https://gitlab.com/groups/gitlab-org/-/issues/?label_name%5B%5D=Editor%20Extensions%3A%3AAll) | Cross-cutting items impacting multiple editor extensions. |
| [`Deliverable`](https://gitlab.com/groups/gitlab-org/-/issues/?label_name%5B%5D=Deliverable) | Committed items for the milestone / must-ship work. |

### Issues' weight

We use three weights to give a rough estimate of the issue's complexity. The weight is made out of two parts:

- `1` - day or two of effort
- `2` - week of effort
- `3` - week and a half of effort

**Notes:**

1. Everything with a base weight above `3` should be a spike that will result in one or more issues with estimated weight.
2. These are estimates for someone familiar with the codebase/system, you can add extra weight `1` or `2` to the base weight if you are new to the team/codebase/system.
3. Weights are assigned in [planning flow, step #2](#flow-1)

---

## 🔗 Useful Links

- **Planning**
  - <a href="https://gitlab.com/groups/gitlab-org/-/boards/9154815?label_name[]=group%3A%3Aeditor%20extensions" target="_blank">Issues Board By Assignee</a>
  - <a href="https://gitlab.com/groups/gitlab-org/-/boards/9651444?label_name%5B%5D=group%3A%3Aeditor%20extensions" target="_blank">Issues Board By Status</a>
  - <a href="https://gitlab.com/gitlab-org/editor-extensions/meta/-/issues/?sort=created_date&state=all&label_name%5B%5D=Planning%20Issue" target="_blank">Milestones Planning Issues</a>
- **Dashboard & Monitoring**
  - <a href="https://10az.online.tableau.com/#/site/gitlab/views/PDCodeSuggestions/IDEMetrics" target="_blank">Tableau code suggestions IDE metrics</a>
  - <a href="https://session-error-rates-dashboard-87d159.gitlab.io/" target="_blank">Agent Platform Session Error Rates</a>
  - <a href="https://dashboards.gitlab.net/dashboards/f/editor-extensions/?orgId=1" target="_blank">Grafana Dashboard</a>
  - <a href="https://10az.online.tableau.com/#/site/gitlab/views/DRAFTCentralizedGMAUDashboard/MetricReporting/48493d6c-cd11-45b9-bdc5-bf5242e0de0b/EditorExtensionsMAU?:iid=2" target="_blank">Tableau MAU</a>

- **Miscellaneous**
  - <a href="https://docs.google.com/document/d/1UJg-Prf5qGjiGImvaYl5HNjMcJddoeE4u33Ri6SxQ6g" target="_blank">Weekly Sync Meeting Agenda</a>
  - <a href="https://www.youtube.com/playlist?list=PL05JrBw4t0KoC0pFfuNOAQjKxe4_ypFKc" target="_blank">Editor Extensions playlist</a> on the GitLab Unfiltered YouTube channel
