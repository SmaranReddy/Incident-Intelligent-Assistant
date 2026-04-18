---
title: "DRI responsibilities"
description: "How the Organizations team handles ownership of team initiatives"
---

As the Handbook [explains](/handbook/people-group/directly-responsible-individuals/), DRI status for a given Epic means that you should “be able to articulate the objectives, check progress and give and receive feedback” for your area of responsibility.

For our team, DRI status means ownership of a piece of work – this extends to refining work within it, making recommendations for prioritisation and scheduling relating to that work, and writing regular updates to share progress, blockers and achievements with the wider team.

Why do this? DRI status is typically used to support quick decision-making. But it also means taking technical leadership over an area where you’re the key decision maker and the most informed expert. As part of your career development and team influence, being able to articulate what work we should pick up next or which areas need escalating due to engineering complexities is hugely valuable – both to you and to GitLab.

## Three responsibilities

An individual engineer may be a DRI for several epics concurrently, representing team objectives and deliverables. They may share this status with others depending on the scale of the work or their seniority.

There are three areas of responsibility within this which a DRI should support the Engineering and Product managers:

### 1. Status updates on previous work

Each week, the named DRI(s) for a given epic will be pinged by the [Epic Issue Summaries](https://gitlab.com/gitlab-com/gl-infra/epic-issue-summaries) bot. This bot asks the DRI(s) to provide a structured weekly update message, which is folded into multiple other places later in the week.

### 2. Refinement of upcoming work

As the milestone progresses, the DRI should select upcoming issues for refinement within their epic. This should be a scheduled activity, perhaps prompted/supported by automation: create placeholder/stub issues early in the milestone, then as the milestone progresses, flesh these out with implementation plans. Involve the wider team where needed. At the end of the milestone, confirm there is enough refined work to proceed with, and continue to add detail where needed.

### 3. Planning for the next milestone

As the end of the milestone approaches, the DRI should have an approximate understanding of what work is likely to slip into the next release, and which new items should be brought in. Make a recommendation (and proactively schedule issues after confirming) for new work to be planned, especially if you’re aware of blockers, dependencies or user needs which make some issues higher priority than others.

These tasks will require some time management to build into your week, but should ensure that we’re prioritising the right things from an engineering perspective, that we’re surfacing blockers or risks as early as possible, and that we celebrate wins (even small ones) on a regular basis.

## Detailed steps for each ask

Here are some more detailed guidelines on the expectation for a DRI for each of the above requirements:

### Status updates on your epics

Each Monday, you’ll be pinged by the [Epic Issue Summaries](https://gitlab.com/gitlab-com/gl-infra/epic-issue-summaries) bot for a status update. These updates are used for a number of things:

* They’re propagated to the parent epic in a nice table summary of child work (see [an example here](https://gitlab.com/groups/gitlab-org/-/epics/19412#project-work))
* They’re used to help populate sync call agendas for things like the Protocells Weekly Standup
* They’re used to assemble the team’s weekly Outlook report which contributes to the department-wide Grand Review.

A well-written status update has the following properties / goals:

* **Stay low-context**: anyone reading it who sits outside our team or only has basic familiarity with our workstreams should be able to read it and understand what each bullet point means.
* **Explain the wins**: if we’ve shipped something with a complex technical background, add some background to outline why it’s important for someone with less familiarity.
* **Use links for additional detail**: people may wish to follow up so link related MRs, blocking issues, or other useful places.

Or to put it another way, here’s what a **bad** status update looks like:

* Completed [!123456](https://gitlab.com), will pick up [#98765](https://gitlab.com) next.
* Shipped the Foo Widget update to improve performance.
* Next: [Unlink dashboard to component factory #546453](https://gitlab.com).

Note that there’s no context, it’s just a list of links, and there’s no attempt to explain the wider impact of the work.

People read these things, so put some time/effort into writing them! If you’re not sure how to phrase something for this audience, try asking Duo to help – we have [a team handbook page with an example prompt for this](/handbook/engineering/infrastructure-platforms/tenant-scale/organizations/ai-prompts/#epic-summary-prompt). There's also [an automation script](https://gitlab.com/kkyrala/gitlab-epic-summarizer) which can automatically generate an epic update which you can review and post.

Here are a couple of good real world examples to base your updates on:

* [https://gitlab.com/groups/gitlab-org/-/epics/19425#note_2850897317](https://gitlab.com/groups/gitlab-org/-/epics/19425#note_2850897317)
* [https://gitlab.com/groups/gitlab-org/-/work_items/19413#note_2826864047](https://gitlab.com/groups/gitlab-org/-/work_items/19413#note_2826864047)
* [https://gitlab.com/groups/gitlab-org/-/epics/19416#note_2914215438](https://gitlab.com/groups/gitlab-org/-/epics/19416#note_2914215438)

### Refining future work

The whole team support one another with refinement. We do this by asking DRIs—subject matter experts—to guide us on where to help.

If you’re the DRI for an epic, you can do this by visiting the new [team Refinement Dashboard](https://refinement-tracker-651035.gitlab.io/) – this tool will list out all epics you’re assigned to, and pick up any child issues with either a Label of `workflow::refinement` or a Status of `Refinement`. Use this tool to find issues in need of refinement – it updates every hour.

* Put some time aside each week to go through the upcoming work for your Epic
* Where an issue needs refinement, [follow the existing instructions](/handbook/engineering/infrastructure-platforms/tenant-scale/organizations/#step-2-refining-issues) to add details (such as an implementation guide, weight, breakdown of issues etc)
* If you need support from other colleagues, mention them directly, or use the team’s group tags:
  * `@gitlab-com/gl-infra/tenant-scale/organizations/backend-engineers`
  * `@gitlab-com/gl-infra/tenant-scale/organizations/frontend-engineers`
* Make sure that, as the milestone progresses, the number of your items in need of refinement decreases.

### Planning issues for upcoming milestones

As a milestone ends and a new one begins, you’ll be asked to recommend work to schedule for the upcoming release, which is tracked in a planning issue. You should already be reporting against the status of your Epic and already flagging things that may slip due to complexities, bugs or other unknowns.

As this happens, be proactive and follow the [bias for action](/teamops/decision-velocity/#bias-for-action) value: you should be empowered to reschedule an issue for a future milestone if it becomes clear that we’ll be unable to get to it when planned. When this happens, leave a comment on the issue explaining your decision so everyone is aware.

Otherwise, as new work is refined, make a recommendation for when we should schedule something. Here are some examples of when you may wish to make a priority call:

* When external factors may limit us (eg. required stops or maintenance windows that we need to fit into)
* When there are sequential deployment steps (eg. add a feature flag in one release so we can enable it in the next one)
* When there are cross-team impacts which mean we unblock another team by releasing a specific feature first

Here is a great example of this kind of ownership of Epic(s) for an impending milestone: https://gitlab.com/gitlab-com/gl-infra/tenant-scale/group-tasks/-/work_items/478#note_2807737845

Remember: you’re the directly responsible individual and you know this area well: the team wants you to own this work and needs you to call out things that will help us get there faster.

## In summary / TL;DR

* Write a weekly update on a Monday with low-context, explanations of why something is good (or bad) and what you need help with
* Refine new work within your Epic on a weekly basis and ask for input from others
* When a new milestone is approaching, help plan the next priorities for your areas and check in on anything that may slip.
