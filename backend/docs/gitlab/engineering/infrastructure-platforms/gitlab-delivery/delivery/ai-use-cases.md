---
title: Release and Deployment LLM/AI Use Cases
summary: This page outlines possible use cases of AI/LLM for Release and Deployments.  These hopefully will help offset repetitive tasks, assist in analysis Release Managers do on a recurring basis or assist in upcoming development improvements we have for the deployment process.  Some cases will require agentic agents to work.
---

## Proposed use cases

### 1. Assist with handovers for Release Managers

Release Managers can have complex daily handovers.  Incident.io provides nice summaries that end up in incident issues.  We could have Duo Agent look over the production issue queue for the `~incident` label and then look for incidents that have summary items or implications for release managers.  A summary comment could be done for each regional handoff (APAC -> EMEA), (EMEA -> AMER), (AMER -> APAC) in an issue.

### 2. Assist in summarizing and reading out weekly release metrics like deployment blockers

In our weekly Delivery meeting, we have a readout on [our metrics](https://about.gitlab.com/handbook/engineering/deployments-and-releases/#weekly-delivery-metrics-review) for how the last week went.  One of those steps is to categorize, describe and look at trends in [Deployment Blockers](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/1496).  This work is currently manual.  The summarization work could be done by asking for a summary on a given release task issue ([example](https://gitlab.com/gitlab-org/release/tasks/-/issues/21317)).

### 3. Future extensions of chatops to ease deployment manager work

We are looking at further extending chatops to add a deployment manager like role.  New chatops commands will need to be added [1](https://gitlab.com/gitlab-org/release/tasks/-/merge_requests/133#note_2746357795), [2](https://gitlab.com/gitlab-org/release/tasks/-/merge_requests/133#note_2747860135). We should attempt to use Duo Agent to help with that development.

### 4. Automation of Backport work

With the extension of the [maintenance policy soft launching](https://gitlab.com/gitlab-com/gl-infra/delivery/-/issues/21474), we are continuing to look at ways to make it easier for development teams to do backports to supported versions.  Part of that work is creating MRs to stable branches after a bug has been fixed for the version currently in development.  We wonder if Duo Agent could help create these backport MRs.

### 5. Assist finding release documents

Release&Deploy team has extensive documentation at https://gitlab.com/gitlab-org/release/docs for knowledge sharing and guiding engineers how a code change makes it to different types of release. Using GitLab Duo Agentic Chat saves time for both stakeholders and Release&Deploy team members by reducing question&answer effort.
