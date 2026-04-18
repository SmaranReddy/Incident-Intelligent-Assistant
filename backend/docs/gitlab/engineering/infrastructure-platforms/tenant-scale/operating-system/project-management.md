---
title: "Group Tenant Scale - Operating System - Project Management"
description: "Group Tenant Scale project management practices and processes."
---

## Overview

Structured and consistent project management practices are important for ensuring that team members and stakeholders understand:

- The status of in-progress and planned work.
- Why the work we are doing is important and how it delivers value for our customers.
- The scope of what we’re working on.

## Expectations

1. We align to the Infrastructure Platforms [guidance](/handbook/engineering/infrastructure-platforms/project-management/#epics) for implementation epics for both epics and issues. Most importantly:

- Epics/issues have a problem statement that explains why we’re doing the work and how it delivers customer value.
- Epics/issues that are in-progress have an assignee.
- Epics/issues have a clearly defined exit criteria.
- Epics that are in-progress and planned have start and due dates set. Due dates are regularly evaluated and updated as needed. Issues that are in-progress or planned have a milestone assigned.
- Epics contain a status section that is updated weekly. We use the [epic-issue-summaries](https://gitlab.com/gitlab-com/gl-infra/epic-issue-summaries) project to ping the epic DRI for updates. Updates are rolled up from child epics into a top-level epic.

1. Epics must have a defined way to measure progress. It’s up to you how you want to measure progress (examples: issues closed, weight finished, % exit criteria completed, etc).
1. Epics/Issues are measurable. Customer facing features always include success/ ROI metrics. Technical deliverables always include availability or performance metrics. This enables quick and accurate prioritization and trade-off decisions.

## Inputs and Artifacts

1. **Deliverable Epics:** A deliverable epic represents work of significant business value that we are committed to completing in a specified timeframe.

- We denote these epics with the “Deliverable” label for tracking purposes.
- Deliverable epics should be scoped to a quarter or milestone. If a deliverable epic is projected to span multiple quarters, it is an indication that we need to narrow the scope or break it down into smaller milestones.
- Deliverable epics align to the Infrastructure Platforms [guidance](/handbook/engineering/infrastructure-platforms/project-management/#epics) for implementation epics.
- We use the epic’s health status field to denote the following broad delivery confidence:
  - On Track: 80%+ confidence to complete by due date
  - Needs Attention: 50-80% confidence to complete by due date. Additional support may be required to be back on track.
  - At Risk: Less than 50% confidence to complete by due date. Urgent action required to meet the due date or timeline needs revision.
- We keep a status log of when an epic has had timeline changes or health status updates. This provides us data that will improve our estimation abilities.

1. **GTS Weekly Update Issue:** Each quarter, we create an issue for EMs to add weekly updates for their teams. We use these issues as input for our weekly Infrastructure Platforms Grand Review update.

## Operating Cadences

1. Protocells Code Yellow Standup (weekly on Tuesdays)
1. GTS Weekly Grand Review Update
1. GTS Weekly CXO Company Update
