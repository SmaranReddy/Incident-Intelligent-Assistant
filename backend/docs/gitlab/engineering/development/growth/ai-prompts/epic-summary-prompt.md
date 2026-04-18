---
title: "GitLab Epic Summary with Duo Agent"
description: "Process for summarizing epic status based on related issues, comments and statuses."
---

## Overview

This document outlines our process for generating weekly summaries for our in-progress epics.

## The Problem

It can be difficult to efficiently see the current status, health and next steps for a given epic.

## The Solution

We use Duo Agent with a standardized prompt to generate a summary in a format that is pertinent to our team as well.  The summaries also serve as snapshots of where we were at a given moment.

- Create a table of in flight issues and their workflow status
- Surface any blockers or impediments
- Call out any missed steps from previous summaries
- Calculate completed progress percent by weight
- Notify concerned parties

## Process

### Step 1: Weekly DRI update

For each epic of which you are the DRI add the epic URL to the prompt and run.

{{% details summary="Epic summary prompt, click me to expand"%}}

````markdown
Fill in epic summary markdown template for this epic and create a top-level comment with an update. 

For the SUMMARY section Provide a descriptive update of what happened since the last update summary of this epic. Evaluate next steps from previous update to ensure they were achieved, or note why they weren't.

The COMPLETE section should be the total weight of all completed issues as a percentage of all issue weights.

STATUS should answer "what is the health status of the epic?" Evaluate the health status of this epic based on: scope increases (more issues added, significant total weight increase in the middle of the project), impediments or blockers encountered and due date delivery at risk. Answer this question using only (:large_green_circle: On track  | :large_yellow_circle: Needs Attention  | :red_circle: At risk).

DUE_DATE_AT_RISK should answer "Is the due date at risk?" Answer this question using only (:large_green_circle: On track  | :large_yellow_circle: Needs Attention  | :red_circle: At risk).

For the NEXT_STEPS section cover what are the next steps that would move this epic forward towards completion.

The NOTIFY section should be a list of users that would be interested in reading this summary. It should be in the format: "@user, @user2, etc"

The Impediments, risks & blockers section should provide an additional summary of blockers and impediments that were encountered. Remove entire section if no impediments

The Remaining work section should be a table with the headings Issue, Weight and Status. It should snapshot of state of the issues at the time of making update. This helps us see the progress in time, especially when scope increases, additional issues are added etc. Only include issues that are in progress and not complete. The status column should contain the current workflow tag of the issue.

The template is as follows

# Status update - Jan 01 2025

## :clipboard: Summary
<!-- Concise, descriptive update of what happened since the last update. In the update also evaluate next steps from previous update to ensure they were achieved, or note why they weren't --> 

1. **Complete:** xx% (change from xx%) <!-- this should be % complete by weight -->
1. **Status:** :large_green_circle: On track  | :large_yellow_circle: Needs Attention  | :red_circle: At risk  
1. **Update:** 

## :construction: Impediments, risks & blockers
<!-- Additional summary of blockers and impediments that were encountered. Remove entire section if no impediments -->
1. 
1. 

## :todo: Remaining work
<!-- Snapshot of state of the issues at the time of making update. This helps us see the progress in time, especially when scope increases, additional issues are added etc. -->

| Issue | Weight | Status |
|-------|--------|--------|
| <!-- please use +s on issue link --> | | ~"workflow::verification" |
|  | | ~"workflow::in dev" |
|  | | ~"workflow::in review" |

* Total issues: 
* Total closed issues: 
* Total weight of estimated issues:
* Total weight of completed issues:
* Currently we are `xx%` complete by issue weight with `x` unestimated issues. 

## 🎯 Next Steps
<!-- What are the next steps to complete in upcoming time period (aligned to the cadence of updates) -->
1. 
1. 

## Notify
<!-- Tag anyone who should be receiving this update - PM/EM, engineers on the team, stakeholders -->
cc: `@...`
````

{{% /details %}}

### Step 2: Review and Post

Review the summary for:

- Sound summary guidance
- Clear next steps
- Accurate notification list

## Team Benefits

- **Clear Understanding**: No guesswork about what needs to be done next and what has been completed
- **Time Savings**: Summaries can be automatically generated and the template can be modified to meet current needs

### When to Use

When you want to communicate the current progress of an epic or snapshot the current progress at a milestone.

## Success Criteria

We measure success through a reduction of comments asking for updates regarding the current state of the epics or its related issues.

## Feedback and Iteration

This process should evolve based on team feedback and results.

- Prompt effectiveness and accuracy
- Duo Agentic performance improvements
