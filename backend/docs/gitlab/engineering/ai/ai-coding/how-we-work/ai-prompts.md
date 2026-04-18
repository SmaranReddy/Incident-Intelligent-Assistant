---
title: "AI Coding AI Prompts for Productivity"
description: "Sharing some common prompts we can use with Duo Chat to be more efficient"
---

With Duo Agent Platform and Duo Chat we now have access to agents and tools that can make our processes more efficient. Here are some of the prompts we have collected to make this easier.

## Researching File History

### Use Case / Impact

When looking to make a change to a file, we can quickly reference the file history to see what has changed, who has worked on it, and what are the related files. That should help more quickly get up to speed on the previous work that has been done on that file.

### Prompt

You will have to replace `FILE_PATH` with the path to the file you are interested in.

```markdown
Analyze the complete history for [FILE_PATH] and provide a report with the following information:

**IMPORTANT: Only include information that is directly found in the actual file history, commits, merge requests, and related documentation. Do not infer, guess, or speculate about any details except for identifying subject matter experts.**

**File History Overview:**
- Total number of changes/modifications to this file
- Date range of activity (first change to most recent)
- Last modification date and time

**Key Contributors & Subject Matter Experts:**
- Top 5 contributors by number of substantive changes (exclude pure refactoring/chores)
- 5 most recent contributors with substantive changes (feature additions, bug fixes, logic changes)
- Subject matter experts by area (infer from their commit patterns and areas of focus)

**Bug History & Considerations:**
- Recent bug fixes affecting this file (last 6 months) - only those explicitly identified as bug fixes
- Common types of issues/bugs found in this file (based on actual commit messages mentioning bugs/fixes)
- Critical bugs or security issues (only if explicitly mentioned as such)

**Related Test Files:**
- Test files that are actually modified in the same commits/merge requests as this file

**Related Files & Dependencies:**
- Files that are actually changed together with this file in the same commits/merge requests
- Only include files that appear in the actual git history alongside this file

**Recent Activity (Last 5 Substantive Changes):**
For each of the last 5 non-trivial modifications:
- Exact change identifier and author
- Exact date and commit message
- Type of change (only if explicitly stated in commit message)
- Brief description based only on commit message and actual changes

If any section has no concrete information available in the history, exclude that section rather than making assumptions. Use bold text for section headers instead of markdown headers for better readability.
```

## Generating Daily Standup Reports

### Use Case / Impact

We post daily standup updates in Slack using Geekbot. We report on what we have done recently, what we plan to do, and if there are any blockers. Collecting this information can be time consuming and prone to missing information, even if we are doing this every day.

By having AI tools look through our recent activity, we can create a first draft of the report quickly.

We think this could save up to an hour per week (10-15 minutes per day) per engineer on the team.

### Prompt

This prompt looks at your activity over the past 72 hours. This is likely more than we need, but it will catch multiple days like over a weekend. LLMs seem to have a hard time knowing the current time and do not know when you last updated your standup, so having a few days of activity will give us a good chance of catching all activity.

```markdown
## Objective

Generate my daily standup summary by reviewing my GitLab activity from the past 72 hours.

## Query Strategy

1. Identify my GitLab user ID for filtering
2. Query merge requests (last 72 hours):
   - `author_id=me` with states: merged, opened, closed
   - `assignee_id=me` or `reviewer_id=me` with state: opened
3. Query commits authored by me in the last 72 hours
4. Query issues with my recent activity (assigned, commented, created) from last 72 hours
5. Fetch recent discussions/comments I've participated in within the last 72 hours

## Output Format

### What have you done since last report?

- Merged MRs with title and number
- Resolved or closed issues with brief context
- Code reviews completed (approved/commented on others' MRs)
- Commits pushed to active branches/MRs
- Progress on open MRs (addressed feedback, updated code)

### What will you do today?

- Continue work on open MRs (specify which ones and next steps)
- MRs assigned to me for review
- Issues to investigate or start
- Follow-ups needed from yesterday's work
- Meetings or planning sessions related to development work

### Anything blocking your progress?

- MRs waiting on reviews from others (include reviewer names)
- Blocked MRs with external dependencies
- Stalled reviews (no activity \> 2 days)
- Waiting on decisions or clarifications
- Technical blockers or access issues

## Formatting Guidelines

- Use bullet points with 1-2 sentances per item
- Do not use sub-bullets
- Always include MR/issue numbers (!XXXX, #XXXX)
- Bold important status indicators (**BLOCKED**, **Failing CI**)
- Group related activities for the same feature
- Include metrics where helpful (e.g., "1 approval received")
```
