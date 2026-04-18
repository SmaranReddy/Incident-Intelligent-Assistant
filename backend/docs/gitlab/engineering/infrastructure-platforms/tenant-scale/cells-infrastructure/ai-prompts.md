---
title: "AI Prompts"
description: "A library of useful prompts for Duo and other AI tools used by our team"
---

← [Back to Cells Infrastructure Team](_index.md)

This page contains AI prompts organized by workflow stages to support our [Development Workflow](_index.md#development-workflow) and team processes.

## Issue Management & Planning

### Issue Readiness

Use this prompt to evaluate if an issue is ready for development, following our [Issue Status Reference](_index.md#issue-status-reference):

```markdown
Evaluate this issue based on the following checklist and provide recommendations

### Core Requirements
- [ ] **Context**: Team/section identified with labels?
- [ ] **Problem**: Clear current vs desired state?
- [ ] **Action**: Next steps with checkboxes or clear options?
- [ ] **Links**: Parent epic/initiative referenced?
- [ ] **Specifics**: Exact component/table/index/config names provided where known?

### Discussion Issues Must Have
- [ ] **Questions**: Specific items needing input listed?
- [ ] **Stakeholders**: Tagged with reason for involvement?
- [ ] **Timeline**: Decision or action timeline indicated?

### Labels Present
- [ ] Category label (e.g., `Category:Cell`)
- [ ] Team/group label (e.g., `group::cells infrastructure`)
- [ ] Type label (e.g., `type::maintenance`)

### Communication
- [ ] Resources/documentation linked?

## Score
- **Ready**: 80%+ checked
- **Needs work**: 50-79% checked
- **Not ready**: <50% checked
```

### Issue Update

Use this prompt for weekly issue updates as part of our [Development Workflow](_index.md#development-workflow):

```markdown
Can you provide a status update for this issue by looking at issue discussion and MR progress to give a very concise summary of last 7 days activity. Include links.
```

### Issue Summary on Close

```markdown
Provide a concise summary to put in issue description upon close, include reference to any threads where research, discussion or data was referenced, create a summary table where appropriate
```

## Development & Code Review

### Manual Testing Plan Creation

This prompt helps create testing plans for merge requests. Usually requires refinement to specify the testing approach:

```markdown
Create a manual testing plan for this merge request so I can run it myself.
```

Example refinement for specific scenarios:

```markdown
Create a manual testing plan using grpcurl for the new protofiles that are added.
```

### Code Review and Explanation

During code reviews, use this prompt to understand changes:

```markdown
Why are the changes in file X needed? What are they doing?
```

Alternative prompt for more comprehensive analysis:

```markdown
explain what these changes are doing, why they are necessary, and what other possible alternatives to the current approach are.
```

### Documentation Updates

During merge requests, check if documentation needs updating:

```markdown
Should I update any documentation anywhere?
```

## Team Reporting & Communication

### Weekly Epic Summaries

Use this prompt for generating status reports for our [Team Meetings](_index.md#team-meetings) and grand review updates:

```markdown
Using this markdown template write a summary for cells infrastructure this week using the most recent epic updates in cells infrastructure. Include links.

2025-xx-xx Update for Grand Review
:issue-blocked: Help Needed / Blockers

None this week

:closed_book: To Be Closed

None this week

:star2: Highlights

None this week

```
