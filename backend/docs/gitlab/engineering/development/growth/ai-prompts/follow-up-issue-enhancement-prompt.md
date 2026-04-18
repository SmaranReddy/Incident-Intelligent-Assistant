---
title: "GitLab Follow-up Issue Enhancement with Duo Agent"
description: "Process for transforming generic GitLab follow-up issues into well-structured, actionable work items using Duo Agent."
---

## Overview

This document outlines our process for improving GitLab follow-up issues using Duo Agent to transform generic, context-poor issues into well-structured, actionable work items.

## The Problem

When creating follow-up issues from merge request discussions, GitLab generates issues with:

- Generic titles like "Follow-up from <MR title>"
- Minimal context (only the original comment)
- No implementation guidance
- Vague acceptance criteria

This may lead to technical debt accumulation through unaddressed, poorly-defined issues. Additionally, it makes it very difficult for engineers other than the original author to pick up these issues due to the lack of context.

## The Solution

We use Duo Agent with a standardized prompt to enhance these issues by:

- Creating meaningful, action-oriented titles
- Adding comprehensive context and background
- Providing implementation suggestions based on code analysis
- Defining clear acceptance criteria
- Linking related dependencies and files

## Process

### Step 1: Identify Follow-up Issues

Look for issues with titles starting with "Follow-up from" or containing only brief comments from MR discussions.

### Step 2: Apply the Enhancement Prompt

Use the Duo Agent prompt below to analyze and enhance the issue:

````markdown
# GitLab Follow-up Issue Enhancement Prompt

You are tasked with improving GitLab follow-up issues created from merge request discussions. These issues typically have generic titles like "Follow-up from <MR title>" and contain only the original comment without proper context.

## Your Objectives

Transform the follow-up issue by:

### 1. Create a Meaningful Title

- Replace the generic "Follow-up from <MR title>" with a specific, actionable title
- Focus on WHAT needs to be done, not WHERE it came from
- Use imperative mood (e.g., "Refactor authentication logic", "Add error handling for API calls")
- Keep it concise but descriptive (50-80 characters ideal)

### 2. Enhance the Issue Description

Structure the description with these sections:

#### **Background & Context**

- Summarize the original merge request purpose
- Explain why this follow-up was created
- Link to the specific MR discussion thread

#### **Current State**

- Describe the current implementation/situation
- Include relevant code snippets or file references
- Highlight what works but could be improved

#### **Requested Changes**

- Clearly articulate what needs to be changed/improved
- Break down complex requests into specific tasks
- Use bullet points for clarity

#### **Proposed Implementation**

- Analyze the referenced code/files
- Lightly suggest possible approaches
- Consider architectural implications
- Identify potential challenges or dependencies

#### **Acceptance Criteria**

- Include both functional and non-functional requirements

#### **Technical Context**

- List affected files/modules/components
- Note any dependencies or related issues
- Suggest implementation timeline/priority

## Analysis Instructions

Before writing the enhanced issue:

1. **Examine the source code** mentioned in the original comment
2. **Understand the broader context** of the area being modified
3. **Identify patterns** that might indicate systemic issues
4. **Consider the impact** of the proposed changes
5. **Research existing solutions** or similar implementations in the codebase

## Quality Guidelines

- Be specific and actionable, not vague
- Provide enough detail for any team member to understand and implement
- Include code examples or pseudo-code where helpful
- Link to relevant documentation, ADRs, or related issues
- Use consistent terminology and formatting
- Stick strictly to the example structure provided in the prompt
- Only include sections that were explicitly mentioned in the guidelines

## Output Format

Present your enhanced issue in standard GitLab markdown format with:

- Clear section headers
- Proper code formatting
- Relevant links
- Appropriate labels suggestions (if applicable)

## Example Structure

```markdown
## Background & Context

[Explanation of why this issue exists]

## Current State

[Description of current implementation]

## Requested Changes

[Specific changes needed]

## Proposed Implementation

[Suggested approach with technical details]

## Acceptance Criteria

- [ ] Criterion 1
- [ ] Criterion 2

## Technical Context

**Affected Files:**

- `path/to/file1.js`
- `path/to/file2.py`

**Dependencies:**

- Related to #issue-number

**Priority:** Medium/High/Low
```

Now, please analyze the provided follow-up issue and update the issue description with enhanced version based on these guidelines.
````

### Step 3: Review and Refine

Review the enhanced issue for:

- Technical accuracy
- Completeness of acceptance criteria
- Appropriate priority and labels
- Clear implementation path

## Before and After Examples

### Before Enhancement

_[Screenshot placeholder - Original follow-up issue]_

**Typical characteristics:**

- Generic title: "Follow-up from Implement user authentication"
- Description contains only: "We should refactor this method for better readability"
- No context about which method or files
- No implementation guidance
- No acceptance criteria

### After Enhancement

_[Screenshot placeholder - Enhanced follow-up issue]_

**Improved characteristics:**

- Specific title: "Refactor UserAuthenticator.validateCredentials() for improved readability"
- Comprehensive description with background, current state, and proposed changes
- Clear acceptance criteria
- Implementation suggestions based on code analysis
- Linked dependencies and affected files

## Team Benefits

Enhanced follow-up issues provide significant improvements for the entire team:

- **Clear Understanding**: No guesswork about what needs to be done - any team member can pick up and work on the issue
- **Better Planning**: Well-defined issues integrate smoothly into sprints with accurate effort estimation
- **Technical Debt Reduction**: Vague issues become actionable improvements that actually get addressed
- **Knowledge Preservation**: Context and implementation guidance prevent information loss over time
- **Time Savings**: Less back-and-forth discussion needed during implementation

## Usage Guidelines

### When to Use

- All follow-up issues created from MR discussions
- Existing vague issues that need clarification
- Issues marked as "needs more information"
- Technical debt items requiring better definition

### When Not to Use

- Issues that are already well-defined
- Simple bug reports with clear reproduction steps
- Issues with comprehensive requirements already provided

### Guidelines

- Focus on actionability over documentation when enhancing issues
- Include code snippets only when they add clarity
- Link to relevant architecture decisions or documentation
- Review enhanced issues for technical accuracy and testable acceptance criteria
- Add appropriate labels and assignees after enhancement

## Success Criteria

We measure success through improved issue resolution times, higher completion rates for follow-up issues, and reduced need for clarification during implementation. Enhanced issues should be clear enough for any team member to pick up and complete efficiently.

## Feedback and Iteration

This process should evolve based on team feedback and results. Regular review sessions should address:

- Prompt effectiveness and accuracy
- Process efficiency and adoption
- Quality improvements and suggestions
- Tool limitations and workarounds

---

**Last Updated**: [28/08/2025]
**Process Owner**: [Kamil Niechajewicz]
