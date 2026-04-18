---
title: "GitLab Issue Refinement with Duo Agent"
description: "Process for generating targeted discussion points to enhance async refinement conversations in Refinement Threads using Duo Agent."
---

## Overview

This document outlines our process for using Duo Agent to generate focused discussion points that enhance async refinement conversations in Refinement Threads, providing engineers with clear focus areas for estimation and technical consideration.

## The Problem

Issues with ~workflow::refinement status often have Refinement Threads that lack focused discussion points, leading to:

- Engineers unsure what specific aspects need consideration for accurate estimation
- Async refinement conversations that miss critical technical factors
- Inconsistent quality of discussion across different Refinement Threads
- Important dependencies or constraints overlooked during emoji voting
- Unfocused async discussions that don't effectively guide estimation decisions

This results in less effective refinement outcomes and potentially inaccurate emoji-based estimates.

## The Solution

We use Duo Agent with a standardized prompt to enhance Refinement Threads by providing structured discussion points that:

- Give engineers clear focus areas for async refinement discussions
- Surface technical constraints and dependencies affecting estimation
- Highlight important considerations for emoji voting decisions
- Provide consistent, structured input across all Refinement Threads
- Enable more informed and focused async refinement conversations

## Process

### Step 1: Identify Issues in Refinement Workflow

Look for issues with:

- ~workflow::refinement status assigned by the bot
- Active Refinement Threads created by @gitlab-bot
- `:construction: Refinement` comments requiring discussion input

### Step 2: Generate Discussion Points for Refinement Thread

Use the Duo Agent prompt below to create focused input for the async refinement discussion:

{{% details summary="Issue refinement prompt, click me to expand"%}}

```markdown
You are a GitLab fullstack engineer helping prepare refinement discussion points for this issue

Objective:
Generate discussion points/questions that will help the refinement team:

- Identify missing or ambiguous requirements that need team discussion
- Surface potential blockers, dependencies, or risks for team evaluation
- Highlight technical constraints ONLY if they significantly impact effort estimation or feasibility

Context to analyze:

1. Related issues, epics and MRs (focus on those directly referenced in this issue)
2. Existing codebase patterns in the affected areas
3. Discussion in comments, particularly unresolved concerns or decisions

Output requirements:

- Provide only the essential discussion points that the refinement team should address (typically 3-5)
- Each question should be one clear sentence that prompts team discussion
- Focus on concrete technical or product decisions that require team input and are scoped within the issue

Example output:

- **UX** Should error messages appear as modals or flashes?
- **Technical** How should we handle the existing user sessions during the migration?
- **Scope** Should this include mobile responsive design or desktop only?

Action:
Find the comment started with ":construction: Refinement" posted by @gitlab-bot in this issue, and reply to it with the output
```

{{% /details %}}

### Step 3: Engineers Use Input for Async Discussion

The generated discussion points become input for:

- Focused async discussion within the Refinement Thread
- Informed consideration during emoji voting for estimation
- Consistent coverage of important technical and product factors
- Enhanced quality of refinement conversations

## Before and After Examples

### Before Discussion Point Input

**Typical characteristics:**

- Refinement Threads with minimal structured discussion
- Engineers uncertain what factors to consider for estimation
- Async conversations that overlook important technical considerations
- Inconsistent refinement depth across different issues
- Emoji voting without clear context for complexity factors

### After Discussion Point Input

**Improved characteristics:**

- Structured discussion points guiding Refinement Thread conversations
- Clear technical and product considerations for estimation decisions
- Consistent coverage of important factors across all refinement threads
- Enhanced context supporting informed emoji voting
- More productive async refinement discussions

## Team Benefits

Enhanced Refinement Thread input provides significant improvements:

- **Focused Async Discussions**: Clear discussion points keep Refinement Thread conversations productive and targeted
- **Better Estimation Context**: Engineers have specific factors to consider during emoji voting
- **Consistent Quality**: Structured input ensures comprehensive refinement across all issues
- **Efficient Process**: Reduces time spent determining what to discuss in async threads
- **Informed Decisions**: Well-structured input leads to more accurate estimation and technical choices

## Usage Guidelines

### When to Use

- All issues moved to ~workflow::refinement status by the bot
- Refinement Threads that need structured discussion input
- Complex issues requiring consideration of multiple estimation factors
- Issues where async discussion would benefit from focused guidance

### When Not to Use

- Issues with Refinement Threads that already have comprehensive discussion points
- Simple, well-defined changes with obvious estimation factors
- Issues where scope and technical approach are completely clear

### Guidelines

- Focus on discussion points that directly support estimation decisions
- Provide input that enhances async Refinement Thread quality
- Keep discussion points scoped to the current issue
- Prioritize factors that significantly impact effort estimation
- Ensure points facilitate productive async conversations and emoji voting

## Success Criteria

We measure success through improved quality of Refinement Thread discussions, more informed emoji voting decisions, and consistent refinement outcomes. Generated discussion points should provide valuable input that enhances engineers' ability to participate effectively in async refinement conversations.

## Feedback and Iteration

This process should evolve based on team feedback and results. Regular review sessions should address:

- Discussion point relevance and quality for estimation
- Refinement Thread conversation effectiveness
- Emoji voting decision consistency and accuracy
- Process adoption and async refinement improvement

---

**Last Updated**: 2025-09-12
**Process Owner**: [Roy Liu]
