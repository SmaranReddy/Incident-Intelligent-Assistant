---
title: "Duo-First Development"
---

## Overview

Development Standards define how we work as engineers at GitLab. These standards establish expected practices, workflows, and tools that help us maintain quality, consistency, and efficiency across our codebase.

Our development standards are informed by our [CREDIT Values](/handbook/values/), customer feedback, and data on what improves our productivity and the quality of our work.

## Standard Development Practices

Going forward, all team members are expected to perform these use cases using Duo as part of our standard development practice:

### 1) Issue and Epic Creation

Always generate issues using Duo, with cross-references to similar issues to reduce duplication.

### 2) MR Generation

Use Duo to draft merge requests before manual editing, including the MR description and addressing the review comments where applicable. This should be added to MR templates and required for all submissions, saving significant time per MR with daily usage.

### 3) Code Review Assistance

Run Duo review before human reviewers to accelerate the review process and catch more issues. Add to review checklists and collect developer feedback when Duo's review doesn't meet quality standards. Ask for specific Duo reviews before sending to reviewers (e.g. "Outline the database-specific implications for the database reviewer", "Pull out and describe the frontend changes to streamline the frontend review")

### 4) Test Case Generation

Start test scaffolding with Duo as part of the Definition of Done for new features, improving coverage and saving time per feature.

### 5) Documentation Generation

Use Duo to generate technical and marketing documentation with specialized skillsets, making it the standard for all documentation updates. Consider doing this before any work has started as part of spec-driven development, then moving on to TDD.

## Our Goal

Our goal is to be a true customer zero of our own product and identify wins and gaps to provide constructive feedback while celebrating victories. Please use [this issue](https://gitlab.com/gitlab-org/core-devops/internal-discussions/-/work_items/14) to share your experiences, and please tag issues and success with prefix "Issue:" or "Success:"
