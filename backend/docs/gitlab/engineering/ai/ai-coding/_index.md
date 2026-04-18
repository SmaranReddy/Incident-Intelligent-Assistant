---
title: "AI Coding Group"
description: The AI Coding Group is part of the AI Engineering organization focused on user-facing AI-powered coding features.
---

## Team Overview

The AI Coding Group develops AI-powered features that help developers write, understand, and improve code. We focus on creating intelligent tools that enhance the developer experience across GitLab.

## Our Projects

The AI Coding team owns and is actively working on the following projects:

- [Code Suggestions](/handbook/engineering/ai/ai-coding/code_suggestions/) - AI-generated code completion and generation within IDEs.
- [Duo Code Review](/handbook/engineering/ai/ai-coding/duo_code_review/) - AI-powered code review assistance and insights.
- [Duo Context Exclusion](/handbook/engineering/ai/ai-coding/duo_context_exclusion/) - Filtering of sensitive or irrelevant code context.
- [Codebase Semantic Indexing](/handbook/engineering/ai/ai-coding/codebase_semantic_indexing/) - Advanced code search and discovery capabilities using embeddings.
- [Code-related Slash Commands](/handbook/engineering/ai/ai-coding/slash_commands/) - Interactive Duo Chat commands including `/explain`, `/refactor`, `/tests`, and `/fix`.
- [Repository X-Ray](/handbook/engineering/ai/ai-coding/repository_xray/) - Repository analysis and metadata extraction for enhanced code suggestions context.
- [AI Assisted Service](/handbook/engineering/ai/ai-coding/ai_assisted_service/) - Core AI infrastructure and services supporting our features.

### Evaluation and Testing

AI Coding is responsible for evaluations across all our features, which includes:

- Creating datasets in LangSmith and registering them in the [Datasets repository](https://gitlab.com/gitlab-org/modelops/ai-model-validation-and-research/ai-evaluation/datasets/-/blob/main/doc/guidelines/register_dataset.md#registration-process). We also have some [Code Creation Datasets](https://gitlab.com/gitlab-org/code-creation).
- Creating evaluators in [Centralized Evaluation Framework](https://gitlab.com/gitlab-org/modelops/ai-model-validation-and-research/ai-evaluation/prompt-library)
- Running evaluations

## Contact Us

Use this information to connect with the AI Coding group:

| Category                 | Name                                  |
|--------------------------|---------------------------------------|
| GitLab Team Handle       | @gitlab-org/code-creation/engineers   |
| Slack Channel            | #g_ai_coding                          |

## Team Members

The following people are permanent members of the AI Coding Team:

{{< team-by-manager-slug manager="mnohr" team="AI Coding">}}

You can reach the whole team on GitLab issues/MRs by using the `@gitlab-org/code-creation/engineers` handle.

## Stable Counterparts

The following members of other functional teams are our stable counterparts:

| Category          | Counterpart                                                                          |
|-------------------|--------------------------------------------------------------------------------------|
| Product Manager   | {{< member-by-name "Jordan Janes" >}}                                                |
| Technical Writing | {{< member-by-name "Uma Chandran" >}}                                                |
| UX                | TBD                                                                                  |
| Support           | [TBD](/handbook/support/support-stable-counterparts/)                                |

## How We Work

For information on how the team works including onboarding, time off, issue boards, meetings, and more, please refer to the [How We Work](/handbook/engineering/ai/ai-coding/how-we-work/) page.

## Dashboards and Monitoring

1. [Metrics Dashboard](https://dashboards.gitlab.net/d/stage-groups-code_creation/stage-groups3a-code-creation3a-group-dashboard?orgId=1) (Grafana)
1. [Error Budget](https://dashboards.gitlab.net/d/stage-groups-detail-code_creation/stage-groups-code-creation-group-error-budget-detail?orgId=1) (Grafana)

## Related Resources

- [AI Coding Product Categories](/handbook/product/categories/#ai-coding-group)
