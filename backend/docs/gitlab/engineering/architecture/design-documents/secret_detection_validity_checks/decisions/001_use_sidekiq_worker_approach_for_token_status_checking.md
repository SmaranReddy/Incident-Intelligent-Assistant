---
title: "GitLab Secret Detection Validity Checks ADR 001: Use sidekiq worker approach for token status checking"
---

## Summary

We chose to implement token status checking using Sidekiq workers triggered after report ingestion, rather than as part of the analyzer pipeline. This approach enabled faster delivery and immediate self-managed customer support without requiring complex external service authentication.

## Context

When implementing validity checks for Secret Detection findings, we needed to decide at what point to check the detected token status.

Two main approaches were considered:

1. **Post-analyzer approach**: Check token status as part of the Secret Detection analyzer using a post-analyzer component that calls the Secret Detection Response Service (SDRS)
2. **Sidekiq worker approach**: Check token status after report ingestion using background jobs triggered from the monolith

The key considerations were:

- Speed of delivery and implementation complexity
- Self-managed customer support requirements
- Need for manual refresh functionality in beta phase
- Authentication complexity with external services
- Ability to batch process GitLab tokens efficiently

## Decision

We will use the **Sidekiq worker approach**, checking token status after report ingestion using background jobs triggered from the monolith.

## Consequences

**Benefits:**

- Faster delivery since SDRS authentication challenges don't block the experiment or beta phases
- Immediately available to self-managed customers without requiring SDRS connectivity
- Manual refresh functionality naturally fits within the worker pattern
- Can efficiently batch process GitLab tokens using direct database queries without exposing API endpoints

**Drawbacks:**

- Token status is not immediately available when reports are first processed, although this time delay is generally not noticeable to users

## Alternatives Considered

**Post-analyzer approach**: This would check token status during the Secret Detection analysis phase and embed results in the gl-secret-detection-report.json. While this would provide immediate token status availability, it would have required solving SDRS authentication, delayed delivery, and would not have been available to self-managed customers in the experiment or beta phase. The manual refresh requirement would still necessitate a worker-based solution.
