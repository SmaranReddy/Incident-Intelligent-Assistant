---
title: "GitLab Secret Detection Validity Checks ADR 003: Use Secret Detection Response Service (SDRS)"
---

## Status

**SUPERSEDED** - This decision has been superseded by [ADR 004: Use direct partner API calls](004_use_direct_partner_api_calls.md)

## Summary

Initially proposed creating a separate Secret Detection Response Service (SDRS) to handle partner token validation requests. This service would act as an intermediary between GitLab instances and partner APIs.

## Context

When designing partner token validation, we initially believed:

- Partner APIs would require protected credentials that couldn't be shared with self-managed instances
- A separate service would provide better security isolation
- Centralized rate limiting and credential management would be beneficial

## Original Decision

Implement a dedicated SDRS service to:

- Manage partner API credentials centrally
- Handle rate limiting across all GitLab instances
- Provide security isolation from the monolith
- Enable custom partner integrations without modifying GitLab

## Why This Was Superseded

After investigation in [#18277](https://gitlab.com/groups/gitlab-org/-/epics/18277), we discovered:

1. All current partner APIs (AWS, GCP, Postman) use public endpoints
2. No protected credentials are required for token validation
3. The additional infrastructure complexity wasn't justified
4. Self-managed customers would face deployment challenges

## Consequences of Original Approach

**Would have provided:**

- Service isolation for third-party interactions
- Centralized security controls
- Fault isolation from GitLab monolith

**Would have required:**

- Additional infrastructure for self-managed instances
- Complex authentication between GitLab and SDRS
- Longer delivery timeline
- Separate deployment and monitoring
