---
title: "GitLab Secret Detection Validity Checks ADR 004: Use direct partner API calls instead of SDRS"
---

## Summary

Implement partner token validation through direct API calls from GitLab's monolith using Sidekiq workers, replacing the previously proposed SDRS service architecture.

## Context

Following discovery work in [#562364](https://gitlab.com/gitlab-org/gitlab/-/issues/562364) and consultation with infrastructure team, we determined that:

- All partner APIs are publicly accessible
- No protected credentials need to be managed
- Monolith-first approach provides better operational support
- Self-managed instances benefit from simpler deployment

## Decision

Make partner API calls directly from Sidekiq workers within the GitLab monolith, using existing rate limiting and retry mechanisms.

## Consequences

**Benefits:**

- Leverages existing SRE on-call support
- Uses established security release process
- Works for self-managed by default
- Faster data access to GitLab DB
- No additional infrastructure required

**Drawbacks:**  

- Partner API changes require monolith changes
- Mixed security contexts in the same codebase
- Need careful rate limit management per partner
