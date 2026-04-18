---
title: 'GitLab WLIF: STS-002 Merge GLGO into IAM Service'
toc_hide: true
---

## Context

GLGO was created as a specialized identity translation layer between GitLab and
external cloud providers. We are now building a new, centralized Identity and
Access Management (IAM) service in Go. This service is intended to become the
authoritative source for identity and access management code at GitLab.

To avoid code duplication, consolidate our services, and make identity features
available to self-managed customers, we need to decide where the functionality
of GLGO should live long-term.

## Decision

We will merge the functionality of the GLGO service into the new IAM service.
The IAM service will incorporate and expand upon GLGO's capabilities, including
the Secure Token Service (STS) for Workload Identity Federation.

## Consequences

**Positive:**

* **Codebase Consolidation:** Merging GLGO into the IAM service centralizes
    our identity management logic, reducing maintenance overhead.
* **Availability for Self-Managed:** Integrating this functionality into a
    core service like IAM makes it easier to package and deliver to
    self-managed GitLab customers.
* **Unified Architecture:** It establishes the IAM service as the single
    place for identity-related features, creating a clearer and more
    maintainable architecture.

**Negative:**

* **Migration Effort:** This decision requires a migration effort to move
    existing GLGO functionality into the new IAM service codebase.

## Alternatives

1. **Keep GLGO as a Standalone Service:** We could continue to develop GLGO as
a separate service. However, this would lead to code duplication and increased
maintenance complexity. It would also complicate efforts to bring this
functionality to self-managed instances.
2. **Integrate GLGO into GitLab Rails:** This is not a viable alternative
because it would not provide sufficient isolation for sensitive authentication
data and signing materials from the rest of the GitLab monolith.
