---
owning-stage: "~devops::tenant scale"
title: 'Organizations ADR 007: Self-managed and Dedicated Single Organization'
status: accepted
creation-date: "2025-11-18"
authors: [ "@tkuah" ]
---

## Context

The Organizations feature introduces a multi-tenant architecture to GitLab, enabling multiple isolated Organizations for GitLab.com. This architecture is designed to support GitLab.com's SaaS model where many customers share the same infrastructure.

However, Self-managed and Dedicated GitLab instances have different operational models:

- **Self-managed**: Customers install and operate their own GitLab instance, typically for a single company or entity
- **Dedicated**: GitLab operates a single-tenant instance exclusively for one customer

Both deployment models serve a single customer organization, unlike GitLab.com which serves multiple distinct customer organizations.

### Current state

Self-managed and Dedicated instances operate with instance-level administration and a flat namespace structure. Customers expect to manage their entire instance as a unified entity. Billing and licensing is applied to the whole instance.

### What's changing

With the introduction of Organizations, the data model and access patterns are shifting to support multi-tenancy. This raises questions about how Self-managed and Dedicated instances should adopt the Organizations architecture.

## Decision

Self-managed and Dedicated instances will have a single Organization for now. This single Organization will be:

- Created automatically during instance setup
- The default container for all Groups, Projects, and Users on the instance

No other organization will be allowed to be created on Self-managed and Dedicated instances.

## Consequences

- Self-managed and Dedicated instances only needs a smaller, transparent migration from instance level to a single Organization. In most cases, the current user experience is maintained.
- Backend code can be unified across all deployment types, using the same Organization-aware data access patterns
- Future migration to multiple Organizations (if needed) will be possible since the architecture supports it
- The Organization context resolution ([ADR 001](001_organization_context_resolution.md)) applies transparently - all operations occur within the single Organization context
- Administration and settings ([ADR 006](006_administration_and_settings.md)) remain at the instance level from the user's perspective, while technically operating at the Organization level in the backend

## Alternatives

### Bypass Organizations entirely for Self-managed and Dedicated

We could maintain separate code paths where Self-managed and Dedicated instances do not use the Organizations architecture at all. This was rejected because:

- It risks accidentally bypassing Organizations for GitLab.com, since the codebase and data access pattern is shared.
- It would create significant code divergence between deployment types.
- It would increase maintenance burden and testing complexity.

### Expose multiple Organizations immediately

We could allow Self-managed and Dedicated customers to create multiple Organizations from the start. This was rejected because:

- There is no clear customer demand for this capability in single-tenant environments.
- It would add unnecessary complexity to the Self-managed and Dedicated experience.
- Implementing cross-Organization access patterns that is coherent with GitLab.com is complex.
