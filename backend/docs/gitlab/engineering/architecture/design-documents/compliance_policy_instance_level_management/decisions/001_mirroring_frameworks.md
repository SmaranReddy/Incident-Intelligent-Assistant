---
title: "Compliance Frameworks ADR 001: CSP Framework Distribution Strategy"
toc_hide: true
---

## Context

We need to make a CSP group's frameworks available to all other root-level groups. Currently, compliance frameworks are scoped to individual groups, but there's a requirement to share frameworks from a designated CSP group across the the other groups.

The system needs to handle frequent read operations (compliance checks, UI loads) efficiently while maintaining data integrity and clear ownership models. Two primary approaches emerged: copying frameworks to all groups versus dynamically combining frameworks at query time.

## Option 1

Copy CSP group frameworks across to all other groups, creating individual framework copies in each group.

This approach would trigger copy operations whenever CSP frameworks are created, updated, or deleted. Each group would maintain its own copy of the CSP frameworks alongside any group-specific frameworks. The copying would need to handle various scenarios including new group creation, group imports, requirements changing and so on.

### Benefits

- Maintaining existing functionality without modification.
- Clear data ownership per group.
- Optimal performance for frequent read operations.
- Straightforward authorization logic.
- Audit trail remains clear since each group owns its framework copies.

### Drawbacks

- Expensive bulk operations during framework changes.
- Storage duplication across potentially hundreds of groups.
- Complex synchronization logic with edge case handling.
- Significant risk of data drift if synchronization fails.
- Approach may not scale well as things grow.
- Synchronization complexity could become substantial technical debt over time.

## Option 2

Union the CSP group frameworks with each group's frameworks dynamically at query time.

This approach would modify the frameworks we retrieve for a given group to include both the CSP group's frameworks and the group's own frameworks.

### Benefits

- Eliminating expensive copy operations.
- Removing storage duplication.
- Instant visibility of CSP framework changes across all groups.
- Maintains a single source of truth and avoids synchronization complexity entirely.
- Approach scales better as the system grows and prevents accumulation of technical debt.

### Drawbacks

- Ensuring union logic is consistently applied across all code paths where frameworks are accessed.
- Potential performance impact on read operations.
- Complex authorization patterns surrounding the assignment of frameworks on project level.

## Decision

We decided to implement Option 2: Union the CSP group frameworks with group frameworks dynamically at query time.

The decision was influenced by concerns about the long-term maintainability and scalability of the copying approach. While the union approach requires more architectural changes upfront, it provides a cleaner solution that avoids the complexity of keeping copied data synchronized across potentially hundreds of groups.

The team assessed that the benefits of maintaining a single source of truth outweigh the potential implementation complexity. The approach aligns better with our scaling requirements and reduces the risk of accumulating technical debt.
