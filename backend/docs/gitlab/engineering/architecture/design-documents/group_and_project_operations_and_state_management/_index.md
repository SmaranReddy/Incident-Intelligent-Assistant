---
# This is the title of your design document. Keep it short, simple, and descriptive. A
# good title can help communicate what the design document is and should be considered
# as part of any review.
title: Group and project operations and state management
status: ongoing
creation-date: "2025-05-26"
authors: [ "@lohrc", "@rymai" ]
dris: [ "@lohrc", "@rymai", "@abdwdd" ]
owning-stage: "~devops::runtime"
participating-stages: []
# Hides this page in the left sidebar. Recommended so we don't pollute it.
toc_hide: true
---

{{< engineering/design-document-header >}}

## Summary

This blueprint proposes a unified state management and tracking system for GitLab namespaces (groups and projects), as well as guidelines on making state-related operations asynchronous.
Currently, groups and projects implement state management (deletion, archival, transfer) as separate features with inconsistent data representations and no historical tracking.
The proposed solution introduces a centralized state management system using `namespaces.state` and `namespace_details.state_metadata` to provide consistent state tracking, metadata storage, and historical records across all namespace types.

## Motivation

Groups and Projects currently have inconsistent state management implementations that create several problems:

**Current Issues:**

- No consistency in group state management
- No consistency in project state management
- No consistency between group and project state management
- State in descendants is sometimes inferred from ancestors inconsistently
- No audit trail for state changes. For instance, it's impossible to know when a project was archived, then unarchived, or when a group was transferred from another namespace
- Different data models for similar operations (for example `group_deletion_schedules` vs `projects.marked_for_deletion_at` to track the "scheduled for deletion" state)
- Performance issues with long-running synchronous operations (99.95th percentile: group transfer 51s, project transfer 27s)

**Business Impact:**

- Poor user experience due to inconsistent behavior and bugs
- Poor user experience due to performance bottlenecks causing timeouts
- Increased load on the Support and Engineering teams to resolve operations that failed due to timeouts or bugs
- Difficulty in auditing and compliance
- Maintenance overhead from duplicated and inconsistent code

### Goals

- Establish a unified state management system for all namespace types
- Provide consistent APIs and behavior across groups and projects
- Enable audit trail for state changes
- Improve performance by making appropriate operations asynchronous
- Support metadata tracking (who initiated actions, error states, inheritance)
- Reduce code duplication and maintenance overhead
- Enable better observability and debugging capabilities

### Non-Goals

- Complete rewrite of existing functionality (iterative migration approach)
- Changes to user-facing APIs or UI in the initial implementation
- Migration of non-state related group/project consolidation work
- Performance optimizations unrelated to state management

## Proposal

Introduce a centralized namespace state management system and asynchronous operation guidelines.

### State management data architecture

See [001: Unified State Management System](decisions/001_unified_state_management.md).

### Asynchronous operation guidelines

See [002: Asynchronous Operations Guidelines](decisions/002_asynchronous_operations.md).

### Migration strategy

See [005: Migration Strategy and Backward Compatibility](decisions/005_migration_strategy_and_backward_compatibility.md).

### Performance considerations

**Asynchronous operations:**

- Group transfers (P1 priority - currently 51s at 99.95th percentile)
- Project transfers (P1 priority - currently 27s at 99.95th percentile)

**Optimization strategies:**

- Reduced database writes by not propagating state to descendants and using state inheritance through ancestor lookup
- Database indexes on `state` and related columns

## Metrics and success criteria

**Performance targets:**

- Reduce P99.95 group transfer time from 51s to <10s
- Reduce P99.95 project transfer time from 27s to <5s
- Maintain deletion operation performance (<2s for scheduling)

**Quality metrics:**

- Zero state inconsistency bugs after full migration
- 100% audit event coverage for state changes
- Unified test coverage across all namespace types

**Developer experience:**

- Single API for all state management operations
- Consistent behavior documentation
- Reduced code duplication (target: 50% reduction in state-related code)

## Decisions

- [001: Unified State Management System](decisions/001_unified_state_management.md)
- [002: Asynchronous Operations Guidelines](decisions/002_asynchronous_operations.md)
- [003: State Inheritance Model](decisions/003_state_inheritance_model.md)
- [004: State Changes Audit Integration](decisions/004_state_changes_audit_integration.md)
- [005: Migration Strategy and Backward Compatibility](decisions/005_migration_strategy_and_backward_compatibility.md)
- [006: State Preservation](decisions/006_state_preservation.md)

## Risks and mitigations

**Risk: Data migration complexity**

- *Mitigation:* Iterative approach with rollback capabilities, extensive testing

**Risk: Performance impact during migration**

- *Mitigation:* Ancestor lookup is similar to the status quo, feature flags, monitoring

**Risk: API breaking changes**

- *Mitigation:* Maintain backward compatibility, versioned APIs

**Risk: State consistency issues**

- *Mitigation:* Bidirectional synchronization during transition, validation checks
