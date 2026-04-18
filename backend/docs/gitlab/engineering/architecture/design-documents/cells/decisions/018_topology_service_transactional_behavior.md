---
owning-stage: "~devops::tenant scale"
title: "Cells ADR 018: Topology Service Transactional Behavior"
status: accepted
creation-date: "2025-08-04"
authors: [ "@ayufan" ]
toc_hide: true
---

{{< engineering/design-document-header >}}

## Context

GitLab Cells architecture requires coordination of globally unique claims (usernames, emails, routes) across multiple distributed cells. The system must prevent conflicts while maintaining high availability and ensuring data consistency, even in the presence of network failures, application crashes, and concurrent operations across cells.

## Decision

1. **Distributed Lease-Based Coordination**: Implement a lease-first, commit-later pattern where cells acquire exclusive leases from the central Topology Service before making local changes.

2. **Atomic Batch Operations**: Process multiple related model changes (User + Email + Route) in single batch requests to ensure all-or-nothing semantics across the distributed system.

3. **Time-Bounded Lease Management**: Use creation timestamps with configurable staleness thresholds (default 10 minutes) instead of explicit expiration times, enabling automatic cleanup through reconciliation.

4. **Rails-Driven Reconciliation**: Implement background reconciliation processes that are Rails-driven with idempotent Topology Service operations to handle failure recovery.

5. **Ownership Security Model**: Enforce that only the cell that created a claim can destroy it, preventing interference between cells and ensuring security isolation.

6. **Three-Phase Transaction Pattern**:
   - **Phase 1**: Pre-flight lease acquisition before local DB transaction
   - **Phase 2**: Local database transaction with lease tracking
   - **Phase 3**: Lease commitment to finalize operations

## Pros

1. **Distributed Locking**: Prevents data corruption from concurrent modifications through exclusive lease access control.
1. **Graceful Recovery**: Distinguishes between permanent conflicts (duplicates) and temporary conflicts (active leases), providing appropriate user feedback.
1. **Self-healing**: Self-healing system with clear failure modes and automatic recovery through background reconciliation.
1. **Easy to use**: Transparent integration with ActiveRecord using declarative configuration.
1. **Network Resilient**: Handles network partitions, cell crashes, and various failure scenarios with error handling and retry logic.
1. **Cross-Cell Independence**: Cells operate independently until conflicts occur, minimizing cross-cell communication and coordination overhead.
1. **Immediate Routing**: Claims become routable immediately after lease acquisition, hiding global replication latency and optimizing user experience.

## Cons

1. **Complexity Introduction**: Adds distributed coordination complexity compared to single-database solutions.
1. **Network Dependency**: Additional network round-trips for every operation that modifies claimed attributes.
1. **Reconciliation Overhead**: Requires background processes to handle failure recovery and data consistency verification.
1. **Operational Monitoring**: Requires monitoring of lease age, conflict rates, and reconciliation processes for production health.
1. **Limited Batch Constraints**: Same claims cannot be both created and destroyed in a single batch operation, which may complicate certain administrative operations.

## Implementation Requirements

### Rails Application Changes

- **CellsUniqueness Concern**: ActiveRecord integration for transparent claim handling
- **Batch Collection Logic**: Aggregate claims from multiple models into single requests
- **Lease Tracking**: Local `leases_outstanding` table for reconciliation synchronization
- **Error Translation**: Convert gRPC errors to Rails validation errors

### Background Reconciliation Services

- **Lost Transaction Recovery**: Minute-level reconciliation for orphaned leases
- **Data Verification Process**: Daily consistency checks between Rails and Topology Service
- **Cursor-Based Processing**: Optimized for large-scale verification operations

### Topology Service Implementation

- **Lease Exclusivity Enforcement**: Database constraints preventing concurrent operations on same objects
- **Atomic Batch Operations**: Single-transaction processing of multiple creates/destroys
- **Idempotent Operations**: CommitUpdate/RollbackUpdate operations safe for retry
- **Security Validation**: Cell ID verification for claim ownership enforcement

### Cloud Spanner Schema Design

- **Integrated Claims Table**: Single table with embedded lease tracking (`lease_id`, `lease_op`)
- **Outstanding Leases Table**: Synchronization point between Rails and Topology Service
- **Performance Indexes**: Optimized for lease operations and verification queries

## Alternative Solutions

### In-Transaction Claims Processing

Execute claims within Rails transaction rather than before it:

```text
1. Local DB: BEGIN
2. Local DB: Operations
3. TS: BeginUpdate() (within transaction)
4. Local DB: COMMIT
5. TS: CommitUpdate()
```

**Trade-offs**: Simpler error handling but potential connection pool exhaustion and increased transaction duration.

### Separate Leased Table Design

Create dedicated `leased` table with UUID cross-references:

**Trade-offs**: Cleaner separation of concerns but increased JOIN complexity and transaction overhead.

### Two-Phase Commit (2PC)

Use traditional distributed transactions:

**Trade-offs**: Proven pattern but overkill for this use case, with significant complexity and performance overhead.
Requires coordination between Cells when making change.

## Failure Recovery Scenarios

### Network Failures

- **During Lease Acquisition**: Safe to retry, no resources allocated
- **During Local Transaction**: Automatic rollback through reconciliation
- **During Commit**: Background reconciliation completes operation

### Application Crashes

- **Before Local Commit**: Reconciliation rolls back orphaned leases
- **After Local Commit**: Reconciliation commits pending operations
- **During Cleanup**: Background processes handle eventual consistency

### Topology Service Outages

- **Graceful Degradation**: Operations fail fast with appropriate user feedback
- **Recovery**: Automatic resume when service becomes available
- **Data Consistency**: Reconciliation ensures no data loss

## Monitoring and Operational Considerations

### Key Metrics

- **Lease Conflict Rates**: Indicates contention levels
- **Reconciliation Frequency**: Health of failure recovery
- **Stale Lease Counts**: System health indicator
- **Operation Latencies**: Performance monitoring

### Administrative Interfaces

- **Outstanding Lease Inspection**: Debugging failed operations
- **Manual Lease Cleanup**: Emergency procedures for edge cases
- **Cell Decommissioning**: Administrative claim cleanup

## Related Documents

- [Topology Service Transactional Behavior design document](../topology_service_transactional_behavior.md)
- [GitLab Cells Infrastructure Architecture](../infrastructure/_index.md)
- [Topology Service Implementation](https://gitlab.com/gitlab-org/cells/topology-service)
