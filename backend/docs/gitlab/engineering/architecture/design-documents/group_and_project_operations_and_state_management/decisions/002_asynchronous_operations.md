---
owning-stage: "~devops::runtime"
title: 'Group and Project Operations ADR 002: Asynchronous Operations Guidelines'
status: accepted
creation-date: "2025-05-26"
authors: [ "@rymai" ]
---

## Context

Current group and project operations suffer from significant performance issues:

- Group transfers: P99.95 performance of 51 seconds
- Project transfers: P99.95 performance of 27 seconds
- These long-running synchronous operations cause timeouts and poor user experience
- Operations involving cascading changes to descendants can be resource-intensive
- Database locks from bulk operations affect concurrent operations

The current synchronous approach creates bottlenecks and reliability issues that impact both user experience and system stability.

## Decision

Operations must be asynchronous if they meet any of the following criteria:

1. **Performance Threshold**: P99.95 performance exceeds 10 seconds
2. **Cascading Changes**: Operation involves cascading changes to descendants
3. **External Dependencies**: Operation requires external service calls or integrations
4. **Bulk Operations**: Operation involves bulk database writes, deletes, or migrations
5. **Locking Risk**: Risk of creating database locks that affect concurrent operations
6. **Resource Intensity**: Memory-intensive operations that could impact system resources

### Implementation Requirements

All asynchronous operations must provide:

- **State Transition**: Transition to appropriate `_in_progress` state
- **Immediate Acknowledgment**: Successful request confirmation and ongoing operation notification
- **Progress Indicators**: Where technically feasible
- **Completion Notifications**: Through appropriate channels (activity, notification center, email)
- **Error Handling**: Comprehensive error handling with user-facing error messages
- **Rollback Capabilities**: For failed operations

### Priority Operations

Based on current performance metrics, these operations require immediate async implementation:

1. **Group transfers** (P99.95: 51s) - Priority 1
2. **Project transfers** (P99.95: 27s) - Priority 1

### Implementation Pattern Example

```ruby
class NamespaceTransferService
  def execute
    namespace.transfer_start!
    NamespaceTransferWorker.perform_async(namespace.id, target_namespace.id, current_user.id)

    ServiceResponse.success(
      message: "Transfer initiated successfully",
      payload: { operation_id: operation_id, estimated_duration: "5-10 minutes" }
    )
  end
end
```

## Consequences

### Positive Consequences

- **Performance**: Eliminates timeout issues for heavy operations
- **User Experience**: Immediate feedback with progress tracking
- **System Stability**: Reduces database contention and resource conflicts
- **Scalability**: Enables handling of larger operations without system impact

### Technical Consequences

- **Complexity**: Requires implementing background job processing
- **State Management**: Need for `_in_progress` states and proper state transitions
- **UI Changes**: Must show operation status and progress indicators
- **Error Handling**: More complex error scenarios to handle
- **Testing**: Additional complexity in testing asynchronous flows

## Alternatives

### Alternative 1: Optimize synchronous operations

- **Pros**: Simpler implementation, no state management complexity
- **Cons**: Limited improvement potential, still subject to timeout risks

### Alternative 2: Hybrid approach (some sync, some async)

- **Pros**: Gradual migration, maintains simple operations as synchronous
- **Cons**: Inconsistent user experience, complex decision logic

### Alternative 3: Make all operations asynchronous

- **Pros**: Consistent experience, maximum performance
- **Cons**: Unnecessary complexity for simple operations, poor UX for fast operations
