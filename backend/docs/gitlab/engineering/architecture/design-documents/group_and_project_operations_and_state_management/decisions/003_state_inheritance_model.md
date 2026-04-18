---
owning-stage: "~devops::runtime"
title: 'Group and Project Operations ADR 003: State Inheritance Model'
status: accepted
creation-date: "2025-09-04"
authors: [ "@rymai" ]
---

## Context

In GitLab's hierarchical namespace structure (Organization > Group > Subgroup > Project), state management needs to handle inheritance efficiently. The current approach has several issues:

- State in descendants is sometimes inferred from ancestors inconsistently
- No clear rules for how state should propagate through the hierarchy
- Performance concerns with propagating state changes to all descendants
- Complexity in maintaining state consistency across large hierarchies

We need a clear model for how state inheritance works that balances consistency, performance, and simplicity.

## Decision

We will implement a **lookup-based state inheritance model** with the following rules:

### State Inheritance Rules

1. **Direct State Assignment**: When a namespace has an explicit `state` value (not `nil` or `ancestor_inherited`), that state takes precedence
2. **Ancestor Lookup**: When a namespace's state is `nil` or `ancestor_inherited`, the effective state is determined by traversing up the hierarchy to find the first ancestor with a defined state
3. **Default Fallback**: If no ancestor has a defined state, the effective state is `active` (normal operational state)

### State Validation Guards

Validation guards prevent inconsistent state combinations that would create confusing user experiences.
These validation rules govern state transitions across three layers: parent state, self state, and children states.

The following tables shows all possible state transitions and whether they are allowed or denied:

#### All State Transitions

| **New State**            | **Old state: active**    | **Old state: archived**   | **Old state: creation_in_progress** | **Old state: deletion_in_progress** | **Old state: deletion_scheduled** | **Old state: transfer_in_progress** |
|--------------------------|--------------------------|---------------------------|-------------------------------------|-------------------------------------|-----------------------------------|-------------------------------------|
| **active**               | -                        | :white_check_mark: Allow  | :white_check_mark: Allow            | :white_check_mark: Allow            | :white_check_mark: Allow          | :white_check_mark: Allow            |
| **archived**             | :white_check_mark: Allow | -                         | :x: Deny                            | :white_check_mark: Allow            | :white_check_mark: Allow          | :white_check_mark: Allow            |
| **creation_in_progress** | :x: Deny                 | :x: Deny                  | -                                   | :x: Deny                            | :x: Deny                          | :x: Deny                            |
| **deletion_in_progress** | :x: Deny                 | :x: Deny                  | :white_check_mark: Allow            | -                                   | :white_check_mark: Allow          | :x: Deny                            |
| **deletion_scheduled**   | :white_check_mark: Allow | :white_check_mark: Allow  | :x: Deny                            | :white_check_mark: Allow            | -                                 | :x: Deny                            |
| **transfer_in_progress** | :white_check_mark: Allow | :white_check_mark: Allow  | :x: Deny                            | :x: Deny                            | :x: Deny                          | -                                   |

#### Valid State Transitions

| Allowed Transition                          | Note/Reason                                                                                        |
|---------------------------------------------|----------------------------------------------------------------------------------------------------|
| archived → active                           | Unarchive                                                                                          |
| creation_in_progress → active               | Creation complete                                                                                  |
| deletion_in_progress → active               | Deletion failure. Skip deletion_scheduled to avoid infinite retry loop.                            |
| deletion_scheduled → active                 | Restore from delete                                                                                |
| transfer_in_progress → active               | Transfer complete                                                                                  |
| active → archived                           | Archive                                                                                            |
| deletion_in_progress → archived             | Deletion triggered from archived namespace failed. Skip deletion_scheduled to prevent retry loop.  |
| deletion_scheduled → archived               | Deletion triggered from archived namespace                                                         |
| transfer_in_progress → archived             | Allow transfer to complete for archived item                                                       |
| creation_in_progress → deletion_in_progress | Fatal creation failure                                                                             |
| deletion_scheduled → deletion_in_progress   | Start deletion of the namespace                                                                    |
| active → deletion_scheduled                 | Schedule deletion                                                                                  |
| archived → deletion_scheduled               | Schedule deletion of archived namespace                                                            |
| deletion_in_progress → deletion_scheduled   | Deletion failure. Requeue for retry                                                                |
| active → transfer_in_progress               | Transfer a namespace                                                                               |
| archived → transfer_in_progress             | Transfer archived namespace                                                                        |

#### Invalid State Transitions

| Denied Transition                           | Note/Reason                                                                               |
|---------------------------------------------|-------------------------------------------------------------------------------------------|
| creation_in_progress → archived             | Must transition to active before archived                                                 |
| active → creation_in_progress               | Cannot revert to creation_in_progress once active. Use creation_in_progress for new only. |
| archived → creation_in_progress             | Cannot revert to creation_in_progress                                                     |
| deletion_in_progress → creation_in_progress | Cannot revert to creation_in_progress                                                     |
| deletion_scheduled → creation_in_progress   | Cannot revert to creation_in_progress                                                     |
| transfer_in_progress → creation_in_progress | Cannot revert to creation_in_progress                                                     |
| active → deletion_in_progress               | Must go through deletion_scheduled first                                                  |
| archived → deletion_in_progress             | Must go through deletion_scheduled first                                                  |
| transfer_in_progress → deletion_in_progress | Transfer must complete before permanent deletion                                          |
| creation_in_progress → deletion_scheduled   | Deletion schedule only allowed after successful creation                                  |
| transfer_in_progress → deletion_scheduled   | Deletion schedule only allowed after successful transfer                                  |
| creation_in_progress → transfer_in_progress | Transfer only allowed after successful creation                                           |
| deletion_in_progress → transfer_in_progress | Cannot transfer while deletion is in progress                                             |
| deletion_scheduled → transfer_in_progress   | Cannot transfer while deletion is scheduled                                               |

#### State Transition Overview

For each allowed transition, we examined dependencies across the state hierarchy.
The following shows the permitted transitions along with the parent and children state requirements that must be satisfied:

| Transition                                  | Parent Check                                                                             | Parent Check Note                                                                               | Children Check                                     | Children Check Note                                                                       |
|---------------------------------------------|------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|----------------------------------------------------|-------------------------------------------------------------------------------------------|
| archived → active                           | NOT deletion_in_progress, NOT deletion_scheduled                                         | For child unarchive, parent cannot be deletion_in_progress, or deletion_scheduled               | -                                                  | Children inherit unarchive or become active                                               |
| creation_in_progress → active               | -                                                                                        | Parent state doesn't restrict creation completion                                               | N/A (no children exist yet)                        | No children exist during creation_in_progress                                             |
| deletion_in_progress → active               | -                                                                                        | Allowed for fatal cases like deleting user loses access                                         | -                                                  | Allowed for recovery scenarios                                                            |
| deletion_scheduled → active                 | -                                                                                        | Restore action - no parent restrictions                                                         | -                                                  | Children restored to previous state                                                       |
| transfer_in_progress → active               | -                                                                                        | Successful transfer completion                                                                  | -                                                  | Transfer completion - children remain in their states                                     |
| active → archived                           | NOT archived, NOT deletion_in_progress, NOT deletion_scheduled, NOT transfer_in_progress | Parent cannot be in archived, deletion_in_progress, deletion_scheduled, or transfer_in_progress | NOT creation_in_progress, NOT transfer_in_progress | Children in creation_in_progress or transfer_in_progress block parent archiving           |
| deletion_in_progress → archived             | NOT archived                                                                             | Parent is already archived, instead use active (ancestor_inherited)                             | -                                                  | Allowed as restore to archived state                                                      |
| deletion_scheduled → archived               | NOT archived                                                                             | Parent is already archived, instead use active (ancestor_inherited)                             | -                                                  | Allowed as restore to archived state                                                      |
| transfer_in_progress → archived             | -                                                                                        | Successful transfer completion of archived namespace                                            | -                                                  | Transfer completion - children remain in their states                                     |
| creation_in_progress → deletion_in_progress | -                                                                                        | Cleanup on fatal creation failure                                                               | -                                                  | No children exist during creation_in_progress                                             |
| deletion_scheduled → deletion_in_progress   | -                                                                                        | Begin deletion process after grace period                                                       | -                                                  | All child states allowed                                                                  |
| active → deletion_scheduled                 | NOT deletion_in_progress, NOT deletion_scheduled, NOT transfer_in_progress               | Parent cannot be in deletion_in_progress, deletion_scheduled, or transfer_in_progress           | NOT creation_in_progress, NOT transfer_in_progress | Children in creation_in_progress or transfer_in_progress block parent deletion scheduling |
| archived → deletion_scheduled               | NOT deletion_in_progress, NOT deletion_scheduled, NOT transfer_in_progress               | Parent cannot be in deletion_in_progress or deletion_scheduled, or transfer_in_progress         | NOT creation_in_progress, NOT transfer_in_progress | Children in creation_in_progress or transfer_in_progress block parent deletion scheduling |
| deletion_in_progress → deletion_scheduled   | -                                                                                        | Rollback on deletion failure for retry                                                          | -                                                  | Rollback scenario - all children states allowed                                           |
| active → transfer_in_progress               | NOT deletion_in_progress, NOT deletion_scheduled, NOT transfer_in_progress               | Parent must be active or archived                                                               | All children must be active or archived only       | All children must be active or archived                                                   |
| archived → transfer_in_progress             | NOT deletion_in_progress, NOT deletion_scheduled, NOT transfer_in_progress               | Parent must be active or archived                                                               | All children must be active or archived only       | All children must be active or archived                                                   |

### Implementation Approach

```ruby
module Namespaces
  module Stateful
    extend ActiveSupport::Concern

    included do
      ...

      state_machine :state, initial: :ancestor_inherited, initialize: false do
        ...

        # Validation guards to prevent inconsistent state updates
        before_transition any => :deletion_scheduled do |namespace|
          if namespace.ancestor_archived?
              raise StateMachine::InvalidTransition, "Cannot schedule deletion when ancestor is archived"
          end
        end

        before_transition any => :archived do |namespace|
          if namespace.ancestor_deletion_scheduled?
            raise StateMachine::InvalidTransition, "Cannot archive when ancestor is scheduled for deletion"
          end
        end

        ...
      end
    end

    def effective_state
      return state if state.present? && state != :ancestor_inherited

      # When state is nil, lookup first ancestor up in the hierarchy with a state
      ancestors
        .select(:id, :state)
        .where.not(state: [nil, STATES[:ancestor_inherited]])
        .last&.state || :active
    end

    def archived?
      effective_state == :archived
    end

    def deletion_scheduled?
      effective_state == :deletion_scheduled
    end
  end
end
```

## Consequences

### Positive Consequences

- **Performance**: No need to propagate state changes to descendants, reducing database writes
- **Consistency**: Clear inheritance rules prevent state conflicts
- **Flexibility**: Descendants can override inherited state when needed
- **Simplicity**: Lookup-based approach is easier to understand and debug

### Technical Consequences

- **Query Overhead**: Effective state calculation requires ancestor lookup for inherited states
- **Caching Needs**: May need caching for frequently accessed state information
- **Validation Complexity**: State transition guards need to check ancestor states
- **Migration Complexity**: Existing state data needs careful migration to new model

## Alternatives

### Alternative 1: Propagation-based inheritance

Store the effective state directly on each namespace by propagating changes down the hierarchy.

- **Pros**: Fast state lookup, no ancestor queries needed
- **Cons**: Complex propagation logic, many database writes, consistency challenges

### Alternative 2: No inheritance

Each namespace manages its own state independently.

- **Pros**: Simple implementation, no inheritance complexity
- **Cons**: Inconsistent user experience, no way to apply organization-wide policies

### Alternative 3: Event-driven inheritance

Use events to notify descendants of state changes.

- **Pros**: Decoupled architecture, eventual consistency
- **Cons**: Complex event handling, potential for temporary inconsistencies
