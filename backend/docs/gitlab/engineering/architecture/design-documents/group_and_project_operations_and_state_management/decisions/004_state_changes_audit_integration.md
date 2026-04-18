---
owning-stage: "~devops::runtime"
title: 'Group and Project Operations ADR 004: State Changes Audit Integration'
status: accepted
creation-date: "2025-09-04"
authors: [ "@rymai" ]
---

## Context

Groups and Projects currently don't have audit trail for state changes (impossible to know when a project was archived, then unarchived, or when a group was transferred).

This makes auditing and compliance difficult.

## Decision

State changes will be tracked through GitLab's existing audit events system rather than a separate table. This provides:

- Consistent audit trail with other GitLab operations
- Built-in compliance and security features
- Unified audit log viewing and filtering
- Existing retention and export capabilities

### Implementation Approach

```ruby
module Namespaces
  module Stateful
    extend ActiveSupport::Concern
    included do
      ...

      state_machine :state, initial: :ancestor_inherited, initialize: false do
        ...


        after_transition any => any do |namespace, transition|
          namespace.run_after_commit do
            # Create audit event for state change
            AuditEventService.new(
              namespace.state_metadata['last_changed_by_user_id'],
              namespace,
              {
                action: :namespace_state_changed,
                custom_message: "Namespace state changed from #{transition.from} to #{transition.to}",
                target_details: namespace.full_path
              }
            ).for_namespace.security_event
          end
        end

        ...
      end
    end

  end
end
```

## Consequences

### Positive Consequences

- **Auditability**: Complete audit trail for state changes through existing audit events system

### Technical Consequences

- **Implementation Needs**: New audit events need to be implemented

## Alternatives

### Alternative 1: Track state history in the database

Store the effective state changes in the database.

- **Pros**: Complete history of state changes stored in a permanent storage
- **Cons**: Might be overkill if we don't plan to build features on top of it

### Alternative 2: No additional audit events

Keep the current audit events.

- **Pros**: Nothing to implement
- **Cons**: Debugging and compliance continue being an impediment
