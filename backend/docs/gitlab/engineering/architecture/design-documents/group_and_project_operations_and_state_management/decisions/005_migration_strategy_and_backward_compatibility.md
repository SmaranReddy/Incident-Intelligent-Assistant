---
owning-stage: "~devops::tenant scale"
title: 'Group and Project Operations ADR 005: Migration Strategy and Backward Compatibility'
status: accepted
creation-date: "2025-09-04"
authors: [ "@rymai" ]
---

## Context

The transition from the current inconsistent state management system to the unified state management system requires careful planning to ensure:

- Zero downtime during migration
- Backward compatibility with existing APIs and user interfaces
- Data integrity throughout the migration process
- Minimal impact on ongoing operations
- Ability to rollback if issues are discovered

Current state management implementations include:

- `projects.marked_for_deletion_at` for project deletion scheduling
- `group_deletion_schedules` table for group deletion scheduling
- Various archived flags and timestamps across different models
- Inconsistent transfer mechanisms between groups and projects

The migration must handle millions of existing namespaces while maintaining system availability and data consistency.

## Decision

We will implement a phased migration strategy with backward compatibility guarantees:

### Phase 1: Infrastructure Setup (Weeks 1-2)

1. **Database Schema Changes**:
   - Add `state` column to `namespaces` table (SMALLINT, default 0)
   - Add `state_metadata` column to `namespace_details` table (JSONB)
   - Create necessary database indexes
   - Deploy schema changes with feature flags disabled

2. **Code Infrastructure**:
   - Implement `Namespaces::Stateful` concern
   - Add state machine definitions
   - Create migration utilities and validation helpers

### Phase 2: Bidirectional Synchronization (Weeks 3-4)

1. **Dual Write Implementation**:
   - Update existing state change operations to write to both old and new systems using ActiveRecord callbacks (like before_save) to write in the same transaction, avoiding consistency issues
   - Implement batched background migration to backfill historical data
   - Add validation to ensure consistency between old and new state representations

2. **Backward Compatibility Layer**:

   ```ruby
   module Namespaces
     module BackwardCompatibility
       extend ActiveSupport::Concern

       # Maintain existing API methods
       def marked_for_deletion_at
         return nil unless deletion_scheduled?
         # Use existing columns if the feature flag is disabled
         return deletion_schedule.marked_for_deletion_on unless Feature.enabled?('namespace_state_management')

         # Use state management if the feature flag is enabled
         state_metadata.dig('scheduled_at').to_time
       end

       def marked_for_deletion_at=
         # Update the existing columns as well as the state management data
         self.deletion_schedule.marked_for_deletion_on = state_metadata['scheduled_at'] = self.deletion_schedule.marked_for_deletion_on
       end
     end
   end
   ```

### Phase 3: Feature Flag Rollout (Weeks 5-8)

1. **Gradual Feature Enablement**:
   - Enable new state system for new operations first
   - Gradually migrate existing operations behind feature flags
   - Monitor performance and consistency metrics

2. **API Compatibility**:
   - Maintain existing REST API responses
   - Add new state fields to API responses (additive changes only)
   - Ensure GraphQL schema backward compatibility

### Phase 4: Data Migration (Weeks 9-12)

1. **Historical Data Migration**:

   ```ruby
   class MigrateNamespaceStatesBatchedMigrationJob < ::Gitlab::BackgroundMigration::BatchedMigrationJob
     def perform
       each_sub_batch.find_each do |namespace|
         migrate_namespace_state(namespace)
       end
     end

     private

     def migrate_namespace_state(namespace)
       new_state = determine_state_from_legacy_data(namespace)
       metadata = build_metadata_from_legacy_data(namespace)

       namespace.update_columns(
         state: new_state,
         state_metadata: metadata
       )
     end
   end
   ```

2. **Validation and Consistency Checks**:
   - Run consistency validation jobs
   - Generate migration reports
   - Handle edge cases and data anomalies

### Phase 5: Legacy Cleanup (Weeks 13-16)

1. **Remove Dual Writes**:
   - Disable backward compatibility layer
   - Remove legacy state management code
   - Clean up unused database columns and tables

2. **Performance Optimization**:
   - Optimize database queries for new state system
   - Remove unnecessary indexes from legacy columns
   - Update documentation and training materials

### Backward Compatibility Guarantees

1. **API Compatibility**:
   - All existing REST API endpoints maintain response format
   - GraphQL schema remains backward compatible
   - Webhook payloads include both old and new state representations during transition

2. **Database Compatibility**:
   - Legacy columns remain readable during migration
   - Existing queries continue to work with compatibility layer
   - No breaking changes to database constraints

3. **UI Compatibility**:
   - Existing UI components continue to function
   - State indicators show consistent information
   - No changes to user workflows during migration

### Rollback Strategy

1. **Immediate Rollback** (if issues detected in Phases 1-3):
   - Disable feature flags
   - Revert to legacy state management
   - No data loss as dual writes maintain legacy system

2. **Data Rollback** (if issues detected in Phase 4):
   - Pause batched background migration [using ChatOps](https://docs.gitlab.com/development/database/batched_background_migrations/#pause-a-batched-background-migration)
   - Restore from database backups if necessary
   - Re-enable legacy system with dual writes

3. **Emergency Rollback** (critical issues):
   - Automated rollback procedures
   - Database failover to read replicas
   - Incident response procedures

## Consequences

### Positive Consequences

- **Zero Downtime**: Phased approach ensures continuous system availability
- **Data Safety**: Dual writes and validation ensure no data loss
- **Gradual Risk**: Issues can be detected and addressed incrementally
- **Rollback Safety**: Multiple rollback options at each phase
- **Team Confidence**: Thorough testing and validation at each step

### Technical Consequences

- **Increased Complexity**: Temporary dual-write system adds complexity
- **Resource Usage**: Additional database writes and background jobs during migration
- **Extended Timeline**: 16-week migration timeline for safety
- **Monitoring Overhead**: Extensive monitoring required during transition
- **Code Maintenance**: Temporary backward compatibility code needs maintenance

### Migration Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Data inconsistency | Dual writes + validation jobs + consistency checks |
| Performance degradation | Gradual rollout + monitoring + rollback procedures |
| API breaking changes | Backward compatibility layer + extensive testing |
| Migration failures | Batch processing + retry logic + manual intervention procedures |
| User experience impact | Feature flags + gradual enablement + communication plan |

## Alternatives

### Alternative 1: Big Bang Migration

- **Pros**: Faster completion, simpler final state
- **Cons**: High risk, potential for extended downtime, difficult rollback

### Alternative 2: Parallel System Approach

- **Pros**: Complete isolation, easy rollback
- **Cons**: Resource intensive, complex data synchronization, longer timeline

### Alternative 3: Gradual Feature-by-Feature Migration

- **Pros**: Lower risk per feature, easier testing
- **Cons**: Inconsistent user experience, complex state management during transition
