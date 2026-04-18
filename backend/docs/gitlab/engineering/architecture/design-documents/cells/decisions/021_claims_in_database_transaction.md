---
title: "Topology Service Claims In Database Transactions"
owning-stage: "~devops::tenant-scale"
group: cells-infrastructure
creation-date: "2025-05-09"
authors: ["@sxuereb"]
coach: "@ayufan"
approvers: ["@ayufan"]
toc_hide: true
---

{{< engineering/design-document-header >}}

## Context

GitLab's Cells architecture requires coordination with the Topology Service to claim cluster-wide unique resources like emails, routes which is described in detail in [Topology Service transactions](../topology_service_transactional_behavior.md).
The critical decision is whether to perform these claims requests to the Topology Service inside or outside Database transactions.

**Key Requirements**:

- Claim resources to prevent conflicts across cells.
- Maintain data consistency between the Rails DB and the Topology Service.
- Support batching of performance claims.
- Handle ~50 claims/sec average, with the ability to 6x our rate to ~300 claims/sec.
- Database stability not affected.

**Technical Constraints**:

- Network calls to Topology Service: [\~200ms P99.95](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/488#note_2767889339)
- Current connection pool: 58 connections support [\~290 claims/sec at 200ms](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/488#note_2773726223)
- Need Postgres primary keys, for example, user.id for users. This exists when saving records to the database inside the transaction.
- Rails automatically wraps creates/updates in transactions

## Decision

Implement claims INSIDE [ActiveRecord](https://guides.rubyonrails.org/active_record_basics.html) transactions using callbacks. The implementation will:

- Use [`after_save`](https://api.rubyonrails.org/v8.0.3/classes/ActiveRecord/Callbacks/ClassMethods.html#method-i-after_save) (officially documented) or [`before_commit`](https://github.com/rails/rails/blob/v7.2.2.2/activerecord/lib/active_record/connection_adapters/abstract/transaction.rb#L135-L137) (not documented) callback that run inside of a transaction.
- Claim resources after INSERT but before COMMIT (when all IDs are available).
- Batch multiple claims from the same transaction when possible.
- Set a client-side 200ms timeout on Topology Service requests.
- Implement a client-side circuit breaker on when there are `N` claims/sec Topology Service requests inside of a transaction.
  `N` should be configurable and takes database load into consideration.
- Part of the rollout of claims will be to use a feature flag and progressively
  increase the percentage of claims we do to observe any negative database
  effects. The exact rollout process is yet to be figured out and will be part
  of [Adopt GitLab.com to Cell Cluster as Legacy Cell](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/1625)

**Monitoring Requirements**:

- Connection pool utilization and wait times.
- Transaction duration for claim-related operations.
- Topology Service request latencies (P50, P99, P99.95).
- Client-side circut-breaker hits.
- Client-side timeout hits.
- Failed claims and rollback rates from the client and server.

## Consequences

**Positive**:

- Simpler implementation: Works within existing Rails patterns and callbacks
- Faster time to production: Minimal architectural changes required
- Atomic rollback: Transaction rolls back cleanly if the claim fails
- No side effects issue: Claims happen before commit, preventing situation
  where we commit on the database before checking with Topology Service.
- Sufficient capacity: Current infrastructure supports 6x peak load
- Batching support: Can batch claims within the same transaction after all IDs are available

**Negative**:

- Connection pool usage: Network calls hold database connections during claim.
- Transaction duration: Longer transactions ~200ms can result in more load on the database:
  - [WAL accumulation](https://www.percona.com/blog/five-reasons-why-wal-segments-accumulate-in-the-pg_wal-directory-in-postgresql/).
  - Connection pool exhaustion since each transactions holds a connection.
- Scalability ceiling: Limited by connection pool size (currently 290 claims/sec).

## Alternatives

### Option 1: Claims Outside Transactions

Deferred because:

- Massive refactoring required, for example thousands of `User.create!` calls would need modification,
  and this would be required per model to claim.
- Breaks Rails conventions of automatic transaction wrapping.
- Higher implementation complexity and error handling.
- Difficult to batch claims when IDs are not yet available.
- Current capacity (6x peak) makes optimization premature.

### Option 2: Database Trigger

Rejected because:

- Cannot handle non-database side effects.
- Adds complexity outside the Rails application layer.
- Difficult to test and maintain.
- Verification loop already handles cascade deletion cases.
- Still doesn't verify whether the resource is claimable cluster-wide.

### Option 3: Override ActiveRecord save/update

Rejected because:

- Cannot batch multiple attributes efficiently.
- Triggers one claim per attribute instead of a batched request.
- Doesn't work with `update_column` and similar methods.
