---
title: "Artifact Registry ADR 003: System Requirements"
owning-stage: "~devops::package"
description: "Infrastructure requirements and performance constraints for the Artifact Registry"
toc_hide: true
---

## Context

The Artifact Registry requires a robust infrastructure foundation to support enterprise-scale artifact management. This ADR documents infrastructure requirements and performance constraints.

The registry must support:

- **Reliable background job processing** for async operations, migrations, and lifecycle policies
- **Persistent metadata storage** for artifacts, repositories, access control, and audit logs
- **Content-addressable blob storage** for artifact data with organization-level deduplication
- **Analytics and insights** for cost tracking, usage patterns, and AI-powered recommendations
- **Distributed caching** for frequently accessed data, rate limiting, and session management
- **Event-driven architecture** for real-time notifications and platform integration

Technology choices fulfilling these requirements are documented in [ADR-006](006_technology_stack.md)<!-- , based on the decision to build within the Rails monolith ([ADR-005](005_implementation_architecture.md)) -->.

## Decision

The Artifact Registry requires the following infrastructure capabilities:

### 1. Background Job Processing

**Purpose**: Asynchronous task execution for long-running operations with reliable retry logic and monitoring.

**Use Cases**:

- Event recording and audit log processing
- Async bulk operations (bulk delete, bulk tag, bulk move)
- Artifact migrations (from external providers, from existing GitLab registries)
- Lifecycle policy execution (retention cleanup, expiration management)
- Cache invalidation and warming
- Vulnerability scanning and security analysis
- Metadata extraction and enrichment

### 2. Relational Database

**Purpose**: Persistent storage for artifact metadata, repositories, workspaces, and access control.

**Use Cases**:

- Artifact, repository, and workspace metadata management
- Access control rules and RBAC definitions
- Audit logs and compliance tracking
- Lifecycle policy definitions and execution state
- Virtual repository configuration and upstream source management
- Organization and workspace hierarchy
- Reference tracking for garbage collection

### 3. Blob Storage

**Purpose**: Content-addressable storage for artifact deduplication, supporting organization scope and GitLab's consolidated object storage.

**Use Cases**:

- Store actual artifact data (container layers, package files, binary blobs)
- Support organization deduplication (see [ADR-002](002_storage_deduplication_scope.md))
- Enable efficient garbage collection and lifecycle management
- Support multi-region replication and disaster recovery

### 4. Content Delivery Network (CDN)

**Purpose**: Edge caching for artifact downloads, reducing latency and origin load for frequently accessed content.

**Use Cases**:

- Cache artifact blob downloads at edge locations close to users
- Reduce load on object storage and origin servers
- Improve download latency for geographically distributed teams
- Cost optimization through reduced egress from primary storage

### 5. Analytics Engine

**Purpose**: Efficiently filter, aggregate, and query across large datasets for analytics, cost tracking, and AI-powered insights.

**Use Cases**:

- Long-term event records (downloads, uploads, access patterns)
- Cost analytics and forecasting
- Usage trends and insights
- Feed AI/ML features for recommendations and optimization
- Performance analysis and bottleneck identification

### 6. Cache

**Purpose**: Cache for frequently accessed data, rate limiting, distributed locking, and ephemeral shared application state.

**Use Cases**:

- Cache metadata and session data
- Rate limiting and quota enforcement
- Distributed locking for concurrent operations
- Temporary upload session tracking
- Ephemeral shared application state
- Cache invalidation coordination

### 7. Event Bus

**Purpose**: Event-driven architecture for real-time notifications and platform integration.

**Use Cases**:

- Publish events for all actions on artifacts (upload, download, delete, tag, etc.)
- Trigger actions from events (cache invalidation, analytics ingestion, security scanning)
- Integrate with other GitLab platform components
- Real-time notifications to subscribers
- Event streaming for audit and compliance

## Performance, Scale, and Operational Constraints

The Artifact Registry targets large enterprise organizations and GitLab.com scale. This section captures high-level constraints; detailed capacity planning will follow.

### Expected Usage Characteristics

- High read volume (downloads, metadata reads) with sustained write activity (uploads, tag updates, lifecycle policy actions).
- Strong skew toward a small number of "hot" repositories and artifacts, with a long tail of rarely accessed content.
- Spiky traffic patterns driven primarily by CI/CD workloads (for example, deploy waves, large pipelines).
- Many large organizations with thousands of repositories and millions of artifacts.

### Key Performance and Reliability Goals

- API and registry operations remain responsive under typical enterprise workloads, with predictable degradation under load rather than widespread failures.
- Core operations (upload, download, garbage collection, lifecycle policies) are safe and correct under high concurrency.
- Metadata operations (listing, searching, resolving versions) perform well at "millions of artifacts per organization" scale.
- Background processing can drain backlogs created during peak events within an acceptable window.

### Capacity and Growth Considerations

The Artifact Registry launches as a new service with no initial data. Growth will be organic as customers adopt the offering. We plan migration tooling for a later phase, enabling customers to move artifacts from existing registries over time.

As customers migrate, load and storage will shift gradually from existing registries (Container Registry, Package Registry) to the Artifact Registry. The combined load across all registries should remain stable, with the unified registry growing as existing registries shrink.

Current GitLab.com registry metrics provide guidance for the scale the Artifact Registry may eventually reach. For detailed metrics, see the [internal capacity planning note](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/17524#note_3024864366).

- **Traffic (GitLab.com projection)**:

  - API traffic: thousands of requests per second steady-state, scaling to low tens of thousands during CI burst peaks.
  - Background job throughput (GC, lifecycle, migrations, analytics): tens to hundreds of jobs per second, with peaks during cleanup operations.
  - Storage throughput: hundreds of MB/s upload, several GB/s download.

- **Data volume (GitLab.com projection)**:

  - Total stored artifact data: tens of petabytes across all customers.
  - Object count: around one billion objects in blob storage.
  - Artifact metadata: millions of artifacts per large organization, resulting in PostgreSQL datasets that can grow into the TB range.
  - Audit/event records retained: estimates to be determined based on usage patterns; typical enterprise retention is 90-365 days.

- **Storage patterns**:

  - Existing registries show that most artifact data is infrequently accessed after initial push/pull cycles, with the majority of storage eventually moving to cold/archive tiers (over 80% in cold storage classes). The Artifact Registry should plan for similar tiered storage to optimize costs.

Detailed infrastructure sizing, deployment topologies, and environment-specific SLAs are out of scope for this ADR.

## References

- [ADR-002: Storage Deduplication Scope](002_storage_deduplication_scope.md) - Deduplication scoped to organizations
<!-- - [ADR-008: Content-Addressable Storage](008_content_addressable_storage.md) - Storage approach enabling deduplication -->
<!-- - [ADR-005: Implementation Architecture](005_implementation_architecture.md) - Rails monolith decision -->
- [ADR-006: Technology Stack](006_technology_stack.md) - Technology choices based on these requirements
- [ADR-010: Data Retention](010_data_retention.md) - Retention policies for artifacts, audit logs, and cached content
