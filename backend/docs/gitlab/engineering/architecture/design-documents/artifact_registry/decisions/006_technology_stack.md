---
title: "Artifact Registry ADR 006: Technology Stack"
owning-stage: "~devops::package"
description: "Technology choices for the Artifact Registry satellite service based on system requirements and architecture decisions"
toc_hide: true
---

## Context

This ADR maps system requirements ([ADR-003](003_system_requirements.md)) to specific technologies for the Artifact Registry satellite service.

The Artifact Registry is built as a standalone service outside the Rails monolith, deployed using the [Runway GKE runtime](https://docs.runway.gitlab.com/runtimes/kubernetes/getting-started/). This required selecting a technology stack that can independently fulfill all system requirements while integrating with the broader GitLab platform.

The technology choice is informed by two factors:

1. **GitLab LabKit**: LabKit is a minimalist library that provides common functionality for GitLab services, including correlation, logging, tracing, and metrics. It is available for both [Go](https://gitlab.com/gitlab-org/labkit) and [Ruby](https://gitlab.com/gitlab-org/ruby/gems/labkit-ruby). Recent developments are porting LabKit to [Rust](https://gitlab.com/gitlab-org/rust/labkit-rs).
1. **Package stage team expertise**: The [Package stage](/handbook/engineering/devops/package/) team has deep experience with both Go (building and operating the Container Registry) and Rails (building and operating the Package Registry within the monolith).

Three alternatives were considered:

1. **Go (Golang)** — Following the pattern of other GitLab satellite services such as Gitaly, Workhorse, and the Container Registry.
1. **Ruby on Rails** — As a standalone Rails API-only application, reusing selected patterns and libraries from the GitLab monolith.
1. **Rust** — Leveraging Rust's performance and safety guarantees, following the path of newer GitLab projects like the Knowledge Graph and the new Auth services.

## Decision

**Go (Golang)** is the recommended language for new GitLab satellite services.

Go is the most used for GitLab satellite services (Gitaly, Runner, Workhorse, Container Registry) and provides the best balance of performance, ecosystem maturity, and organizational alignment for this project.

**Pros**:

1. **Well-established**: Most used across GitLab satellite services (Gitaly, Runner, Workhorse, Container Registry), with proven Go libraries and patterns.
2. **Organizational investment**: LabKit provides observability, logging, tracing, usage events, billing events and others out of the box. Supporting tooling and infrastructure built around Go is more likely to benefit existing services. Tooling and deployment improvements built for the Artifact Registry may be retrofitted to existing Go satellite services to reduce their maintenance burden. If Go continues to be the most used option, then the upfront investment may be more valuable than creating or adapting existing tooling for a Rails service.
3. **Performance characteristics**: Excellent performance for high-throughput, concurrent I/O workloads (artifact uploads/downloads), low memory footprint, and efficient resource utilization. Strong concurrency primitives (goroutines, channels) well-suited for handling many simultaneous artifact transfers, including long-lived streaming connections.
4. **Type safety for agentic development**: Compile-time type safety and static typing give AI agents clearer contracts to reason about compared to dynamically typed languages, though Go's guarantees are weaker than Rust's (for example, Go does not prevent unintended mutation of structures). Go's toolchain also provides runtime quality gates for agentic workflows, such as the race and goroutine leak detectors.
5. **Cloud-native alignment**: Static binaries simplify deployment and distribution across all installation types.
6. **Broader contributor pool**: Go has a larger pool of skilled contributors within GitLab compared to Rust, and is more familiar to teams working on satellite services than a standalone Rails application.
7. **Vendor-supported SDK ecosystem**: All major cloud providers ([AWS](https://github.com/aws/aws-sdk-go-v2), [GCP](https://pkg.go.dev/cloud.google.com/go/storage), [Azure](https://github.com/Azure/azure-sdk-for-go)) maintain official Go SDKs, ensuring long-term support for object storage and other cloud-native integrations.

**Cons**:

1. **Tooling gap**: Need to build or adopt libraries for database migrations, database load balancing, background jobs or feature flags. Existing monolith tooling cannot be directly extracted, and the Database Frameworks team has no plans to provide equivalent functionality for satellite services.
   - Partial mitigation: the database load balancer used in the Container Registry may be extracted and re-used
2. **Smaller contributor pool**: Fewer Go contributors within GitLab compared to Ruby/Rails, which may slow cross-team contributions.
3. **Virtual registries rewrite**: The current virtual registries in the Rails monolith will require a full rewrite.
   - Partial mitigation: a [PoC](https://gitlab.com/jdrpereira/artifact-registry-poc/-/blob/main/README.md) proved that a rewrite is a viable option.

## Components and Dependencies

Libraries listed below are initial candidates, not final choices. Follow-up MRs will investigate these and other options to identify the best fit for each component, and this ADR will be updated accordingly.

### 1. Background Job Processing: Redis-backed Queue

The Artifact Registry requires background job processing for lifecycle policy execution, garbage collection, event processing, analytics aggregation, and cleanup operations. This includes reliable scheduling (cron-style and one-off), retry and failure handling, priority queues, and monitoring and observability.

The background job processing library will be wrapped with a LabKit abstraction for consistent usage across satellite services.

**Candidate library: [asynq](https://github.com/hibiken/asynq)**:

- MIT-licensed, actively maintained, Redis-only by design. No additional infrastructure dependency.
- Sidekiq-like semantics familiar to GitLab engineers (queues, retries, dead-letter queue)
- Supports priority queues, cron-style scheduling, and unique job deduplication
- Built-in Prometheus metrics exporter and web UI ([asynqmon](https://github.com/hibiken/asynqmon))

**Complementary option: [River](https://github.com/riverqueue/river)**:

- Transactional enqueueing: jobs are enqueued within the same PostgreSQL transaction as data changes, eliminating race conditions where jobs execute before the originating transaction commits or after it rolls back.
- May be relevant for operations that need atomicity between metadata writes and background job scheduling.
- Supports priority queues, cron-style scheduling, unique jobs, and snoozing
- Go-native design using generics for strongly typed job arguments

### 2. Relational Database: PostgreSQL

**Why PostgreSQL**:

- Aligns with GitLab's existing infrastructure and database standards
- Proven at scale with GitLab.com's massive dataset
- Rich feature set: JSONB for flexible metadata, full-text search, advanced indexing
- Existing expertise, operational runbooks, and backup/recovery procedures
- Strong consistency guarantees for critical metadata

**Database Isolation**: The Artifact Registry satellite service will use its own dedicated PostgreSQL database, providing full isolation from the Rails monolith without requiring the [multiple databases](https://docs.gitlab.com/ee/development/database/multiple_databases.html) decomposition pattern.

#### 2.1 Database Schema Migrations

Satellite services lack the advanced migration tooling impacting development velocity. Database schema migrations will be wrapped with a LabKit abstraction as reusable infrastructure for all satellite services, providing zero-downtime migrations, safe schema changes (including partitioned tables), rollback capabilities, and migration state tracking. Migrations should be self-contained and triggered at pod startup using distributed locking (PostgreSQL advisory locks or Redis), avoiding the need for external orchestration or the monolith's post-deployment migration model. The application must enforce backwards schema compatibility around upgrades.

**Candidate libraries**: [goose](https://github.com/pressly/goose) (supports PostgreSQL advisory locks for safe startup-time migrations), [Atlas](https://atlasgo.io/) (open-core, with an Apache 2.0 [Kubernetes operator](https://atlasgo.io/integrations/kubernetes)), and [pgroll](https://github.com/xataio/pgroll) (zero-downtime, reversible schema migrations).

#### 2.2 Database Query Layer

The Go ecosystem offers several approaches to database interaction, ranging from full ORMs to type-safe query builders and code generation from SQL.

**Candidate libraries**:

- **ORMs**: [Ent](https://entgo.io/) (graph-based, code-generated schema), [Bun](https://bun.uptrace.dev/) (lightweight, SQL-first ORM)
- **Type-safe query builders**: [go-jet/jet](https://github.com/go-jet/jet) (generates type-safe SQL from the database schema)
- **SQL code generation**: [sqlc](https://sqlc.dev/) (generates Go code from raw SQL queries)

### 3. Blob Storage: Object Storage

**Why Object Storage**:

- Aligns with GitLab's [consolidated object storage](https://docs.gitlab.com/administration/object_storage/#configure-a-single-storage-connection-for-all-object-types-consolidated-form) strategy
- Compatible with all [supported providers](https://docs.gitlab.com/administration/object_storage/#supported-object-storage-providers) (AWS S3, Google Cloud Storage, Azure Blob Storage, MinIO, etc.)
- Proven at scale with existing GitLab registries and CI artifacts
- Cost-effective for large-scale blob storage
- Existing operational expertise and integration patterns

**Deployment-Specific Choices**:

- **GitLab.com (SaaS)**: Google Cloud Storage (GCS), leveraging existing infrastructure and operational expertise
- **Dedicated**: Amazon S3, aligning with established Dedicated deployment patterns
- **Self-Managed**: GCS, S3, or S3-compatible providers. Local filesystem is not supported as the service is Kubernetes-only. Additional [providers](https://docs.gitlab.com/administration/object_storage/#supported-object-storage-providers) may be added later.

**Native SDKs**: The Artifact Registry will use native provider SDKs behind an internal interface, rather than third-party clients and abstractions. This guarantees access to provider-specific primitives (for example, GCS atomic object move, resumable uploads, S3 multipart upload with fine-grained parallelism) that abstraction libraries may hide behind lowest-common-denominator APIs. Long-term vendor maintenance of these SDKs is also essential for a service with this dependency profile.

**Bucket Isolation**: Although the design supports shared buckets using path prefixes, the Artifact Registry should use a dedicated bucket. This isolates lifecycle policies, access controls, cost tracking, and operational management from other GitLab storage.

### 4. Content Delivery Network (CDN)

CDN is deployment-specific and optional:

- **GitLab.com (SaaS)**: Google Cloud CDN, as the Container Registry currently does (achieving [~87% cache hit rate](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/17524#note_3023542021)). AWS CloudFront may be required for Cells.
- **Dedicated**: Optional use of AWS CloudFront, also supported by the Container Registry
- **Self-Managed**: Optional; customers can configure either Cloud CDN or CloudFront

### 5. Analytics Engine: ClickHouse

ClickHouse is out of scope for MVP. It will be needed later for advanced analytics features (long-term event records, usage trends, and AI-powered insights). This is an optional dependency — the Artifact Registry must be fully functional without ClickHouse available.

### 6. Cache: Redis Protocol Compatible Service

A Redis protocol compatible service (for example, Redis, Valkey) is already required for background job processing, so no additional infrastructure dependency is introduced. It covers caching, rate limiting, distributed locking, and ephemeral shared application state.

**Candidate library: [go-redis](https://github.com/redis/go-redis)**. Wrapping the Redis client in a LabKit abstraction may be relevant for consistent usage across satellite services.

### 7. Event Bus

For MVP, inter-service communication (primarily with the Rails monolith) will use direct API calls.

**Future Alternative: NATS**

As the registry evolves beyond MVP, **NATS** (specifically [NATS JetStream](/handbook/engineering/architecture/abstractions/candidate/nats/)) may be considered when:

- Event persistence and replay become critical requirements
- Audit and compliance requirements demand event sourcing
- Complex event routing and filtering patterns emerge
- GitLab's infrastructure includes NATS as a standard component

**Why NATS (for future consideration)**:

- Persistent event streaming with JetStream
- Message replay and consumer groups
- Better for event sourcing patterns
- Aligns with [GitLab's NATS abstraction](/handbook/engineering/architecture/abstractions/candidate/nats/)
- Supports complex event routing and filtering

**Current NATS Status**:

- **Proposal stage**: There is a [proposal to package NATS](https://gitlab.com/gitlab-org/architecture/gitlab-data-analytics/design-doc/-/work_items/184) for Self-Managed and Dedicated installations
- **Production use**: NATS is currently deployed in production for the Data Insights Platform (DIP) on CustomersDot environments, with [operational runbooks](https://runbooks.gitlab.com/nats/) available
- **Standard installations**: NATS is not yet available as standard infrastructure across all GitLab installation types, so it remains a future option for the Artifact Registry
- **Abstraction maturity**: The [NATS abstraction page](/handbook/engineering/architecture/abstractions/candidate/nats/) is under development

## Alternatives

### Ruby on Rails (Standalone Application)

Building the satellite service as a standalone Rails application, separate from the monolith but reusing the Rails ecosystem.

**Pros**:

- API-only application mode natively supported by Rails
- Familiar technology for the majority of GitLab engineers
- Can reuse known patterns and gems from the monolith (for example, ActiveRecord additions, Sidekiq)
- Existing database and background job tooling that could be extracted and re-used (zero-downtime database migrations, background migrations, database load balancer, Sidekiq reliable fetcher). A [PoC](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/18455#note_3105411409) demonstrated this is technically feasible.
- Existing GraphQL tooling (complexity analysis, batch loaders) that could be extracted and re-used
- Existing authentication logic (declarative policy) can be re-used directly since it's a standalone gem
- Reuse existing expertise in Rails performance optimization at GitLab scale
- Virtual Registries logic could be extracted and re-used

**Cons**:

- Rails standalone services are not an established pattern at GitLab. Satellite services are written in Go or Rust, meaning time spent extracting and/or building supporting infrastructure and tooling would likely yield a lower ROI, unless there is a desire to start investing in Rails services moving forward.
- Higher memory footprint per process, slower boot times compared to compiled runtimes, and less efficient for high-concurrency I/O workloads like streaming file uploads/downloads.
- Extracting tooling from the monolith into standalone libraries is non-trivial work with unclear scope. Background data migrations in particular are coupled to the upgrade lifecycle in ways that Helm cannot natively express, requiring init container scripts and custom orchestration, and depend on Sidekiq as a separate long-running process.
  - Mitigation: AI tooling can accelerate the extraction and adaptation effort
- Delivery [actively discourages](https://docs.google.com/document/d/1tZLY4px_wiWBe9F2PEcE3M8avp52TCyc1hAS0dF34YM/edit?tab=t.0) replicating the monolith's migration patterns in new services, as they are considered operationally burdensome and incompatible with cloud-native deployments.
- Dependency management overhead (Bundler, native extensions) complicates packaging and distribution.
- Risk of accruing tight coupling to monolith patterns, making the service harder to evolve independently.
  - Mitigation: Careful selection of which monolith components to extract can avoid inheriting problematic patterns
- Requires a proxy sidecar (similar to Workhorse's function) to handle long-lived uploads, since some clients (for example, `npm`) do not support signed URLs for direct uploads, and self-managed setups may restrict object storage access to GitLab components only. This increases the total number of runtime components (Redis, PostgreSQL, Rails, Proxy sidecar) and adds an additional hard dependency that may affect availability (beyond the monolith, which is already required for authentication).
- The proxy sidecar introduces a dual software supply chain: one for the proxy and one for the Rails application, increasing the surface area for dependency management, security audits, and vulnerability remediation.
- The proxy sidecar adds an additional network hop to every proxied request. While the per-request overhead may be small, it compounds with the existing proxy layers (load balancer, ingress, monolith for authentication) and affects tail latency under load.
- Weaker object storage SDK ecosystem: Microsoft [retired](https://azure.microsoft.com/en-us/updates?id=retirement-notice-the-azure-storage-ruby-client-libraries-will-be-retired-on-13-september-2024) the official Azure Storage Ruby client library. A community replacement exists but is not vendor-maintained.
- Ruby is an interpreted, dynamically typed language, which provides weaker quality gates for agentic development workflows compared to compiled, statically typed languages.

**Why it was rejected**: Not an established pattern for satellite services, requires a proxy sidecar for long-lived uploads, and the investment in extracting monolith tooling would likely have lower organizational ROI than investing in Go.

### Rust

Building the satellite service in Rust, leveraging its performance characteristics and type system guarantees.

Rust is gaining traction within GitLab: the [Knowledge Graph](https://gitlab.com/gitlab-org/rust/knowledge-graph), GLQL, and the GLFM gem are built in Rust. A LabKit for Rust is being developed, and the Auth Architecture team has selected Rust for their services.

**Pros**:

- No garbage collector overhead, with zero-cost abstractions. While Go's GC has improved significantly, the absence of GC pauses is a meaningful advantage for latency-sensitive workloads.
- Compiler-enforced memory safety eliminates entire classes of bugs (use-after-free, data races, null pointer dereferences) without runtime overhead
- Growing GitLab investment in Rust (Knowledge Graph, GLQL, GLFM, Auth Architecture services) provides internal precedent and expanding expertise
- LabKit for Rust is under development
- Rust's compiler acts as a natural quality gate for AI-generated code

**Cons**:

- Go already provides the type safety and compilation benefits needed for this use case. There is no concrete need for Rust's additional guarantees (for example, memory safety without a garbage collector) in this context.
- Smallest contributor pool of the three options within GitLab. Go has a significantly larger pool of skilled contributors.
  - Mitigation: AI tooling can accelerate onboarding and reduce the learning curve
- The immediately available ecosystem and existing integrations across GitLab products (LabKit, observability, deployment tooling) strongly favor Go. LabKit for Rust is still under development; platform integration is not yet production-ready.
- Rust is better suited for exceptional cases that require higher performance or a more expressive type system, such as data engineering (Knowledge Graph) or compilers. The Artifact Registry does not fall into that category.
- Rust's async runtime is not part of the language and requires external crates (for example, tokio). Go's concurrency model (goroutines, channels) is a first-class language feature, making it more straightforward.
- Infrastructure tooling built around Go benefits the wider organization, whereas Rust tooling would serve a smaller set of services.
- Current virtual registries will require a rewrite.

**Why it was rejected**: Go has a larger contributor pool within GitLab and a more mature ecosystem for satellite services. While Rust offers stronger guarantees, the practical benefits do not outweigh the ecosystem and contributor advantages of Go for this use case.

## References

- [ADR-003: System Requirements](003_system_requirements.md)
<!-- - [ADR-005: Implementation Architecture](005_implementation_architecture.md) -->
<!-- - [ADR-010: Data Retention](010_data_retention.md) - Audit log retention policies stored in ClickHouse -->
- [GitLab Object Storage Documentation](https://docs.gitlab.com/administration/object_storage/)
- [GitLab NATS Abstraction](/handbook/engineering/architecture/abstractions/candidate/nats/)
- [Multiple Databases Documentation](https://docs.gitlab.com/ee/development/database/multiple_databases.html)
- [Sidekiq Development Guidelines](https://docs.gitlab.com/ee/development/sidekiq/)
- [Lab Bench: GitLab SOA Architecture](https://docs.google.com/document/d/11Zj918LuZeY3fPcU50ZPhzJtcqzvyXaO0SDamW7cDc8/edit?tab=t.0)
