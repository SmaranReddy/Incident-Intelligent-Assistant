---
title: "Artifact Registry"
status: proposed
creation-date: "2025-12-15"
authors: [ "@jdrpereira", "@10io" ]
coaches: [ "@sxuereb" ]
dris: [ "@trizzi", "@crystalpoole" ]
owning-stage: "~devops::package"
participating-stages: []
toc_hide: true
---

<!--
Before you start:

- Copy this file to a sub-directory and call it `_index.md` for it to appear in
  the design documents list.
- Remove comment blocks for sections you've filled in.
  When your document ready for review, all of these comment blocks should be
  removed.

To get started with a document you can use this template to inform you about
what you may want to document in it at the beginning. This content will change
/ evolve as you move forward with the proposal.  You are not constrained by the
content in this template. If you have a good idea about what should be in your
document, you can ignore the template, but if you don't know yet what should
be in it, this template might be handy.

- **Fill out this file as best you can.** At minimum, you should fill in the
  "Summary", and "Motivation" sections.  These can be brief and may be a copy
  of issue or epic descriptions if the initiative is already on Product's
  roadmap.
- **Create a MR for this document.** Assign it to an Architecture Evolution
  Coach (i.e. a Principal+ engineer).
- **Merge early and iterate.** Avoid getting hung up on specific details and
  instead aim to get the goals of the document clarified and merged quickly.
  The best way to do this is to just start with the high-level sections and fill
  out details incrementally in subsequent MRs.

Just because a document is merged does not mean it is complete or approved.
Any document is a working document and subject to change at any time.

When editing documents, aim for tightly-scoped, single-topic MRs to keep
discussions focused. If you disagree with what is already in a document, open a
new MR with suggested changes.

If there are new details that belong in the document, edit the document. Once
a feature has become "implemented", major changes should get new blueprints.

The canonical place for the latest set of instructions (and the likely source
of this file) is
[content/handbook/engineering/architecture/design-documents/_template.md](https://gitlab.com/gitlab-com/content-sites/handbook/-/blob/main/content/handbook/engineering/architecture/design-documents/_template.md).

Document statuses you can use:

- "proposed"
- "accepted"
- "ongoing"
- "implemented"
- "postponed"
- "rejected"

-->

<!-- Design Documents often contain forward-looking statements -->
<!-- vale gitlab.FutureTense = NO -->

<!-- This renders the design document header on the detail page, so don't remove it-->
{{< engineering/design-document-header >}}

<!--
Don't add a h1 headline. It'll be added automatically from the title front matter attribute.

For long pages, consider creating a table of contents.
-->

## Summary

<!--
This section is very important, because very often it is the only section that
will be read by team members. We sometimes call it an "Executive summary",
because executives usually don't have time to read entire documents like this.
Focus on writing this section in a way that anyone can understand what it says,
the audience here is everyone: executives, product managers, engineers, wider
community members.

A good summary is probably at least a paragraph in length.
-->

The Artifact Registry is a **new SKU** for GitLab providing centralized artifact management. It competes directly with JFrog Artifactory and Sonatype Nexus.

Unlike GitLab's current project-level package and container registries, the Artifact Registry gives enterprises centralized control, visibility, and management of all artifacts (containers, packages, ML models).

The registry targets large enterprises (1,000+ users) who rely on external tools and need tool consolidation, cost optimization, and centralized governance. It offers:

- **Organization-level artifact management** with a unified control plane
- **Virtual repositories** that aggregate multiple upstream sources (public registries, cloud providers, legacy tools)
- **AI-powered configuration assistance** and cost optimization
- **Migration tooling** to onboard from other providers and existing GitLab package and container registries
- **Content-addressable storage** with organization-scoped deduplication
- **Multi-format support** starting with Docker, Maven, and npm

We are building a new registry from the ground up, free from backward compatibility constraints, with modern event-driven architecture from the start.

This initiative began with an [internal proposal](https://gitlab.com/gitlab-org/gitlab/-/issues/568349) and evolved into an [extended specification](https://gitlab.com/gitlab-org/ci-cd/package-stage/unified-artifact-management/-/blob/main/blueprint.md) (internal) with user journeys, capabilities, database schemas, and implementation roadmap. This blueprint covers high-level design and architectural decisions suitable for public disclosure; the internal document contains additional strategic details.

## Glossary

- **Organization**: The highest-level container entity in GitLab, serving as the primary isolation and sharding boundary. Contains workspaces, repositories, and all artifacts. Each has independent storage, cost attribution, and deduplication scope.
- **Workspace**: A logical grouping of repositories within an organization, organizing artifacts by team, security domain, or product line. Enables organizational structure mapping and workspace-scoped access controls. (Post-MVP feature; name subject to UX validation.)
- **Repository**: A typed container for artifacts of a specific format (Docker, Maven, npm). Either Local (stores the organization's own artifacts) or Virtual (aggregates upstream sources).
- **Local Repository**: Stores artifacts published by the organization. Content-addressable storage persists artifacts; lifecycle policies manage them.
- **Virtual Repository**: A proxy/aggregation layer caching artifacts from upstream sources (public registries, cloud providers, legacy tools). Provides unified access to local and remote artifacts.
- **Content-Addressable Storage (CAS)**: Artifacts use their SHA256 hash as both identifier and storage key, enabling deduplication, immutable paths, and integrity verification.
- **Artifact**: A versioned software package (Docker image, Maven package, npm module) stored in the registry, consisting of metadata and one or more blobs.
- **Blob**: Raw artifact content (container layers, package files). Stored and deduplicated within each organization.
- **Upstream Source**: An external registry providing cached content to virtual repositories (Maven Central, npmjs.com, Docker Hub, AWS ECR, JFrog Artifactory, other GitLab repositories).
- **Deduplication**: Automatic elimination of duplicate storage within an organization. Identical content (same SHA256 hash) stores only once.
- **OCI (Open Container Initiative)**: Industry standards for container formats and runtimes, extending beyond Docker's original format.
- **MVP (Minimum Viable Product)**: Phase 1, covering ~30% of enterprise [use cases](https://gitlab.com/gitlab-org/ci-cd/package-stage/unified-artifact-management/-/blob/55c4a5f6af4c1049da70062a5d39561a5d1ca189/blueprint.md#core-use-cases-by-user-journey-phase) (internal) with core storage, virtual repositories, and basic lifecycle management.
- **GA (General Availability)**: Production-ready status for all installation types (GitLab.com, Self-Managed, Dedicated).
- **SKU (Stock Keeping Unit)**: A distinct product offering. The Artifact Registry is a new premium SKU, separate from existing free-tier registries.
- **RBAC (Role-Based Access Control)**: Permissions assigned based on roles at organization, workspace, or repository level.

## Motivation

<!--
This section is for explicitly listing the motivation, goals and non-goals of
this document. Describe why the change is important, all the opportunities,
and the benefits to users.

The motivation section can optionally provide links to issues that demonstrate
interest in a document within the wider GitLab community. Links to
documentation for competing products and services is also encouraged in cases
where they demonstrate clear gaps in the functionality GitLab provides.

For concrete proposals we recommend laying out goals and non-goals explicitly,
but this section may be framed in terms of problem statements, challenges, or
opportunities. The latter may be a more suitable framework in cases where the
problem is not well-defined or design details not yet established.
-->

GitLab's current artifact management is fragmented and project-centric:

- **Project-level coupling**: All artifacts bind to individual projects, creating a 1:N management burden where N can reach thousands of projects
- **No centralized visibility**: Customers cannot see or manage artifacts across projects. Questions like "What artifacts do we have?" or "What's our total storage cost?" are hard to answer
- **Separate implementations**: Package registry (Rails) and container registry (Go) are separate applications with different architectures, preventing unified experiences
- **Repetitive configuration**: Security policies, lifecycle rules, and access controls must be configured per-project with limited inheritance
- **Event data silos**: Usage and security events scatter across services with no centralized harvesting. Security features like container scanning are reactive rather than proactive
- **Storage inefficiencies**: Package registry lacks deduplication; container registry's instance-wide deduplication creates operational complexity with expensive cross-partition operations
- **Development velocity gap**: Container registry has limited access to Rails monolith tooling (migrations, feature flags, automated releases), slowing feature delivery <!-- (see [ADR-005](decisions/005_implementation_architecture.md)) -->
- **Limited enterprise features**: No virtual repositories, limited upstream proxying, no migration path from JFrog/Nexus

**Target Market:**

Enterprises using competing tools who want to:

- **Consolidate tools** into GitLab
- **Centralize management** of their artifact landscape
- **Optimize costs** with visibility and AI-powered recommendations
- **Migrate** from legacy tools

**Why Now:**

- **Team expertise**: The Package stage has 6+ years operating registries, understanding what works, what doesn't, and what customers need
- **Technology maturity**: Modern storage, event streaming, and AI/ML capabilities now exist that weren't available when current registries were designed
- **Clean break advantage**: Starting fresh provides velocity and architectural freedom for today's requirements
- **Market opportunity**: We can leapfrog competitors with an AI-native, top-level scoped solution

**Strategic Alignment:**

- **AI Integration**: Event-rich foundation enables AI-powered configuration, cost optimization, and predictive capabilities (see [AI-Enhanced Artifact Management](https://unified-artifact-managment-965acd.gitlab.io))
- **Enterprise Positioning**: Competes directly with JFrog and Nexus
- **Platform Consolidation**: Eliminates external artifact management tools

### Goals

<!--
List the specific goals / opportunities of the document.

- What is it trying to achieve?
- How will we know that this has succeeded?
- What are other less tangible opportunities here?
-->

1. **Provide organization-level artifact management** eliminating project-level fragmentation
2. **Enable migration** from JFrog Artifactory and Sonatype Nexus with tooling, validation, and safety controls
3. **Deliver AI-powered optimization**: configuration assistance, cost recommendations, predictive insights
4. **Support virtual repositories** aggregating upstream sources with caching
5. **Reduce platform engineering effort** through automation, templates, and AI assistance
6. **Achieve cost savings** through deduplication, lifecycle policies, and optimization
7. **Align with Cells architecture** using Organizations as the isolation and sharding boundary
8. **Support multi-format artifacts**: containers (Docker, OCI), packages (Maven, npm), and future formats

### Non-Goals

<!--
Listing non-goals helps to focus discussion and make progress. This section is
optional.

- What is out of scope for this document?
-->

1. **Enhancing existing registries**: This is a new SKU, not an evolution of current registries. Existing registries continue in parallel with no sunset plans. Free tier keeps current tools; Artifact Registry is a premium feature. Building new avoids disrupting tier 0 tools and eliminates backward compatibility constraints
2. **Supporting nested group artifact management**: Organizations are the isolation boundary, not top-level groups or projects
3. **Providing instance-wide deduplication**: Deduplication scopes to organizations for clear cost attribution
4. **Supporting all artifact formats in MVP**: Starting with Docker, Maven, npm; additional formats in v1.0+

## Proposal

<!--
This is where we get down to the specifics of what the proposal actually is,
but keep it simple!  This should have enough detail that reviewers can
understand exactly what you're proposing, but should not include things like
API designs or implementation. The "Design Details" section below is for the
real nitty-gritty.

You might want to consider including the pros and cons of the proposed solution so that they can be
compared with the pros and cons of alternatives.
-->

## Design and implementation details

<!--
This section should contain enough information that the specifics of your
change are understandable. This may include API specs (though not always
required) or even code snippets. If there's any ambiguity about HOW your
proposal will be implemented, this is the place to discuss them.

If you are not sure how many implementation details you should include in the
document, the rule of thumb here is to provide enough context for people to
understand the proposal. As you move forward with the implementation, you may
need to add more implementation details to the document, as those may become
valuable context for important technical decisions made along the way. A
document is also a register of such technical decisions. If a technical
decision requires additional context before it can be made, you probably should
document this context in a document. If it is a small technical decision that
can be made in a merge request by an author and a maintainer, you probably do
not need to document it here. The impact a technical decision will have is
another helpful information - if a technical decision is very impactful,
documenting it, along with associated implementation details, is advisable.

If it's helpful to include workflow diagrams or any other related images.
Diagrams authored in GitLab flavored markdown are preferred. In cases where
that is not feasible, images should be placed under `images/` in the same
directory as the `index.md` for the proposal.
-->

### Architecture Overview

The Artifact Registry is implemented as a separate service.

Deployment is Kubernetes-only through [Runway GKE](https://docs.runway.gitlab.com/runtimes/kubernetes/getting-started/).

Key architectural decisions:

- TBD

<!-- - **Implementation**: Rails monolith integration for development velocity and operational simplicity (see [ADR-005](decisions/005_implementation_architecture.md))
- **Technology Stack**: PostgreSQL, Redis, Sidekiq, Object Storage, ClickHouse for analytics (see [ADR-006](decisions/006_technology_stack.md))
- **Storage**: Content-addressable with organization scoped deduplication (see [ADR-008](decisions/008_content_addressable_storage.md))
- **Database**: Format-specific tables avoiding centralized table limitations (see [ADR-007](decisions/007_database_schema.md))
- **APIs**: Management APIs (REST/GraphQL) and format-specific client APIs (OCI, Maven, npm) (see [ADR-009](decisions/009_api_design.md))-->

### Scalability Requirements

The registry targets GitLab.com scale and large enterprise organizations. For detailed requirements, see [ADR-003: System Requirements](decisions/003_system_requirements.md).

### Phased Implementation

Implementation follows three phases, prioritized for customer adoption while building differentiation capabilities.

#### MVP (Phase 1)

**Goal**: Table stakes for customer adoption, covering ~30% of enterprise use cases with organization-level management.

**Artifact Storage & Management:**

- Multi-format support (Docker, Maven, npm)
- Content-addressable storage with organization-scoped deduplication
- Metadata management
- Basic artifact operations (upload, download, delete, tag)
- Version management
- GitLab CI metadata capture (platform captures pipeline, commit, and job context in artifact metadata)

**Repository Management:**

- Repository types (local/virtual)
- Typed repositories per format
- Organization repositories (no workspaces)
- Repository configuration

**Virtual Repositories:**

- Public upstream proxying (Maven Central, npmjs.com, Docker Hub)
- Private upstream proxying with credentials management
- Multiple upstreams support where public and private upstreams can be mixed in a priority-based list
- Shareable upstream configurations
- Connection testing and health monitoring
- Basic cache management with TTL

**Access Control:**

- Organization RBAC
- Authentication methods (personal access tokens, deploy tokens, CI/CD job tokens)
- Visibility controls (public, internal, private)

**Lifecycle Management:**

- Automated retention policies (basic)
- Expiration management
- Soft deletion with restoration capability
- Version cleanup
- Quarantine management

**Analytics & Observability:**

- Organization dashboard with unified visibility
- Download tracking
- Storage usage and trends
- Basic audit logging

**Integration:**

- RESTful API
- GraphQL API
- Native client support (Maven, npm, Docker)
- CI/CD integration/metadata

**Success Criteria for MVP:**

- Customers can publish and retrieve Docker, Maven, and npm artifacts through organization repositories
- Virtual repositories successfully proxy and cache artifacts from public upstream sources
- Organization administrators can configure lifecycle policies that apply across all repositories
- API performance meets or exceeds existing package/container registry benchmarks
- GitLab CI/CD pipelines can seamlessly publish artifacts with embedded metadata
- Early adopter customers validate the platform with production workloads

#### v1.0 and Beyond

**v1.0 Goal**: Expand format support and advanced features to reach ~60% of enterprise use cases.

Key v1.0 capabilities:

- Additional artifact formats (PyPI, NuGet, RubyGems, Go modules)
- Workspace support for organizational structure mapping
- Enhanced virtual repository features (cloud provider integration, multi-upstream aggregation)
- Advanced lifecycle policies with pattern matching and usage-based rules
- Enhanced analytics and cost attribution

**Success Criteria for v1.0:**

- Support for 7+ artifact formats with consistent management experience
- Workspace-based structure enables large enterprises to map their team hierarchy
- Customers successfully migrate from JFrog/Nexus using automated tooling
- AI-powered recommendations reduce configuration time by measurable percentage

For comprehensive capability prioritization including v1.0 (Phase 2) and Future (Phase 3+) features, see the [Capability Prioritization Matrix](https://gitlab.com/gitlab-org/ci-cd/package-stage/unified-artifact-management/-/blob/main/blueprint.md#capability-prioritization-matrix) in the extended blueprint (internal).

### Migration Strategy

Migration prioritizes service stability before introducing migration tools.

#### MVP Approach: Organic Migration

The MVP **excludes migration tools**. The registry will stabilize first, incentivizing organic adoption for new workflows:

- Service matures and establishes reliability
- Early adopters validate the platform with new projects
- Avoids premature migration tooling complexity
- Time to gather feedback and refine migration requirements

Users publish artifacts manually using native clients (npm, Maven, Docker). Virtual repositories facilitate gradual adoption by proxying upstream sources during migration.

#### Post-MVP: Migration Tools

After stabilization, migration capabilities accelerate adoption:

**From External Providers:**

- Bulk import from specific competitors
- Metadata extraction and preservation
- Dry-run validation before actual migration
- Progress tracking and error reporting
- Rollback capabilities
- Checksum verification and metadata completeness validation
- Dependency resolution validation and pre-flight compatibility checks
- Transitive dependencies support

**From GitLab Package and Container Registries:**

- Migration tooling to move artifacts from existing GitLab project-level registries to the unified registry
- Seamless transition path for GitLab customers

For migration timeline and prioritization, see the [Capability Prioritization Matrix](https://gitlab.com/gitlab-org/ci-cd/package-stage/unified-artifact-management/-/blob/main/blueprint.md#capability-prioritization-matrix) (internal).

## Team Dependencies

The Artifact Registry requires collaboration with several GitLab teams to ensure successful implementation:

TBD

<!--| Team | What's Needed | Output | Criticality |
|------|---------------|--------|-------------|
| **[Database Frameworks](/handbook/engineering/data-engineering/database-excellence/database-frameworks/)** | **Schema design**: Review schema proposed in [ADR-007](decisions/007_database_schema.md)-format-specific tables vs. cross-format operations, deduplication logic, blob reference tracking, access and cleanup patterns. **Sharding**: Validate sharding key. **Performance**: Query patterns for key operations, cleanup tasks, storage attribution. **Scale**: Validate for GitLab.com scale (billions of records, TB-range metadata). **Partitioning/isolation**: Table partitioning strategy and logical database isolation (new database alongside `main` and `ci`). | ADR-007 revised, expanded, and approved. | **Critical** - Large number of new tables with complex deduplication, reference tracking, and cleanup logic. Schema must be correct from the start; large-scale refactoring would be extremely costly. |
| **[Platform Insights](/handbook/engineering/data-engineering/analytics/platform-insights/)** | **ClickHouse/DIP integration**: Event collection patterns for artifact operations, schema design for analytics tables, query optimization for cost tracking and usage reporting. **Event instrumentation**: Patterns for capturing and storing long-term event records for audit and feeding AI/ML features (recommendations, optimization). | New ADR with strategy for event collection and processing. | **High** - Core value proposition includes cost analytics, usage tracking, and AI-powered insights. Getting this right from the start is key. |
| **[Fulfillment:Utilization](/handbook/engineering/development/fulfillment/utilization/)** | **Billing integration**: Billing model, integration with CustomersDot for invoicing and payment processing. **Storage quotas**: Quota enforcement patterns, usage tracking per organization, integration with existing consumables management. **Cost attribution**: Mechanisms for tracking and reporting storage costs. **Usage notifications**: Alert mechanisms when approaching limits. | New ADR with strategy for consumption tracking and usage billing. | **High** - New paid SKU requiring monetization. Without billing integration the product cannot be sold. |
| **[Geo](/handbook/engineering/infrastructure-platforms/tenant-scale/geo/)** | **Replication validation**: Confirm existing Geo Self-Service Framework can replicate all relevant registry data for self-managed and Dedicated installations. **Gap analysis**: Identify any limitations or additional work needed beyond the framework. | Update blueprint to confirm full compatibility. Follow-up issues for any gaps. | **Medium** - Early validation prevents costly rework and ensures feature parity across all installation types. |-->

## Alternative Solutions

TBD
<!--
It might be a good idea to include a list of alternative solutions or paths considered, although it is not required. Include pros and cons for
each alternative solution/path.

"Do nothing" and its pros and cons could be included in the list too.
-->

## Links

- **Direction**: [Package Stage Direction](https://about.gitlab.com/direction/package/)
- **Top-level Epic**: [Artifact Registry](https://gitlab.com/groups/gitlab-org/-/epics/19844)
- **Original Proposal**: [Internal Proposal Issue](https://gitlab.com/gitlab-org/gitlab/-/issues/568349)
- **Extended Blueprint**: [Detailed Specification](https://gitlab.com/gitlab-org/ci-cd/package-stage/unified-artifact-management/-/blob/main/blueprint.md) (internal)
- **AI Vision**: [AI-Enhanced Artifact Management](https://unified-artifact-managment-965acd.gitlab.io)

## Decisions

Key architectural decisions are documented as Architecture Decision Records (ADRs).

{{< note >}}
Some ADRs include an **Open Questions** section capturing details not yet decided during the initial proposal. These do not block review and will be addressed in subsequent updates or new ADRs.
{{< /note >}}

1. [ADR-001: Organizations as Anchor Point](decisions/001_organizations_as_anchor_point.md) - Why the registry anchors to Organizations
2. [ADR-002: Storage Deduplication Scope](decisions/002_storage_deduplication_scope.md) - Deduplication scoped to Organizations rather than instance-wide
3. [ADR-003: System Requirements](decisions/003_system_requirements.md) - Infrastructure requirements and performance constraints
<!--
4. [ADR-005: Implementation Architecture](decisions/005_implementation_architecture.md) - Rails monolith vs satellite service decision
-->
1. [ADR-006: Technology Stack](decisions/006_technology_stack.md) - Technology choices based on requirements and architecture
<!--
6. [ADR-007: Database Schema](decisions/007_database_schema.md) - Data tables organization for the registry
7. [ADR-008: Content-Addressable Storage](decisions/008_content_addressable_storage.md) - SHA256-based identification for deduplication and integrity verification
8. [ADR-009: API Design](decisions/009_api_design.md) - API endpoints organization for the registry
-->
1. [ADR-010: Data Retention](decisions/010_data_retention.md) - Retention policies for artifacts, audit logs, and cached content
