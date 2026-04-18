---
title: "Artifact Registry ADR 001: Organizations as Anchor Point"
owning-stage: "~devops::package"
description: "Decision to anchor the Artifact Registry to Organizations"
toc_hide: true
---

<!-- Design Documents often contain forward-looking statements -->
<!-- vale gitlab.FutureTense = NO -->

## Context

The Artifact Registry requires a GitLab entity to serve as its anchor point - the primary boundary for registry instances, storage, cost attribution, and access control. This foundational decision affects every other architectural choice in the system.

Two candidates exist:

1. **[Organizations](https://docs.gitlab.com/user/organization/)**: A new GitLab entity designed as the top-level container for groups and projects, intended to replace top-level groups as the primary organizational boundary
2. **Top-level groups**: The highest-level group entity currently available in GitLab, serving as the root of the group hierarchy

Organizations align perfectly with the enterprise use case: a single entity representing an entire company or business unit, containing all its groups, projects, and now artifacts.

### Organizations Availability

Organizations are technically available but not yet GA. After consultation with the [Organizations team](/handbook/engineering/infrastructure-platforms/tenant-scale/organizations/), we have updated timeline information:

1. **Availability**: Organizations are technically available across GitLab installation types
2. **GA timeline**: Organizations are expected to launch within a year with a customer opt-in approach
3. **Strategic direction**: Organizations represent GitLab's long-term direction for enterprise organizational boundaries

Building for Organizations from the start provides significant advantages:

1. **No future migration burden**: Customers won't need to restructure from top-level groups to Organizations later
2. **Alignment with GitLab direction**: The Artifact Registry will be ready as Organizations adoption grows
3. **Conceptual correctness**: Organizations represent the natural boundary for enterprise artifact management

## Decision

**The Artifact Registry will anchor exclusively to Organizations.**

The feature will only be available at the Organization level - not at both Organization and top-level group levels. This avoids the complexity of maintaining dual implementations and ensures a single, clear architectural path.

This means:

- All repositories, artifacts, lifecycle policies, and access controls belong to an Organization
- Storage and cost attribution are calculated at the Organization boundary
- Deduplication is scoped to the Organization (see [ADR-002](002_storage_deduplication_scope.md))
- Organizations serve as the isolation and sharding boundary

## Consequences

### Positive

1. **Aligned with GitLab's target organizational model**: Building for Organizations from day one means no architectural pivots later
2. **No future migration burden**: Customers adopt the target architecture immediately, avoiding disruptive migrations
3. **Conceptually correct anchor point**: Organizations represent entire companies or business units - the natural boundary for enterprise artifact management
4. **Clean enterprise mapping**: One Organization equals one artifact registry instance, simplifying customer mental models
5. **Forward-compatible with Cells architecture**: Organizations are the intended sharding boundary for Cells, ensuring alignment
6. **Improved storage deduplication**: Organizations sit above top-level groups in the hierarchy. Identical blobs across multiple top-level groups within the same Organization are deduplicated, providing greater storage efficiency compared to top-level-group-scoped deduplication

### Negative

1. **Customer opt-in required**: During the Organizations rollout period, customers must explicitly enable Organizations
2. **Dependency on Organizations timeline**: Registry availability coordinates with Organizations readiness across installation types

## Implications by Installation Type

### GitLab.com (SaaS)

Customers create or use an Organization for their artifact management. This aligns with GitLab.com's direction toward Organizations as the primary organizational boundary.

### Self-Managed and Dedicated

Organizations will be available across all installation types. Customers enable Organizations to use the Artifact Registry. This provides a consistent experience across all GitLab deployment models.

## Blob Storage Deduplication

Anchoring to Organizations has direct implications for storage deduplication (see [ADR-002](002_storage_deduplication_scope.md) for detailed deduplication design).

Organizations sit above top-level groups in the GitLab hierarchy. By scoping deduplication to Organizations rather than top-level groups:

- **Broader deduplication scope**: Identical blobs across multiple top-level groups within the same Organization are stored only once
- **Greater storage efficiency**: Common artifacts (base images, shared libraries, public packages) used across different top-level groups benefit from deduplication
- **Improved cost attribution**: Storage costs are calculated at the Organization level, providing clearer billing boundaries

## Alternatives Considered

### Alternative: Top-Level Groups

#### Approach

Anchor the registry to top-level groups - the highest-level group entity currently available - with a plan to migrate to Organizations once they become stable and widely adopted.

**Note**: Top-level groups were considered during early planning when Organizations timeline had more uncertainty. The updated Organizations timeline and customer opt-in approach make building for the target architecture viable.

#### Why Not Chosen

1. **Future migration burden**: Customers would need to migrate from top-level groups to Organizations later, requiring careful planning and execution
2. **Conceptual mismatch**: Top-level groups are hierarchical containers, while the artifact registry conceptually serves an entire organization (which may span multiple top-level groups)
3. **Organizations timeline alignment**: Organizations are expected to launch within the Artifact Registry project timeline, making an interim approach unnecessary
4. **Avoiding precedent pitfalls**: Other features that chose top-level groups (Security Dashboard, Compliance Center, Value Stream Analytics) now face migration considerations; building for Organizations from the start avoids this pattern
5. **Reduced deduplication efficiency**: Top-level group scoped deduplication misses opportunities to deduplicate identical blobs across top-level groups within the same Organization
6. **Complex blob migration**: Migrating from top-level groups to Organizations would require consolidating blobs across top-level groups, adding significant migration complexity
7. **Consolidation not feasible**: Not all customers with multiple top-level groups (for example, GitLab itself uses `gitlab-org`, `gitlab-com`, etc.) can consolidate into a single group. Creating a separate top-level group exclusively for artifacts was considered but rejected due to the complexity of maintaining synchronized permissions across groups.

## References

- [Organizations Development Documentation](https://docs.gitlab.com/ee/development/organization/)
- [Organizations Team Handbook](/handbook/engineering/infrastructure-platforms/tenant-scale/organizations/)
- [Cells Design Document](../../cells/)
- [ADR-002: Storage Deduplication Scope](002_storage_deduplication_scope.md) - Deduplication boundary
<!-- - [ADR-007: Database Schema](007_database_schema.md) -->
