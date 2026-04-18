---
title: "Artifact Registry ADR 002: Storage Deduplication Scope"
owning-stage: "~devops::package"
description: "Decision to scope storage deduplication to individual organizations rather than instance-wide"
toc_hide: true
---

<!-- Design Documents often contain forward-looking statements -->
<!-- vale gitlab.FutureTense = NO -->

## Context

Content-addressable storage <!-- (see [ADR-008](008_content_addressable_storage.md)) --> deduplicates identical content. We must choose the deduplication level: instance-wide, organization, top-level group, or repository.

This choice affects:

- **Cost attribution**: How easily we calculate and bill storage at the selected level
- **Performance**: Garbage collection and query complexity
- **Security**: Whether shared blobs leak information across boundaries
- **Operations**: Disaster recovery, backups, and cross-boundary coordination

### Current State

The container registry uses instance-wide deduplication and suffers [operational problems](https://gitlab.com/gitlab-org/container-registry/-/issues/1242):

- Cross-partition queries for cost calculations
- Garbage collection blocked by cross-namespace dependencies
- Storage costs difficult to attribute to specific organizations

The Package registry has no deduplication.

### Analysis

We evaluated three boundaries: instance-wide, top-level group, and repository.

Note: this analysis was carried out when the top-level group was the considered level for object storage deduplication. Since then, the selected [anchor level is organizations](001_organizations_as_anchor_point.md). Therefore, deduplication will be scoped to organizations.

**Container Registry** ([analysis](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/17524#note_3023542021)):

| Deduplication scope | Total storage | Overhead vs instance-wide |
| ------------------- | ------------- | ------------------------- |
| Instance-wide | 13 PB | baseline |
| Top-level group | 13.5 PB | +4% (+530 TB) |
| Repository | ~17 PB | +36% (~4 PB) |

At least ~95% of blobs appear in exactly one top-level group. Only ~5% span multiple groups. Top-level group deduplication adds only 4% storage overhead compared to instance-wide, while repository-level loses significant benefit (+36%).

**Maven Package Registry** ([analysis](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/17524#note_3023014429)):

| Deduplication scope | Storage savings (deduplication benefit) |
| ------------------- | --------------------------------------- |
| Instance-wide | 3.68% |
| Top-level group | 3.62% |
| Repository | 2.11% |

Top-level group deduplication captures nearly the same benefit as instance-wide (3.62% vs 3.68%). The difference is only 0.06 percentage points. Repository-level loses almost half the benefit (2.11% vs 3.68%).

**Maven Virtual Registry** ([analysis](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/17524#note_3027109326)):

| Metric | Value |
| ------ | ----- |
| Cache entries analyzed | 161,832 |
| Cache storage | ~44 GB |
| Duplicated with Package Registry (same group) | ~8-9% |

~92% of Virtual Registry cache entries are external dependencies not found in the same group's Package Registry. This confirms Virtual Registries primarily cache upstream content rather than content already stored locally. The Maven Virtual Registry is in beta with limited adoption, so these figures have lower confidence.

**Conclusion**: As we can see, the top-level group scope captures nearly all deduplication benefit (96%+ for containers, 98%+ for Maven) while avoiding cross-group complexity. The small additional storage overhead is an acceptable tradeoff for simpler operations, clearer cost attribution, and security isolation. Organization-level scope is closer to instance-wide, which provides the most storage-efficient deduplication.

## Decision

**Scope deduplication to individual organizations.**

Identical content (same SHA256 hash) within an organization stores once. Identical content across different organizations stores separately in each.

This applies to all artifact types (Docker images, Maven packages, npm modules) and all content (container layers, package files, binary blobs).

## Consequences

### Positive

1. **Clear cost attribution**: Sum blob sizes per organization. No cross-level calculations.
2. **Fair customer billing**: Customers pay once per unique blob.
3. **Predictable performance**: Operations (queries, GC, backups) stay within one organization. No cross-level dependencies.
4. **Security isolation**: Organizations cannot reference other organizations' blobs. No information leakage.
5. **Simpler garbage collection**: GC checks references within one organization only. No cross-level coordination.
6. **Self-contained disaster recovery**: Restore one organization without touching others.
7. **Independent scaling**: Each organization scales storage independently.
8. **Best deduplication level across platforms**: GitLab.com will use organizations. Dedicated and self-managed, since they will operate under a single organization, will effectively have instance-level deduplication.

### Negative

1. **Cross-organization duplication**: Popular content (base images, public packages) stores separately in each organization.
2. **Higher total instance storage**: Organization scope adds overhead vs instance-wide. This does *not* translate to an extra net expense as storage is billed to customers.
3. **Broader scope than repository-level**: Storage usage must track deduplicated blobs across repositories. GC must operate across all repositories, handling concurrent operations on shared blobs.

## Alternatives

### Alternative 1: Instance-Wide Deduplication

Store identical content once per instance-wide scope, regardless of which organization uploads it.

**Pros:**

- Minimum overall storage cost
- Fewer physical objects in object storage

**Cons:**

- **Cost attribution requires cross-organization algorithms**: Reference counting across all organizations to calculate each organization's usage
- **GC requires cross-organization coordination**: Deleting content forces checking all organizations for remaining references
- **Unbounded query scope**: Operations may cascade across partitions unpredictably
- **Information leakage risk**: Shared references reveal what content other organizations use
- **Disaster recovery couples organizations**: Restoring one organization may require content from others
- **First uploader subsidizes others**: Organizations uploading popular content first pay for everyone

**Why rejected:**

- Analysis shows only low single-digit savings over top-level group scope (container ~4%; Maven is much smaller)
- Container registry's instance-wide approach proved [operationally expensive](https://gitlab.com/gitlab-org/container-registry/-/issues/1242)
- Security risk from cross-organization blob sharing

### Alternative 2: Top-Level Group Deduplication

**Pros:**

- Reasonable overhead vs instance-level.
- Already a well-established entity in GitLab.

**Cons:**

- **Storage increase** over organization scope
- **Roll-up usage metrics** required for the organization level
- **Doesn't map naturally** to the [anchor level](001_organizations_as_anchor_point.md) of the feature, which is organization

**Why rejected:**

- Potential savings loss compared to organization-level scope
- Cognitive complexity from mixing deduplication scope with other features that operate at the organization level
- Roll-up metrics pose a challenge at scale. [Past experience with namespace statistics](https://gitlab.com/groups/gitlab-org/-/work_items/8627)

### Alternative 3: Repository-Scoped Deduplication

Deduplicate only within a single repository.

**Pros:**

- Strongest isolation (no cross-repository references)
- Simplest GC and cost attribution

**Cons:**

- **~36% storage increase** over top-level group scope
- **Unfair customer billing**: Customers charged multiple times for identical content across repositories

**Why rejected:**

- Analysis shows substantial savings loss at repository scope
- Billing customers repeatedly for the same blob is likely unacceptable
- Organization scope provides better cost-benefit balance

## Implementation Notes

1. Content-addressable storage with SHA256 hashing <!-- (see [ADR-008](008_content_addressable_storage.md)) -->
2. Reference tracking scoped to individual organizations
3. Cost calculation: sum blob sizes per organization

## References

- [Container Registry Deduplication Analysis](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/17524#note_3023542021)
- [Maven Package Registry Deduplication Analysis](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/17524#note_3023014429)
- [Maven Virtual Registry Cache Analysis](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/17524#note_3027109326)
<!-- - [ADR-008: Content-Addressable Storage](008_content_addressable_storage.md) -->
- [Container Registry Deduplication Complexity](https://gitlab.com/gitlab-org/container-registry/-/issues/1242)
