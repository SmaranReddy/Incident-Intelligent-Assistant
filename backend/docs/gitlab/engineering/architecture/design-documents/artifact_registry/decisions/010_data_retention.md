---
title: "Artifact Registry ADR 010: Data Retention"
owning-stage: "~devops::package"
description: "Decision on data retention policies for artifacts, audit logs, and cached content"
toc_hide: true
---

<!-- Design Documents often contain forward-looking statements -->
<!-- vale gitlab.FutureTense = NO -->

## Context

The Artifact Registry manages data through two complementary mechanisms:

1. **Retention**: Time-based expiration that sets the maximum lifespan for artifacts
2. **Cleanup**: Active processes that remove data (lifecycle policies, garbage collection)

These mechanisms operate independently. Retention defines when artifacts become eligible for deletion; cleanup executes the deletion.

| Mechanism          | Purpose                            | Scope                    | Configurable |
|--------------------|------------------------------------|--------------------------|--------------|
| Default retention  | Maximum artifact lifespan          | Organization-wide floor  | Yes          |
| Lifecycle policies | User-defined cleanup rules         | Repository or Org level  | Yes          |
| Garbage collection | Infrastructure cleanup             | Organization-wide        | No           |
| Soft delete        | Recovery window before hard delete | Organization-wide        | No           |

### Deletion Precedence

When multiple mechanisms apply to the same artifact, the following precedence determines behavior:

| Deletion trigger     | Soft delete applies? | Rationale                                                                 |
|----------------------|----------------------|---------------------------------------------------------------------------|
| Manual deletion      | Yes                  | User-initiated; recoverable within the soft delete window                 |
| Lifecycle policy     | Yes                  | Policy-driven; recoverable within the soft delete window                  |
| Retention expiration | Yes                  | Protects against misconfiguration; garbage collection handles hard delete |

All deletion paths go through soft delete. This provides a consistent recovery window regardless of the deletion trigger, protecting against bugs or misconfigured retention periods. Garbage collection permanently removes soft-deleted artifacts after the soft delete window expires.

## Artifact Retention

Retention defines the maximum time an artifact can exist without being accessed.

### Default Retention Period

Artifacts unused for 365 days are automatically deleted. This default prevents unbounded storage growth, while the extension mechanism (below) accommodates customers with longer retention needs. This Organization-level default:

- Applies to all artifacts regardless of tags or lifecycle policies
- Acts as a floor: lifecycle policies can delete sooner, but cannot extend retention beyond this limit
- Resets when an artifact is downloaded

Organization administrators can adjust the default period. The 365-day value applies unless explicitly changed.

### Extending Retention

By default, all repositories must comply with the Organization's retention policy through strict enforcement. Organization Owners can optionally enable:

- Repository-level exceptions: Allow repositories to extend retention beyond the Organization default
- Indefinite retention: Allow critical artifacts to be retained indefinitely

These mechanisms are designed for regulated industries where multi-year retention is required (for example, SOX or HIPAA).

## Lifecycle Policies

Lifecycle policies are user-defined rules that actively delete artifacts matching specific criteria. They execute more aggressive cleanup than the default retention period:

- Lifecycle policies can delete artifacts before the retention period expires
- Retention deletes artifacts that lifecycle policies miss or ignore
- Neither mechanism can resurrect destroyed artifacts

### Supported Strategies

| Strategy                | Description                             | Example                                    |
|-------------------------|-----------------------------------------|--------------------------------------------|
| Keep last N versions    | Retain the N most recent versions       | Keep last 10 versions                      |
| Age since last download | Delete artifacts unused within a period | Delete if unused for 90 days               |
| Pattern matching        | Retain artifacts matching a pattern     | Keep `release-*` names; keep `v*` versions |

Lifecycle policies use **keep semantics only**: patterns define what to retain, and everything not matched is eligible for deletion. This avoids the complexity of conflicting keep/expire rules, a [lesson learned](https://docs.gitlab.com/user/packages/container_registry/reduce_container_registry_storage/#how-the-cleanup-policy-works) from container registry cleanup policies.

### Artifact Targeting

Pattern matching uses a consistent targeting model shared with access rules and virtual repository filters. Each policy rule specifies a **pattern** and a **target field**. Available target fields vary by artifact type:

| Artifact type    | Target fields                            |
|------------------|------------------------------------------|
| npm              | Scope, package name, version             |
| Maven            | `groupId`, `artifactId`, version         |
| Container images | Image name, tag                          |
| Generic          | Package name, version                    |

Policies can combine multiple target fields in a single rule (for example, keep artifacts where name matches `release-*` and version matches `v*`). Using the same targeting model across lifecycle policies, access rules, and virtual repository allow/deny filters enables implementation reuse and a consistent user experience.

## Soft Delete

Soft delete is always enabled and not user-configurable. All deletions (manual, lifecycle policy, retention expiration) mark artifacts for deletion rather than removing them immediately. Users can restore soft-deleted artifacts until the hard deletion window expires. This is a [lesson learned](https://gitlab.com/gitlab-org/gitlab/-/work_items/561680) from GitLab's project delayed deletion, where making soft delete optional led to accidental permanent data loss.

Whether soft-deleted artifacts are visible in the UI is a display concern, not a retention concern.

## Notifications

The Artifact Registry notifies users when artifacts approach deletion:

| Trigger                | Description                                                  |
|------------------------|--------------------------------------------------------------|
| Retention expiration   | Artifacts approaching the retention limit                    |
| Lifecycle policy match | Artifacts matching a lifecycle policy scheduled for deletion |
| Soft delete expiry     | Soft-deleted artifacts approaching permanent removal         |

A minimum advance notice of 30 days before retention expiration should be enforced to give users time to download or extend retention on critical artifacts. Exact notification channels and display locations to be defined during UX design.

## Virtual Repository Caches

Virtual repositories cache packages from upstream registries. Cached content has separate retention and cleanup rules from local artifacts.

### Cache Validity

Each upstream has a configurable validity period that determines when cached metadata needs refreshing. Default: 24 hours. This controls staleness, not deletion.

Some upstreams linked to registries known to have immutable artifacts will automatically disable the cache validity setting unless specified by the user.

### Cache Retention

Cached artifacts unused for 7 days are automatically deleted. This is more aggressive than local artifact retention because cached content can be re-fetched from upstream.

| Setting          | Options                        | Default |
|------------------|--------------------------------|---------|
| Retention period | 1-365 days since last download | 7 days  |
| Cleanup cadence  | Daily, weekly, monthly         | Daily   |

Cache retention is configured at the Organization level and enabled by default.

## Audit Log Retention

Audit logs capture security-relevant operations:

| Log Category                                                    | Default Retention | Rationale               |
|-----------------------------------------------------------------|-------------------|-------------------------|
| Security events (auth failures, permission changes)             | 365 days          | Matches GitLab's [Records Retention & Disposal](/handbook/security/policies_and_standards/records-retention-deletion/) policy for production audit logs (1 year) |
| Administrative operations (policy changes, repository creation) | 365 days          | Matches production audit log retention for consistent change tracking |
| Artifact operations (upload, download, delete)                  | 60 days           | Most operational issues surface within this window; Meets the 60-day minimum for critical system activity logs |

Deletion audit entries must record the trigger (manual, lifecycle policy, retention expiration, or garbage collection) to support debugging and compliance investigations.

Background jobs purge logs exceeding retention periods.

## Garbage Collection

Garbage collection removes orphaned data within an Organization's deduplicated storage pool (see [ADR-002](002_storage_deduplication_scope.md)). Unlike retention and lifecycle policies, garbage collection is not configurable.

### Blob Cleanup

Each Organization has its own blob storage pool. A blob is deleted only when no artifacts within that Organization reference it:

1. Deleting an artifact removes its attachment record
2. A background job finds blobs with zero attachments within the Organization
3. The job deletes both the blob record and the physical file

Orphaned blobs are deleted within 24 hours. This delay accommodates transaction rollbacks, concurrent uploads, and batch operations.

### Soft Delete Expiry

Background jobs permanently remove soft-deleted artifacts after 30 days. This window balances recovery time with storage cost.

### Secure Disposal

All permanent deletions (blob cleanup, soft delete expiry, retention expiration) rely on the storage provider's secure deletion mechanisms, in accordance with the [Records Retention & Disposal](/handbook/security/policies_and_standards/records-retention-deletion/) policy. For object storage, this means the physical file is deleted through the provider's API (for example, GCS or S3 object deletion). Database records are hard-deleted from PostgreSQL. Audit log entries for the deletion itself are retained per the audit log retention periods defined in this ADR.

## Backup Retention

Backup retention for the Artifact Registry's PostgreSQL database and object storage is a platform responsibility and follows the deployment's standard backup policies:

- **GitLab.com (SaaS)**: Follows GitLab's existing database and object storage backup schedules
- **Dedicated/Self-Managed**: Backup retention is configured by deployment administrators

Backups may contain artifacts that have already been deleted by retention policies, lifecycle policies, or garbage collection. Restoring from backup may reintroduce previously deleted artifacts. This is expected behavior and the normal retention/cleanup cycle will re-process them.

## Storage Class Transitions

Storage class management is a **platform responsibility**. The Artifact Registry does not manipulate storage classes; deployment administrators configure transitions on their storage provider when supported. The service is able to see and operate with artifacts regardless of their storage class.

Storage class transitions are independent of retention and cleanup:

- Artifacts in Archive storage are still subject to retention limits and lifecycle policies
- Soft-deleted artifacts retain their storage class until garbage collection
- Virtual repository caches follow the same transition schedule

### Transition Schedule

| Days Since Last Download | Storage Class | Use Case                    |
|--------------------------|---------------|-----------------------------|
| 0-30                     | Standard      | Frequently accessed         |
| 31-90                    | Nearline      | Monthly access patterns     |
| 91-365                   | Coldline      | Quarterly access patterns   |
| 366+                     | Archive       | Rarely accessed, compliance |

Objects transition to Standard storage immediately when accessed, regardless of current storage class.

### Platform Configuration

Deployment administrators enable storage class transitions at the infrastructure level:

| Provider | Mechanism                                                                                             |
|----------|-------------------------------------------------------------------------------------------------------|
| GCS      | [Autoclass](https://cloud.google.com/storage/docs/autoclass) with Archive as terminal class           |
| AWS S3   | [Intelligent-Tiering](https://aws.amazon.com/s3/storage-classes/intelligent-tiering/)                 |
| Azure    | [Access tier automation](https://learn.microsoft.com/en-us/azure/storage/blobs/access-tiers-overview) |

For **GitLab.com**, GCS Autoclass provides:

- Automatic transitions based on access patterns
- No retrieval fees (included in Autoclass pricing)
- No early deletion fees

## Cross-Cutting Concerns

### Multi-Region Replication

When artifacts are deleted by retention policies, lifecycle policies, or garbage collection, the deletion must propagate to all replicas. The replication mechanism (whether multi-region replication or another approach, see [multi-regionality direction](https://gitlab.com/gitlab-org/gitlab/-/work_items/590300)) must ensure that deleted blobs and metadata are removed from all replicas to avoid storage drift and stale data.

### Export and Import

GitLab project export/import may reference artifacts stored in the Artifact Registry. If artifacts have been deleted by retention or lifecycle policies before an export, the export should handle missing artifacts gracefully (for example, by excluding them and logging the omission rather than failing).

### Dependent Features

CI pipelines, merge requests, and other GitLab features may reference artifacts managed by the Artifact Registry. When retention or lifecycle policies delete these artifacts, the **service or application that holds the reference** is responsible for handling the missing artifact gracefully:

- Links to deleted artifacts should display clear messaging rather than generic errors
- API responses for deleted artifacts should return appropriate status codes
- Soft-deleted artifacts should remain resolvable (with a "scheduled for deletion" indicator) until hard deletion

The Artifact Registry does not proactively notify dependent features when an artifact is deleted. Dependent services resolve references at query time and handle missing artifacts accordingly. An event-driven approach (e.g., publishing deletion events for interested subscribers) is a potential future improvement but is out of scope for this ADR.

## Consequences

### Positive

- **Predictable costs**: Default expiration prevents unbounded storage growth
- **Automatic cost optimization**: Storage class transitions reduce costs for aging artifacts without manual intervention
- **Flexibility**: Organization and repository overrides accommodate diverse needs
- **Recovery capability**: Soft delete protects against accidental deletion
- **Compliance support**: Configurable retention meets regulatory requirements
- **Cache efficiency**: TTL-based management balances freshness with storage

### Negative

- **Onboarding friction**: Users must understand default expiration to avoid surprise deletions
- **Complexity**: Multiple retention levels (Organization, repository, rule) require clear documentation
- **Background load**: Cleanup jobs consume additional resources

## Alternatives Considered

### Require Explicit Policy Before Publishing

Users must configure a lifecycle policy before publishing artifacts.

**Trade-offs**: Prevents accidental storage growth but increases onboarding friction and may discourage adoption.

### Indefinite Retention by Default

Artifacts persist forever unless explicitly deleted.

**Why rejected**: Leads to unbounded storage growth and cost surprises. Most users forget to configure cleanup policies, resulting in accumulated stale artifacts.

## References

- [ADR-001: Organizations as Anchor Point](001_organizations_as_anchor_point.md)
<!-- - [ADR-006: Technology Stack](006_technology_stack.md) - ClickHouse for audit logs -->
<!-- - [ADR-007: Database Schema](007_database_schema.md) - Lifecycle policies and blob cleanup -->
- [GitLab Audit Events Documentation](https://docs.gitlab.com/ee/administration/audit_event_reports.html)
- [GCS Autoclass](https://cloud.google.com/storage/docs/autoclass) - Automatic storage class transitions
- [GCS Storage Classes](https://cloud.google.com/storage/docs/storage-classes) - Standard, Nearline, Coldline, Archive
- [Records Retention & Disposal](/handbook/security/policies_and_standards/records-retention-deletion/) - GitLab security policy for records retention and secure disposal
- [Data Retention Guidelines for Feature Development](/handbook/engineering/architecture/guidelines/data_lifecycle/data_retention/) - Architecture guidelines for data retention in new features
- [Competitive analysis (internal)](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/18459#note_3089332399)
