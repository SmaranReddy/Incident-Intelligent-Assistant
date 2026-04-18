---
title: 'GitLab Secrets Manager ADR 013: OpenBao Separate Logical Database'
owning-stage: "~sec::software supply chain security"
toc_hide: true
---

## Context

When deploying GitLab Secrets Manager with OpenBao, a critical architectural decision must be made regarding where OpenBao stores its data. OpenBao requires a PostgreSQL database to persist its state, including:

- Key-value secrets stored in the KV engine
- Authentication and authorization policies
- High availability (HA) locks for cluster coordination

Two primary options exist:

1. **Shared Database**: OpenBao shares the same PostgreSQL database as GitLab Rails (the `main` database)
2. **Separate Logical Database**: OpenBao uses a separate logical database on the same PostgreSQL server as GitLab Rails

GitLab Secrets Manager is currently as a closed beta,
and OpenBao uses the `main` database.
However, this can be reverted,
and a final decision needs to be made before the feature becomes
Generally Available.

This decision doesn't apply to GitLab.com.
OpenBao is deployed to that platform using Runway,
and it is connected to a dedicated Google Cloud SQL database.

## Shared Database (Rails `main`)

### Pros

- **Simplified deployment**: Single database connection string and configuration
- **Easier backup/restore**: All data backed up together using existing GitLab backup procedures
- **Lower operational burden**: Customers don't need to manage separate database credentials or connections
- **Reduced adoption friction**: Particularly beneficial for on-premises customers without managed database solutions
- **Easy path to GA**: Customers joining the closed beta today don't need to migrate their deployment.

Additionally, this option is already in place and documened, and it requires no additional development.

### Cons

- **Data isolation concerns**: OpenBao secrets (tier-0 data) are stored alongside application data, reducing security separation, and separation of concerns. Note: OpenBao creates and manages its own tables independently of Rails migrations.
- **Accidental data loss**: Rails `db:drop_tables` rake task could inadvertently affect OpenBao tables. **However**, OpenBao data can't be used once the corresponding groups and projects have been removed.

Additionally, sharing the database prevented Rake tasks from seeding the `main` database. However, this has been mitigated as part of issue [#582640](https://gitlab.com/gitlab-org/gitlab/-/work_items/582640).

## Separate Logical Database

### Pros

- **Data isolation**: OpenBao secrets are isolated from Rails application data, improving security posture
- **Cleaner separation of concerns**: OpenBao manages its own database and schema. OpenBao maintains full control over its table creation and migrations.
- **Safer operations**: Rails rake tasks cannot accidentally affect OpenBao tables
- **Future scalability**: The OpenBao database can grow independently of the `main` database.

### Cons

- **Adoption friction**: Requires additional setup steps, particularly for on-premises customers without managed database solutions
- **Incomplete tooling**: The backup tool need to be changed to handle the OpenBao database, otherwise customers must backup/restore OpenBao separately (using the backup tool)

Note: We have communicated to beta customers that the feature is not ready for production, and that data won't be persisted.
As a result, we don't need to provide tooling to migrate OpenBao data from the `main` database to a separate database.

## Decision

OpenBao data will be stored in a **separate logical database** dedicated to OpenBao, rather than sharing the GitLab Rails `main` database.

This decision prioritizes:

1. **Security**: Tier-0 secrets data is isolated from application data
2. **Operational safety**: Prevents accidental data loss or conflicts with Rails operations
3. **Architectural clarity**: Each service owns and manages its own data store

While a shared database offers operational simplicity, the security and architectural benefits of separation outweigh the additional operational burden.

## Implementation Notes

### For Self-Managed Deployments

- OpenBao uses a separate logical database on the same PostgreSQL server as the Rails `main` database
- Database creation and user provisioning must be handled separately from GitLab's standard setup
- Backup and restore procedures must account for both databases
- Geo replicates all logical databases, including the OpenBao database.

The following work is planned:

1. **Backup/Restore Enhancement**: Extend GitLab's backup and restore rake tasks to handle multiple databases
2. **Documentation**: Provide clear guidance on database setup and backup/restore
3. **Testing:** Test backup procedures provided in admin documentation.
4. **Changing default behavior:** GitLab chart changes the OpenBao configuration to not default to the `main` DB anymore. 
5. **Determine** the conditions under which a separate physical database instance is required to prevent noisy neighbor resource contention

### For GitLab.com

GitLab Secrets Manager already uses a separate database on GitLab.com.

- OpenBao uses a dedicated Cloud SQL instance
- Complete isolation from Rails infrastructure
- Separate backup, restore, and replication procedures

## References

- [Work Item #582640: DB not seeded when GitLab chart installed with OpenBao enabled](https://gitlab.com/gitlab-org/gitlab/-/work_items/582640)
