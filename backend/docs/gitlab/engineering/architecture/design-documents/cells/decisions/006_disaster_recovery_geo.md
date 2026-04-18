---
owning-stage: "~devops::tenant services"
title: 'Cells ADR 006: Use Geo for Disaster Recovery (DEPRECATED)'
toc_hide: true
status: deprecated
---

{{< alert title="Deprecated" color="warning" >}}
This ADR has been deprecated and replaced by [ADR 024: Use Backup and Restore for Disaster Recovery](024_disaster_recovery_cells.md).

The decision to use Geo for disaster recovery has been superseded by a backup and restore strategy that better aligns with the Cells isolation model and provides more flexibility for regional recovery. See ADR 024 for the current approach.
{{< /alert >}}

## Context

We discussed whether we should use Geo for Disaster Recovery in [this issue](https://gitlab.com/gitlab-com/gl-infra/production-engineering/-/issues/25246).

## Decision

~~It was decided that for Cells, we will use Geo for Disaster Recovery.~~
~~This is the same approach we take for GitLab Dedicated.~~

**This decision has been reversed.** The current approach uses backup and restore as documented in [ADR 024](024_disaster_recovery_cells.md).

## Consequences

~~This decision means that it will increase the initial cloud spend for Cells.~~
~~We estimate that it will double the spend of our first Cells deployments, which will be limited in number for the first Cells deployment.~~

These consequences no longer apply. See [ADR 024](024_disaster_recovery_cells.md) for the consequences of the backup and restore approach.

## Alternatives

~~The alternatives we discussed was to come up with a new process specific to Dedicated tooling for restoring from backup.~~

The backup and restore approach (originally considered as an alternative) has now become the primary decision. See:

- [ADR 024: Use Backup and Restore for Disaster Recovery](024_disaster_recovery_cells.md) for the overall DR strategy
- [ADR 013: Cell Restore from Backup](013_cell_restore_from_backup.md) for cell restoration procedures
- [ADR 020: Spanner Backup Strategy](020_spanner_backup_strategy.md) for Topology Service backup details
