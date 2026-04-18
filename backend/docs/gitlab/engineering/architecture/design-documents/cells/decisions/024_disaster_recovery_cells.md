---
owning-stage: "~devops::tenant services" # because Tenant Scale is under this
title: 'Cells ADR 024: Use Backup and Restore for Disaster Recovery'
toc_hide: true
---

## Context

This ADR states the use of a backup and restore strategy as Cells disaster recovery model.

Today, we regularly back up cell data and store those backups in AWS Vault. As discussed in
[this issue](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/work_items/483),
these backups are intended to be the primary mechanism for Disaster Recovery (DR) in a Cells-based architecture.

## Decision

We will use **backup and restore as the primary Disaster Recovery mechanism for Cells**.

Each cell is responsible for producing consistent backups of its stateful components (for example databases, object storage, and configuration state). In a disaster scenario,
a new cell can be provisioned potentially in the same or different AWS region and restored from the most recent valid backup.

This approach implies:

- Backups are treated as first-class recovery artifacts.
- Restore workflows must be automated, repeatable, and regularly exercised.
- Recovery is performed by creating a new cell rather than attempting in-place repair of a failed one.

Backup data stored in AWS will be considered the source of truth for disaster recovery operations.

## Consequences

### Positive

- Strong alignment with the Cells isolation model: failures and recoveries are scoped to individual cells
- Reduced operational complexity/cost compared to global synchronous replication as we have with geo.
- Flexibility to restore into a new region or account, helping mitigate AWS quota exhaustion and regional outages.
- Clear and testable recovery procedures that can be validated through drills.

### Negative

- Unlike Geo, RTO will be bounded by tenant provision and restore time which will be longer than a Geo failover.
- Unlike Geo, RPO will be bounded by the time of the last successful backup which is configurable and will be longer than a Geo failover.
- Restore workflows must be reliable; failures in restore tooling directly impact RTO.

### Operational Implications

- Backup freshness and integrity must be continuously monitored
- Restore procedures must be documented and automated.
- Regular disaster recovery exercises are required to ensure backups are usable under real conditions
