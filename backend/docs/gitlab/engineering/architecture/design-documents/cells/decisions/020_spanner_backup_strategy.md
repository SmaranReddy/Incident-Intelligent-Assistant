---
title: "Cloud spanner backup strategy selection for Topology Service"
owning-stage: "~devops::tenant-scale"
group: cells-infrastructure
creation-date: "2025-05-09"
authors: ["@daveyleach"]
coach: "@sxuereb"
approvers: []
toc_hide: true
---

{{< engineering/design-document-header >}}

## Summary

This ADR documents the decision to implement a comprehensive backup strategy for the Cloud Spanner instance supporting the Topology Service, using **multi-region configuration**, **36-hour Point-in-Time Recovery (PITR)**, and **daily incremental backups with 90-day retention** to achieve robust disaster recovery capabilities while maintaining acceptable performance.

## Context

The Topology Service uses Cloud Spanner as its primary database for storing critical infrastructure metadata including:

- Cell configurations and metadata
- Sequence ranges for ID allocation
- Claim records for uniqueness enforcement (usernames, emails, routes)
- Classification data for routing requests

As documented in the [Topology Service design](../topology_service.md), this service is critical for Cells infrastructure operation, requiring robust backup and disaster recovery capabilities.

### Technical Constraints

- [Extending point in time recovery has performance implications](https://cloud.google.com/spanner/docs/pitr#performance)
- [Maximum window for point in time recovery is 7 days](https://cloud.google.com/spanner/docs/pitr)
- [The minimum interval is 12 hours for full backups and 4 hours for incremental backups.](https://cloud.google.com/spanner/docs/backup#backup-schedules)
- [Spanner permits up to 13 incremental backups per chain of backups](https://cloud.google.com/spanner/docs/backup#incremental-backups)
- [Full and incremental backups do not include point-in-time recovery capabilities](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/301#note_2679808248)

## Problem Statement

We need to determine the optimal backup strategy for our Topology Service Spanner instance that:

- Provides comprehensive protection against various failure scenarios
- Maintains compliance with GitLab's disaster recovery requirements. [GitLab.com current state](../../../../gitlab-com/policies/backup/), [Gitlab.com targets](../../disaster_recovery/), [Cells targets](../infrastructure/disaster_recovery.md)
- Minimizes performance impact on production workloads
- Balances backup persistence with operational flexibility
- Has restore validation performed on a regular basis

## Testing Methodology

### Performance Impact Testing

Google Cloud's official documentation warns that extending Spanner's point-in-time recovery retention beyond the default one-hour period may impact performance. However, our internal load testing contradicts this concern for our specific use case. When we tested our exact database configuration at production-equivalent scale, extending the retention period from 1 hour to 36 hours produced no detectable impact on either performance metrics or CPU utilization.

Load testing was conducted in [Issue #474](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/474) to evaluate PITR impact.

### Backup Recovery Testing

Testing performed in [project spanner_backups](https://gitlab.com/daveyleach/spanner_backups) validated:

- Backup creation and restoration procedures
- PITR capabilities and limitations
- Recovery time objectives
- Data consistency after recovery

## Results

### Performance Impact Analysis

Testing details in [Issue](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/474)

### Failure Scenario Coverage Analysis

Based on testing and analysis documented in [Spanner Failure Scenarios and Backup Protection](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/301#note_2691882252)

| Failure Scenario | Description | Multi-Region Config | Full/Incremental Backup | Point-in-Time Recovery (PITR) | Best Solution | RPO | RTO |
|-----------------|-------------|-------------------|-------------|------|---------------|-----|-----|
| **Logical Corruption (<1.5 days)** | Application bug corrupts data within PITR window | ❌ Unprotected | ✅ Protected | ✅ **Protected** | **PITR** - Restore to exact moment before corruption | < 1 minute | ~Hours |
| **Failed Migration** | Schema migration fails and corrupts data | ❌ Unprotected | ✅ **Protected** | ✅ **Protected** | **PITR** - Immediate rollback capability | < 1 minute | ~Hours |
| **Accidental Table Drop** | Production table dropped by mistake | ❌ Unprotected | ✅ Protected | ⚠️ Partially Protected (within PITR window) | **Full Backup** - Recreate table with original data | 24 hours | ~Hours |
| **Logical Corruption (>1.5 days)** | Data corruption discovered after PITR expires | ❌ Unprotected | ✅ **Protected** | ❌ Unprotected | **Full Backup** - Longer retention period | 24 hours | ~Hours |
| **Accidental Database Deletion** | Admin accidentally deletes entire database | ❌ Unprotected | ✅ **Protected** | ❌ Unprotected | **Full Backup** - Deletion protection in place | 24 hours | ~Hours |
| **Regional Failure** | Entire region becomes unavailable | ✅ **Protected** | ❌ Unprotected | ❌ Unprotected | **Multi-Region Config** - Automatic failover | < 1 minute | < 1 minute  |
| **Multi-Region Disaster** | Natural disaster affects multiple regions | ✅ **Protected** | ❌ Unprotected | ❌ Unprotected | **Multi-Region Config** - Geographic redundancy | < 1 minute | ~Hours if both writer regions fail otherwise < 1 minute |

## Storage Cost Analysis

### Cloud Spanner Pricing Reference

Source: https://cloud.google.com/spanner/pricing

| Region | Type | Count | Rate per 100GB/hour |
|--------|------|-------|---------------------|
| us-east4 | Read-write | 2x | $0.03014 |
| us-east1 | Read-write | 2x | $0.02740 |
| us-west2 | Read-only | 1x | $0.01644 |
| europe-west1 | Read-only | 1x | $0.01370 |
| asia-southeast1 | Read-only | 1x | $0.01567 |

Backup storage: $0.30/GB/month (flat rate for multi region backups)

### Usage Cost Calculation

Conservative estimates assume **75.96 GB** database size (2x current projections) to account for incomplete table and index designs.

| Component | Storage Size | Monthly Cost | Annual Cost | Notes |
|-----------|-------------|--------------|-------------|--------|
| **Multi-region base storage** | 75.96 GB | $89.20 | $1,070.40 | 4 read-write + 3 read-only replicas |
| **36-hour PITR overhead** | ~7.6 GB (+10%) | $8.92 | $107.04 | MVCC versions for 6.48M changes |
| **Backup Strategy Options:** | | | | |
| Option A: Daily full schedule (90-day) | 6,836.4 GB | $2,050.92 | $24,611.04 | 90× base storage |
| Option B: Incremental schedule (90-day) | 1,230.6 GB | $369.18 | $4,430.16 | 8 fulls + 82 incrementals @ 10% |
| **Total with Option A** | - | **$2,149.04** | **$25,788.48** | Full backup strategy |
| **Total with Option B** | - | **$467.30** | **$5,607.60** | **Incremental strategy (78% savings)** |

**Key Assumptions:**

- Database doubled from 37.98 GB projection to account for design uncertainty
- 10% annual growth rate based on historical trends of Postgres growth
- Incremental sizing: 1 full backup every 14 days (8 total) + daily incrementals at 10% of full size
- All costs include multi-region replication across 7 total replicas
- Backup storage charged at $0.30/GB/month

### Key Findings

1. **PITR Performance Impact**: Testing showed no significant performance degradation with 36-hour PITR compared to the 1-hour baseline based on an approximation of our database size
2. **Backup Storage**: According to [Cloud Spanner documentation](https://cloud.google.com/spanner/docs/backup#key-features), backups are automatically replicated across all configured regions
3. **Recovery Time**: Database restoration takes approximately 20-30 minutes, plus ~1 hour for redeployment, totaling ~2 hours RTO (conservative estimate). More thorough testing planned in [issue #483](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/483)
4. **Version Retention**: PITR maintains multiple versions using Multi-Version Concurrency Control (MVCC) with minimal overhead at our scale

## Decision

We will implement the following backup strategy for the Cloud Spanner Topology Service:

1. **Configure production as multi-region** as per the [existing ADR](015_spanner_multiregional.md) to handle regional failures automatically and multi-regional failures in combination with backups
2. **Enable Point-in-Time Recovery for 36 hours** to provide immediate recovery capabilities for recent issues and protection from most identified failure scenarios that aren't covered by being multi region. Data corruption is likely to cause issues that will be discovered immediately and the 36 hour window provides buffer for discovery and response window
3. **Implement daily incremental backup schedule with 90-day retention** to leverage Spanner's automatic full backup management (creates full backups as needed, then up to 13 incrementals per chain) while matching our current backup policies and protecting against long-term unnoticed data corruption

### Rationale

1. **Comprehensive Coverage**: This strategy provides protection against all identified failure scenarios
2. **Performance Validated**: Load testing confirms 36-hour PITR has no significant performance impact at our scale. Longer PITR retention period provides minimal benefit with increased risk around performance.
3. **Alignment with Standards**: Meets or exceeds GitLab's existing [PostgreSQL backup policies](../../../../gitlab-com/policies/backup/)
4. **Cost-Effective**: Balances storage costs with recovery capabilities. Longer PITR retention period provides minimal benefit with increased storage cost.
5. **Operational Flexibility**: Daily backups provide recovery options beyond PITR window

### Implementation Details

**PITR Configuration:**

- Retention Period: 36 hours
- Provides overlap with 24-hour incremental backup windows
- Enables precise recovery to any point within the window

**Backup Schedule:**

- Incremental Backups: Daily at 02:00 UTC (Full Backups Automatically created by Spanner as needed)
- Retention: 90 days
- Location: Multi-region replication (automatic)

**Access Control:**

- Restricted to broken glass escalation procedures
- IAM-based authentication (no username/password)
- Audit logging for all backup operations

## Consequences

### Positive Consequences

1. **Robust Disaster Recovery**: Comprehensive protection against hardware failures, data corruption, and human errors
2. **Validated Performance**: Load testing proved 36-hour PITR adds negligible latency (<0.25ms) at our scale
3. **Geographic Redundancy**: Multi-region configuration provides continental-level disaster protection with automatic failover
4. **Flexible Recovery Options**: 36-hour PITR for precise recent recovery, 90-day backups for older incidents
5. **Compliance Ready**: Meets enterprise DR requirements with <1 minute RTO/RPO for regional failures, ~2 hours RTO for backup restoration scenarios (including redeployment time)
6. **Operational Efficiency**: Fully managed service eliminates backup maintenance overhead (saves ~0.5 FTE)

### Negative Consequences

1. **Regional Change Limitation**: Cannot change instance regional configuration while backups exist
2. **Performance Implications**: Possible long term performance implications of having longer PITR than default
3. **Recovery Point Objective**: For data corruption discovered after 36 hours, recovery point is limited to backup intervals (up to 24 hours of data loss)
4. **Storage Costs**: Maintaining 90 days of daily backups incurs additional storage expenses
5. **Recovery Time for Backup Restores**: Database restoration from backups takes 20-30 minutes, plus ~1 hour for redeployment (~2 hours total conservative estimate), with no point-in-time recovery capabilities on restored backups

### Mitigations

1. **Regional Changes**: Implement controlled process for regional migrations using change management process
2. **Recovery Testing**: [Regular restore drills to optimize procedures and reduce recovery time](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/483)
3. **Monitoring and Alerts**: [Implement monitoring to ensure backup changes or drop protection removal don't go undetected](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/471)

## References

- [Cloud Spanner Backup Strategy Discussion](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/301)
- [Spanner Load Testing Results](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/474)
- [Cloud Spanner Disaster Recovery Overview](https://cloud.google.com/spanner/docs/backup/disaster-recovery-overview)
