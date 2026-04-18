---
title: "AWS Primary Region Selection for Cells Infrastructure"
owning-stage: "~devops::tenant-scale"
group: cells-infrastructure
creation-date: "2025-05-08"
authors: ["@tkhandelwal3"]
coach: "@sxuereb"
approvers: []
toc_hide: true
---

{{< engineering/design-document-header >}}

## Summary

This ADR documents the decision to use **`us-east-1` (Northern Virginia)** as the primary AWS region for Cells infrastructure based on comprehensive latency testing against the Topology Service hosted in GCP.

## Context

As part of the Cells architecture evolution, we are using AWS to provision our first Cells while maintaining critical infrastructure components like the Topology Service in GCP. This cross-cloud architecture requires careful consideration of network latency to ensure optimal performance.

The cross-cloud communication between AWS Cells and GCP infrastructure impacts:

- Organization classification and routing.
- Data migration when moving organizations from Legacy Cell to Cells.
- Cell discovery and health checks.
- Configuration management.

Given this architecture, the latency between AWS-hosted Cells and the GCP-hosted Topology Service directly impacts:

- The latency to [Claim a resource](../topology_service.md#claim-service).
- Data transfer speed when migrating organizations from Legacy Cell to Cell.
- SSH Routing for Git traffic.

## Problem Statement

We need to determine the optimal AWS region for hosting Cells that minimizes latency to the Topology Service while considering:

- Network performance and latency characteristics
- Future scalability requirements
- Geographic distribution of traffic
- Disaster recovery capabilities
- Availbility of features

## Testing Methodology

### Test Infrastructure

- **Load Testing Tool**: k6 with [custom script](https://gitlab.com/sxuereb/cells-aws/-/blob/2fdcf7bd6fbee0431b06683bec40fde4d83a1a88/classify.js) for generating consistent load.
- **Test Duration**: 5 runs of 360 seconds each per region
- **Virtual Users**: 100 concurrent connections
- **Target Endpoint**: `https://topology-rest.gitlab.net/v1/classify` (Production Topology Service)
- **Instance Type**: `m5a.xlarge` (4 vCPUs, 16GB RAM, up to 10Gbps network)

### Regions Tested

- **us-east-1**: Northern Virginia (geographically closest to GCP us-east1)
- **us-east-2**: Ohio

Note: `us-central-1` does not exist in AWS; only `us-east` and `us-west` regions are available.

### Test Configuration

```javascript
// k6 test configuration
export const options = {
  vus: 100,           // 100 virtual users
  duration: '360s',   // 6 minutes per run
};
```

Testing was performed and documented in [GitLab Issue #475 Comment](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/475#note_2705749726).

## Results

### Latency Measurements

The following results represent the average across 5 test runs for each region:

| Region | Median | P95 | P99.9 | Requests/sec |
|--------|--------|-----|-------|--------------|
| **us-east-1** | 23.608ms | **32.31ms** | **80.88ms** | 3,589/s |
| **us-east-2** | 43.768ms | 51.234ms | 106.462ms | 2,031/s |

> [!note]
> See the complete test data at: <https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/475#note_2705749726>

### Performance Analysis

1. **P95 Latency**: `us-east-1` demonstrates **40% better performance** than `us-east-2`
   - us-east-1: 32.31ms
   - us-east-2: 51.234ms
   - Improvement: ~37% reduction

2. **P99.9 Latency**: `us-east-1` shows **~30% better performance** for tail latencies
   - us-east-1: 80.88ms
   - us-east-2: 106.462ms
   - Improvement: ~24% reduction

3. **Throughput**: `us-east-1` achieved **76% higher request throughput**
   - us-east-1: ~3,589 requests/second
   - us-east-2: ~2,031 requests/second

## Decision

We will use **`us-east-1` (Northern Virginia)** as the primary region for AWS Cells deployment.

### Rationale

1. **Superior Performance**: 40% better P95 latency ensures better user experience for the majority of requests
2. **Higher Throughput**: 76% higher request processing capability provides better scalability
3. **Geographic Proximity**: Closest AWS region to GCP's us-east1 where Topology Service is hosted
4. **Network Path Optimization**: Shorter network paths between Northern Virginia and GCP's eastern regions
5. **Consistency**: More stable performance characteristics across test runs

## Consequences

### Positive Consequences

1. **Improved Performance**: Lower latency translates to faster response times for Cell operations.
2. **Better Scalability**: Higher throughput capacity allows for more efficient resource utilization.
3. **Optimal Data Transfer**: Reduced latency benefits organization migrations from Legacy Cell to Cells.
4. **Feature Availability**: Access to the latest AWS features as they typically launch in us-east-1 first.

### Negative Consequences

1. **Regional Concentration Risk**: `us-east-1` historically experiences the most AWS incidents due to its high concentration of infrastructure.
2. **Reliability Concerns**: Higher incident rate compared to other AWS regions may impact overall system reliability.
3. **Quota Request Delays**: As the busiest AWS region, us-east-1 may have longer processing times for quota increase requests needed for Cell provisioning, potentially impacting deployment timelines

### Mitigations

1. **Multi-AZ Deployment**: Deploy across multiple availability zones within us-east-1 to mitigate AZ-specific failures
2. **Secondary Region Readiness**: Maintain us-east-2 as a backup_region with Geo.
3. **Proactive Quota Management**: Submit quota increase requests well in advance of Cell provisioning needs and maintain buffer capacity to account for potential delays
