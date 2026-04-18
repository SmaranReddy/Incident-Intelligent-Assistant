---
title: "GitLab Runner Job Router"
status: proposed
creation-date: "2025-09-19"
authors: [ "@josephburnett", "@ash2k" ]
coach: []
dris: [ "@ash2k" ]
approvers: []
owning-stage: "~devops::verify"
participating-stages: []
toc_hide: true
---

{{< engineering/design-document-header >}}

## KAS Job Router Module Design

*Note: "KAS Job Router Module" will be referred to as just "KAS" going forward.*

## Overview

KAS transforms GitLab's CI job scheduling from runner polling to intelligent, KAS-mediated routing. This evolution occurs across three phases: smart proxy with runner prioritization, distributed autoscaling coordination, and full job orchestration with KAS ownership of job lifecycle.

This design reframes the GitLab Runner as an Agent which connects to KAS. The module aligns with the planned reverse gRPC tunnel infrastructure for network ingress to job environments. Job routing will complement the tunnel design by optimizing which runners execute jobs based on cost, performance, and capacity considerations.

## Architecture Evolution

### Current State

```text
GitLab Rails ←→ GitLab Runner (HTTP polling every few seconds)
```

*Note: Polling goes through Workhorse, which continues in all phases but is omitted from diagrams for clarity.*

### Phase 1 Target State

```text
 Steps Debug Clients/Admission Controllers
                    ↓
GitLab Rails ← KAS Job Router ← GitLab Runner (jobs)
Job API             ↓
and Runner      Redis Coordination
Prority
```

### Phase 2 Target State

```text
 Steps Debug Clients/Admission Controllers
                    ↓
GitLab Rails ← KAS Job Router ←→ GitLab Runner (jobs and autoscaling signals)
Job API             ↓
and Runner      Redis Coordination
Priority
and GraphQL
Metrics
```

### Phase 3 Target State

```text
 Steps Debug Clients/Admission Controllers
                    ↓
GitLab Rails → KAS Job Router ←→ GitLab Runner (jobs and autoscaling signals)
 (job push)         ↓
                Redis Coordination
```

## Capability Fingerprinting

Central to the job router design is **capability fingerprinting** - a stable hash that uniquely identifies all factors determining job-runner compatibility:

- **Tags**: Exact tag matching requirements
- **Runner Type**: Instance, group, or project-specific runners
- **Protected Status**: Whether runner can handle protected references
- **Project Access**: Which projects/groups the runner can access

Runners with identical fingerprints can handle the exact same set of jobs, enabling:

- **Efficient Grouping**: Runners sharing capabilities are managed together
- **Priority Routing**: Within capability groups, apply cost/performance preferences
- **Autoscaling Coordination**: Calculate demand per capability group across KAS instances
- **Deduplication**: Avoid double-counting jobs/runners across distributed systems

The fingerprint will be available from GitLab's runner check endpoint and GraphQL APIs.

## Network Ingress Integration

KAS aligns with the planned reverse gRPC tunnel design:

- **Job Environment Access**: The tunnel blueprint enables HTTP/SSH access to running jobs for debugging
- **Step-Runner Integration**: Planned connectivity to step-runner services in job environments
- **Operational Benefits**: Eliminates external network setup requirements (domains, SSL, ingress)
- **Shared Infrastructure**: Both features leverage KAS's gRPC connection pooling and routing

Job routing decisions can consider network topology when assigning jobs to runners, optimizing both performance and operational access.

## Three-Phase Implementation

### Phase 1: Smart Proxy with Runner Prioritization

**Goal**: Replace runner HTTP polling with runner polling over bidirectional gRPC stream and implement spillover-based runner prioritization

#### Core Features

- **Bidirectional KAS-Runner gRPC Streaming**: Long running Runner->KAS reverse tunnel connections replace HTTP polling
- **Spillover Algorithm**: Higher priority runners get first choice within capability groups
- **Admission Control**: gRPC-based access restrictions and resource quotas
- **Direct Log Streaming**: Job logs stream directly to GitLab, not through KAS

#### gRPC Protocol

Runner makes two RPCs to KAS:

- Registration/heartbeat/status update RPC.
  Carries all information about the Runner and lets KAS know there is still such a Runner with certain capabilities and capacity available
- Reverse tunnel connections.
  All communication from KAS to Runner happen via these tunnels.
  KAS RPCs: job assignments, wait responses for spillover control, autoscaling signals

All Runner->KAS RPCs use Runner credentials plus machine_id for individual runner tracking.

#### Job Retrieval

- When a KAS instance receives a runner registration RPC, the instance broadcasts via Redis that it is handling this runner from now on
  and all other KAS instances stop polling for this runner id.
  A Redis lock is used to ensure only a single KAS is polling.
- KAS polls Rails using the runner credentials for jobs then schedules them on runners according to capabilities and capacity

#### Routing Algorithm

Routing happens based on runner ID.
i.e. the runner whose credentials were used for polling gets the jobs.

- Priority configuration per capability fingerprint stored in GitLab
- KAS retrieves configuration through configuration streaming from GitLab
- Concurrency limits continue to be configured in individual runner `config.toml`
- Runner reports concurrency saturation to KAS through registration RPC
- Runner rejects requests to schedule jobs with `ResourceExhausted` status code if at capacity.
  This deals with any coordination races and allows KAS to schedule the job to a different runner.
  Normally this shouldn't happen because KAS is aware of a runner's available capacity

**Spillover Logic**:

- Group runners by capability fingerprint
- Within each group, apply priority ordering
- Let job requests through to GitLab based on runner priority
- Lower priority runners are not proxied until higher priority runners reach capacity

#### Admission Control

- Multiple Controllers: Support multiple admission controllers configured in GitLab for each instance, group and project
- gRPC Streaming: Controllers establish gRPC reverse tunnel connections to KAS
- Sequential Processing: Jobs make their way through relevant controllers from instance, to group, to project. All must pass
- Batch Processing: Controllers can receive batches of jobs for validation decisions
- Availability-Based: Jobs requiring decisions look for appropriately connected controllers
- Timeout Handling: Jobs time out and fail if no controllers are available or responsive

#### GitLab API Enhancement

- Add capability fingerprint to runner check endpoint response
- Enables priority configuration per capability fingerprint

#### Redis State Management

- Priority-based runner tracking per capability fingerprint
- Individual runner capacity tracking per runner token and machine_id
- Current job counts and concurrent limits per runner

### Phase 2: Distributed Autoscaling Coordination

**Goal**: Enable queue depth autoscaling coordination across multiple KAS instances without taking over job orchestration yet

#### GitLab API Enhancement

- Extend GraphQL jobs resolver with capability fingerprinting
- Add stable hash of job-runner compatibility factors for autoscaling coordination
- Enables efficient grouping of runners by capabilities

#### Redis Coordination Schema

- Distributed runner tracking per capability fingerprint
- Job tracking with Redis sets for deduplication across KAS instances
- TTL strategy: longer for runners (config changes infrequently), shorter for jobs (queue changes frequently)

#### Job Retrieval

Same as in Phase 1.

#### Routing Algorithm

Similar to Phase 1, but routing happens based on the capability fingerprint rather than on runner id.

#### Autoscaling Algorithm

**Per capability fingerprint calculation:**

```pseudocode
for each capability_fingerprint:
    total_runners = count_runners(fingerprint)
    total_jobs = count_jobs(fingerprint)

    if total_runners > 0:
        demand_per_runner = total_jobs / total_runners
    else:
        demand_per_runner = 0

    generate_autoscaling_signal(fingerprint, demand_per_runner)
```

#### Scalability Properties

- **Write Complexity**: O(J×R) per KAS instance
- **Read Complexity**: O(F) where F = unique capability fingerprints
- **Memory Usage**: O(F×(R+J)) with automatic deduplication via Redis sets

### Phase 3: Full Job Orchestration

**Goal**: KAS owns complete job lifecycle with GitLab pushing all actionable jobs

#### New Job States

```text
pending → routing → admitted → assigned → running → completed/failed
                 ↘ rejected
```

#### Job Push Protocol

- GitLab pushes all actionable jobs to KAS via gRPC
- Pushed jobs include the capability fingerprint
- KAS reports job state changes back to GitLab
- Bidirectional job lifecycle management

#### Complete Job Selection Engine

Based on GitLab's `Ci::Queue::BuildQueueService`, implements:

1. Tag Matching: Exact match between job and runner tags
2. Runner Type Filtering: instance/group/project type validation
3. Protected Reference Handling: Protected jobs require protected runners
4. Project Access Control: Verify runner permissions for job's project
5. Resource Validation: Memory, CPU, Docker requirements
6. Advanced Features: Cost optimization, performance history, affinity rules

#### Distributed Job Management

- Job deduplication across KAS instances
- Priority queue for admitted jobs
- Runner assignment tracking
- Runner status and capacity monitoring
- Intra and inter cell job routing for GitLab.com cost savings

## Summary

KAS evolves GitLab CI from polling-based to intelligent job routing across three phases: smart proxy with spillover, distributed autoscaling coordination, and full job orchestration. Each phase builds incrementally while maintaining backward compatibility.

## References

**Key GitLab Issues:**

- [GitLab Runner Admissions Controller (MVC) - Issue #378322](https://gitlab.com/gitlab-org/gitlab/-/issues/378322)
- [Runner Priority - Issue #14976](https://gitlab.com/gitlab-org/gitlab/-/issues/14976)
- [Runner Load Balance - Issue #15963](https://gitlab.com/gitlab-org/gitlab/-/issues/15963)

**Architecture Blueprints:**

- [Reverse gRPC Tunnel for Workspaces and CI](../reverse-grpc-tunnel-workspaces-and-ci/)
- [Runner Admission Controller Blueprint](../runner_admission_controller/)

**Step-Runner Integration:**

- [Step-Runner Debug API and Demo - MR !168](https://gitlab.com/gitlab-org/step-runner/-/merge_requests/168)
