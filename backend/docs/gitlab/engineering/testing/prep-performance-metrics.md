---
title: PREP Performance Metrics Guide
---

## PREP Performance Metrics Guide

This guide helps teams identify and measure performance metrics for their features as part of the [PREP (Platform Readiness Enablement Process) readiness process](https://gitlab.com/gitlab-org/architecture/readiness). Rather than creating new metrics, we will map features to existing production SLIs (Service Level Indicators) that are already monitored in production environments.

## Overview

There are two main sources for production metrics:

- Our SaaS service makes use of the [metrics catalog](https://gitlab.com/gitlab-com/runbooks/-/tree/master/metrics-catalog) which defines how GitLab.com measures service health through SLIs and SLOs (Service Level Objectives). These thresholds are battle-tested in production and represent what users expect.
- Our Self-Managed customers rely on our [Reference Architectures](https://docs.gitlab.com/administration/reference_architectures/) and corresponding published [benchmarks](https://gitlab.com/gitlab-org/reference-architectures/-/wikis/Benchmarks/Latest)

By mapping your feature to the relevant production SLIs / benchmarks, you gain several advantages:

- **Readiness checks tied to production criteria** - You measure what we've already identified as important to production health, not arbitrary metrics
- **Consistency between testing and production** - You avoid measuring different things in PREP than we do in production, reducing surprises at deployment
- **Feedback loop for metrics** - When PREP testing reveals gaps in our metrics, we can identify new metrics to add to production monitoring

## How to Use This Guide

**Prerequisite:** Performance testing assumes functional correctness of the code paths being tested.

1. **Identify your SLI(s)** - Use the [decision tree](#sli-component-decision-tree) below to find which SLI component your feature touches, also review the [Self-Managed Metrics](#self-managed-metrics-sources) for relevant metrics to be SLIs
2. **Understand what matters** - Review the [metrics selection guide](#metrics-selection-guide) to see what production measures and what you should test
3. **Measure underlying performance** - Gather latency, resource, and correctness metrics during PREP testing
4. **Document your results** - Record your findings in the PREP issue template

## Metrics Sources for Your PREP Report

Your PREP report should include metrics from both GitLab.com (SaaS) and self-managed deployment contexts. These sources provide complementary information:

**GitLab.com (SaaS) Metrics:**

- Source: Production SLI thresholds from the [metrics catalog](https://gitlab.com/gitlab-com/runbooks/-/tree/master/metrics-catalog)
- Why it matters: Shows how your feature performs against what we've learned is important in production at scale
- What to measure: Latency percentiles, error rates, and resource consumption compared to production thresholds
- Relevance: Direct indicator of production readiness for SaaS deployments

**Self-Managed Metrics:**

- Source: [Reference benchmarks](https://gitlab.com/gitlab-org/reference-architectures/-/wikis/Benchmarks/Latest)
- Why it matters: Shows how your feature performs in environments with different hardware, scale, and configurations
- What to measure: Latency percentiles, error rates, and resource consumption compared to reference benchmarks
- Relevance: Indicates how your feature will behave across the range of self-managed deployments

**Why Both Matter:**

- SaaS metrics tell you if your feature is production-ready for GitLab.com
- Self-managed metrics tell you if your feature will perform reasonably across diverse deployment environments
- Together, they give confidence that your feature works well regardless of deployment type

**Stitching Them Together in Your PREP Report:**

Include both contexts in your results:

1. **SaaS Performance:** "Against production SLI thresholds, my feature achieves [metrics]. This indicates [readiness level]."
2. **Self-Managed Performance:** "Against reference benchmarks for [deployment size/type], my feature achieves [metrics]. This indicates [readiness level]."
3. **Variance Analysis:** "The difference between SaaS and self-managed contexts is [explanation]. This is expected because [reasons]."

**Example:**

```plaintext
SaaS Context: p95 latency is 450ms (production threshold: < 1s) ✓
Self-Managed Context: p95 latency is 520ms (reference benchmark for a 200 RPS / 10,000 user deployment: < 800ms) ✓
Variance: Self-managed is slightly higher due to no background load optimization
Conclusion: Feature is ready for both deployment types
```

This approach ensures your PREP report demonstrates readiness across deployment contexts, not just one.

## SLI Component Decision Tree

```mermaid
graph TD
    WORK["What type of work does your feature do?"] --> HTTP["HTTP Request<br/>Web or API"]
    WORK --> BACK["Background Job<br/>Async Processing"]
    WORK --> DB["Database Operation<br/>Direct DB Access"]
    WORK --> SEARCH["Search Operation<br/>Elasticsearch"]
    WORK --> AI["AI/LLM Operation<br/>Language Model"]

    HTTP --> HTTP_TYPE{"What sort of<br/>HTTP operation?"}
    HTTP_TYPE -->|GraphQL| GRAPHQL_SLI["GraphQL Query SLI<br/>api.jsonnet"]
    HTTP_TYPE -->|Web Page| RAILS_WEB["Rails Request SLI<br/>web.jsonnet"]
    HTTP_TYPE -->|API Endpoint| RAILS_API["Rails Request SLI<br/>api.jsonnet"]

    BACK --> SIDEKIQ["Sidekiq Execution SLI<br/>sidekiq.jsonnet"]

    DB --> CLIENT_DB["Client Database<br/>Transaction SLI<br/>patroni.jsonnet"]

    SEARCH --> GLOBAL_SEARCH["Global Search SLI<br/>search.jsonnet"]

    AI --> AI_SLI["LLM Operation SLI<br/>ai-assisted.jsonnet"]

    style GRAPHQL_SLI fill:#e1f5ff
    style RAILS_WEB fill:#e1f5ff
    style RAILS_API fill:#e1f5ff
    style SIDEKIQ fill:#f3e5f5
    style CLIENT_DB fill:#e8f5e9
    style GLOBAL_SEARCH fill:#fff3e0
    style AI_SLI fill:#fce4ec

    click GRAPHQL_SLI href "https://gitlab.com/gitlab-com/runbooks/-/blob/master/metrics-catalog/services/api.jsonnet"
    click RAILS_WEB href "https://gitlab.com/gitlab-com/runbooks/-/blob/master/metrics-catalog/services/web.jsonnet"
    click RAILS_API href "https://gitlab.com/gitlab-com/runbooks/-/blob/master/metrics-catalog/services/api.jsonnet"
    click SIDEKIQ href "https://gitlab.com/gitlab-com/runbooks/-/blob/master/metrics-catalog/services/sidekiq.jsonnet"
    click CLIENT_DB href "https://gitlab.com/gitlab-com/runbooks/-/blob/master/metrics-catalog/services/patroni.jsonnet"
    click GLOBAL_SEARCH href "https://gitlab.com/gitlab-com/runbooks/-/blob/master/metrics-catalog/services/search.jsonnet"
    click AI_SLI href "https://gitlab.com/gitlab-com/runbooks/-/blob/master/metrics-catalog/services/ai-assisted.jsonnet"
```

## Understanding Your Service's SLIs

| Decision Tree Output | Guide Section | Key Metrics |
| --- | --- | --- |
| Rails Request SLI (web.jsonnet) | [HTTP Request SLIs](#http-request-slis-rails-request-graphql-query) | Latency (p95, p99), Error Rate |
| Rails Request SLI (api.jsonnet) | [HTTP Request SLIs](#http-request-slis-rails-request-graphql-query) | Latency (p95, p99), Error Rate |
| GraphQL Query SLI (api.jsonnet) | [HTTP Request SLIs](#http-request-slis-rails-request-graphql-query) | Query Latency (p95, p99), Error Rate |
| Sidekiq Execution SLI | [Sidekiq Job SLIs](#sidekiq-job-slis-execution--queueing) | Execution Duration, Resource Usage |
| Database Transaction SLI | [Database Transaction SLI](#database-transaction-sli) | Transaction Duration, Connection Pool |
| Global Search SLI | [Global Search SLI](#global-search-sli) | Search Latency, Index Size |
| LLM Operation SLI | [AI/LLM Operation SLIs](#aillm-operation-slis) | TTFT, Completion Latency |

## Metrics Selection Guide

Once you've identified which service your feature maps to, use the appropriate section below to understand what the production SLI measures and what underlying metrics you should gather during PREP testing.

**Key Principle:** Production SLIs are designed to detect degradation in a live system. PREP testing focuses on the *underlying measurements* that prove your feature won't cause that degradation. We're not trying to replicate production SLO thresholds; we're measuring the fundamental performance characteristics that indicate readiness.

**Note on Thresholds:** The production values shown in each section are reference points to understand what's acceptable in production. Your test environment won't match production exactly, but these help you understand the performance characteristics that matter. Use them as context when interpreting your test results.

## Understanding Percentile Metrics: p90 vs p95 vs p99

Different tools and services measure different percentiles. Understanding the difference helps you choose the right metric for your context:

- **p90** - What 90% of operations are faster than. Used by [GitLab Performance Tool (GPT)](https://gitlab.com/gitlab-org/quality/performance) as the base metric. It gives a good balance of real world timing while not getting completely skewed by a [long tail distribution](https://en.wikipedia.org/wiki/Long_tail) of timings.
- **p95** - What 95% of operations are faster than. The standard for most production SLIs in the [metrics catalog](https://gitlab.com/gitlab-com/runbooks/-/tree/master/metrics-catalog), balancing sensitivity to degradation with noise tolerance.
- **p99** - What 99% of operations are faster than. Captures worst-case scenarios and tail latency; useful for identifying systemic issues.

**Note:** Most SLIs use p90, p95, or p99, but some use other percentiles. For example, Global Search uses the 99.95th percentile. Check your SLI's description in the metrics catalog to see which percentile it uses.

**When measuring your feature:** Use the percentile specified in your SLI section. If you're comparing against production metrics or using GPT for testing, note which percentile each tool measures and adjust your interpretation accordingly. You don't need to measure all three — measure what your SLI requires and what your testing tool provides.

## Understanding Underlying Metrics for Your SLI

The metrics catalog is organized in two layers:

**Layer 1: SLI Definitions** ([`metrics-catalog/gitlab-slis/library.libsonnet`](https://gitlab.com/gitlab-com/runbooks/-/tree/master/metrics-catalog//gitlab-slis/library.libsonnet))

- Defines reusable SLI templates with descriptions and significant labels
- Examples: `rails_request`, `global_search`, `sidekiq_execution`, `llm_completion`
- These are the *conceptual* definitions of what each SLI measures, but not the actual thresholds

**Layer 2: Service Implementations** ([`metrics-catalog/services/*.jsonnet`](https://gitlab.com/gitlab-com/runbooks/-/tree/master/metrics-catalog/services))

- Each service file (e.g., `api.jsonnet`, `web.jsonnet`) instantiates SLIs from the library
- Uses `sliLibrary.get('rails_request').generateServiceLevelIndicator(selector, overrides)` to create concrete implementations
- The `overrides` parameter sets service-specific thresholds (e.g., `monitoringThresholds: { apdexScore: 0.99 }`)
- The `capacityPlanning` section identifies underlying metrics to monitor

Once you've identified your SLI, you can find the underlying metrics to monitor by looking at the `capacityPlanning` section in the corresponding service jsonnet file in the [metrics catalog](https://gitlab.com/gitlab-com/runbooks/-/tree/master/metrics-catalog/services).
For example, in [`api.jsonnet` the `capacityPlanning` section](https://gitlab.com/gitlab-com/runbooks/-/blob/master/metrics-catalog/services/api.jsonnet#L298) identifies:

- `ruby_thread_contention` - Monitors if Ruby threads are blocked
- `rails_db_connection_pool` - Monitors database connection availability
- `kube_go_memory` - Monitors memory usage of the service

The metrics in the `capacityPlanning` section have proven useful enough to be tracked in our Observabilty so they are worth tracking as part of testing. They don't directly compose your SLI (apdex score and error rate), but they're prerequisites for meeting latency and error rate targets. If any of these are exhausted or degraded, the SLI will suffer.

**For PREP testing**, measure these underlying metrics to ensure your feature can sustain the primary SLI targets under load. They help you understand whether your feature will continue to meet its SLI thresholds as load increases.

## System-Level Metrics to Monitor

Regardless of which SLI you're measuring, always monitor system-level metrics during performance testing. This enables tuning and troubleshooting. Some common metrics are:

- **CPU utilization** - Should scale linearly with load; watch for plateaus indicating bottlenecks
- **Memory usage** - Should be stable; watch for leaks or sudden spikes
- **Disk I/O** - For database/search operations, monitor read/write latency and throughput
- **Network throughput** - For services handling large payloads, monitor bandwidth utilization
- **Connection pools** - Database connections, thread pools, etc. should not be exhausted

**Note:** This list covers common system-level metrics. Your feature may require monitoring additional metrics depending on its specific implementation.

## Measuring Your SLI

The sections below detail what to measure for each of the SaaS SLI types. For each one, you'll find:

- **What Production Measures** - The primary SLI metrics (apdex, error rate, etc.)
- **What You Should Measure in PREP** - The underlying metrics that prove you can sustain the primary SLI
- **Example** - A concrete scenario showing how to test

### HTTP Request SLIs (Rails Request, GraphQL Query)

**Applies to:** Web pages, API endpoints, GraphQL queries

**What Production Measures:**
The production SLI tracks apdex score (latency compliance) and error rate to detect when the service is degrading. The production thresholds are:

- **Apdex Score:** 99.8% of requests should meet latency expectations (satisfied < 1s, tolerated < 10s)
- **Error Rate:** < 0.01% of requests should fail

**What You Should Measure in PREP:**

1. **Request Latency Percentiles** - How fast does your feature respond?
   - Measured as: milliseconds or seconds
   - Why it matters: Slow features degrade user experience and can cascade into other failures
   - What to measure: see [Understanding Percentile Metrics](#understanding-percentile-metrics-p90-vs-p95-vs-p99) to understand which percentile to capture
   - Production context: Production considers < 1s "satisfied" and < 10s "tolerated"
   - PREP gate: Establish a baseline in your test environment. Your p90/p95 should be consistent and predictable; p99 should not exceed p95 by more than 5-10x. Compare against similar existing features — your feature should not be significantly slower (use production thresholds as a reference point).
   - How to gather: See [Performance Testing Tools](performance-tools.md) for guidance on selecting and using the right tool, or use application instrumentation

2. **Resource Consumption** - Does your feature consume reasonable resources?
   - Measured as: CPU %, memory usage, database connections
   - Why it matters: Resource exhaustion can cause cascading failures
   - What to measure:
     - CPU usage during peak load
     - Memory usage (no memory leaks)
     - Database connection pool utilization
   - PREP gate: Monitor resource usage as you increase load. CPU, memory, and connections should scale predictably. Watch for superlinear resource growth (for example: 2x load causing 4x memory), sudden spikes, or plateaus that indicate bottlenecks.
   - How to gather: Monitor system metrics during load testing

**Significant Labels to Track:**

- `endpoint_id` - Which specific endpoint/route is being called
- `feature_category` - What feature area does this belong to
- `request_urgency` - Is this a high-priority or low-priority request

**Example:** A new API endpoint for user profile updates would be tested by:

- Load testing with 100-1000 concurrent users
- Measuring p95 and p99 latency (should be < 1s and < 10s respectively)
- Verifying all requests return correct data
- Monitoring CPU/memory to ensure no resource exhaustion

### Sidekiq Job SLIs (Execution & Queueing)

**Applies to:** Background jobs, async workers, scheduled tasks

**Important:** Sidekiq jobs are classified by [urgency level](https://docs.gitlab.com/development/sidekiq/worker_attributes/#job-urgency). You must determine your job's urgency before setting thresholds. When job queues are under load, [latency-sensitive jobs](https://docs.gitlab.com/development/sidekiq/worker_attributes/#latency-sensitive-jobs) are prioritized for execution.

**Urgency Levels:**

- **High Urgency** - User-facing operations that should complete quickly (for example, sending notifications, updating user preferences)
- **Low Urgency** - Background operations that can take longer (for example, batch processing, cleanup jobs)

**Determining Your Job's Urgency:**

Ask: "Will users notice if this job is delayed by 5 minutes?"

- If yes → High urgency
- If no → Low urgency

**What Production Measures:**
The production SLI tracks execution duration and queueing duration to detect when jobs are taking too long or piling up. The production thresholds are:

- **Execution Duration (apdex satisfied):**
  - High urgency: < 5 seconds
  - Low urgency: < 300 seconds
- **Queueing Duration (apdex satisfied):**
  - High urgency: < 5 seconds
  - Low urgency: < 5 minutes
- **Error Rate:** < 0.01% of jobs should fail

**What You Should Measure in PREP:**

1. **Execution Duration Percentiles** - How long does your job take to run?
   - Measured as: seconds
   - Why it matters: Long-running jobs can block other work and consume resources
   - What to measure: execution duration (see [Understanding Percentile Metrics](#understanding-percentile-metrics-p90-vs-p95-vs-p99) to understand which percentile to capture)
   - Production context: High urgency jobs should be < 5s, low urgency < 300s
   - PREP gate: Execution duration should be well under your job's production threshold
   - How to gather: Instrument your worker with timing metrics; enqueue test jobs

2. **Resource Consumption** - Does your job consume reasonable resources?
   - Measured as: CPU %, memory usage, database connections
   - Why it matters: Resource exhaustion can cause cascading failures
   - What to measure:
     - CPU usage during peak job processing
     - Memory usage (no memory leaks)
     - Database connection pool utilization
   - PREP gate: No resource exhaustion; should scale linearly with job count
   - How to gather: Monitor system metrics while processing test jobs

**Significant Labels to Track:**

- `worker` - The job class name
- `feature_category` - What feature area does this job belong to
- `urgency` - Is this high or low urgency work
- `queue` - Which queue is the job on

**Example:** A new background job that processes user exports would be tested by:

- Enqueuing 1000 jobs with varying data sizes
- Measuring p95 and p99 execution time (should be well under your job's production threshold)
- Verifying all jobs complete successfully and produce correct results
- Monitoring CPU/memory to ensure no resource exhaustion

### Database Transaction SLI

**Applies to:** Features that perform direct database operations

**What Production Measures:**
The production SLI tracks transaction duration to detect when database operations are taking too long. Long transactions lock resources and can block other operations. The production threshold is typically < 1 second for most operations.

**What You Should Measure in PREP:**

1. **Transaction Duration Percentiles** - How long do database transactions take?
   - Measured as: milliseconds or seconds
   - Why it matters: Long transactions lock resources and can block other operations
   - What to measure: transaction duration (see [Understanding Percentile Metrics](#understanding-percentile-metrics-p90-vs-p95-vs-p99) to understand which percentile to capture)
   - Production context: Most operations should be < 1 second
   - PREP gate: Transaction duration should be under the production threshold
   - How to gather: Instrument database calls with timing metrics; run queries with test data

2. **Resource Consumption** - Does your feature consume reasonable database resources?
   - Measured as: connection pool utilization, query complexity
   - Why it matters: Resource exhaustion can cause cascading failures
   - What to measure:
     - Database connection pool utilization
     - Query execution plans (no full table scans)
     - Lock contention
   - PREP gate: No connection pool exhaustion
   - How to gather: Monitor database metrics during testing; review query plans

**Significant Labels to Track:**

- `db_config_name` - Which database (primary, replica, etc.)
- `feature_category` - What feature area does this belong to

**Example:** A feature that adds a new database table and queries it would be tested by:

- Running queries with test data and measuring p95/p99 latency
- Verifying queries return correct results and maintain data consistency
- Monitoring connection pool utilization and reviewing query execution plans

### Global Search SLI

**Applies to:** Features that use Elasticsearch or global search functionality

**What Production Measures:**
The production SLI tracks search latency to detect when searches are taking too long. Search is user-facing, so slow searches directly degrade the experience. The production threshold is based on the 99.95th percentile of similar searches.

**What You Should Measure in PREP:**

1. **Search Latency Percentiles** - How long do search queries take?
   - Measured as: milliseconds or seconds
   - Why it matters: Slow searches degrade user experience and can cascade into other failures
   - What to measure: Latency (see [Understanding Percentile Metrics](#understanding-percentile-metrics-p90-vs-p95-vs-p99) to understand which percentile to capture) Global Search uses the 99.95th percentile
   - Production context: Typical searches should be < 2 seconds
   - PREP gate: Latency should not exceed your baseline values
   - How to gather: Run searches against test data; measure query execution time

2. **Resource Consumption** - Does your search feature consume reasonable resources?
   - Measured as: Elasticsearch resource usage, index size
   - Why it matters: Resource exhaustion can cause cascading failures
   - What to measure:
     - Elasticsearch CPU and memory usage
     - Index size growth
     - Query complexity
   - PREP gate: No resource exhaustion; index size should be reasonable
   - How to gather: Monitor Elasticsearch metrics during testing; review index statistics

**Significant Labels to Track:**

- `search_scope` - What's being searched (projects, issues, etc.)
- `search_type` - Type of search (basic, advanced, etc.)
- `feature_category` - What feature area does this belong to

**Example:** A feature that adds a new searchable field would be tested by:

- Running searches against test data and measuring p90 latency
- Verifying searches return correct results with the new field
- Monitoring Elasticsearch resource usage and index size

### AI/LLM Operation SLIs

**Applies to:** Features that call language models or AI services

**Important:** Depending on your AI feature's design, you may measure different aspects:

- **Chat/Streaming features** - Prioritize Time to First Token (TTFT) since users perceive responsiveness based on how quickly they see a response
- **Completion-based features** - Prioritize total completion latency since the entire operation must complete before the user sees results

Both approaches measure the same underlying SLI in `ai-assisted.jsonnet`, but the emphasis differs based on your feature's interaction model.

**What Production Measures:**

The production SLI tracks Time to First Token (TTFT) and completion latency to detect when AI operations are taking too long. The production thresholds are:

- **TTFT:** < 5 seconds (apdex satisfied)
- **Completion Latency:** < 20 seconds (apdex satisfied)
- **Error Rate:** < 0.01% of operations should fail

**What You Should Measure in PREP:**

1. **Time to First Token (TTFT) Percentiles** - How long until the first response token arrives?
   - Measured as: milliseconds or seconds
   - Why it matters: Users perceive responsiveness based on how quickly they see a response
   - What to measure: Latency (see [Understanding Percentile Metrics](#understanding-percentile-metrics-p90-vs-p95-vs-p99) to understand which percentile to capture).
   - Production context: Should be < 5 seconds
   - PREP gate: Latency should not exceed your baseline values
   - How to gather: Measure time from request to first token received

2. **Completion Latency Percentiles** - How long does the full operation take?
   - Measured as: milliseconds or seconds
   - Why it matters: Indicates overall performance and resource consumption
   - What to measure: Latency (see [Understanding Percentile Metrics](#understanding-percentile-metrics-p90-vs-p95-vs-p99) to understand which percentile to capture).
   - Production context: Should be < 20 seconds
   - PREP gate: Latency should not exceed your baseline values
   - How to gather: Measure total operation duration from request to completion

3. **Resource Consumption** - Does your AI feature consume reasonable resources?
   - Measured as: API rate limits, token usage, memory
   - Why it matters: Resource exhaustion can cause cascading failures
   - What to measure:
     - API rate limit utilization
     - Token usage per operation
     - Memory usage during streaming
   - PREP gate: No resource exhaustion; should stay within API limits
   - How to gather: Monitor API usage and system metrics during testing

**Significant Labels to Track:**

- `service_class` - Which AI service is being used
- `feature_category` - What feature area does this belong to

**Example:** A new Duo Chat feature would be tested by:

- Measuring p90 TTFT and completion latency
- Verifying responses are coherent and relevant
- Monitoring API rate limit usage and token consumption

## Gathering Metrics During Development

### Testing Approaches

**Synthetic Testing** - Simulate user behavior with controlled requests

- Best for: Establishing baseline performance under known conditions
- How: Write functional tests that mimic user behavior, then run them to establish baseline performance. See [Performance Testing Tools](performance-tools.md) for load testing guidance if you need to combine synthetic tests with background load
- Metrics to gather: Operation rate, latency percentiles, error rate

**Load Testing** - Gradually increase load to find breaking points

- Best for: Understanding how your feature scales
- How: See [Performance Testing Tools](performance-tools.md) for guidance on selecting and using the right tool
- Metrics to gather: Latency at different load levels, error rate under stress

**Manual Testing with Instrumentation** - Measure real usage patterns

- Best for: Understanding actual user behavior
- How: Add logging/metrics to your feature and test manually
- Metrics to gather: Real-world latency, error patterns, resource usage

### Key Percentiles to Measure

When gathering latency metrics, measure the percentiles specified in your SLI section (see [Understanding Percentile Metrics](#understanding-percentile-metrics-p90-vs-p95-vs-p99) to understand which percentile to capture).

### When Existing Metrics Don't Fit

If your feature doesn't map to an existing SLI:

1. **Use closest match** - Pick the nearest SLI and document limitations
2. **Propose new metrics** - Follow the process in the Observability Documentation for [adding a new SLI](https://gitlab-com.gitlab.io/gl-infra/observability/docs-hub/service-level-indicators/howto-implementing-slis/)
3. **Include both in PREP** - Show standard SLI metrics + your proposed metrics

Document in PREP report: "Measured against [SLI] but feature differs in [ways]. Proposed [metrics] for future use."

### Accounting for Test Environment Differences

Your test environment differs from production. When interpreting results:

- **Smaller data** → Focus on query patterns and scaling, not absolute times
- **Different hardware** → Compare to other features in the same test environment, not production numbers
- **No background load** → Consider additional buffer to load to account for production competition

**What still matters:** Query patterns, algorithm complexity, resource scaling behavior

## Self-Managed Metrics Sources

To ensure your feature supports self-managed deployments, establish performance baselines by:

1. **Define your SLI** - Identify which metrics should be used as a SLI (some possible sources include the [metrics catalog](#sli-component-decision-tree), the [GitLab Reference Architecture Benchmarks](https://gitlab.com/gitlab-org/reference-architectures/-/wikis/Benchmarks/Latest), and other sources defined by your team)
2. **Define good/bad results** - Determine what acceptable performance looks like for your deployment context
3. **Run benchmark on your environment** - Test your feature in a representative self-managed environment with your target data and configuration
4. **Evaluate results** - Compare against your defined thresholds

**Reference Resources:**

- [GitLab Reference Architecture Benchmarks](https://gitlab.com/gitlab-org/reference-architectures/-/wikis/Benchmarks/Latest) - Published performance data for standard deployment sizes
- [Self-Service Performance Regression Testing](self-service-performance-regression-testing.md) - Process for running performance regression tests, it includes guidance on how to determine success criteria (SLIs)

**Note:** Benchmarks represent typical configurations; your deployment may differ. Use them as reference points, not absolute requirements.

## Interpreting Results

### Green Light (Ready for Deployment)

Your feature is ready if:

- **Latency:** Metrics are under the production threshold
- **Resources:** No resource exhaustion (CPU, memory, database connections, API limits)
- **Scalability:** Performance degrades gracefully under load; no sudden failures

### Yellow Light (Needs Optimization)

Your feature needs work if:

- **Latency:** Metrics are 1-2x the production threshold
- **Resources:** Approaching resource limits under peak load

Consider:

- Optimizing hot paths
- Adding caching
- Reducing database queries or API calls
- Improving algorithm efficiency
- Batching operations

### Red Light (Not Ready)

Your feature is not ready if:

- **Latency:** Metrics are > 2x the production threshold
- **Resources:** Resource exhaustion occurring (connection pool full, out of memory, API rate limits exceeded)
- **Scalability:** Performance degrades non-linearly or fails under expected load

Do not proceed to deployment. Investigate root causes and optimize before retesting.

### If You Get Yellow or Red

1. Profile before optimizing - don't guess at bottlenecks
2. Fix highest-impact issue
3. Re-test with same test to measure improvement

## Documenting Your Results

Add the results to your [Performance Validation](https://gitlab.com/gitlab-org/architecture/readiness/-/blob/main/templates/performance_strategy/performance_validation.md) readiness issue, and share with reviewers.

Make sure to include any variances or new metrics used.

## Common Scenarios

**Note:** The production thresholds shown in these scenarios are illustrative examples to demonstrate how to apply the guide. Your actual thresholds depend on your specific SLI implementation. Check your service's jsonnet file in [`metrics-catalog/services/`](https://gitlab.com/gitlab-com/runbooks/-/tree/master/metrics-catalog/services) for the actual monitoring thresholds.

### Scenario 1: New API Endpoint

**Feature:** Add a new endpoint to retrieve user activity

**SLI Mapping:** Rails Request SLI (api.jsonnet)

**Production Context:**

- Production threshold: p95 latency < 1 second, p99 < 10 seconds
- Error rate in production: < 0.01%

**Testing Approach:**

- Load test with 100-1000 concurrent users
- Measure latency percentiles (p95 and p99)
- Verify all requests return correct data
- Monitor CPU/memory during peak load

**Success Criteria (Green Light):**

- p95 latency is consistent and predictable across test runs
- p99 latency does not exceed p95 by more than 5-10x (indicating tail behavior, not systemic issues)
- Performance is comparable to similar existing endpoints (use production thresholds as a reference point)
- Resource usage scales linearly with load

### Scenario 2: Background Job for Data Processing

**Feature:** New Sidekiq worker to process user exports

**SLI Mapping:** Sidekiq Execution SLI (sidekiq.jsonnet)

**Job Urgency:** Low urgency (users won't notice if delayed by 5 minutes)

**Production Context:**

- Production threshold: p95 execution < 300 seconds, p99 < 600 seconds
- Error rate in production: < 0.01%

**Testing Approach:**

- Enqueue 1000+ jobs with varying data sizes
- Measure execution time percentiles (p95 and p99)
- Verify all jobs complete successfully and produce correct results
- Test edge cases (empty data, very large data, malformed data)
- Monitor CPU/memory during peak job processing

**Success Criteria (Green Light):**

- p95 execution time is consistent and predictable across test runs
- p99 execution time does not exceed p95 by more than 5-10x
- Performance is comparable to similar existing workers (use production thresholds as a reference point)
- Resource usage scales linearly with job count

### Scenario 3: Search Feature Enhancement

**Feature:** Add a new searchable field to global search

**SLI Mapping:** Global Search SLI (search.jsonnet)

**Production Context:**

- Production threshold: p95 latency < 2 seconds, p99 < 5 seconds
- Error rate in production: < 0.5%

**Testing Approach:**

- Run searches against test data with known results
- Measure search latency percentiles (p95 and p99)
- Verify searches return correct results with the new field
- Test edge cases (empty results, special characters, large result sets)
- Monitor Elasticsearch resource usage and index size
- Compare performance with and without the new field

**Success Criteria (Green Light):**

- p95 search latency is consistent and predictable across test runs
- p99 search latency does not exceed p95 by more than 5-10x
- Performance is comparable to similar existing search features (use production thresholds as a reference point)
- No degradation to existing search performance
- Index size growth is reasonable
- Elasticsearch resources scale linearly with data size

## References

- [Metrics Catalog README](https://gitlab.com/gitlab-com/runbooks/-/blob/master/metrics-catalog/README.md) - Overview of the metrics catalog structure
- [Service Definitions](https://gitlab.com/gitlab-com/runbooks/-/tree/master/metrics-catalog/services) - Detailed SLI definitions for each service
- [SLI Library](https://gitlab.com/gitlab-com/runbooks/-/blob/master/metrics-catalog/gitlab-slis/library.libsonnet) - Reusable SLI patterns
- [Self-Service Performance Regression Testing](self-service-performance-regression-testing.md) - Related testing guidance
- [GitLab Performance Testing Tool Selection Guide](performance-tools.md) - Selection process for performance tools
- [Sitespeed Runway](https://gitlab.com/gitlab-org/quality/sitespeed-runway) - [SiteSpeed](https://www.sitespeed.io/) wrapper which measures frontend performance in browsers
- [GitLab Component Performance Tool](https://gitlab.com/gitlab-org/quality/component-performance-testing) -  [k6](https://grafana.com/docs/k6/latest/) wrapper which leverages containerization and automated testing to provide insights on individual component performance.
- [GitLab Performance Tool](https://gitlab.com/gitlab-org/quality/performance) - [k6](https://grafana.com/docs/k6/latest/) wrapper to provide performance testing of any GitLab instance.
