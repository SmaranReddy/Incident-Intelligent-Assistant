---
title: "Separate Cloudflare Worker for /api/v4/jobs/request endpoint"
owning-stage: "~devops::tenant-scale"
group: cells-infrastructure
creation-date: "2026-02-03"
authors: ["@tkhandelwal3"]
approvers: ["@sxuereb"]
coach: "@sxuereb"
toc_hide: true
---

## Context

The `/api/v4/jobs/request` endpoint is used by GitLab Runner for [long-polling CI job requests](https://docs.gitlab.com/ci/runners/long_polling/), with connections staying open for up to 50 seconds. Multiple concurrent long-polling connections routed to the same Cloudflare Worker isolate at the same edge location cause the isolate to hit its memory limit before connections complete, resulting in `exceededMemory` exceptions.

While these exceptions don't impact users (clients retry automatically and the runtime spins up new isolates), they create significant noise in the HTTP Router error metrics. During [incident INC-6665](https://app.incident.io/gitlab/incidents/6665), this noise made it difficult to identify genuine issues, contributing to a 10-hour resolution time. The error ratio showed normal levels despite 502 errors occurring because the `/api/v4/jobs/request` errors masked the real problem.

Having accurate error rate metrics for HTTP Router alerting is essential for operational visibility. The current state makes it impossible to alert on high error ratios as the baseline noise from the jobs endpoint is too high.

Reference: [gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team#632](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/632)

## Decision

We will create a separate Cloudflare Worker deployment specifically to handle the `/api/v4/jobs/request` endpoint. This worker will be deployed alongside the main HTTP Router but with its own metrics and error tracking.

Implementation involves:

1. Add additional environments in the HTTP Router configuration (e.g., `gstg-jobs-requests`, `gprd-jobs-requests`).
2. Update the deployment pipeline to deploy the two new environments for the jobs-requests endpoint.
3. Add an explicit worker route for `/api/v4/jobs/request` in Cloudflare configuration. Cloudflare's route matching behavior is from most specific to general, so this route will take precedence over the general HTTP Router route.

Once the [Job Router](../../runner_job_router/_index.md) is implemented and all runners are switched away from long-polling, we can remove this separate worker and consolidate routing back to the main HTTP Router.

## Consequences

### Positive

- **Clean error metrics**: The main HTTP Router will have accurate error rates without noise from the jobs endpoint, enabling proper alerting for genuine issues.
- **Direct signal vs inferred**: We get clean, direct metrics for the HTTP Router rather than relying on edge error rate as a proxy for worker health.
- **Cleaner deprecation path**: Once the WebSocket migration is complete, we simply retire the dedicated worker without touching the main HTTP Router or its metrics pipeline.
- **Cost-neutral**: Traffic is split, not duplicated, so total requests and CPU time remain the same.
- **Foundation for future workers**: Lays groundwork for similar separations needed for [Container Registry routing](../container_registry_routing_service.md) and KAS in the future.
- **No fork maintenance**: Unlike alternatives, this doesn't require maintaining a forked exporter.

### Negative

- **Maintenance overhead**: Requires deploying and managing an additional worker.
- **Increased complexity**: Adds another component in the networking stack that every Engineer On-Call (EOC) needs to understand.

## Alternatives

### 1. Switch to WebSockets for the endpoint

Eliminate long-polling by switching to WebSocket connections. Once a Worker arranges to proxy through a WebSocket, [the Worker does not remain in use during proxying, avoiding memory accumulation](https://blog.cloudflare.com/workers-optimization-reduces-your-bill/#but-it-doesnt-stop-there).

Not immediately viable because:

- Already under development as part of [Job Router](../../runner_job_router/_index.md).
- Will take significant time to implement in production.
- Must still support long-polling for backward compatibility.

### 2. Reduce polling time

Reduce polling interval from 50 seconds to 5 seconds to prevent request accumulation that leads to `exceededMemory`.

Rejected because:

- Would increase connections from runners to the server by 10x.
- Likely to burn through the pending job queue SLO.

### 3. Use Tail Worker + Workers Analytics Engine

Use a Tail Worker to classify errors at the source, marking `exceededMemory` on `/api/v4/jobs/request` as expected noise while alerting only on real errors. Write classified data points to [Worker Analytics Engine](https://developers.cloudflare.com/analytics/analytics-engine/get-started/) and [query from Grafana](https://developers.cloudflare.com/analytics/analytics-engine/grafana/).

Rejected because:

- Estimated [cost of ~$25,000/month](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/632#note_3043315473) for writes alone (90 billion requests/month).
- Additional cost for the tail worker CPU time.
- Adds complexity with another worker component.

### 4. Fork cloudflare-exporter with path dimension

Add a new metric `cloudflare_zone_edge_errors_by_path` to the [lablabs/cloudflare-exporter](https://github.com/lablabs/cloudflare-exporter) with a path dimension, then use Prometheus metric relabel configs to filter out noise from the `/api/v4/jobs/request` endpoint.

The implementation would involve:

1. Adding a new metric with `zone`, `account`, `status`, `host`, and `path` labels
2. Using Prometheus rewrite rules to drop metrics for the jobs/request endpoint while collapsing cardinality for other paths:

```yaml
metric_relabel_configs:
  # Drop metrics for the jobs/request endpoint entirely (expected noise)
  - source_labels: [__name__, path]
    regex: 'cloudflare_zone_edge_errors_by_path;/api/v4/jobs/request'
    action: drop

  # For remaining metrics, replace path with empty string to collapse cardinality
  - source_labels: [__name__]
    regex: 'cloudflare_zone_edge_errors_by_path'
    target_label: path
    replacement: ''
    action: replace
```

An [upstream PR was created](https://github.com/lablabs/cloudflare-exporter/pull/193) to add this metric support.

Rejected because:

- Requires maintaining a fork of `lablabs/cloudflare-exporter` indefinitely as the upstream PR is unlikely to be accepted for this specific use case.
- The path dimension creates high cardinality even with normalization, which could impact Prometheus performance.
- Relies on edge error rate as a proxy for worker health, which could cause false positives (e.g., if KAS fails at the edge, we'd page for HTTP Router).
- Long-term maintenance burden for a shared component that other teams also rely on.
