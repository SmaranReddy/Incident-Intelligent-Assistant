---
title: "Tutorial: Add observability metrics to a CI-based analyzer"
---

Observability metrics help you understand how CI-based security analyzers perform in production.
Metrics like scan duration, exit codes, and findings count, flow through the security report
artifact to GitLab's internal event tracking system.

For an implementation example, see
[Secret Detection metrics](/handbook/engineering/development/sec/secure/secret-detection/metrics/#ci-based-analyzer-observability-metrics).

## Architecture

```mermaid
flowchart LR
    A[CI Analyzer] --> B[Security Report]
    B --> C[Post-analyzer]
    C --> D[Internal Event Tracking]
    D --> E[Snowflake]
```

1. The CI analyzer collects metrics during the scan and writes them to the security report artifact.
1. The post-analyzer processes the report and extracts observability events.
1. Allowed events are sent to the internal event tracking system.
1. Events are stored in Snowflake for analysis and dashboards.

## Design considerations

The observability system uses a decentralized events pattern. Each analyzer defines its own
event structure rather than using a shared registry.

This design provides:

- **Independent development**: Add or modify metrics without coordinating releases
  across repositories.
- **Version tolerance**: Older post-analyzers skip unknown events gracefully instead of failing.
- **Faster iteration**: Test new metrics locally without infrastructure updates.
- **Decentralized ownership**: Each analyzer team owns their event definitions.

The trade-offs are:

- No single location lists all possible events. Use the monolith event definitions in
  [`config/events/`](https://gitlab.com/gitlab-org/gitlab/-/tree/v18.6.4-ee/config/events) or
  [`ee/config/events/`](https://gitlab.com/gitlab-org/gitlab/-/tree/v18.6.4-ee/ee/config/events)
  as the source of truth.
- New events aren't processed until the post-analyzer and backend are updated.

## Prerequisites

The observability feature requires:

- [Report](https://gitlab.com/gitlab-org/security-products/analyzers/report) package v7 or later
- [Command](https://gitlab.com/gitlab-org/security-products/analyzers/command) package v5 or later

## Event design guidance

- Create multiple events rather than using many extra properties. JSON columns with extra
  properties are slower to query.
- To join related events, include a UUID in a column value (for example, `property`).
- Before adding a variable field, check with maintainers of the
  [integration-test](https://gitlab.com/gitlab-org/security-products/tests/integration-test)
  project to ensure it will be handled correctly.

## Register the internal event

Events require a corresponding definition in the GitLab monolith and must be added to the
backend allow list. Only allowed events are processed.

1. Create event definition in `config/events/` or `ee/config/events/`.
1. Update the allow list in
   [ProcessScanEventsService](https://gitlab.com/gitlab-org/gitlab/-/blob/master/ee/app/services/security/process_scan_events_service.rb#L9).
1. Add an rspec test using actual report output from the analyzer.

For more information, see
[Quick start for internal event tracking](https://docs.gitlab.com/ee/development/internal_analytics/internal_event_instrumentation/quick_start.html).

## Validation

After deployment:

1. Query Snowflake to verify events are collected correctly.
1. Check Sentry for exceptions during report processing.
1. Request dashboard creation from the Analytics Instrumentation team.
