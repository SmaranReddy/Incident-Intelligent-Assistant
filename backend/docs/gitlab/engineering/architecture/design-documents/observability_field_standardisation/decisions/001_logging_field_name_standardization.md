---
title: 'Logging ADR 001: Standardized application field naming strategy'
owning-stage: "~devops::developer experience"
description: 'Logging ADR 001: Standardized application field naming strategy'
toc_hide: true
---

## Context

GitLab's logging infrastructure has inconsistent field organization across services. Fields are scattered at the top level, within `meta.*`, `extra.*`, and message bodies. Different services use different conventions (for example, `remote_ip` vs `meta.remote_ip`), making it difficult for developers and operators to discover and filter logs.

This ADR focuses on application logging. HTTP access logging is out of scope.

## Decision

Adopt a field naming strategy inspired by the [OpenTelemetry Log Data Model](https://opentelemetry.io/docs/specs/otel/logs/data-model/), providing an opinionated approach to log structure across GitLab services. Where pragmatic, we diverge from OTel conventions (for example, keeping `meta` rather than `resource`, using snake_case field names, and keeping human-readable timestamps).

### Structure

```json
{
  "time": "2026-01-27T15:44:30.446Z",
  "level": "INFO",
  "body": "Member created successfully",
  "attributes": {
    "endpoint_id": "Projects::MembersController#create",
    "status_code": 201,
    "method": "POST",
    "duration_s": 0.234,
    "action": "create_member",
    "target_user_id": 171554,
    "role": "developer"
  },
  "meta": {
    "gl_user_id": 89012,
    "gl_project_path": "gitlab-org/gitlab",
    "caller_id": "Projects::MembersController#create",
    "feature_category": "member_management"
  }
}
```

### Field placement

| Field | Purpose | Examples |
|-------|---------|----------|
| `time` | ISO 8601 timestamp when the event occurred | `2026-01-27T15:44:30.446Z` |
| `level` | Log level | `DEBUG`, `INFO`, `WARN`, `ERROR`, `FATAL` |
| `body` | Log message, either a string or structured data | `"Member created successfully"` |
| `attributes` | Event-specific information that varies per log event | `endpoint_id`, `status_code`, `duration_s` |
| `meta` | Application context that propagates between services | `gl_user_id`, `gl_project_path`, `feature_category` |

When deciding where a field belongs:

- **`meta`**: Context that identifies *where* the log originated. These fields propagate across service boundaries and remain constant for a request (for example, user identity, project, feature category).
- **`attributes`**: Event-specific data about *what* happened. These fields vary per log event and describe the action or outcome (for example, status codes, durations, operation-specific IDs).
- **`body`**: The log message. Can be a simple string or structured data. Structured data in `body` is queryable in some stores (for example, ClickHouse with `JSONExtract`) and is flattened for indexing in others (for example, Elasticsearch).

Fields must appear in only one location. LabKit enforces that only known fields are permitted in `meta` and `attributes`, preventing duplication.

### Developer experience

LabKit handles field placement internally. Developers do not need to manage the structure directly:

```ruby
logger.info("Member created successfully", action: "create_member", gl_user_id: 171554)
# => Outputs structured JSON
```

## Migration

See [Safe Migrations](../#safe-migrations) in the parent design document for guidance on migrating existing logs without breaking dashboards and automations.

### Phased rollout

1. **Existing services**: Retain `meta` while adopting `attributes` and `body` structure incrementally.
1. **New services**: Use `meta`, `attributes`, and `body` from day one.

Kibana does not work well with nested fields. The Observability team must flatten nested structures in the ingest pipeline to maintain query performance.

## Consequences

- Consistent field organization across services improves log discoverability and filtering
- Clear developer guidance reduces confusion about field placement
- LabKit SDKs enforce the naming strategy through validation rules
- Requires migration of existing logs without breaking dashboards and automations
- Implementation effort needed across multiple SDKs (Ruby, Go, Rust, Python)

## Alternatives

### Flat structure with prefixes

All fields at top level with prefixes (for example, `meta_user_id`, `attr_action`). Simpler initially but leads to field name pollution and is harder to extend.

### Do nothing

Continue with the current inconsistent approach. Avoids implementation effort but perpetuates developer confusion and operational friction.
