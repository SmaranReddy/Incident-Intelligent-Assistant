---
title: 'GitLab Secrets Manager ADR 011: Use Streaming Audit Events for Customer Audit Ingress'
owning-stage: "~sec::software supply chain security"
toc_hide: true
---

## Context

OpenBao generates its own audit logging that is separate from the GitLab Rails
format. For hosted platforms such as .Com and GitLab Dedicated, GitLab must
collect audit events to be able to monitor overall system integrity. However,
on all platforms and especially for self-managed, tenants will want the
ability to collect audit logging for their own purposes. This document
proposes an approach to that.

OpenBao currently has three native audit devices:

1. `file`, which we use to stream to `stdout`,
2. `socket`, to stream via `tcp` or `unix` sockets, and
3. `syslog`, to write logs via syslog.

The audit subsystem is not presently pluggable, though this could be adjusted
in the future. Currently _any_ audit device may succeed; if no audit devices
succeed, the request will be denied. This makes streaming difficult: if http
streaming is the only audit device, we would potentially block the request
due to network issues unless we implement internal queuing. The upstream audit
[device API](https://github.com/openbao/openbao/blob/293e62223e8617baf33be916a7cc281f7dd1e1e1/audit/audit.go#L51-L52)
has storage support, though, so we can implement a queue _if_ we add the HTTP
device as optional. This will not be possible to use on standby nodes though,
so the queue must be non-persistent by default.

## Decision

A new `http` audit device will be proposed to upstream OpenBao. This will
support sending a POST response with the JSON audit log directly as the
body. This will have the ability to set custom headers to be sent along with
the request, mirroring [GitLab's webhook authorization model](https://docs.gitlab.com/user/project/integrations/webhooks/#custom-headers).

This audit device will be added to all environments. We'll use a custom
header, `X-GitLab-Token`, with a value provisioned from the environment
(either via environment variable or file). This will allow it to be rotated
as necessary.

This audit device will be considered to send a request to a new,
unauthenticated Rails endpoint which checks the header, correlates the tenant
to a Rails tenant, and emits the audit log via either the streaming audit
event API or the stored audit event API. The latter will used sparingly per
[discussion on volume](https://gitlab.com/gitlab-org/gitlab/-/issues/470461).

### Format Coercion

<details>

<summary><strong>Sample OpenBao audit event</strong></summary>

```json
{
  "request": {
    "operation": "read",
    "client_token": "hmac-sha256:<REDACTED>",
    "path": "group_123456/project_123456/<REDACTED>",
    "mount_type": "kv",
    "mount_class": "secret",
    "namespace": {
      "id": "root"
    },
    "mount_point": "group_123456/project_123456/<REDACTED>",
    "remote_address": "<REDACTED>"
  },
  "type": "response",
  "response": {
    "mount_point": "group_123456/project_123456/<REDACTED>",
    "mount_class": "secret",
    "data": {
      "metadata_cas_required": false,
      "versions": {
        "1": {
          "destroyed": false,
          "created_time": "hmac-sha256:<REDACTED>",
          "deletion_time": "hmac-sha256:<REDACTED>"
        }
      },
      "delete_version_after": "hmac-sha256:<REDACTED>",
      "current_version": 1,
      "created_time": "hmac-sha256:<REDACTED>",
      "cas_required": false,
      "current_metadata_version": 0,
      "max_versions": 0,
      "custom_metadata": {
        "branch": "hmac-sha256:<REDACTED>",
        "description": "hmac-sha256:<REDACTED>",
        "environment": "hmac-sha256:<REDACTED>"
      },
      "updated_time": "hmac-sha256:<REDACTED>",
      "oldest_version": 0
    },
    "mount_type": "kv",
  },
  "auth": {
    "token_issue_time": "2010-10-10T10:10:10Z",
    "policies": [
      "default",
      "project_123456/users/direct/group_123456",
      "project_123456/users/direct/user_123456",
      "project_123456/users/roles/123456"
    ],
    "metadata": {
      "role": "project_123456"
    },
    "client_token": "hmac-sha256:<REDACTED>",
    "token_policies": [
      "default",
      "project_123456/users/direct/group_123456",
      "project_123456/users/direct/user_123456",
      "project_123456/users/roles/123456"
    ],
    "policy_results": {
      "allowed": true,
      "granting_policies": [
        {
          "name": "project_123456/users/roles/123456",
          "namespace_id": "root",
          "type": "acl"
        }
      ]
    },
    "token_type": "service",
    "entity_id": "<REDACTED>",
    "display_name": "group_123456-user_jwt-123456",
    "token_ttl": 900
  }
}
```

</details>

See [GitLab documentation](https://docs.gitlab.com/user/compliance/audit_event_schema/)
for equivalent GitLab audit event schema.

This can be computed from the above as follows:

1. `author_id` can be computed from the direct token policies (`user_<id>`
   policy under `auth.policies`).
2. `created_at` can be the audit request receive time and is not natively in
   the OpenBao format.
3. `details` can be the raw OpenBao audit event.
4. `entity_id` and `entity_type` can be computed from the `request.path`,
   inferring the project or group identifier, with `target_id` and
   `target_type` matching.
5. `ip_address` can be computed from the `request.remote_address` field.
6. `event_type` can be computed based on the combination of `request.path`
   and `request.operation`; distinct paths+operation combinations will be
   different events.

We can positively and negatively filter a mix of paths:

1. Certain operations will match to a unique `event_type` such as
   `update_secret`, `read_secret`, &c.
2. Certain operations will not need to be audited, such as listing secrets.
3. Unmatched operations could have a `raw_secret_operation` to be later
   classified accordingly by operators.

This will ensure a reasonable amount of emitted audit events.

### Table of Audit Events

| Path | Operation | Event Type | Stored |
| :--- | :-------- | :--------- | :----- |
| `*/project_<id>/secrets/kv/data/explicit/*` | `read` | `read_project_secret` | No |
| `*/project_<id>/secrets/kv/data/explicit/*` | `update` | `update_project_secret` | Yes |
| `*/project_<id>/secrets/kv/data/explicit/*` | `create` | `create_project_secret` | Yes |
| `*/project_<id>/secrets/kv/data/explicit/*` | `delete` | `delete_project_secret` | Yes |
| `*/group_<id>/secrets/kv/data/explicit/*` | `read` | `read_group_secret` | No |
| `*/group_<id>/secrets/kv/data/explicit/*` | `update` | `update_group_secret` | Yes |
| `*/group_<id>/secrets/kv/data/explicit/*` | `create` | `create_group_secret` | Yes |
| `*/group_<id>/secrets/kv/data/explicit/*` | `delete` | `delete_group_secret` | Yes |

## Consequences

Users _should_ have a way of capturing audit logs written to `/dev/stdout` for
fully reliable operation. These will be cross-tenant and thus should be
somewhat privileged (e.g., limited only to the instance operator) despite
still being HMAC'd to remove sensitive data. Loss of data or out-of-order
audit events may occur due to network outages. Audit device metrics should
be monitored to ensure audit device remains operational and does not consume
endless memory.
