---
owning-stage: "~devops::tenant scale"
title: "Cells ADR 017: Container Registry Routing Service"
status: accepted
creation-date: "2025-07-29"
authors: [ "@ayufan" ]
toc_hide: true
---

{{< engineering/design-document-header >}}

## Context

The GitLab Container Registry needs to work within the Cells architecture, where projects and repositories are distributed across multiple Cells. The HTTP Router must route requests to the correct Cell based on project ownership while maintaining Docker client compatibility and enabling access to public repositories across Cells.

## Decision

1. **Separate HTTP Router Deployment**: Container Registry will use a dedicated HTTP Router deployment (`registry.gitlab.com`) separate from the main GitLab Rails HTTP Router (`gitlab.com`).

2. **Path-Based Routing**: Registry requests will be routed based on project paths extracted from URLs (`/v2/{project_path}/...`) and JWT authentication scopes (`repository:{project_path}:actions`).

3. **Target-Based Classification**: The Topology Service will support target-based classification (`web` vs `registry`) to return appropriate Cell addresses for different services.

4. **JWT Token Enhancement**: GitLab Rails will include `organization_id` in generated JWT tokens to enable routing of token validation requests (`/v2/` endpoint) to the correct Cell.

5. **Public Repository Access**: JWT authentication routing based on project path enables cross-Cell access to public repositories, even when user credentials are not valid on the target Cell.

6. **Legacy Cell Fallback**: Non-classified requests (like `/v2/_catalog`) will route to the first Cell using `FIRST_CELL` classification type.

## Pros

1. **Service Separation**: Dedicated HTTP Router eliminates complexity of conditional routing and improves performance by avoiding unnecessary rule processing.

2. **Cross-Cell Public Access**: Users can access public container repositories from any Cell without needing to know the hosting Cell location.

3. **Docker Compatibility**: Maintains full compatibility with existing Docker client workflows and authentication flows.

4. **Minimal Cell Changes**: No (or minimal) changes required to GitLab Rails or Container Registry components - all routing complexity is handled by the HTTP Router.

5. **Scalable Architecture**: Each service can maintain its own ruleset and scale independently.

6. **Fault Isolation**: Running Container Registry within each Cell provides fault isolation - if one Cell becomes unavailable, it only affects repositories hosted in that Cell, while other Cells continue to serve their repositories normally.

## Cons

1. **JWT Scope Dependency**: Routing depends on the first repository scope in JWT requests, which may be problematic since Docker Registry spec doesn't define scope order.

2. **Cross-Cell Limitations**: Repository linking and blob mounting across Cells will not work, limiting some advanced Docker features. However, due to how GitLab is used, this is acceptable tradeoff.

3. **Additional Infrastructure**: Requires separate HTTP Router deployment and maintenance.

## Implementation Requirements

### HTTP Router Changes

- Support for `target` parameter in classification requests
- Multiple classification matches processed in sequence
- Separate ruleset for Container Registry (`container_registry.json`)

### Topology Service Updates

- Extended `ClassifyRequest` interface with `target` field
- Support for `FIRST_CELL` classification type
- Registry-specific address configuration per Cell

### GitLab Rails Changes

- Include `organization_id` in JWT tokens for routing
- Minimal configuration changes for Cell-specific registry endpoints

### Alternative Solutions

If first-scope dependency proves problematic, alternative JWT routing approaches include:

1. Query parameter approach: `/jwt/auth?cell_id=X` (preferred)
2. Path prefix approach: `/c/cell-id/jwt/auth`
3. Path suffix approach: `/jwt/auth/cell/cell_id`

## Related Documents

- [Container Registry Routing Service design document](../container_registry_routing_service.md)
- [GitLab Cells Infrastructure Architecture](../infrastructure/_index.md)
- [HTTP Router Configuration](https://gitlab.com/gitlab-org/cells/http-router/-/blob/main/docs/config.md)
- [HTTP Router Rulesets](https://gitlab.com/gitlab-org/cells/http-router/-/tree/main/config/ruleset)
- [Topology Service Implementation](https://gitlab.com/gitlab-org/cells/topology-service)
- [Container Registry Auth Request Flow](https://gitlab.com/gitlab-org/container-registry/-/blob/master/docs/auth-request-flow.md)
