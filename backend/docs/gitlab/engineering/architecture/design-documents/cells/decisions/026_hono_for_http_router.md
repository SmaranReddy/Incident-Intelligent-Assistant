---
title: 'Cells ADR 026: Using Hono for HTTP Router Path-Based Routing'
owning-stage: "~devops::data stores"
group: cells-infrastructure
creation-date: "2026-02-20"
authors: ["@daveyleach"]
approvers: ["@samihiltunen"]
toc_hide: true
---

## Context

As part of implementing path-based routing in the HTTP Router, we anticipate a significant increase in the number of routing rules. The current JSON-based rule engine supports `path` type routing rules, but this approach leads to significant duplication and verbosity in route definitions.

For example, with the current approach, two path-based rules look like:

```json
{
  "id": "o_prefix",
  "action": "classify",
  "classify": {
    "type": "ORGANIZATION_PATH",
    "value": "${organization_path}"
  },
  "match": {
    "type": "path",
    "regex_value": "^/o/(?<organization_path>[^/]+)"
  }
},
{
  "id": "groups_prefix",
  "action": "classify",
  "classify": {
    "type": "GROUP_PATH",
    "value": "${group_path}"
  },
  "match": {
    "type": "path",
    "regex_value": "^/groups/(?<group_path>[^/]+)"
  }
}
```

Most of the definition is duplication - the only real differences are the regex value and capture group. A more compact representation could be:

```plaintext
o/:ORGANIZATION_PATH
groups/:GROUP_PATH
```

Additionally, the custom rule engine applies rules one after another in a linear manner, which becomes increasingly inefficient as the number of rules grows.

## Decision

Integrate [Hono](https://hono.dev) as the routing framework for path-based routing rules in the HTTP Router.

Hono is a popular, fast, and simple routing implementation that addresses our needs:

- **Simple path variable syntax**: Supports the simple path variable-based rules we want (e.g., `/o/:organization_path`) without requiring custom route matching logic
- **Performance**: Hono provides [performance benchmarks](https://hono.dev/docs/concepts/benchmarks#cloudflare-workers) for Cloudflare Workers, offering better performance than a other off-the-shelf solutions.
- **Proven and optimized**: Uses an off-the-shelf, well-tested router implementation rather than building and maintaining our own
- **Progressive migration**: We can integrate Hono alongside the existing routing engine and progressively migrate rules, falling through to the custom engine for rules not yet migrated

## Consequences

- **Simpler rule definitions**: Path-based routing rules will be significantly more concise and easier to maintain
- **Better performance**: Hono's optimized routing will handle the growing number of rules more efficiently than linear rule processing
- **Reduced maintenance burden**: We don't need to build and optimize our own router implementation
- **Gradual migration**: Existing routing rules continue to work while we progressively migrate to Hono-based definitions
- **Dependency addition**: We add Hono as a new dependency to the HTTP Router codebase

## Implementation

[The initial integration](https://gitlab.com/gitlab-org/cells/http-router/-/merge_requests/1039)  adds Hono to the HTTP Router and migrates the version endpoint as the first path-based rule. For routes not yet defined in Hono, requests fall through to the existing custom routing engine, allowing for progressive migration of routing rules.

## Alternatives

- **Custom router implementation**: Build our own optimized router with support for compact rule syntax. This would require significant development and testing effort, and likely wouldn't match the performance or maturity of established router libraries.
- **Other routing libraries**: Other JavaScript/TypeScript routing libraries exist (e.g., Express Router, Fastify), but Hono is specifically optimized for edge runtime environments like Cloudflare Workers and offers superior performance characteristics.
- **Keep existing rule format**: Continue with the verbose JSON-based rule definitions. This would make maintenance increasingly difficult as the number of path-based routing rules grows.
