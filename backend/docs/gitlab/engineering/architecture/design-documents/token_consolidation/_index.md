---
# This is the title of your design document. Keep it short, simple, and descriptive. A
# good title can help communicate what the design document is and should be considered
# as part of any review.
title: Token Consolidation
status: proposed
creation-date: "2025-02-23"
authors: [ "@ifarkas" ]
coaches: [ "@grzesiek" ]
dris: [ "@hsutor", "@adil.farrukh" ]
owning-stage: "~devops::software_supply_chain_security"
participating-stages: []
# Hides this page in the left sidebar. Recommended so we don't pollute it.
toc_hide: true
---

<!-- Design Documents often contain forward-looking statements -->
<!-- vale gitlab.FutureTense = NO -->

<!-- This renders the design document header on the detail page, so don't remove it-->
{{< engineering/design-document-header >}}

## Summary

Token consolidation aims to transform our fragmented token landscape into a
unified framework, bringing consistent security and token management features
across all token types, while addressing critical security and operational
risks.

By implementing a standardized token framework, we will enhance security, reduce
maintenance overhead, and improve user experience.

The proposed approach consists of three phases:

1. Building the core framework and architecture.
2. Migrating existing token types to the unified framework.
3. Deprecating legacy user flows.

## Motivation

GitLab offers more than 20 different token types.
Rather than leveraging and adapting existing token types, teams have often
created new ones. This lead to a fragmented token landscape with inconsistent
security, and management features, resulting in security gaps, and overlapping
functionality, and increased complexity.

Despite our established token standards, many essential security features are
either missing, or inconsistently implemented across different token types,
including:

- encrypted storage
- support for the shortest reasonable expiration time
- fine-grained access controls
- logging, auditing, and visibility into token usage
- configurable token prefix
- rotation
- automatic reuse detection
- retention controls
- routability for Cells and Organizations
- expiration

This inconsistency across token implementations has resulted in greater system
complexity, making it difficult to enforce uniform security standards across the
organization.

Maintaining a large number of token types significantly increases the long-term
maintenance burden.
When security updates or new features are required, each token type must be
individually modified, tested, and maintained — resulting in ongoing maintenance
challenges that slow down development and increase the risk of inconsistencies,
security gaps, and potential vulnerabilities.

### Goals

- Establish a unified token as the foundational building block for all token
  types
- Standardize authentication and authorization model around tokens
- Enhance security by enabling consistent features across all token types
- Improve maintainability by eliminating redundant token implementations
- Improve user experience by simplifying token usage and management

### Non-Goals

- Define a Token Exchange Service

## Proposal

Token consolidation integrates with the new auth architecture, which introduces
a layered model: L0 (realm), L1 (region), and L2 (deployment unit). The unified
token is a signed JWT minted by the L2 Secure Token Service (STS).

### Token Format

Benefits of using a signed JWT:

- Built-in support for claims and other metadata
- Digital signature ensures authenticity and prevents tampering
- Self-contained authentication and authorization data
- Routing metadata can be embedded directly in the token

### Storage

Token data is stored across two database systems:

- IAM database (L1 and L2): Token hashes for authentication and validation,
  replicated to L1 via L0.
- `gitlab-rails` database (L2): Token metadata including resource
  associations, name, and usage tracking.
- L2 is the authoritative data source.

### Service Integration

Within GATE, the IAM service delegates to STS for:

- Minting tokens
- Token revocation

### Revocation

Revocation lists are maintained at L1/L2, with OAuth service coordination at L0:

- L2: Immediate revocation in the authoritative deployment unit database
- L1: Revocation list replicated to the regional distributed database
- L0: OAuth service handles realm-wide coordination

### Expiration

Non-expiring tokens will be deprecated. The unified token will support a range
of expiration options:

- Short-lived tokens like OAuth access tokens (2-hour expiry)
- Long-lived tokens like Personal access tokens (up to 1 year)

Existing non-expiring tokens will require migration to use expiration. This
enhances security by ensuring all tokens have a bounded lifetime.

### Permissions

The token's permissions are set based on its `scope` claim, ensuring flexibility
while maintaining least-privilege access when required. Scopes can be generic,
similar to the current `api` scope, or fine-grained for specific use cases
(e.g., project-level access):

```json
"scope": {
  "read_project": ["gid://gitlab/Project/42"]
}
```

### Cells Compatibility

The unified token includes routing metadata to enable efficient routing to the
owning Cell without additional lookup overhead. This metadata is embedded
directly in the JWT claims, allowing the routing layer to direct requests to the
correct Cell without requiring a database lookup.

### Consolidation of Existing Tokens

A centralized interface for managing tokens will be introduced, providing a
unified way to create, manage, and monitor tokens.

Existing tokens will be migrated to use the unified token as their backend,
standardizing their implementation. The migration path for each token type will
be evaluated independently, taking into account feature parity, security
implications, and user impact. Once migrated, their management will be available
through the unified token management interface.

As user workflows transition to the unified token model, original tokens and
legacy token flows can be deprecated and removed from the UI. This phased
approach will minimize disruption while ensuring seamless transition to the new
token framework.

## Alternative Solutions

- Do nothing: Existing issues remain unresolved; fragmented token landscape
  persists.
