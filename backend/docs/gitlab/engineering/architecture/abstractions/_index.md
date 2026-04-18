---
title: "Key Abstractions"
---

## Overview

Key Abstractions (also known as Key Primitives) are foundational technologies,
patterns, and components that have been approved for use across GitLab's
engineering organization. Once a key abstraction is approved, engineers must use
it when building functionality that requires that capability.

## Governance

- **Approval Process**: Adding a new key abstraction requires VP approval
- **Ownership**: Each approved key abstraction must have a designated owner
  within the company
- **Compliance**: If a technology or abstraction is in the approved catalog,
  teams must use it rather than introducing alternatives
- **Architecture Board**: The Architecture Board reviews and approves items for
  the catalog of key abstractions

If a needed capability is not in the catalog, teams should discuss it with the
Architecture Board.

## Architecture Board

The Architecture Board is a lightweight, informally structured group that
reviews and stewards the catalog of key abstractions. This group ensures that
architectural decisions are made collaboratively while maintaining technical
consistency across GitLab's engineering organization.

The board operates using a lightweight, informal approach that prioritizes
collaboration and rapid iteration while ensuring proper oversight of
foundational technology decisions. This represents the starting point for the
Architecture Board. As the organization's needs evolve, the board's structure,
scope, and processes may be expanded and formalized.

### Composition

- VPs appoint Architecture Board members
- We initially started with appointing all Distinguished Engineer+ as members.
- Membership is managed through CODEOWNERS for the `architecture/abstractions`
  directory
- Only Architecture Board members can approve and merge changes to the key
  abstractions catalog

### Responsibilities

The Architecture Board reviews and approves:

- Proposals for new key abstractions
- Promotion of candidate abstractions to graduated (approved) status
- Updates to existing key abstraction documentation
- Deprecation or removal of abstractions from the catalog

## Maturity States

Key abstractions can be in one of the following states:

- **Candidate**: Under consideration, requires Architecture Board review
- **Graduated**: Approved for use, requires an assigned owner

All graduated key abstractions must have a designated owner.

## Approved Key Abstractions

| Abstraction | Category | Description | Owner |
|-------------|----------|-------------|-------|
| [**PostgreSQL**](postgresql/) | Data & Storage | OLTP Database | TBD |

## Candidate Key Abstractions

| Abstraction | Category | Description | Notes |
|-------------|----------|-------------|-------|
| **Siphon** | Data & Storage | Data ingestion and transformation | |
| **ClickHouse** | Data & Storage | Analytics / OLAP queries | |
| **Redis** | Data & Storage | Cache and global locking | |
| **Object Storage** | Data & Storage | Binary and large file storage | |
| **Elasticsearch/OpenSearch** | Data & Storage | Natural language search | |
| **GitLab Zoekt** | Data & Storage | Code search | |
| [**NATS**](candidate/nats/) | Messaging & Processing | Message bus for high-throughput event streaming | Use for very high throughput and durable messages. |
| [**Sidekiq**](candidate/sidekiq/) | Messaging & Processing | Background processing | Default choice for background jobs. |
| **Active Context** | Search & AI | Semantic search (embeddings) | |
| **Ruby, Go, Python, Rust** | Languages & Frameworks | Backend languages with LabKit support | |
| **VueJS, JavaScript, TypeScript** | Languages & Frameworks | Frontend technologies | |
| **gRPC, REST** | Languages & Frameworks | Communication protocols | |
| **RAFT** | Distributed Systems | Consensus algorithm | |
| **GitLab Zoekt** | Distributed Systems | Pattern for stateful services | |
| **Secrets Storage** | Security | Secure secrets management | Includes proper database encryption (avoid initialization vector reuse) |
| **JSON Schema** | Security | Data validation | |
| **OpenMetrics** | Observability | Metrics collection | Prometheus compatible |
| **OpenTelemetry** | Observability | Distributed tracing | Configured via LabKit |
| **Rate Limiter** | Architecture Patterns | Request rate limiting | Implemented through LabKit where possible |
| **Circuit Breaker** | Architecture Patterns | Fault tolerance pattern | Implemented through LabKit where possible |
| **UBI-9** | Infrastructure | Supported Cloud Native GitLab images | |

## Services we don't want to use

The following technologies should **not** be used:

- **Consul**
- **Kafka** - use [NATS](candidate/nats/) or [Sidekiq](candidate/sidekiq/) instead

## Contributing

To propose a new key abstraction or promote a candidate to graduated status:

1. Prepare documentation including:
   - Clear description and use cases
   - Why it should be a key abstraction
   - Proposed owner
   - Integration considerations
2. Present to the Architecture Board for review
3. Obtain approval from VP of Data Engineering and VP of Infrastructure

## Related Resources

- [Architecture Workflow](/handbook/engineering/architecture/workflow/)
- [Scalability Practice](/handbook/engineering/architecture/practice/scalability/)
