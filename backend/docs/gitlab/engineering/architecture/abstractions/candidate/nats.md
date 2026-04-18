---
title: "NATS - Message Bus"
---

## Status

**Candidate** - Under consideration for approval

## Owner

[To be assigned]

## Description

NATS is a high-performance message bus for high-throughput event streaming and real-time communication between distributed systems.

## Why This Should Be a Key Abstraction

NATS is being considered as the standard message bus for GitLab because it provides:

1. **High Throughput**: Handles millions of messages per second
2. **Low Latency**: Optimized for real-time event streaming
3. **Cloud Native**: Well-suited for distributed architectures

## Use Cases

Use NATS for:

- Durable message bus
- Very high throughput event streaming
- Real-time data processing pipelines
- Event-driven architectures requiring low latency

## Do Not Use For

**Do not use NATS for:**

- Core GitLab functionality that must work in all deployment environments
- Standard background processing that Sidekiq can handle

## Integration Considerations

### Self-Managed Limitations

NATS is not suitable for self-managed foundational (core) features because:

- Not all self-managed instances may have NATS deployed (yet)
- Adds operational complexity for customers

## Related Key Abstractions

- [**Sidekiq**](sidekiq/) - Alternative for standard background processing
- **PostgreSQL** - May be used alongside NATS for persistent data storage
- **Redis** - Used by Sidekiq; may complement NATS for caching
