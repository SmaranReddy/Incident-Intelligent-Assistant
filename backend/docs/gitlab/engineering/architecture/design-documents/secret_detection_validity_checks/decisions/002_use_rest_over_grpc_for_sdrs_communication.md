---
title: "GitLab Secret Detection Validity Checks ADR 002: Use REST over gRPC for SDRS communication"
---

## Summary

We chose to implement a REST API rather than gRPC for communication between GitLab instances and the Secret Detection Response Service (SDRS). This decision prioritizes using the right tool for the job based on SDRS's specific requirements.

## Context

When designing the communication protocol between GitLab instances and the Secret Detection Response Service (SDRS) for token validity checking, we needed to choose between REST and gRPC.

The key considerations were:

- Payload size and complexity of token validation requests
- Need for streaming capabilities
- Implementation complexity and team familiarity
- Integration requirements with partner services
- Performance optimization needs

## Decision

We will implement a **REST API** for SDRS communication rather than gRPC.

## Consequences

**Benefits:**

- Simpler implementation and integration
- Standard HTTP tooling and debugging capabilities available
- Easier partner integration if needed in the future

**Drawbacks:**

- Does not provide gRPC's advanced features like built-in streaming or binary serialization
- Slightly less efficient for high-volume scenarios (though not relevant for our use case)

## Alternatives Considered

**gRPC approach**: While gRPC offers streaming capabilities and performance optimizations, these benefits are not significant for our use case. The service will only need to accept individual tokens or small arrays of tokens, so we won't benefit from gRPC's streaming capabilities (which are more valuable in cases like sending git diffs). Given the small payload sizes, request optimization isn't a critical factor, making REST's simplicity a better fit for our needs.
