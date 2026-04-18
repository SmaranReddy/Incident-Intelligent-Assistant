---
title: "Duo Workflow ADR 005: Deprecate checkpoint field in favor of duo_messages"
owning-stage: "~devops::ai_powered"
toc_hide: true
---

## Context

When Duo Workflow was initially developed, the GraphQL API exposed raw checkpoint data from LangGraph directly to clients through a `checkpoint` field. This approach was taken as a shortcut to enable rapid iteration and get the feature working quickly. The checkpoint data is an internal LangGraph representation of workflow state stored as nested JSON structures.

Clients (Web UI, IDE extensions, CLI) consumed this raw checkpoint data to:

- Display chat history to users
- Show workflow status and progress
- Render tool calls and their results
- Present error states

The checkpoint field was available on both:

- `duoWorkflowWorkflows` query (for fetching complete workflow history)
- `duoWorkflowEvents` subscription (for streaming real-time updates)

## Problems

The raw checkpoint data exposed internal LangGraph state directly to clients, creating three main issues:

1. **Developer experience**: Deeply nested JSON with no clear schema made it very difficult for developers to work with our APIs ([original issue](https://gitlab.com/gitlab-org/gitlab/-/work_items/535898#note_2447486353)). Most checkpoint data was internal state irrelevant to the UI and required to be parsed and then access with no schema to guaranteed properties would be present. This lead to hard to maintain and fragile client side code.

2. **Performance**: A simple workflow transferred **3 MiB** of data with individual checkpoint updates reaching **150 KiB**. This caused significant egress costs and database growth in the `p_duo_workflows_checkpoints` table.

3. **Architectural coupling**: Clients depended on LangGraph internals with no abstraction layer, making it difficult to evolve the backend without breaking frontend code.

## Decision

Introduce a new `duo_messages` GraphQL field that provides a clean, structured API for workflow chat history. The existing `checkpoint` field will be marked as deprecated but remain available for backward compatibility and debugging.

The `duo_messages` field sources data from the `ui_chat_log` that Duo Workflow Service already maintains separately from LangGraph state, exposing only UI-relevant data with a well-defined schema. This field is available on both `duoWorkflowWorkflows` (for complete history) and `duoWorkflowEvents` (for streaming updates).

See the implementation details in the referenced MRs below.

## Benefits

- **Reduced payload sizes**: Only UI-relevant data is transmitted, significantly reducing bandwidth and egress costs
- **Clear schema**: Well-defined types with explicit fields make frontend development straightforward and provide type safety
- **Clean separation**: Clear boundary between persistence layer (checkpoints) and presentation layer (duo_messages), allowing backend optimization without affecting clients
- **Single implementation**: Checkpoint parsing logic lives in one place (backend) instead of duplicated across clients (Web, IDE, CLI)
- **Graceful transition** Current approach allows clients to migrate at their own pace while we validate duo_messages meets all needs

## Alternatives Considered

### Client-side checkpoint parsing library

**Rejected** because:

- Doesn't solve payload size or performance issues
- Doesn't provide a stable API contract
- Duplicates parsing logic across multiple environments (Web, IDE, CLI, etc)

### Separate streaming API for incremental updates

**Deferred** for future iteration because:

- Can be built on top of duo_messages without breaking changes
- Requires more complex client-side state management
- Current approach provides immediate benefits while keeping this option open
- As noted in the decision, we can "iterate on the schema to create a much nicer to consume API for subscriptions"

## Implementation References

- [Original issue: Simplify DAP api to be able to request ui_chat_log](https://gitlab.com/gitlab-org/gitlab/-/work_items/535898)
- [Backend implementation: Expose duo_messages field](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/214015)
- [GraphQL schema deprecation](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/213959)
- [Web UI migration to duo_messages](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/214016)
- [IDE extension migration](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/214017)
- [Related: Reduce Duo Workflow Service checkpoint payload sizes](https://gitlab.com/gitlab-org/gitlab/-/work_items/571913)
