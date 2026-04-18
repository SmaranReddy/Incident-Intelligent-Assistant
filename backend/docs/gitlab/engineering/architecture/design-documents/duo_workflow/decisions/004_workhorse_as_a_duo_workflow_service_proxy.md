---
title: "Duo Workflow ADR 004: Use Workhorse as a Duo Workflow Service proxy"
owning-stage: "~devops::ai_powered"
toc_hide: true
---

## Context

The Duo Workflow Service is a gRPC server that runs in Python, and orchestrates multi-step LLM interactions via LangGraph. Duo Workflow Executor is a client that runs in a local environment that dynamically sends additional information about the environment to Duo Workflow Service and processes the response. Currently:

- GitLab Rails generates a short-lived token for Duo Workflow Executor to communicate with Duo Workflow Service
- Duo Workflow Executor communicates with Duo Workflow Service directly to send additional information and get the response
- Duo Workflow Executor acts as a proxy when GitLab API requests should be performed: for getting GitLab related information (projects, merge requests), but also for internal purposes to save the current state of a workflow

![Current Architecture](/images/engineering/architecture/design-documents/duo_workflow/diagrams/duo-workflow-workhorse-current-architecture.png)

## Problems

- The requests are proxied by Duo Workflow Executor, which causes additional delays and restricts us from sending sensitive information from GitLab Rails to Duo Workflow Service (for example, API keys for self-hosted models)
- The environment that is running Duo Workflow Executor must have access to an additional location (Duo Workflow Service) which creates a burden on individual users to have their system correctly configured for access rather than just an admin if the connection runs via the Workhorse
- The current solution requires the executor to communicate with GRPC, we can add WebSockets as a Fallback, but GRPC might still be preferred for communication between deployed services. More information about gRPC vs Websockets in [this ADR](002_add_websocket_support)

## Decision

Use [Workhorse](https://docs.gitlab.com/development/workhorse/) as a proxy for Duo Workflow Service. GitLab Workhorse is a smart reverse proxy for GitLab intended to handle resource-intensive and long-running requests. It can provide a Websocket endpoint to handle communication between Duo Workflow Executor, Duo Workflow Service, and GitLab Rails:

- Duo Workflow Executor opens a Websocket connection with Workhorse when a workflow starts and sends the GitLab private token for authentication and authorization
- Workhorse authorizes the connection by sending an HTTP request to Rails
- Workhorse initiates a gRPC connection with Duo Workflow Service using the auth information provided by GitLab Rails and starts bidirectional communication
- When a gRPC action is received:
  - If it's a run-http request action, Workhorse performs the request directly to Rails using the auth information provided on Websocket initialization
  - Otherwise, Workhorse serializes the action as a JSON and propagates it to Duo Workflow Executor. When the response is received, Workhorse serializes the JSON to gRPC binary data and sends it to Duo Workflow Service

![Proposed Architecture](/images/engineering/architecture/design-documents/duo_workflow/diagrams/duo-workflow-workhorse-proposed-architecture.png)

## PoCs

The MRs demonstrate communication between Rails, Workhorse, Duo Workflow Service and IDE without issuing a Duo Workflow Service token:

- [Rails + Workhorse](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/191755):
  - Adds a handler for ws route `api/v4/ai/duo_workflows/workflows/<id>/connect` in Workhorse
  - Adds an authorization handler for ``api/v4/ai/duo_workflows/workflows/<id>/connect`` in Rails that passes to Workhorse the information related to Duo Workflow Service
  - In Workhorse, it implements the gRPC communication with Duo Workflow Service and Websockets handling with an executor
- [GitLab LSP](https://gitlab.com/gitlab-org/editor-extensions/gitlab-lsp/-/merge_requests/1720)
  - Initializes a WebSocket connection with Workhorse using the GitLab API private token
  - Processes the Workflow actions in Node Executor

## Pros

- The short-lived Duo Workflow token is no longer necessary; a general GitLab personal access token can be used
- Connecting to an additional service via gRPC is no longer necessary; Duo Workflow Executor connects to a GitLab instance using Websockets
- Duo Workflow Executor doesn't proxy GitLab API requests:
  - API requests are executed significantly faster
  - Internal information, like state management, is hidden from a client
  - Sensitive information can be exchanged between Rails and Duo Workflow Service
- Duo Workflow service continues to be a gRPC server only, without providing additional complexity like Websockets connection handling
- Workhorse can act as a deployed Duo Workflow Executor for read-only operations:
  - When a workflow only operates with read-only GitLab API requests, for example, it can be an Agentic Chat in Web UI
  - In this case, a client doesn't need to have any executor logic; it can just react to the results sent by Workhorse via Websockets

## Cons

- Performance-related risks. With part of the executor logic running in a deployed environment, the performance issues will be both more visible and more impactful
- The executor logic is now leaked into a third component (Golang Duo Workflow Executor, Node Duo Workflow Executor in GitLab LSP and now part of it in Workhorse), but on the other hand, it simplifies the implementations because they no longer need to communicate with Duo Workflow Service directly
- The API requests performed by Workhorse use the passed private token for auth. If we want to have a composite identify for these calls, we'll have to generate an OAuth token and pass it to Workhorse. It's possible, but it adds an additional responsibility and unnecessary complexity
- Since there needs to be a stateful connection all the way from the Executor to the Duo Workflow Service we would need to drop the gRPC connection every time the websocket connection drops (and vice versa). Without this the websocket reconnects might arrive at the wrong workhorse process which doesn't have the persistent gRPC connection available. Depending on how reliable websockets (without reconnection) are it may mean that workflows get dropped mid way through more often. It also doubles the chance of a dropped connection dropping a workflow. Initially we will always gracefully drop the gRPC connection if websockets drops, the clients will have to resume the workflow on reconnection. If it proves to disconnect too frequently we'll explore using Redis as a reliable transport across any Workhorse server per https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/13685#note_2515723211
