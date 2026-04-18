---
title: "Engineering Overview"
description: "Technical and engineering overview of the Code Suggestions feature"

---

## Introduction

Welcome to the technical overview of GitLab's Code Suggestions, a feature designed to enhance the coding experience by integrating advanced AI technologies directly within your development environment. This page serves as your guide to understanding the architecture and interactions behind our innovative Code Suggestions feature, which significantly streamlines coding processes through intelligent completions and generative coding capabilities.

At its core, Code Suggestions operates through a sophisticated workflow involving multiple components such as IDE extensions, the Language Server, GitLab Workhorse, and our AI Gateway, all culminating in providing you with real-time, context-aware coding suggestions. From simple code completions that speed up your typing tasks to complex code generations that craft entire code blocks, our system is designed to support a wide array of coding activities and enhance productivity.

Below, we detail each component's role in this ecosystem, describe the flow of data through our system, and explain how different types of coding interactions are handled to provide both quick suggestions and detailed code generation.

## Code Suggestions Technical Overview

Code Suggestions generally follow the sequence described in the following diagram.

```mermaid
sequenceDiagram
    actor USR as  User
    participant IDE
    participant EXT as IDE Extension
    participant LS as Language Server
    participant GLW as GitLab Workhorse
    participant GLR as GitLab Rails
    participant AIGW as AI Gateway
    participant LLM as Large Language Model

    USR->>IDE: types: "def add(a, b)"
    IDE->>EXT: notify about document change def add(a, b)
    EXT->>LS: register document change def add(a, b)
    LS->>GLW: sends code suggestion requests
    GLW->>GLR: proxies code suggestion request
    GLR->>GLW: enriches code suggestion request
    GLW->>AIGW: enriched code suggestion request
    AIGW->>LLM:  enriched code suggestion request
    LLM->>AIGW: "a + b"
    AIGW->>GLW: suggestion: "a + b"
    GLW->>LS:  suggestion: "a + b"
    LS->>EXT: triggers IDE code suggestion UI: "a + b"
    EXT->>IDE: triggers IDE code suggestion UI: "a + b"
```

Components pictured on diagram are as follow:

1. IDE extensions: GitLab offers number of IDE extensions (aka plugins) that among other features also provide integration with Language Server
   1. VSCode Extension: https://gitlab.com/gitlab-org/gitlab-vscode-extension/
   1. JetBrains Extension: https://gitlab.com/gitlab-org/editor-extensions/gitlab-jetbrains-plugin
   1. NeoVim Extension: https://gitlab.com/gitlab-org/editor-extensions/gitlab.vim
1. [Language Server](https://gitlab.com/gitlab-org/editor-extensions/gitlab-lsp): it is a unified way of delivering features that can be shared across different IDEs reducing duplication. Language Server is a component that uses the [LSP protocol](https://microsoft.github.io/language-server-protocol/) for communication with IDE extensions.
1. [GitLab Workhorse](https://docs.gitlab.com/ee/development/workhorse/) - GitLab Workhorse is a smart reverse proxy for GitLab intended to handle resource-intensive and long-running requests.
1. [GitLab Rails](https://gitlab.com/gitlab-org/gitlab) - main GitLab component providing majority of features.
1. [AI Gateway](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist) - a standalone-service that will give access to AI features to all users of GitLab, no matter which instance they are using: self-managed, dedicated or GitLab.com. For more conceptual information refer to [architecture blueprint](https://docs.gitlab.com/ee/architecture/blueprints/ai_gateway/index.html)
1. Large Language Model - a AI model that provides code generation capabilities

Code Suggestions includes two types of interactions:

- **[Code Completion](#code-completion)**: A short AI-generated suggestion intended to complete an existing line or block of code
- **[Code Generation](#code-generation)**: A longer AI-generated suggestion intended to create entire functions, classes, code blocks, etc.

Each code suggestion request is catogrised into a single category. Request categorization is performed by the Language Server before request is sent to GitLab Workhorse. If categorization is not done by Language Server then this categorization is performed by GitLab Rails.

## Code Completion

Code completion interaction is one of two code suggestion requests that can be triggered by IDE. Its goal is to provide very fast responses (< 1 second)
at the cost of smaller suggestion size, and less context awareness of surrounding source code or repository files.

By default, the code completion request is sent directly to the AI Gateway with direct connection details fetched from the GitLab Rails monolith,
as shown in [the sequence described below](#code-completion-direct-connection-diagram).
Alternatively, the request can be routed from the GitLab Rails monolith, as shown in
[the diagram in the Code Suggestions technical overview](#code-suggestions-technical-overview).

A request prepared by the Language Server is proxied in mostly unmodified form without any additional context being attached. GitLab Rails' role in this feature is limited to mostly an authorisation entity assuring that the given user is allowed to use the Code Suggestions feature.

### Code Completion Direct Connection Diagram

```mermaid
sequenceDiagram
    actor USR as  User
    participant IDE
    participant EXT as IDE Extension
    participant LS as Language Server
    participant GLR as GitLab Rails
    participant AIGW as AI Gateway

    USR->>IDE: types: "def add(a, b)"
    IDE->>EXT: notify about document change def add(a, b)
    EXT->>LS: register document change def add(a, b)
    LS->>LS: triggers code suggestion request
    alt there is unexpired direct connection details in cache?
    LS->>LS: fetches direct connection details from cache
    else
    LS->>GLR: requests for direct connection details
    GLR->>LS: returns direct connection details (AIGW url and token, expiry, model details)
    LS->>LS: caches direct connection details, with 1 hour expiry
    end
    LS->>AIGW: sends code suggestion requests using direct connection details
    AIGW->>LS:  suggestion: "a + b"
    LS->>EXT: triggers IDE code suggestion UI: "a + b"
    EXT->>IDE: triggers IDE code suggestion UI: "a + b"
```

The components pictured on the diagram are described in the [technical overview](#code-suggestions-technical-overview) section.

## Code Generation

Code Generation interaction is another type of code suggestion request that can be triggered by IDE. Its goal is to provide long and extensive responses generating
complete blocks of code like functions or classes. It has a much longer response time than code completions (up to 30 seconds). This type of code suggestion request
takes extended context into account when resolving the user task. This context comes from current files in IDE as well as [Repository X-Ray](https://docs.gitlab.com/ee/user/project/repository/code_suggestions/repository_xray.html) report.

```mermaid
sequenceDiagram
    actor USR as User
    participant IDE
    participant PG as GitLab PostgreSQL DB
    participant GLR as GitLab Rails
    participant AIGW as AI Gateway

    USR->>+IDE: types: "#35; generate a function that transposes a matrix"
    IDE->>+GLR: trigger code generation for line ` "#35; generate function `
    GLR->>PG: fetch X-Ray report for project and language
    PG->>GLR:  xray_reports record
    GLR->>GLR: include first 300 entities from xray report into code generation prompt
    GLR->>-AIGW: trigger code generation ` "#35; generate function `
```

In above diagram some components (inc: GitLab Workhorse or Language Server) are ommitted for brevity reasons. However high level flow of requests shown in [technical overview](#code-suggestions-technical-overview) section
remains unchanged.

## API Reference

For a complete overview of the Code Suggestion API Endpoints that are used
on both Rails Monolith and the AI Gateway Project, please refer:

- [Rails app Code Suggestions API Reference](https://docs.gitlab.com/api/code_suggestions/)
- [AI Gateway API Reference](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/docs/api.md?ref_type=heads)
