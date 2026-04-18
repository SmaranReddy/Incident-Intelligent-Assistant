---
title: "Codebase Semantic Indexing"
description: "Advanced code search and discovery capabilities"
---

## Overview

Codebase Semantic Indexing enhances code search and discovery by understanding the meaning and context of code, rather than relying solely on keyword matching. This enables developers to find relevant code more effectively and discover related functionality across large codebases. This is done by indexing projects and creating code embeddings which are searchable.

This project is also known as:

- Chat with your Codebase
- Codebase as Chat Context
- Codebase Embeddings
- Codebase Semantic Search

## Contact Us

| Category                | Name                               |
|-------------------------|------------------------------------|
| Group Slack Channel     | #g_context_systems                 |
| Knowledge Graph Channel | #subteam-codebase-as-chat-context |

## Objectives

- Build semantic understanding of code content and purpose
- Enable natural language queries for code search
- Improve code discovery and reusability
- Support intelligent code recommendations
- Make the search available through the Duo Agent Platform

## What are Embeddings?

Embeddings are numerical representations that capture the semantic meaning of text or code. When we process a code repository, we break it into smaller, manageable chunks (such as functions, classes, or code blocks) and convert each chunk into an embedding — essentially a vector of numbers that represents its meaning and context. These embeddings allow us to understand not just exact keyword matches, but semantic similarity between code segments. For example, a search for "user authentication" could surface relevant results even if the actual code uses terms like "login validation" or "credential verification." By storing these embeddings in a searchable format, we enable developers to find relevant code based on intent and functionality rather than relying solely on exact text matching, making code discovery more intuitive and comprehensive.

## Documentation

- [Epic](https://gitlab.com/groups/gitlab-org/-/epics/16910)
- [Blueprint: Codebase as Chat Context](../../architecture/design-documents/codebase_as_chat_context/_index.md)
- [Blueprint: Code Embeddings](../../architecture/design-documents/codebase_as_chat_context/code_embeddings.md)

## Runbook and Monitoring

- [Runbook](https://gitlab.com/gitlab-com/runbooks/-/blob/master/docs/ai-active-context/README.md#monitoring)
- [Dashboard](https://log.gprd.gitlab.net/app/r/s/tyyS8)
