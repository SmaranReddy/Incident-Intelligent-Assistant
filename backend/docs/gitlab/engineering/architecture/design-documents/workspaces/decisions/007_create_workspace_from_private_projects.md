---
title: "Workspaces ADR 007: Create workspaces from private projects"
---

## Context

We want users to be able to create a workspace from private projects.

## Decision

We will create Personal Access Token(PAT) associated to the user for each new workspace and inject it into the workspace.
This token will be used to clone the project as the workspace starts.
This token will be used for all git operations that the user performs while using the workspace.

More details can be found [here](https://gitlab.com/groups/gitlab-org/-/epics/10882).

## Consequences

The token will available within the workspace as an environment variable/file.
This can result in token exfiltration if an unstrusted container image is used.

This is acceptable for now because users are expected to provision workspaces from projects they trust, like CI jobs.

## Alternatives

We considered using [ephemeral tokens(JWTs/OAuth/OIDC/etc.)](https://gitlab.com/gitlab-org/gitlab/-/issues/421289#note_1511631931)
instead of PAT but each came with its own set of challenges.
This revealed the larger need to have a some [standardized authentication and authorization between internal services and GitLab Rails monolith](https://gitlab.com/gitlab-org/gitlab/-/issues/421983).
