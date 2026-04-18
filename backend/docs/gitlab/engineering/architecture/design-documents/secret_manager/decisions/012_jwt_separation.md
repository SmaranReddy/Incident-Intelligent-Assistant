---
title: 'GitLab Secrets Manager ADR 012: Separate JWT Domains'
owning-stage: "~sec::software supply chain security"
toc_hide: true
---

## Context

GitLab's use of OpenBao relies on careful domain separation of JWTs issued
by GitLab Rails and validated by OpenBao. Notably, GitLab CI only has a
single signing key; thus we need to rely on claim differences to ensure proper
authorization controls. We have a few scopes we wish to separate:

1. Tokens for other external secrets managers; e.g., for Vault.
   - JWT tokens issued via the external secrets integration have customizable
    `aud` claims, allowing a user to manually set the OpenBao address here.
     This would mean we wouldn't reject this token on the basis of `aud` alone
     (and `iss` and key material also match), meaning we have to rely on
     additional claims to prevent usage. Here, `sub` is not enough to
     differentiate the pipeline secret fetching use case either, meaning we
     have to rely on `secrets_manager_scope` as the required claim.
   - Furthermore, to make the requirement explicit, if in the future
     customization of claims was afforded to pipeline definitions, we should
     ensure that `secrets_manager_scope` is never attached to external secrets
     manager integration tokens.
1. Tokens for highly privileged Rails actions, like creating new namespaces or
   assigning permissions. Often these have a user initiating the request.
   - As a sub-category of this, Rake and Sidekiq tasks often lack a direct
     user assignment.
1. Tokens internally issued to Rails, for use with user-scoped actions,
   like updating or creating secrets.
1. Tokens issued for pipelines to execute.

With the introduction of group-level secrets, we additionally have to separate
two distinct use-cases:

1. Project secrets
1. Group secrets

## Decision

We'll use two major fields for scoping access:

1. `sub`, the default claim, must be bound on all JWT roles to prevent
   cross-purpose requests. Therefore, it must maintain a unique prefix
   across all uses. Currently tracked use cases are:
   1. `gitlab_secrets_manager` for system-scoped tokens
   2. `user:#{current_user.username}` for user-scoped JWTs
   3. `project_path:{group}/{project}:ref_type:{type}:ref:{branch_name}` for pipeline JWTs
1. `secrets_manager_scope` as a defense-in-depth, easily verifiable claim
   which contains one of the following literal strings:
   - `privileged`, for the highly privileged top-level mount at
     `auth/gitlab_rails_jwt/`
   - `pipeline`, for a pipeline executing in a project, fetching secrets
     from a project or group namespace.
   - `user`, for a user-scoped token under a project or group namespace.

In this way, if `sub` is _not_ a unique prefix we have a fallback method of
verifying token scope.

Further, within a tenant-, group- or project-scoped auth method, we rely on
the following claims:

1. `project_id`, if scoped to a project, or
2. `group_id` if scoped to a group.

In particular, Rails should validate that the requesting pipeline or user is
related to this project _before_ affixing the claim (see above notes on
tokens for external secrets managers). As actual policy decisions are made
based on other claims (e.g., `user_id`, `role_id`, `ref_type`, `environment`,
etc.), the issued token may not actually be granted any permissions within
OpenBao. That is, Rails validates the _relationship_ to the project (as that
lies strictly in the Rails database), but OpenBao retains the single source
of truth over permissions granted within the secrets manager.

### Single role, multiple uses

For the highly privileged `auth/gitlab_rails_jwt` role, the key authorization
decision resolves around the combination of `sub` and `secrets_manager_scope`;
while values like `project_id`, `group_id`, and others _should_ exist (or
rather, **must** exist when the request is a user-driven action to a specific
location), scenarios like Rake tasks and potentially migrations or Sidekiq
jobs will mean that exact user information may not actually exist.
