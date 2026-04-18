---
title: "GitLab Secrets Manager"
status: ongoing
creation-date: "2023-08-07"
authors: [ "@alberts-gitlab", "@iamricecake", "@cipherboy-gitlab" ]
coach: [ "@grzesiek", "@fabiopitino" ]
approvers: [ "@jocelynjane", "@shampton" ]
owning-stage: "~sec::govern"
participating-stages: []
toc_hide: true
---

{{< engineering/design-document-header >}}

## Summary

GitLab users need a secure and easy-to-use solution to
store their sensitive credentials that should be kept confidential ("secret").
GitLab Secrets Manager is the desired system that provides GitLab users
to meet that need without having to access third party tools.

## Motivation

The current de-facto approach used by many to store a sensitive credential in GitLab is
using a [Masked Variable](https://docs.gitlab.com/ee/ci/variables/index.html#mask-a-cicd-variable) or a
[File Variable](https://docs.gitlab.com/ee/ci/variables/index.html#use-file-type-cicd-variables).
However, data stored in variables (masked or file variables) can be inadvertently exposed even with masking.
A more secure solution would be to use native integration
with external secret managers such as HashiCorp Vault or Azure Key Vault.

Integration with external secret managers requires GitLab to maintain the integration
with the third-party products and to assist customers in troubleshooting configuration issues.
In addition, customer's engineering teams using these external secret managers
may need to maintain these systems themselves, adding to the operational burden.

Having a GitLab native secret manager would provide customers a secure method to store and access secrets
without the overhead of third party tools and to leverage the tight integration with other GitLab features.

### Goals

Provide GitLab users with a way to:

- Securely store secrets in GitLab
- Use the stored secrets in GitLab components (for example, CI Runner)
- Use the stored secrets in external environments (for example, production infrastructure).
- Manage access to secrets across a root namespace, subgroups and projects.

#### Use cases

To help design the architecture, we need to understand how users, in their roles, would
operate and use the system. Here are significant use case scenarios that can help drive our
design decisions:

1. As a Compliance Manager or Security Ops engineer, I want to use dynamic secrets to provide an additional layer of security for my system.
1. As a Compliance Manager or Security Ops engineer, I want to use automatic secret rotation to provide an additional layer of security for my system.
1. As a Compliance Manager or Security Ops engineer, I want to set secrets expiration policies to provide an additional layer of security for my system.
1. As a Developer, I want sensitive credentials to be fully encrypted so I am not accidentally leaking this information in my deliverables.
1. As a Developer, I want to use secrets to store sensitive credentials to avoid storing this information inside of my code.
1. As a GitLab Admin and as a GitLab Group Owner, I want to have the ability to disable secrets management feature altogether if my organization requires the use of a 3rd party tool.
1. As a Compliance Manager or Security Ops engineer, I need to have audit tools which will show me the status of the secrets and their usage so I can identify any suspicious behavior and ensure our security and compliance of our sensitive credentials.
1. As a Compliance Manager, I need to audit log of secrets usage and management to ensure my organization is following compliance policy.
1. As a DevOps Engineer, I want the deployment process to fetch secrets necessary for deployment directly from a secrets manager.
1. As a Customer of a regulated industry, I need a secrets manager with HSM support so I can meet FIPS requirements.
1. As a US Public Sector customer, I need a secrets manager that is FedRAMP certified to meet security requirements.
1. As a Customer with an air-gapped system, I need a secrets managers that can be installed on-prem and managed on-site to meet network requirements.
1. As a Secret Owner, I need to quickly update or rotate the secret in case there has been a vulnerability to ensure the security of my software supply chain.
1. As a Secret Owner, I need the ability to rollback change to a secret in the event of an accidental change.
1. As a Security Engineer, I want to ensure all secrets are properly deleted/destroyed if they are no longer needed.

#### Non-functional requirements

- Security
- Compliance
- Auditability

### Non-goals

This blueprint does not cover the following:

- Secrets created within GitLab to allow external resources to access GitLab personal access tokens, such as personal access tokens.

## Decisions

- [ADR-005: Non-hierarchical key structure for secrets in OpenBao](decisions/005_secrets_key_structure/)
- [ADR-007: Use OpenBao as the secrets management service](decisions/007_openbao/)
- [ADR-008: Redesigning secrets manager without a Rails database table](decisions/008_no_database.md)
- [ADR-009: Request Flow & Architecture Diagrams](decisions/009_request_flows.md)
- [ADR-010: Using Rails ActiveRecord for Secret Rotation Metadata](decisions/010_secret_rotation_metadata_storage.md)
- [ADR-011: Use Streaming Audit Events for Customer Audit Ingress](decisions/011_http_audit.md)
- [ADR-012: Separate JWT Domains](decisions/012_jwt_separation.md)
- [ADR-013: OpenBao Separate Logical Database](decisions/013_openbao_separate_database.md)

### Superseded

These documents are part of the initial iteration of this blueprint.

- [ADR-001: Use envelope encryption](decisions/001_envelop_encryption/)
- [ADR-002: Use GCP Key Management Service](decisions/002_gcp_kms/)
- [ADR-003: Build Secrets Manager in Go](decisions/003_go_service/)
- [ADR-004: Stateless Key Management Service](decisions/004_stateless_kms/)
- [ADR-006: Use AppRole authentication method between Rails and OpenBao](decisions/006_approle_authentication_rails/)

## Proposal

The secrets manager feature will be available on both SaaS and Self-Managed installations
and will consist of two core components:

1. GitLab Rails
1. OpenBao Server

```mermaid
flowchart LR

u([User]) --interacts with--> glab[GitLab Rails]
glab --provisions secrets<br/>(internal URL)--> o{OpenBao}
glab --starts--> p[Pipeline]

p --fetches secrets<br/>(external URL)--> lb[Load Balancer]
lb --routes to--> o
```

Because pipelines can be external to GitLab infrastructure, OpenBao thus
needs to be open to externally accessible and ACL protected. However, we'll
initially restrict the provisioning of secrets and authentication roles
through the GitLab Rails UI and limit JWT token creation to pipeline
execution, until we can build multi-tenancy extensions into OpenBao.

<details>

<summary> Digression on OpenBao visibility </summary>

Technically there are two approaches to make OpenBao visible:

 1. Directly, making the API address listen globally to the public internet.
 2. With GitLab Rails proxying requests on behalf of callers.

In particular, with the latter, while we have the option of replacing the
backend secret store with a different provider in the future, we incur
additional effort for every dynamic secret provider we support: we can either
implement a thin proxy over the OpenBao API (making switching harder as we're
still tightly coupling to OpenBao), or we can implement a bespoke translation
layer per call.

Outside of a few proprietary vendors offering dynamic secrets, we'd thus
likely have to rewrite the functionality from scratch anyways if we were to
switch. Thus, we could stick to a vaguely OpenBao/Vault-compatible API design
(for clients) but modify it as necessary.

With this trade-off, and given there's interest in expanding past static K/V
secrets (into dynamic secrets -- whether for databases, cloud providers, or
even GitLab tokens), we suggest using the former approach, exposing OpenBao
publicly. This enables non-Pipeline workloads in the future (that simply read
secrets, leaving policy creation to GitLab Rails) and lets us use more
advanced features such as Transit for other solutions inside GitLab Rails.

</details>

Within OpenBao, we'll use two authentication engines:

 1. [JWT](https://openbao.org/docs/auth/jwt/), to authenticate Rails to OpenBao
    and created pipeline jobs to OpenBao. All of these JWTs will be issued
    by GitLab Rails, using GitLab [OIDC ID Tokens](https://docs.gitlab.com/ee/ci/secrets/id_token_authentication.html)
    supported by the existing [HashiCorp Vault Runner integration](https://docs.gitlab.com/ee/ci/secrets/hashicorp_vault.html).
    Claims on tokens for administrative use by Rails will have different
    values than those issued for GitLab's OIDC ID tokens.

```mermaid
flowchart LR

glab[GitLab Rails]
o{OpenBao}
p[Pipeline]

glab --authenticated secrets management-->o

glab --issues JWT--> p --authenticated secrets fetch-->o
```

Importantly, while GitLab Rails authorizes the pipeline's execution of a
particular job, the pipeline's token must be adequately scoped to a
particular execution context. A pipeline's execution context is defined
as follows:

```mermaid
flowchart LR

u(Owner)
g(Groups)
sg(Group Namespaces)
o(Organization)
og(Org Groups)
sog(Org Namespaces)
proj(Project)
pln[Pipelines]
env[Environments]
branch[Branches]

u --contains--> proj
g -- optionally contains --> sg
sg --contains--> proj
g --contains--> proj
o --contains--> og
og -- optionally contains --> sog
sog --contains--> proj
og --contains--> proj
proj --contains--> pln
pln --contains--> env
pln --contains--> branch
```

Both groups and organization groups can contain many nested subgroups
("namespaces"). The existing OIDC ID token contains sufficient context
information for these purposes.

From the context of an execution of a pipeline, a secret may come from
many sources:

```mermaid
flowchart LR

u(Owner)
g(Groups)
o(Organization)
og(Org Groups)
proj(Project)
pln[Pipeline]
env[Environments]
branch[Branches]
exec{Execution}

exec -- of a--> pln
exec --contains multiple--> branch --in the context of--> proj
exec --contains multiple--> env --in the context of--> proj
exec --in the context of--> proj

proj --in the context of --> og --in the context of --> o
proj --in the context of --> u
proj --in the context of --> g
```

Here, an execution of a pipeline can, in the restriction of a set of branch
or environment match patterns (for example, `main`, `*`, or `release/*`), can get
secrets from:

 1. A project,
 1. The (user) owner of a project or equivalently,
 1. The hierarchical groups up to and including the organization owner of a
    project, or a parent group (prior to the introduction of organizations).

That is, if a secret is to be used by multiple projects, it should be
provisioned up a level and the branch and/or environment restricted
appropriately, ensuring that the relevant environment does not conflict
across projects it should not be visible in.

Similarly, in the context of user access and management of secrets, the
following hierarchy applies:

```mermaid
flowchart LR

u{User}

secret[Secret]
projTeam[Project Team]
groupTeam[Group Team]
orgTeam[Organization Team]

da(Direct Access)
sm(Secrets Management role)

u -- given direct access to --> secret
u -- belongs to --> projTeam
u -- belongs to --> groupTeam
u -- belongs to --> orgTeam

projTeam -- granted --> da
projTeam -- granted --> sm
groupTeam -- granted --> da
groupTeam -- granted --> sm
orgTeam -- granted --> da
orgTeam -- granted --> sm
```

A team at any level (project, group, or organization) can be given permission
to manage all secrets at or below its level ("Secrets Management capability"
above). However, on an individual secret, a user or team can also be given
direct access even if they don't otherwise have secrets management
capabilities.

Initially, users will manage secrets through GitLab Rails and its broader
privileged JWT tokens, but the intent longer-term is to use per-user JWTs. The
current design necessitates that GitLab Rails sees the value of the secret,
proxying it on behalf of the Browser UI to OpenBao. By using a per-user JWT
issued to the browser session in LocalStorage, the UI can directly contact
the public OpenBao instance and GitLab Rails will not have access to the
secret during provisioning and thus, risk inadvertently logging it. However,
the privileged JWT token will still be used to safely limit changes to
the ACL policies.

### Configuration Options

```yaml
openbao:
  url: "https://openbao.example.com:8200"         # External URL for runners
  internal_host: "http://openbao-internal:8200"   # Internal URL for Rails (optional)
```

Fallback behavior: When internal_host is not configured, Rails will use the standard URL for all connections.

### Secret and authentication hierarchy

Design of OpenBao's mount path will greatly affect available security
parameters. When encoding user values (`{value}`), we'll use hex: this
lets us have a unique, canonical transformation of potentially
non-path-safe components into individual path segments. For components with
dynamic names (like user, organization, and group names), which aren't
glob-aware and which are subject to renaming, we'll use their internal
integer database identifiers to prevent having to rename the underlying
mounts if their human-readable path or name changes.

#### Tenant definition

We assume every project has a parent component: this is either a user, a
group, or an organization. In the event of legacy groups with `org_id=1`
we will use the owning group instead.

To separate tenants, we will use [OpenBao namespaces](https://github.com/openbao/openbao/issues/787).
These will be at the following levels:

- `/{tenant}_{tenantid}`, to separate each tenant from the others as mentioned
  below, and
- `/{tenant}_{tenantid}/{scope}_{scopeid}`, to separate individual
  secret scopes (such as groups or projects) within a tenant.

At the top tenant level, namespaces may in the future have per-namespace
[seal mechanisms](https://github.com/openbao/openbao/issues/1170), allowing
multi-tenant crypto-key separation.

The additional per-scope namespace allows us to achieve mount splitting and
moving for [Cells](#cells).

#### Secrets

We propose the following structure for mounts for user-owned projects:

- `/user_{userid}/secrets`
- `/user_{userid}/proj_{projectid}/secrets`

And for organization-owned projects:

- `/org_{orgid}/secrets`
- `/org_{orgid}/{namespace}_{nsid}/secrets`
- `/org_{orgid}/proj_{projectid}/secrets`

For groups which use the existing group-based system, the top-level entity
would be the parent group:

- `/group_{groupid}/secrets`
- `/group_{groupid}/{namespace}_{nsid}/secrets`
- `/group_{groupid}/proj_{projectid}/secrets`

Here, the first `group_{groupid}` would be the top-most group-id, but would
not be repeated for the nested sub-group secrets.

`{namespace}_{nsid}` indicates a nested entity (whether a group, subgroup, or
user namespace within a top-level organization or group). With the eventual
introduction of top-level organizations, if groups belong to orgid=1 meta-org,
we'll use group-based tenant separation for these.

Collectively, this path structure is referred to in the general template sense
as `{tenant}_{tenantid}/{scope}_{scopeid}`, where `group`, `user`, and `org`
are examples of tenants and `proj` and `group` are examples of scopes.

Within each `secrets` folder, we'll initially mount a K/V secrets engine at
`/kv`; eventually other types of secrets engines can also be mounted to
provide dynamic secrets, with some consideration as to naming engines.

Within each K/V secrets engine, we'll use the following nesting structure to
restrict access to secrets (with a given `<name>` in pipelines:

- `.../secrets/kv/data/explicit/<name>`

Similarly, when using dynamic secrets engines with roles, we'll suggest using
the following structure of mounts and allow configuring secret names inside of
these mounts as appropriate to the specified dynamic secret engine:

- `.../secrets/<mount-name>/.../explicit-<role-name>`

Succinctly, the requirement that any given path a secret could live at is
prefix-free: we use `explicit` to indicate that provisioning auth engines must
_explicitly_ grant secret-level access to specific fields. If later types of
secret protection schemes are used (such as placing environment or branch
protections in the URL to allow broad access to any secret for that branch
or environment), we suggest using `env-` and `branch-` as prefixes.

#### Authentication

For more information on JWT tokens, see [ADR-012: Separate JWT Domains](decisions/012_jwt_separation.md).

Authentication uses three sets of mounts:

- `/auth/{tenant}_{tenantid}/{scope}_{scopeid}/pipeline_jwt`
- `/auth/{tenant}_{tenantid}/{scope}_{scopeid}/user_jwt`
- `/auth/gitlab_jwt`

In particular, because pipelines may need access to nested secrets, but
won't need access to anything outside the particular scope, we will provision
ACL policies and authentication at the scope namespace (`proj_` or `group_`).
With namespaces, this will help with restricting the pipeline from escaping its
sandbox and accessing other tenant's or scope's secrets.

Later, user authentication can be added to each scope as well (such as at
`/auth/org_{orgid}/project_{projectid}/user_jwt`) and the user can request
scoped JWTs (and subsequent OpenBao tokens) so that they aren't requesting
tokens with broad access beyond the scope they are modifying the secrets of.

Aside: presently the order is auth and then tenant segment, but when adding
proper namespace support, auth mounts could be inside of a tenant and thus
the order will be swapped to e.g., `/user_{userid}/proj_{projectid}/auth/pipeline_jwt`.

##### GitLab Privileged JWT

Initially using the same JWT issuer, though potentially migrating to a
purpose-built issuer, GitLab will issue itself JWTs to authenticate against
the privileged `/auth/gitlab_jwt` path. The subject of this token and other
fields will be uniquely different than the OIDC ID tokens issued to pipelines,
preventing them from successfully authenticating against this endpoint.

Long-term, we'll attempt to tie inbound GitLab Rails requests to outbound,
user-initiated management actions via custom claims on these JWTs, allowing
auditing and attribution from source request through to OpenBao.

Using JWTs instead of AppRole removes the need for additional, external
secrets management in the operational side of GitLab Rails: it already has
access to the provisioned JWT issuer and thus can self-issue JWTs to auth
against OpenBao as it remains in a privileged place in the design.

#### ACL design

We propose a group-based approach to ACL policies: each scope (for a pipeline)
and role (for users, when not using explicit grants) will grant access to
specific subsets of secrets. These policies are maintained and stored in
OpenBao, but GitLab Rails is tasked with managing and provisioning them.

Notably, no Rails-initiated [operations](decisions/008_no_database.md#types-of-operations)
are expected to span multiple tenant contexts. This allows us to add
per-namespace ACLs in the future and create smaller `path->policy` indices in
the future. Furthermore, we can use nested paths to segment different policies
and build per-segment indices, reducing the list operation overhead as well.

For each policy, we'll present create a [group alias](https://openbao.org/api-docs/secret/identity/group-alias/)
to allow a `groups_claim` on the Rails-issued JWT to select applicable ACL
policies based. A [future enhancement](decisions/008_no_database.md#jwt-direct-profiles)
will allow us to get rid of all but
[glob-based group matches](decisions/008_no_database.md#group-alias-glob-matching).

##### Hierarchy of policies

Policies will be placed in their respective tenant (e.g., `user_{id}/`)
and scope namespace (e.g., `user_{userid}/proj_{projectid}/`).

In the future, hierarchical secrets can be supported by also supporting
`group_{id}` and other constructs as top-level categories. With multi-tenant
support, we'll have fewer top-level items (as they'll be explicitly bounded
by the tenant and scope), making iterating over all such items easier.
However, we'll usually have fairly few items within each scope. Hierarchical
secrets will require the pipeline JWT be valid in multiple contexts (e.g.,
at both the group and project level).

Note that these do not necessary reflect the secret's path (the contents of
the ACL policy) and only notate _where_ ACL policies exist in OpenBao.

##### Pipeline ACL

To restrict a pipeline's JWT token to only allowed paths, we'll use GitLab
Rail to provision an ACL policy for the pipeline and a JWT role which allows
templating of policies based on [token metadata](https://openbao.org/api-docs/auth/jwt/#acl-policy-templating-examples).
In the future more granular authentication decisions can occur via use of
the [CEL JWT policy engine](https://openbao.org/docs/rfcs/cel-jwt/).

OpenBao's ACLs are directly [stored on disk](https://github.com/openbao/openbao/blob/7fca5c0baebd3f55254da06d26bc160f465a7e1a/vault/policy_store.go#L287-L345),
with a [moderate cache](https://github.com/openbao/openbao/blob/7fca5c0baebd3f55254da06d26bc160f465a7e1a/vault/policy_store.go#L28-L29)
for recently-used policies. Policies are loaded and evaluated at
[request validation time](https://github.com/openbao/openbao/blob/7fca5c0baebd3f55254da06d26bc160f465a7e1a/vault/policy_store.go#L605-L629),
meaning that if a pipeline ACL name is modified and reused, the latest
version (at the time of the request) will win.

For naming roles, we suggest identifier `all-pipelines` as the because the
auth mount is within the project scope.

###### Layout

Within the `project_{id}/` scoped namespace, we'll provision an additional
path segment, `pipelines/`, to separate pipeline-related policies for a
project from other types of access.

We create the following types of pipeline policies with Rails:

- `global`: secrets any pipeline can access
- `env/{context}`: environment restriction policies
- `branch/{context}`: branch restriction policies
- `combined/env/{context}/branch/{context}`: combined environment and branch restrictions (`AND`)

A full path of an ACL would thus look like the following examples:

- `pipelines/global`,
- `pipelines/env/prod-*`,
- `pipelines/branch/release/*`, or
- `pipelines/combined/env/prod-*/branch/release/*`.

These would be accessible under the namespace policy store at (for the sake
of this example) `user_2341/project_12345/sys/policies/acl/:path`.

Notably, the direct encoding of restriction to path allows for us to create
groups with [the same encoding](decisions/008_no_database.md#group-alias-glob-matching), reducing the
need for GitLab Rails to query the set of restrictions before issuing the
JWT. Use of `combined` as a prefix ensures we cannot have confusion attacks
with poorly named environments and branches.

In the future, explicit grants could also occur: stages could have a path
segment and policies (`stages/{name}/global`) and each job in a stage could
also have direct secret access (`stages/{name}/job/{name}/global`). Or, we
could even support ANDing between stage, name, and the above restrictions
(environment/branch) to support rather granular execution contexts for these
jobs. Note that this information (stage/job name) does not yet exist on the
id token and thus cannot be used for ACLing. However, this can easily be
added as the information exists on the `Ci::Build` object (`stage` and
`name` fields).

###### Contents

Each ACL policy will be an explicit grant of capabilities over a path. We
will set these with GitLab Rails and will not use globbing (`+` or `*`) to
grant access to multiple secrets; only access to specific secrets will be
allowed.

For example, if any pipeline running with an `env/prod-<DATE>` context is to
have access to the production database credentials, we will create a policy
named `pipelines/env/prod-*` with the contents:

```hcl
path "secrets/kv/data/explicit/PROD_DB_PASS" {
    capabilities = [ "read" ]
}
```

as the path to the namespace is
[implicitly appended by OpenBao](https://github.com/openbao/openbao/blob/fork-point/vault/policy.go#L401-L402).

Notably, because a pipeline will have multiple contexts which might provision
different ACL policies, we'll eventually want to implement something similar
to [policy unions](decisions/008_no_database.md#policy-unions) or a similar
solution [with CEL](https://openbao.org/docs/rfcs/cel-best-practices/).

##### User ACL

While the user wouldn't initially contact OpenBao directly (and instead have
secrets management operations proxied through GitLab Rails), we want to switch
GitLab Rails to using scoped user-level JWTs and then ultimately to the client
having direct access to OpenBao (without necessarily contacting GitLab Rails),
so that no user-confusion attacks are viable and so that GitLab Rails does
not see the secrets during provisioning, preventing them from accidentally
being logged in the future.

Notably, a user token would be rather privileged: it would have complete
access to all secrets within the users' purview and thus should be suitably
short-lived to avoid potential side-channel attacks. Further hardening (like
[auto-binding to CIDRs](https://github.com/openbao/openbao/issues/32)) would
help avoid these issues. To solve this, we'd need to scope access along team
lines (e.g., any teams with access to manage a certain category of secrets
would have a separate ACL policy) along with any location-specific secrets
granted in a given view and expand OpenBao to allow templating of profiles
(so that the JWT could contain a concrete list of profiles to grant this
user). This would be rather involved but would help scope user JWTs more
specifically.

The upgrade path (from privileged JWT token to incrementally using tightly
scoped policies) would simply be changing the Rails code to generate a user
JWT and updating the Ruby client's request to authenticate and use the
subsequent token for a particular request instead of the privileged JWT token.

Note that users lack read permissions on the actual secret; they can only set
the value. When rolling out dynamic secrets, users would be granted full
permissions on the role, but no permissions on the dynamic secret endpoint
(e.g., for PKI, they could have full permissions on `pki/roles/:name`, but
none on `pki/issue/:name`).

###### Layout

Similar to pipelines, we'll create policies under the relevant top-level
path, with a `roles/` or `direct/` subcomponent. Here, `roles/` will have
various default or custom roles which get access to secrets at various
levels. For instance, `roles/maintainers` and `roles/owners`: this allows
each project to control what scope of access these roles get, if the defaults
are not ideal.

Further, with `direct/user_{id}` roles, users will be given explicit grants
to certain secrets, beyond what they might normally see given their role.

###### Contents

For both roles and direct access, there are two types of grants:

1. Broad grants to the entire category of secrets.
2. Specific grants to individual secrets.

For the former, the policies might look like:

```hcl
path "secrets/kv/data/explicit/+" {
    capabilities = [ "sudo", "create", "update", "patch", "delete", "list" ]
}
```

for read-write access and

```hcl
path "secrets/kv/data/explicit/+" {
    capabilities = [ "list" ]
}
```

for view-only access.

For the latter (specific grants), the policies might look like the above,
just with explicit names in them (e.g., `DB_PASS_PROD`).

When accessing a secrets management page, GitLab Rails will internally issue
a JWT with the associated user's details which will contain the relevant
`groups_claims` to groups with specific policies within the project. Rails
will subsequently use this JWT internally for access to OpenBao, effectively
limiting the scope of access to just what the user can see. In the future,
since OpenBao is externally accessible, this token could be given to the user
for direct provisioning of the underlying secret without GitLab Rails needing
access to it.

##### Modifying ACL policies

Notably, ACL policies are written in HCL: this presents a problem since
[Ruby lacks good HCL support](https://rubygems.org/search?query=hcl).
However, [HCL is JSON-compatible](https://developer.hashicorp.com/vault/tutorials/policies/policies#hashicorp-configuration-language-hcl).

This means that, for any given ACL policy above, we can construct its
equivalent in JSON, giving GitLab Rails native capabilities to edit it
without pulling in another dependency or building our own parser.

For example, the policy in HCL:

```hcl
path "secrets/kv/data/explicit/+" {
    capabilities = [ "list" ]
}
```

would be equivalent to the following JSON:

```json
{
    "path": {
        "secrets/kv/data/explicit/+": {
            "capabilities": [
                "list"
            ]
        }
    }
}
```

This becomes much easier for GitLab Rails to query and update. See relevant
[OpenBao documentation](https://openbao.org/docs/concepts/policies/#policy-syntax).

### Static vs Dynamic Secrets

GitLab Secrets Manager will initially only support static secrets, stored in
a KVv2 engine. Some improvements to OpenBao can be added to static secrets,
such as:

 1. Using leases, to support one-time use secrets. These could be written by
    another workflow and removed on use.
 1. Supporting automatic deletion of secrets: a given expiration date could be
    strictly enforced, preventing use of the secret and requiring rotation of
    even long-lived manual secrets.

However, ultimately automatic rotation is an important consideration. By using
fully dynamic secrets, the problem of rotation is avoided: secrets are scoped
to a particular access and are automatically revoked (trough lease expiration
or revocation) once the pipeline or access finishes. This prevents reliance on
static values and ensures compromise of values require time-limited use and
prevent persistence. We can build several improvement to OpenBao to help the
adoption of dynamic secrets:

 1. Add a plugin registry to allow custom, external plugins or allow
    containerized execution of plugins.
 1. Create additional backend-pluggable plugin types beyond database: this
    could allow various access-token providers to be more easily created by
    exposing common APIs in a standardized format.

We ultimately wish to push more users towards dynamic secrets. These are
more secure and don't have long-term data retention and security concerns.

### Sources of Truth

GitLab Rails is the source of truth for:

- JWT issuance via OIDC
  - Pipeline issuance
  - User-level issuance (eventually)

OpenBao is the source of truth for:

- Definitions of secrets
  - Canonical locations inferable from the pipeline definition and contextual
    information like project, namespace, &c.
  - Ownership information (direct access, role access, &c)
  - Scope of pipeline access (environments, branches, &c)
  - Any textual description of the secret available to operators
- Values of secrets
- ACL policies; these will be pushed from GitLab Rails based on ownership
  information (just before usage in the case of pipelines).
- Configuration of dynamic secrets, including K/V extensions for one-time use
  secrets, and automatic deletion.

### GitLab Rails

GitLab Rails would be the main interface that users would interact with when
managing secrets using the Secrets Manager feature.

This component is a facade to OpenBao server, initiating several requests and
workflows to OpenBao:

```mermaid
flowchart LR

gl[GitLab Rails]
u{{User}}
p(Pipeline)
ob{OpenBao}
lb[Load Balancer]

gl --CUD secret (internal URL)--> ob
gl --manage mounts (internal URL)--> ob
gl --manage auth roles (internal URL)--> ob
gl --manage ACLs (internal URL)--> ob
gl-. issue user JWT .->u-. CUD secret (internal URL) .->ob
gl --issue pipeline JWT (external URL)--> p

p --JWT auth--> lb
p --fetch secrets--> lb
lb --routes to--> ob

ob --audit--> gl
```

Initially all operations will through GitLab Rails including CUD operations
on secrets, but not requests for secrets from pipelines (hence, excluding the
R in CRUD -- though this action is of course initiated by a pipeline job
started from, and with a JWT issued by, GitLab Rails). Eventually the
privileged token for CUD operation will be replaced with a user JWT and
subsequently directly by the user (through a JavaScript OpenBao client running
in the context of the GitLab UI).

Within the [GitLab Rails monolith](https://docs.gitlab.com/ee/development/architecture.html#component-diagram),
two components will need access to the secrets manager: [Puma](https://docs.gitlab.com/ee/development/architecture.html#puma),
where the majority of the GitLab backend code executes; and
[Sidekiq](https://docs.gitlab.com/ee/development/architecture.html#sidekiq),
a background job processor to allow long-lived background operations (such
as initial provisioning of OpenBao and secret engine mounts).

See [ADR 009](decisions/009_request_flows.md) for more request flow diagrams.

#### Management of secrets

Because OpenBao remains the source of truth for many decisions, GitLab can
retain only a minimum table indicating the enablement status of secrets on
projects.

In `SQL`, this could look like:

```sql
CREATE TABLE secrets (
    id bigint NOT NULL,

    // Scope information; only one is set. Currently only project-level
    // secrets are allowed.
    // user_id bigint,
    // organization_id bigint,
    // group_id bigint,
    project_id bigint,

    // enablement status: provisioning, active, or disabled.
    status state_machine,
);
```

Here, `group_id` references the closet-most subgroup that owns the secret
instance. Parent subgroups and the namespace can then be queried.

#### Management of authentication

GitLab Rails needs to issue three types of JWTs:

1. JWTs within the scope of a project for a particular pipeline execution.
1. JWTs for user authentication.
1. JWTs for its own privileged operations, such as creating policies or
   mounts.

GitLab already supports issuance of both JWT types through its OIDC
for [CI/CD](https://docs.gitlab.com/ee/ci/secrets/id_token_authentication.html)
and as an [Identity Provider (IDP)](https://docs.gitlab.com/ee/integration/openid_connect_provider.html).
Within each [auth method](#authentication), we'll need to ensure that we
create the necessary roles to ensure OIDC tokens get exchanged for OpenBao
tokens with appropriate JWT-specific policies attached. As discussed in
[ADR-008: Redesigning secrets manager without a Rails database table](decisions/008_no_database.md),
this will require simple extensions to OpenBao to do.

#### Necessary cleanup hooks

The following places need to be hooked to update secrets (both internally to
the database and in OpenBao):

- Deletion of project
- Deletion of group
- Deletion of organization
- Deletion of user
- Cancellation of a pipeline
- Disabling of secrets on a project
- Non-renewal of secrets management subscription

Because IDs are used in place of names, few mount move operations will be
incurred except when enabling multi-tenant namespaces for the first time.

### OpenBao server

[OpenBao](https://openbao.org/) server will be a new component in the GitLab
overall architecture. This component provides all the secrets management
capabilities including storing the secrets themselves. An overview of
[its internals](https://openbao.org/docs/internals/) and
[threat model](https://openbao.org/docs/internals/security/) are available
on its project website.

In summary, storage is encrypted with a root key, which in turn encrypts a
barrier keyring, which in turn encrypts storage. Access to the root key is
protected through either a [Shamir's derived AES key](https://openbao.org/docs/concepts/seal/#shamir-seals)
(which requires a quorum present to unseal) or an automatic unseal mechanism
(such as [GCP KMS](https://openbao.org/docs/configuration/seal/gcpckms/)).

#### Storage backend

OpenBao on Runway will initially be deployed with the
[PostgreSQL engine](https://openbao.org/docs/configuration/storage/postgresql/).
This is a sane default for self-hosted as well, as GitLab Rails already
requires a database and can create tables in it. However, support for the
more widely deployed Raft backend can also be added in the future fairly
easily.

#### Seal mechanisms

For GitLab.Com deployments, OpenBao will use the
[GCP KMS auto-unseal mechanism](https://openbao.org/docs/configuration/seal/gcpckms/).
For self-hosted environments, if an existing supported KMS mechanism is not
provided, we can use the [static seal](https://openbao.org/docs/configuration/seal/static/)
tying into an environment-provided secrets manager like in Kubernetes.

#### Audit logging

OpenBao supports audit logging through file, syslog, or socket mechanisms.
Because OpenBao will not necessarily be co-located with GitLab Rails or
our audit logging (into Postgres), it is suggested to either use both the
file (to `/dev/stdout`) and a custom audit logging backend (to emit
[streaming audit events](https://docs.gitlab.com/administration/compliance/audit_event_streaming/)
that customers can consume).

Note that, in the event audit logging goes down, OpenBao will not be able to
handle any requests until this is fixed.

See [ADR-011: Use Streaming Audit Events for Customer Audit Ingress](decisions/011_http_audit.md)
for more information.

#### High availability

GitLab Rails includes [Geo support](https://docs.gitlab.com/ee/administration/geo/)
for self-hosted clustering, allowing multiple sites to exist, each with its
own set of nodes. Each site has its own [PostgreSQL replica](https://docs.gitlab.com/ee/administration/geo/#architecture).
One site is designated primary and nodes in the site are allowed to perform
write operations to the Postgres database. All PostgreSQL data is available
on all sites. While Gitaly has selective syncing of repository data, projects
are globally accessible from all sites and nodes. While currently only
secondary nodes talk to the primary site,
[this should be a bidirectional channel](https://docs.gitlab.com/ee/administration/geo/replication/multiple_servers.html#architecture-overview).

OpenBao natively includes [High Availability](https://openbao.org/docs/internals/high-availability/)
support; this is either provided by [Raft](https://openbao.org/docs/internals/integrated-storage/)
or the [PostgreSQL](https://openbao.org/docs/configuration/storage/postgresql/)
storage backends. The latter uses a lock table to implement leadership
election. Only one node is marked active at a time, with secondary nodes
not performing any operations except forwarding requests to the primary.

Notably, the semantics of Geo and OpenBao roughly align. We propose that Geo
will need no additional enhancements to support GitLab Secrets Manager and
that replication will be handled by Geo when using PostgreSQL.

On all front-end service nodes, we'll start the self-hosted OpenBao server
instance. One node will be designated primary by OpenBao HA election: this
is a random node on the primary site as the secondary Geo sites will have
a read-only PostgreSQL replica which cannot acquire the OpenBao lock.

When runners contact the OpenBao instance, if their request does not hit the
active node, OpenBao will route the request through its GRPC request forwarding
mechanism. This means it should work with the existing OpenBao HA support
regardless of whether it hits a Geo Primary or Secondary site.

In the event of a failover, Geo will be able to bring up the new site
designated as primary (due to the PostgreSQL database being marked writable)
and data will already have been replicated (via PostgreSQL replication)
without configuration changes to OpenBao.

The net result is that Geo is responsible for data replication for OpenBao
using a known PostgreSQL database (unless the customer brings their own
database) and OpenBao should integrate with Geo semantics.

#### Cells

Initially we will support one global OpenBao cluster per instance. In the
future [with cells](/handbook/engineering/architecture/design-documents/cells/),
we'd expect each tenant to have cluster affinity. Each GitLab cell will have
its own OpenBao cluster, as each cell will have its own local database (whether
part of the central database or a secondary PostgreSQL instance for secrets
manager). From an engineering PoV, all data is currently isolated within a
project or group, which will live on exactly one Cell, so no cross-cell data
access patterns exist. Cells integration is thus mostly operational.

The biggest issue thus is supporting the ability to
[migrate](/handbook/engineering/architecture/design-documents/cells/iterations/cells-1.5/)
customers into the Cells architecture, to allow operators to rebalance Cells.
This requires two pieces:

1. Being able to [split](https://gitlab.com/gitlab-org/gitlab/-/issues/543014)
   an existing organization or set of repositories with secrets.
2. Being able to [move](https://gitlab.com/gitlab-org/gitlab/-/issues/543001)
   an organization across various Cells.

From discussion with the Tenant Scale team, we've opted to push this
functionality into OpenBao; this spawned two upstream
[RFCs](https://gist.github.com/cipherboy/d658f06b871d5bee0d9a5cd130b1173b)
(federation of namespaces and cross-cluster namespace moves) to allow for
near-zero-downtime migrations.

Currently we focus just on running pipelines: as the organization is moving
between Cells, [Org Mover](https://gitlab.com/groups/gitlab-org/-/epics/12859)
can first set up federation on all namespaces in the destination Cell,
pointing back to the namespaces on the existing parent cell. This allows
pipelines to seamlessly execute across either Cell and Rails to maintain
read-only access to OpenBao via the use of
[inline authentication](https://openbao.org/docs/rfcs/inline-auth/).

After the rest of the organization has moved, Org Mover can then trigger the
OpenBao data move. This will be handled by OpenBao via the cross-cluster
namespace move RFC mentioned above. When completed, Org Mover or Rails will
then be able to finalize the move, enabling the cloned namespace on the
destination Cell's OpenBao instance and removing the namespace off of the
original source Cell's OpenBao instance (potentially leaving reverse
federation intact on the source cell instance to allow any straggling jobs to
continue to talk to the namespace on the new OpenBao instance).

To support splitting, we introduce additional namespaces in OpenBao: rather
than a single namespace at the GitLab tenant level, each scope of secrets
(project or group) will also have its own namespace, meaning namespaces are
never split and are kept atomic. This approach is reflected in the above
hierarchy.

#### Dual Network Path Architecture

OpenBao operates with a single service endpoint that is accessible through two distinct network routes:

1. **Internal Route (Rails)**: Direct cluster communication bypassing external infrastructure
2. **External Route (Runners)**: Internet-accessible path through load balancers and public endpoints

This architecture allows the same OpenBao instance to serve both internal GitLab Rails requests efficiently while remaining accessible to external runners through proper network infrastructure.

#### Required enhancements

OpenBao will require enhancements to improve security prior to GA of this
feature. These are discussed more below.

##### Multi-unseal capabilities

In the event an unseal method has faulted and is unrecoverable, OpenBao today
will be unable to decrypt its secrets and thus fail to start. In the event
of infrastructure outages, this makes recovery hard. Upstream Vault Enterprise
has introduced [Seal HA capabilities](https://developer.hashicorp.com/vault/docs/configuration/seal/seal-ha),
whose design was influenced by the additional requirements of
[Seal Wrapping](https://developer.hashicorp.com/vault/docs/enterprise/sealwrap).
OpenBao does not have Seal Wrapping and thus can rely on a design similar
to an [earlier, retracted PR](https://github.com/hashicorp/vault/pull/18683).

We can allow several parallel unseal mechanisms, each of which encrypts the
same root key. This would allow a mix of auto-unseal and Shamir's, allowing
a hybrid recovery procedure in the event the auto-unseal mechanism were
unavailable or the key was lost. Notably, reuse of the root key isn't bad:
any loss of unseal key material (that compromises root key integrity) with
separate root keys would still allow recovery of the barrier encryption key.
Rotation of the root key would require all unseal mechanisms to either be
online or removed prior to use. However, we wouldn't have to contend with
secrets encrypted using a combination of mechanisms, as they only encrypted
entry is the root key.

##### Multi-tenancy through namespaces

This landed [in OpenBao v2.3.0](https://github.com/openbao/openbao/releases/tag/v2.3.0).

<details>

<summary>Legacy explanation of OpenBao namespaces</summary>

In the current design document, separate tenants' secrets would be encrypted
using the same barrier encryption keys. This means compromise of a single
tenant would result in compromise of all tenants.

OpenBao lacks [namespaces from Vault Enterprise](https://developer.hashicorp.com/vault/docs/enterprise/namespaces).
These allow for logical [separation of tenants](https://github.com/orgs/openbao/discussions/293),
as discussed upstream. In particular, we can grant limited, scoped `sys/`
capabilities to admins within a namespaces:

- [`sys/auth`](https://openbao.org/api-docs/system/auth/) to mount new auth plugins.
- [`sys/mounts`](https://openbao.org/api-docs/system/mounts/) to mount new secret engine plugins.
- [`sys/namespaces`](https://openbao.org/api-docs/system/namespaces/) to allow nested namespaces.
- [`sys/policies`](https://openbao.org/api-docs/system/policies/) to control namespace-restricted policies.
- Optionally [`sys/quotas/config`](https://openbao.org/api-docs/system/quotas-config/) and [`sys/quotas/rate-limit`](https://openbao.org/api-docs/system/rate-limit-quotas/) to control namespace-restricted quotas (if tenants want to set their own, subject to global quoata limits).

The intention of this change would be to safely allow customers to directly
interact with their OpenBao instances, bringing non-Pipeline workloads to
use the same secrets store.

Further, we can add layered seal encryption to namespaces, creating a separate
root key and barrier keyring encrypted by both the parent namespace's barrier
key but also a separate unseal mechanism. This would likely expose the
following scoped `sys/` routes within a namespace:

- [`sys/rotate`](https://openbao.org/api-docs/system/rotate/) to rotate the per-namespace barrier encryption key.
- [`sys/rekey`](https://openbao.org/api-docs/system/rekey/) and [`sys/rekey-recovery-key`](https://openbao.org/api-docs/system/rekey-recovery-key/) to rotate the per-namespace root and recovery keys (if any).
- [`sys/seal`](https://openbao.org/api-docs/system/seal/) and [`sys/unseal`](https://openbao.org/api-docs/system/unseal/) to seal and unseal the namespace. Note that sealing an auto-unseal mount would require the recovery keys.

In the event a customer wanted to revoke access to secrets from GitLab,
they could seal their namespace and prevent GitLab from accessing the
unseal mechanism of choice.

Lastly, we can strengthen the memory isolation of customers by only allowing
plugin multiplexing within a namespace. When coupled with external runners
for plugins, such as a container or cgroups, we could further isolate tenants
data in memory.

</details>

##### Chosen leader

When working in a Geo cluster, we'd ideally like the OpenBao primary node
to align with the Geo cluster's definition of the primary site. We'll want
update OpenBao to give non-voting status on a PostgreSQL read-only replica
so that the secondary site's OpenBao does not attempt to become the leader.

This is proposed in [an upstream PR](https://github.com/openbao/openbao/pull/1284).

### Deployment Scenarios

Single Instance Deployment:

```yaml
openbao:
  url: "https://secrets.gitlab.com"
  # internal_host not needed - Rails and Runners use same URL
```

Kubernetes Deployment:

```yaml
openbao:
  url: "https://secrets-manager.example.com"                     # External load balancer for runners
  internal_host: "http://secrets-manager-internal.example.net"   # Kubernetes service for rails
```

GitLab.com Production:

```yaml
openbao:
  url: "https://secrets.gitlab.com"                            # Public endpoint
  internal_host: "http://secrets-manager-internal.gitlab.net"  # Internal cluster
```

### Backup and Restore

Backup and restore of this solution boils down to two aspects:

1. Database
2. Seal mechanism

Because of [our choice of PostgreSQL](#storage-backend) as the storage
backend, refer to your database provider's [documentation](https://www.postgresql.org/docs/current/backup.html)
for backup and restore of the database. For instance, on Runway this
[is performed automatically](https://docs.runway.gitlab.com/runtimes/cloud-run/reference/blueprints/cloudsql-for-postgres/#backup-for-an-instance).

In the case of an external auto-unseal mechanism, refer to the provider's
documentations. For the case of using the `static` auto-unseal mechanism
transparently with Kubernetes secrets, refer to your Kubernetes secrets
manager provider documentation.

The combination of database and seal backup is sufficient to backup and
restore OpenBao due to the use of [transactions](https://openbao.org/docs/rfcs/transactions/).

After a restore, restart the OpenBao service to pick up the new data.

### Use case studies

- [Using secrets in a CI job](studies/ci_job_secrets/)

### Further investigations required

1. Authentication of clients other than GitLab Runner.
   GitLab Runner authenticates using JWT, for other types of clients, we need a secure and reliable method to authenticate requests to decrypt a secret.
1. Infrastructure setup, including how OpenBao will be installed for self-managed instances.
1. How to best implement sharing of secrets between multiple groups in GitLab.
1. Establish our protocol and processes for incidents that may require sealing the secrets vault.
1. How to support protected and environment specific rules for secrets.
1. Should the secrets be revoked if a project or subgroup is moved under a different top-level group/organization?

## Alternative solutions

Other solutions we have explored:

- Separating secrets from CI/CD variables as a separate model with limited access, to avoid unintended exposure of the secret.
- [Secure Files](https://docs.gitlab.com/ee/ci/secure_files/index.html)
- Implementing secrets management from scratch (see [Superseded](#superseded)).

## References

The following links provide additional information that may be relevant to secret management concepts.

- [OWASP Secrets Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [OWASP Key Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Key_Management_Cheat_Sheet.html)

## Who

DRIs:

<!-- vale gitlab.Spelling = NO -->

| Role                | Who                                            |
|---------------------|------------------------------------------------|
| Author              | Erick Bajao, Senior Engineer                   |
| Co-Author           | Alex Scheel, Staff Engineer                    |
| Recommender         | Fabio Pitino, Principal Engineer               |
| Product Leadership  | Jocelyn Eillis , Product Manager               |
| Engineering Leadership | Scott Hampton, Engineering Manager          |
| Lead Engineer       | Erick Bajao, Senior Backend Engineer           |
| Senior Engineer     | Maxime Orefice, Senior Backend Engineer        |
| Engineer            | Shabini Rajadas, Backend Engineer              |

<!-- vale gitlab.Spelling = YES -->
