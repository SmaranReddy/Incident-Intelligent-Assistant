---
title: 'GitLab Secrets Manager ADR 009: Request flow diagrams'
owning-stage: "~devops::verify"
toc_hide: true
---

## Context

As part of an offsite, the following request flows and Kubernetes-specific
architecture diagrams were created.

## Request Flow

### Runner Fetch Secret

```mermaid
sequenceDiagram
    Runner->>+Puma: Fetch Next Job
    Puma->>+Runner: Resp: JWT + job metadata (auth, secret fetch URIs)
    Runner->>+OpenBao: Auth with JWT
    OpenBao->>+Runner: Resp: with Token
    Runner->>+OpenBao: Fetch Secret
    OpenBao->>+Runner: Resp: Secret Value
```

### User->Rails Management Interactions

```mermaid
sequenceDiagram
    User->>+Puma (GraphQL): List Secrets for Project
    Puma (GraphQL)->>+OpenBao: List Secrets for Project (User-specific Signed JWT)
    OpenBao->>+Puma (GraphQL): Resp: Secret List
    Puma (GraphQL)->>+User: Resp: Secret List
```

### User->Rails Provisioning Step

```mermaid
sequenceDiagram
    User->>+Puma (GraphQL): Enable Secrets Management for Project
    Puma (GraphQL)->>+Puma (GraphQL): Enqueue Provision Job
    Puma (GraphQL)->>+User: Resp: Enqueue Success
    Sidekiq->>+Sidekiq: Dequeue Job
    Sidekiq->>+OpenBao: Create Project Resources (auth, policies, secret engine)
    OpenBao->>+Sidekiq: Resp: OK
    User->>+Puma (GraphQL): Get Provisioning Status
    Puma (GraphQL)->>+Puma (GraphQL): Lookup Job Status
    Puma (GraphQL)->>+User: Resp: Finished
```

### Request Forwarding Flow

```mermaid
sequenceDiagram
    Client->>+OpenBao Standby: Request
    OpenBao Standby->>+OpenBao Standby: Refresh active node GRPC connection (background, continuous)
    OpenBao Standby->>+OpenBao Active: Forward Request (GRPC)
    OpenBao Active->>+OpenBao Standby: Respond Request (GRPC)
    OpenBao Standby->>+Client: Response
```

Presently (through upcoming OpenBao v2.3.0; ending likely v2.4.0), this flow happens regardless of request type; every request is handled exclusively by the active node.

### Internal Active Node Request Flow

For GitLab, everything labeled `OpenBao` will be part of the single active node process.

```mermaid
sequenceDiagram
    Client->>+OpenBao Core: Request (may be proxied by standby)
    OpenBao Core->>+OpenBao Core: Check authentication (token store)
    OpenBao Core->>+OpenBao Core: Check authorization (policy store)
    OpenBao Core->>+OpenBao Plugin: Route Request
    OpenBao Plugin->>+OpenBao Cache: Storage Operation
    OpenBao Cache->>+PostgreSQL: Storage Requests (cache miss and writes)
    PostgreSQL->>+OpenBao Cache: Result: Storage Operation
    OpenBao Cache->>+OpenBao Plugin: Result: Storage Operation
    OpenBao Plugin->>+OpenBao Core: Response
    OpenBao Core->>+OpenBao Core: Create Token (login request only)
    OpenBao Core->>+Client: Response
```

Cache is separated so we know (in a scalable environment) that we'll need to handle invalidations when storage has changed under us. This is handled by Raft today in a clustered manner. In PostgreSQL, we do not have a mechanism to do so. This may be GRPC but will be determined by the upstream OpenBao Horizontal Scalability Working Group.

### OIDC Registration Behavior

```mermaid
sequenceDiagram
    Initializer->>+OpenBao: Set JWT Config with OIDCDiscoveryURL=gitlab
    OpenBao->>+GitLab Puma: Fetch OIDC Issuer Info
    GitLab Puma->>+OpenBao: Return OIDC Issuer Info
    OpenBao->>+Initializer: Return OK
```

When doing authentication:

```mermaid
sequenceDiagram
    Client->>+OpenBao: Login with JWT
    OpenBao->>+GitLab Puma: Fetch OIDC Issuer Info (Once Per Startup)
    GitLab Puma->>+OpenBao: Return OIDC Issuer Info (Once Per Startup)
    OpenBao->>+OpenBao: Cache OIDC Issuer (Once Per Startup)
    OpenBao->>+OpenBao: Validate JWT against cached issuer
    OpenBao->>+Client: Return Token
```

---

In the future, we'd like this to be:

```mermaid
sequenceDiagram
    Initializer->>+OpenBao: Set JWT Config with OIDCDiscoveryURL=gitlab
    OpenBao->>+Initializer: Return OK
```

and only fetch issuer info when doing authentication or on a manual `jwt/config/verify` endpoint. That should be handled by [openbao#1306](https://github.com/openbao/openbao/pull/1306) and part of v2.3.0.

### Audit Log

```mermaid
sequenceDiagram
    OpenBao->>+OpenBao: Enqueue Audit Log
    OpenBao->>+Puma: Emit Audit Log
    Puma->>+PostgreSQL: Persit Audit Log
    Puma->>+Destination: Stream Audit Log
```

See [audit event streaming](https://docs.gitlab.com/administration/compliance/audit_event_streaming/)
documentation for more information.

## Architecture

### Kubernetes without KMS

```mermaid
flowchart TD
    Ingress


        Service_OB([HTTP API])

    subgraph OpenBao
        OB_1[Primary]
        OB_2[Standby A]
        OB_3[Standby B]

        Service_Primary([Primary gRPC])
        end

    Ingress --> Service_OB
    Service_OB --> OB_1
    Service_OB --> OB_2
    Service_OB --> OB_3

    OB_2 -. forward .-> Service_Primary
    OB_3 -. forward .-> Service_Primary

    Service_Primary --> OB_1

    OB_1 -->Service_DB
    OB_1 -. lock maintenance .->Service_DB
    OB_2 -. lock monitor .->Service_DB
    OB_3 -. lock monitor .->Service_DB

    Service_DB([PostgreSQL]) -->    DB[(PostgreSQL)]

```

In this case, Kubernetes secrets will be used to provision static secrets for
auto-unseal in addition to database access credentials.

### Kubernetes with KMS

Additional diagram with KMS or HSM:

```mermaid
flowchart TD
    Ingress


        Service_OB([HTTP API])

    subgraph OpenBao
        OB_1[Primary]
        OB_2[Standby A]
        OB_3[Standby B]

        Service_Primary([Primary gRPC])
        end

    Ingress --> Service_OB
    Service_OB --> OB_1
    Service_OB --> OB_2
    Service_OB --> OB_3

    OB_2 -. forward .-> Service_Primary
    OB_3 -. forward .-> Service_Primary

    Service_Primary --> OB_1

    OB_1 -->Service_DB
    OB_1 -. lock maintenance .->Service_DB
    OB_2 -. lock monitor .->Service_DB
    OB_3 -. lock monitor .->Service_DB

    Service_DB([PostgreSQL]) -->    DB[(PostgreSQL)]

    OB_1 -- auto-unseal --> KMS
    OB_2 -- auto-unseal --> KMS
    OB_3 -- auto-unseal --> KMS
```

Functionally, all KMS/HSM flows are equivalent. KMS may be separate instances _as long_ as all key material is the same.

Standby nodes require auto-unseal functionality to ensure that they are ready to participate in failover resiliency in a timely fashion.
