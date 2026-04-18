---
owning-stage: "~devops::tenant scale"
title: 'Organizations ADR 001: Organization context resolution'
---

## Context

The GitLab data paradigm will shift from a global pool of instance wide data, to a multi-tenant pattern of isolated organizations within multiple instances.

To accomodate this change Organization access will have to occur in an isolated environment through the full data stack. The solution ideally needs to have minimal impact on the engineering team.

We have been through a number of Organization/Cell roadmaps over the last couple of years which have informed this decision.

## Decision

All GitLab execution paths will be expected to operate within the context of a designated Organization. These paths include web requests, scheduled tasks, cron jobs, and data migrations. It will be possible to operate without an Organization context but this will be considered an exceptional circumstance.

Database interactions will filter according to the currently designated Organization. This interaction will be transparent for engineers performing day to day operations. Access to the global data pool will be possible but will be exceptional.

## Consequences

- We need to ensure all entry points define a Current Organization.
- The database will filter view of the data according to the Organization context.
- All other direct/indirect services will need to accomodate similar contextual data access.
- We allow global access for exceptional cases.

## Alternatives

The alternative to a global Organization context is to specify the context selectively. This would require a lot of application level changes that would be very difficult to complete.
