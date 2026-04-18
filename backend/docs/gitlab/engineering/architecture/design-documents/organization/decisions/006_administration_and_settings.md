---
owning-stage: "~devops::tenant scale"
title: 'Organizations ADR 006: Administration and Settings'
status: accepted
creation-date: "2025-08-26"
authors: [ "@alexpooley" ]
---

## Context

### The current state

Administration of GitLab occurs at the Instance Level. Administration
functionality includes:

- Management of customer data such as [User management](https://docs.gitlab.com/administration/administer_users/)
- Application settings such as [Restrict visibility levels](https://docs.gitlab.com/administration/settings/visibility_and_access_controls/#restrict-visibility-levels)
- Instance configuration like [Sign-up Restrictions](https://docs.gitlab.com/administration/settings/sign_up_restrictions/)

### What's changing

The Organization Level will replace most of the Instance Level and the above
functionality will move accordingly. "Management of customer data" and
"Application settings" will move to the Organization Level, while "Instance
configuration" will remain at the Instance Level.

Similarly, the Admin role is an Instance Level role. The Organization Owner role
will replace most of the Admin role. While the Admin role can access all data,
the typical use of this role will be to modify Instance configuration.
Organization Owners will reside within the Organization Level where "Management
of customer data" and "Application settings" will now reside.

These Instance and Organization Levels will exist across all GitLab platforms.

### How this affects Self Managed and Dedicated

Self Managed (SM) and Dedicated will only have a single Organization for now.
With a single Organization the Organization experience is a transparent one and
their experience will remain the same as today.

The Admins within SM and Dedicated will also be the Organization Owners of this
single Organization, though both the Organization Owner role and the
Organization will remain transparent to customers.

### How this affects GitLab.com

On GitLab.com select GitLab Team Members will have Admin roles, and customers
will receive Organization Owner roles. Admins will have rights to administer
Organizations. Organization Owners can only administer their own Organizations.

Organizations and the Organization Owner role will be visible on GitLab.com.
Customers on GitLab.com will have access to the Organization Owner role within
their own Organizations. Any features added to Organization Level such as
features moved down from Instance Level or entirely new features will also be
accessible to customers.

### Execution challenges

The various components of the Organization Level will be developed by the
Organizations group, but each GitLab team will need to move their own features.
This means we will encounter communication and co-ordination challenges.

As the primary goal is data migration to the ProtoCell we are under a time
and capacity constraint. The solution must be feasible within these constraints.

## Decision

Perform a port of the Admin section of GitLab where access and functionality is
defined by the user's role. The key idea behind the port is to re-use as much
existing functionality as possible to avoid any rewrites.

Instance Level administration by Admin users will continue to occur on `/admin`.

Organization level administration by Organization Owners will occur on
`/o/my-org/admin`.

There used to be a one to one mapping with a GitLab Instance having one Instance Level.
Now there are many Organization Levels per GitLab Instance. The port will need to
accommodate this change, while adhering to the above context.

From a backend code perspective an Organization Admin Controller will be
established that will be a landing point for ported Admin Controllers, and will
denote Organization compatibility of that controller and it's associated
actions.

## Consequences

A new UX design will have to be established that facilitates:

1. The existing SM and Dedicated experience to continue.
2. That same SM and Dedicated Admin experience but as an Organization Owner on GitLab.com

A set of developer guides and examples will be constructed to aid other teams to
port and/or develop new features within the Organizations Admin area.

## Alternatives

We were previously building a new parallel Organizations Administration area.
There was work here that duplicated existing work in the Global Admin area. The
handling of Organizations visibility was also more prominent that it should have
been.
