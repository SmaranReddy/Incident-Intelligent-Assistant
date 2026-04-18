---
owning-stage: "~devops::tenant scale"
title: 'Organizations ADR 004: Organization path scope'
---

## Context

With the existing product it's not always possible to determine the current
Organization context. We need to modify the current system to properly namespace
Organizations.

## Decision

We have decided to implement path based Organization scopes. Specifically we
will prefix Organization routes with an `o` such as:

`https://gitlab.com/o/my-organization/my-group/my-project/-/issues/1234`

We will maintain existing routes to ensure backward compatibility.

The Default Organization will not be represented within the
Organization path scope. This allows single Organization installations such as
on Self Managed and Dedicated to avoid verbose paths.

REST and GraphQL requests will remain at `/api/v4` and `/api/graphql` and Organization will be specified with an `organization-id` parameter.

## Consequences

- All existing routes will remain to ensure backward compatibility.
- Organization routes will exist within the `/o/` scope.
- REST and GraphQL routes will remain as they are today, and `organization_id` will be supplied through existing methods.
- The `o` can't be used by a Group or User.
- If the `o` is taken on on-premise we would fail creating the Organization through a model validation and ask the user to release the `o`.
- There is a [decision tree](https://lucid.app/lucidspark/42a3cb25-7b85-49e2-8539-952df0781e1e/edit?beaconFlowId=C3B5807A2EB70840&invitationId=inv_9908f2a1-446d-4f33-9469-07e8b9ff8c76&page=0_0#) that details the mappings.

## Alternatives

### Segmentation alternatives

Alternative Organization context deleneation methods were detailed in a
[comprehensive spreadsheet](https://docs.google.com/document/d/1cssSKFJiy-wm07S5ThyacsOhrlTYJmwU13oLnlroEH4/edit?tab=t.0).

Of the items in the spreadsheet, the custom sub-domain was the next closest fit
but this created a number of problems:

- We would have to build this ability into the product at the tenant level. It is currently an instance level feature.
- The scheme was not compatible with a number of our services. For example, SSH is not able to determine Organization by hostname and would require another identifier such as by username.
- Organizations promoted out of the Default Organization and into their own Organization would not maintain backward compatibility without some kind of hybrid domain approach.
- Having to add subdomains for adjacent services if those will use separate hostnames.
- There were many other concerns.

### Scoping identifier

The `o/` is a typeable identifier.

Besides `o`, the tilde (`~`) symbol was also considered as it's compatible with all GitLab services and has some historical precedence as a URL scoping mechanism. However the `~` character can be confusing for less technical users and it's also difficult to enter on mobile devices and some keyboards so was disregarded.

Others were rejected like: `/orgs/` and `/-/organizations/`.
