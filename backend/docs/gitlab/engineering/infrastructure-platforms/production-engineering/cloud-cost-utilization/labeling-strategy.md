---
title: "Labeling Strategy"
description: "Core labels for cost allocation and resource attribution in GCP"
---

## Overview

This page defines what is considered *compliant* with our cost allocation and attribution strategy.

{{% alert color="warning" %}}
The scope for this document is GCP only.
{{% /alert %}}

This page documents the required labels for resources and projects.
These requirements are a subset of the [Infrastructure Labels and Tags](/handbook/company/infrastructure-standards/labels-tags/) standard.

### Compliance with this strategy

Compliance is evaluated at the level of GCP billing *line items*.

A line item is considered *compliant* if it falls into one of these buckets (defined in more detail below):

1. Resource level `gl_service` label with a valid reference to the service catalog
2. Project level `gl_service` label with a valid reference to the service catalog
3. General Networking costs: `service.description = 'Networking'`
4. Support costs: `service.description = 'Support'`

Line items are evaluated in this precedence order — the first matching bucket wins.

### Service catalog based allocation

The primary goal of this strategy is to associate line items with services in the [GitLab service catalog](https://gitlab.com/gitlab-com/runbooks/-/blob/master/services/service-catalog.yml).

We use the `gl_service` label to reference a service in the catalog. This label enables:

- **Cost allocation** - Understanding which services are driving costs
- **Resource attribution** - Identifying ownership and operational responsibility

The label can be defined at two levels:

1. **Resource level**: Most granular and preferred
2. **Project level**: Fallback for the entire GCP project when resource-level labels are absent

Project-level labels provide a faster way to attribute all project costs to a single service without labeling individual resources.

All projects with a clear mapping to a single service should have a project-level `gl_service` label. This also ensures non-resource costs (for example, networking) are attributed to the correct service.

#### How to implement

All services should:

- Have an entry in the [GitLab Service Catalog](https://gitlab.com/gitlab-com/runbooks/blob/master/services/service-catalog.yml)
- Be managed in Terraform, with labels applied to resources and projects

**Adding or updating services in the service catalog**

To add a new service or update an existing one, follow the [service catalog documentation](https://gitlab.com/gitlab-com/runbooks/-/blob/master/services/README.md?ref_type=heads). For reference, see this [example MR](https://gitlab.com/gitlab-com/runbooks/-/merge_requests/10145/diffs) that adds a new team and service.

**Adding the `gl_service` label to existing infrastructure via Terraform**

To add `gl_service` to existing infrastructure, update the relevant Terraform configuration. For reference, see this [example MR](https://ops.gitlab.net/gitlab-com/gl-infra/config-mgmt/-/merge_requests/12985/diffs?commit_id=ea659632c3c3c66b80893faaec17e69a72cd8611).

Values must reference an entry in the GitLab service catalog. Example values include:

- `ci-runners` - GitLab Runner service
- `ci-jobs-api` - CI Jobs API service
- `frontend` - Frontend application service
- `patroni-ci` - Patroni cluster for CI

> Please consult the [service catalog](https://gitlab.com/gitlab-com/runbooks/-/blob/master/services/service-catalog.yml) for the complete list of valid service catalog entries. Consider [adding to the catalog](https://gitlab.com/gitlab-com/runbooks/-/blob/master/services/README.md?ref_type=heads) if a service is not represented yet.

### Non-resource cost: Networking, Support

Networking and Support-related cost can't have resource labels.

It is possible to associate these cost with a service using a project-level `gl_service` label.
However, for generic projects (e.g. `gitlab-production`), there is no 1:1 mapping between the project and a service in the catalog.

For these cases, we capture Networking and Support cost using separate buckets (3, 4 in above list).

These buckets are not associated with a service and don't have an explicit owner.

## Related Documentation

For the complete infrastructure labels and tags standard, including additional labels and realm-specific requirements, see the [Infrastructure Labels and Tags](/handbook/company/infrastructure-standards/labels-tags/) handbook page.
