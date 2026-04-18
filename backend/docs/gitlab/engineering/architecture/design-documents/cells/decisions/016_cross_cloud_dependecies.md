---
title: "Cross Cloud Dependencies"
owning-stage: "~devops::tenant-scale"
group: cells-infrastructure
creation-date: "2025-07-21"
authors: ["@sxuereb"]
coach:
approvers: []
toc_hide: true
---

{{< engineering/design-document-header >}}

## Context

Cells will be running both on AWS and GCP cloud providers.
This adds complexity since we have to provision a Cell on both cloud providers
independently.

## Decision

To provision a Cell in one cloud provider, we should not depend on the other cloud provider.

For example:

- **Root CA**: Maintain separate root CAs for each cloud provider instead of having a
  single shared root CA in GCP.
- **Runtime artifacts**: Push Docker images required to run the GitLab
  application to both [Google Artifact Registry](https://cloud.google.com/artifact-registry/docs) and [Amazon Elastic Container Registry](https://aws.amazon.com/ecr/).

+Benefits of this separation:

- **Clean architecture**: Our codebase will maintain clear separation between cloud providers.
- **Cost optimization**: Eliminates cross-cloud ingress/egress costs.
- **Improved resiliency**: If there is an outage on a single cloud provider, we can still
  operate Cells on the other provider.

## Consequences

- **Data duplication**: We will need to duplicate some data across clouds (Docker images, Helm
  charts), but this represents only a few GB of data.
- **Increased maintenance**: Managing duplicate infrastructure components will require additional
  operational overhead.
- **Synchronization complexity**: We must ensure consistency of artifacts across both cloud providers. This increases the risk of configuration drift between AWS and GCP environments which could result in difficult to debug issues or incidents.

## Alternatives

Use one cloud provider as the primary source for shared
resources (example; root CA in GCP only). This was rejected due to cross-cloud dependencies
and potential single points of failure.
