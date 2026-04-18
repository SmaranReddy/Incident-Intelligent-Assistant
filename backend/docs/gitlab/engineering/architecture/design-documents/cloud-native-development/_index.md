---
title: Cloud-native Development
status: proposed
creation-date: "2025-11-12"
authors: ["@gitlab.mschoenlaub", "@kkloss"]
coaches: ["@splattael"]
dris: ["@mgamea"]
owning-stage: "~devops::developer experience"
participating-stages: []
# Hides this page in the left sidebar. Recommended so we don't pollute it.
toc_hide: true
---

<!-- Design Documents often contain forward-looking statements -->
<!-- vale gitlab.FutureTense = NO -->

<!-- This renders the design document header on the detail page, so don't remove it-->

{{< engineering/design-document-header >}}

<!--
Don't add a h1 headline. It'll be added automatically from the title front matter attribute.

For long pages, consider creating a table of contents.
-->

## Glossary

| Term      | Meaning |
|-----------| ------ |
| GDK       | The [GitLab Development Kit](https://gitlab.com/gitlab-org/gitlab-development-kit/), the development tool to set up local GitLab instances. |
| Component | Top-level project, such as Postgres, Gitaly, or Workhorse. It can have multiple services (like `praefect-gitaly-1`, `praefect-gitaly-2`, ...) and is associated with zero or one repositories. |
| Service   | The runtime blueprint of a component, such as the Rails web service and Sidekiq background jobs for the Rails component. |
| Runtime   | The service process managed by GDK. We usually refer to a runtime as a running service. |
| GDK core  | The proposed core framework of GDK. A development orchestrator, similar to `docker-compose` but much more powerful and GitLab-specific. |
| SDLC      | Software Development Life Cycle |
| CNG       | [Cloud-native GitLab](https://gitlab.com/gitlab-org/build/CNG) images |
| UBI       | A distribution of GitLab based on Red Hat Universal Base Image |

## Summary

Today, team members, customers, and community contributors (contributors)
use the GitLab Development Kit (GDK) to set up a GitLab development
instance on their local computers.

GDK was created as a side-project in 2014, meant to reduce the manual
labor involved with configuring GitLab development instances. Since then,
the GitLab application has grown massively in features and complexity.
Our production environments employ modern, cloud-native tech
to deploy GitLab in an isolated and reproducible way.

However, development environments have not kept up because GDK didn't have a dedicated owner and relied mainly on voluntary contributions.
This significant under-investment in development environments now poses scalability and stability risks as the GitLab application continues to scale in complexity.

To address this, we have improved stability reactively during FY26 Q2 but this
doesn't prevent or mitigate future instabilities in development environments.
Therefore, we still see a major opportunity to iterate
towards adopting cloud-native standards in GDK.
This would solve critical problems like service ownership, stability, and resource sharing
conflicts, ultimately making development more efficient and scalable.

This proposal also better aligns development environments with production
environments that already use [Cloud Native GitLab](https://gitlab.com/gitlab-org/build/CNG)
and other initiatives like the [Component Ownership Model](../../../infrastructure-platforms/production/component-ownership-model.md),
[self-managed segmentation](../selfmanaged_segmentation/_index.md),
and [self-managed Runway](https://gitlab.com/gitlab-com/gl-infra/platform/runway/team/-/issues/413).

## Motivation

A recurring problem source with development at GitLab has been
platform-related instabilities, such as missing or incomplete compilation
dependencies that are managed globally on the systems of users.
Furthermore, a lot of services are compiled from source and run without any form of
isolation, causing resource sharing conflicts, which makes services less stable.

[Our efforts to improve stability during FY26 Q2](https://gitlab.com/groups/gitlab-org/quality/tooling/-/epics/44#note_2887297127),
resulted in a reduction of weekly GDK-using team members' affected by errors by 30 percentage points.
Nevertheless, 20% of GDK-using team members still encounter an error on a weekly basis.

Likewise, stability of development environments is still entirely owned by
the Development Tooling team, even though GDK is orchestrating
services owned by multiple other teams.
Due to architectural constraints in GDK, we cannot attribute instabilities to certain
components or component owners.
This causes reactive work for the Development Tooling team to remediate instabilities,
even though they are not experts in these components.

Both the architectural constraints and this lack of ownership
causes teams not to engage with the Development Tooling team to improve long-standing challenges,
such as the challenging state of setting up the AI components in GDK.

Furthermore, the "DX Surveys" held in 2025 [show the need](https://gitlab.com/gitlab-org/quality/tooling/team/-/issues/126) (internal)
for more stable and easier to set up development environments.
If we keep the ownership and architecture as-is, we can only alleviate a fraction of
these goals because collaboration with component owners is difficult.

### Goals

- Establish clear component ownership and stability attribution.
- Switch to a self-service model with a streamlined development setup.
- Run GDK services in isolated environments for better stability.
- Adopt cloud-native tech that is already being used in production environments.
- Preserve developer experience with an edit mode for rapid and efficient code changes and debugging.

### Non-Goals

- Create parity with production environments.
- Create review environments with GDK.
- Allow GDK to manage a Kubernetes cluster for development.

## Proposal

### Decisions

- [001: Support development mode on component-level](decisions/001_component_level_editing_mode.md)
- [002: Containerized Service Runtime](decisions/002_containerized_service_runtime.md)

We propose a three-phase approach to future-proof development of the
GitLab application using GDK:

1. [Modularization (logical isolation)](phases/1-modularization)
2. Isolation (physical isolation)
3. Production alignment

Each phase is discussed in its own sub-blueprint. We are including a
high-level, minimal technical summary of each sub-blueprint in this
blueprint.

A major advantage of this phased approach is that we are iteratively delivering a tangible improvement for GDK users after each phase.

### Modularization

First, we modularize all GDK services into logical components and create
a declarative interface that component owners can use to define
installation, configuration, and runtime metadata.

By separating the **GDK core** engine from the **GDK components** that it
orchestrates, we establish a clear border between the orchestration layer and component definition.
This ensures the scalability of GDK, streamlines adding new GDK components,
and allows teams to own their components in development like they own them in production.
Additionally, this approach allows us to attribute stability of GDK to individual components,
helping us designate error budgets similar to how we approach this for production environments.

Our vision is that GDK core feels like an orchestrator—similar to
`docker-compose`—where any team can add components owned by them without
tight coupling, by declaratively fulfilling an interface.

### Isolation

Second, we incrementally migrate services from running natively on the
machines of contributors to a cloud-native approach. Services can run in two modes:

- **Edit mode**: Services developed by the GDK user run natively on the host system with direct access to source code for live editing and debugging.
- **Production mode**: Services run isolated using production-like Docker containers, providing reproducibility and stability.

By default, services run in production mode using cloud-native tech. A lot of
complex stability-related problems are caused by system-level
dependencies being installed incorrectly or resource (sharing) conflicts.

If a service misbehaves, right now a user might have to reinstall system
dependencies or change a system setting. By using an isolated environment
per service, all steps needed to restore it to original is to delete the
service instance and recreate it from the definition established in
the previous phase.

By using cloud-native tech by default for all service runtimes, this
simplifies the setup for essential components like the GitLab Runner,
which can currently not be enabled by default because not every user has
their device configured to run containers. Moreover, we're also aligning
development environments with production environments that already use
one or more containers for each individual service.

To ensure that we're not degrading the developer experience due to the
complexity challenges of cloud-native setups, we will not be using
Kubernetes in local development. To make changing code and debugging
service as easy as possible, this phase requires components to provide
an _edit mode_ that installs dependencies directly on the system.

### Production alignment

Once all services have been containerized and run in isolated
environments, this creates an opportunity to even further align with
production environments. At time of writing, however, the exact
differences between development and production environments that cause
defects from being missed in development are unclear.

While implementing the previous two phases, we propose to analyze the
vectors that cause defects to be missed in development or only surface
in production environments. It is likely to turn out that some defects
are physically impossible to be caught in development environments
because production loads are (near) impossible to be reproduced in
non-production environments.

The idea of production parity is also contentious because across
software industry, production environments are typically optimized for
performance at scale and stability. Developing in a production
environment must necessarily be slower than in a development environment
because production environments lack tools like hot reloading that make
development significantly more efficient.

Instead of full production parity, we propose a middle ground, where we
make a reasonable effort to not diverge from production too much without
a benefit for development, which still allows us to obsess over developer
experience.

In case a developer needs to verify a change in an environment that has
production parity, we should optimize the validation using CNG or other
tools, such as Runway, providing a fast review option through GDK. These tools
already accurately reflect production environments and adopting them for
feature validation requires little changes to GDK.

## Design and implementation details

Design and implementation details are discussed separately in each
sub-blueprint.

## Alternative Solutions

### Do nothing

Retaining the status quo or only eliminating minor technical debt, such
as the component instructions being defined in boilerplate-ridden
Makefiles, increases the risk for the maintenance cost of managing
development environments, ultimately causing more instability and
engineering hours wasted.

Today, product engineering teams are already hesitant to own their
components in GDK, as was recently demonstrated by the strenuous state of
setting up the AI components in GDK, which prompted a [reactive effort](https://gitlab.com/groups/gitlab-org/quality/tooling/-/epics/86)
to remediate this.

- **Pros**: immediate reduced investment in development environments, freeing resources for other development initiatives, such as Feature Gates and LabKit.
- **Cons**: long-term increased cost of GDK maintenance and engineering inefficiency.

### Adopt CNG Helm charts and images for development

We've considered using the [CNG Helm charts](https://gitlab.com/gitlab-org/charts/gitlab) and [container images](https://gitlab.com/gitlab-org/build/CNG) for
development.
A key problem with these images and the Helm charts is that they are
[fundamentally optimized for production usage](https://gitlab.com/gitlab-org/gitlab-development-kit/-/issues/3055#note_2852833789) by customers. Overcoming
this design challenge would require us to add development libraries to
these images and development-related settings to the Helm charts, which
will notably complicate the CNG image project. Further, doing so could
encourage customers to run unsupported setups or risk security
vulnerabilities from development libraries and settings impacting
production environments.

Using CNG (or [UBI](https://docs.gitlab.com/charts/advanced/ubi/)) images for development might still be feasible for the
production mode suggested in the second or third sub-blueprint. That
would represent a compromise that balances production parity with development efficiency.

- **Pros**: a single source of truth for environment setups.
- **Cons**: split ownership of CNG between distribution and development tooling, potential security risk to customer installations, and highly complex development setups.

### Develop directly using CNG or Omnibus instances

To facilitate the largest possible production parity, we could force
development environments to run solely on a production instance powered
by CNG or Omnibus. This would, however, have a huge negative impact on
the velocity of engineers because developer experience features like hot
module/class reload or incremental compilation of frontend assets and Go
services aren't available in production.

Engineering time is among the most expensive time across the whole SDLC,
so we should optimize for developer experience and find other, cheaper
safeguards that prevent defects from escaping into production.

- **Pros**: a single source of truth for environment setups and most optimal production parity.
- **Cons**: inefficient development due to physical limitations of production environments.
