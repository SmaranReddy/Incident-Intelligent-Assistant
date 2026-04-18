---
title: Isolated service components
status: proposed
creation-date: 2025-11-26
authors:
  - "@gitlab.mschoenlaub"
coaches:
  - "@splattael"
dris:
  - "@kkloss"
owning-stage: ~devops::developer experience
participating-stages: []
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

> [!note]
> This is a sub-blueprint for [Cloud-native development](../).

## Summary

This design document proposes **containerizing GitLab components** running in GitLab Development Kit (GDK) to address
current pain points stemming from GDK's tight coupling to developers' local systems
by isolating running services from each other and the developer's bare-metal.

## Motivation

### The Tight Coupling Problem

GDK's current architecture is tightly coupled to the developer's host system, meaning it assumes direct control over and access to system-level resources.
GDK requires specific versions of system packages (PostgreSQL, Redis, Ruby) installed directly on the host, competes for shared resources like ports and file paths, and depends on OS-specific installation procedures. This tight coupling creates the "works on my machine" syndrome and makes GDK fragile in the face of OS updates, package manager changes, or other tools competing for the same resources, impacting developers' experience and velocity.

### Current Pain Points

**Unpredictable Installation:** Setup time varies from minutes to hours based on system conflicts, and same GDK version behaves differently across machines due to environmental differences.

**Runtime Instability:** OS updates break GDK by changing system libraries, while port conflicts and environment variable pollution interfere with other development tools.

**High Maintenance Cost:** System updates can break GDK leading to Development Tooling team spending 30-40% of capacity on environment-specific debugging, and developers losing hours to days resolving machine-specific installation issues.

### Why Previous Solutions Fall Short

Recent attempts to reduce installation friction - such as pre-building binaries for satellite services in CI/CD pipelines - have proven [insufficient](https://gitlab.com/gitlab-org/gitlab-development-kit/-/issues/3111). Pre-compilation remains [unreliable](https://gitlab.com/gitlab-org/gitlab-development-kit/-/jobs?statuses=FAILED&name=compile&kind=BUILD) with frequent pipeline failures, consumes significant compute resources for architecture-specific builds (Mac arm64) that aren't useful for production, and shifts binary compilation ownership to Development Tooling for services we don't maintain. An earlier containerization effort [GitLab Compose Kit](https://gitlab.com/gitlab-org/gitlab-compose-kit) attempted isolation but lacked macOS support, limiting its adoption.

### What We Need

An isolation layer that decouples GDK from the host system by providing consistent runtime environments, eliminating conflicts with local tools, and removing the need for system-level package installation. This enables predictable setup across all platforms and eliminates system-specific bugs.

### Goals

- Improve the stability of GDK workflows
- Isolate third-party and satellite services from the hardware architecture on which the GDK runs
- Ease the integration of future components into GDK

### Non-Goals

- Create a development-platform built for or on top of CNG
- Isolate the monolith from the hardware architecture of developers' local system
- Manage Kubernetes clusters or any kind of physical or virtual infrastructure in GDK
- Support remote Docker hosts to run containers
- Create parity with production environments.

## Proposal

We propose introducing a **Component Runtime** abstraction layer that decouples GDK from how services are executed. Currently, GDK directly manages services as host processes. We will refactor this into a generic **Service Runtime interface** with two implementations:

1. **Process Runtime**: Manages services as native host processes (the current approach, preserved for backward compatibility)
2. **Container Runtime**: Manages services as isolated containers (the new default for improved stability)

This abstraction allows GDK to switch between execution modes without changing its core orchestration logic, enabling a gradual migration path from process-based to container-based service management.

### Service Runtime

A minimum viable component runtime exposes the functionality to:

1. Start a (sub)set of services belonging to a component.
2. Stop a (sub)set of services belonging to a component.
3. Restart a (sub)set of services belonging to a component.
4. Tail the service logs for a component.
5. Get the status (last exit code, running, stopped, ...) of any or all services.

The service runtime **must** be decoupled from the services it runs,
to ensure a high degree of testability by separation of concerns, and would have to work from the configuration of a service.

This interface should be a formalization of what GDK currently offers through its integration to [runit](https://smarden.org/runit/runsv.8).
The configured `version` of a service generally refers to a git reference, which can either be a tag, branch or commit reference.

> [!note]
> While a subset of services is also still run by virtue of a Procfile for legacy reasons, we do intend to [fully deprecate](https://gitlab.com/gitlab-org/gitlab-development-kit/-/work_items/904) this in an effort to provide a golden path forward.

### A Runit Runtime

As mentioned above, the GDK currently executes most services by using [runit](https://smarden.org/runit/runsv.8),
orchestrating the creation of various templated files and starting one process per service on the local system of users.
The configured `version` is used to either download a pre-built binary or checkout and build a specific git reference of the associated repository.
This is intentionally modeled after the feature set we currently offer, and so rather constitutes a refactoring.

### Docker Service Runtime

To facilitate a solution working on Linux and MacOS in a transparent and decoupled way,
we propose an implementation of the Service Runtime,
which can communicate through a socket-based Docker API.
For the time being we propose to only support Docker hosts running on the same physical machine
to avoid an entire class of failure modes related to real-world networking.
This will be a implemented using a to-be-selected Ruby library.
While [docker-api](https://github.com/upserve/docker-api) is an unofficial client library it is at least mentioned in the [official documentation](https://docs.docker.com/reference/api/engine/sdk/) and such should be considered a candidate.
This service runtime will expect the corresponding component to define full image references to services container images.
In the context of this runtime, the user-configurable `version` can generally be any kind of git reference at the moment, as long as a container image exists for this git reference.

### Runway Support

Runway is a Platform-as-a-Service [supporting the deployment of services](https://docs.runway.gitlab.com/runtimes/cloud-run/supported-features/) on Cloud-Run and Kubernetes.
While it is currently moderately [coupled to GCP](https://docs.runway.gitlab.com/team/guiding_principles/), it is expected to mature and be widely adopted across the organization.

While the Docker Service Runtime is the defined goal of the first iteration, our design is meant to be open to extension.
Future iterations of GDK could support a subset of the service configuration of Runway and map this to Docker or Kubernetes concepts
resulting in a Runway-supporting Service Runtime.

### Container Images

As production-parity is out of scope, viable container images can theoretically come from a variety of sources.

> [!note]
> These images are not development-optimized. They are simply the images needed to deploy a service in a containerized environment. As such, [following the Component Ownership Model](/handbook/engineering/infrastructure-platforms/production/component-ownership-model/#roles), they are artifacts intended to be owned by the Component Owner.

1. Ideally, CNG images would be great candidates for being used in a cloud-enabled GDK.
2. A subset of satellite services, such as the GitLab Runner, are readily available in the form of a [container image](https://gitlab.com/gitlab-org/gitlab-runner/container_registry/29383).
3. Third-party services, such as PostgreSQL or NGINX, have officially backed and maintained container images readily available for newer releases.

For GitLab-developed services we intend to strongly focus on using the images provided by CNG.

Using CNG images for GitLab-developed services is a great way to dogfood the multi-arch images we already build for our customers.
Additionally, it avoids costs associated with developing and maintaining separate container image pipelines.

We have to be cognizant of some limitations at the moment, which will effect the degree to which users can make use of the container-based Service Runtime.
Where a particular service or service version is not supported by CNG, GDK will provide graceful degradation in the sense that existing source-based Service Runner will be used.

#### Only a subset of services is currently supported by CNG

Some services, such as the AI Gateway are currently [not integrated with CNG](https://gitlab.com/gitlab-org/gitlab-development-kit/-/issues/3058#note_2838647818).
Despite this, the longer-term goal is to consolidate our service landscape and increase adoption of CNG across the organization and this proposal might support this endeavour.
If a Component Owner would like to see their component integrated with containerized runtime support, they would need to integrate their service with CNG first.

Existing components which are not integrated with CNG will not be able to benefit from running as containerized services.
They will still be supported as source-based components in a backwards-compatible way.

I assume that the adoption of CNG will increase over time. Any necessary cross-collaboration is likely more meaningful than reinventing another image pipeline in parallel to CNG's infrastructure, both from an initial implementation and maintenance point of view.

#### CNG will not provide images for all commits of a satellite service

Because building images for every commit on all our services would simply be cost-prohibitive, CNG builds
only certain versions of satellite services.

For some components, such as the [GitLab Registry](https://gitlab.com/gitlab-org/gitlab-development-kit/-/blob/main/gdk.example.yml?ref_type=heads#L527), the GDK currently allows users to configure a specific version to be used.

In general, this version can currently be an arbitrary git reference - even commits not associated with any tag.
While we would still support this use-case, **the user will not be able to benefit from containerized services**.

For other components, such as the GitLab HTTP Router, the GDK does not support the version to be directly configured by the user.
Instead, it is read from specific file, similar to the [version files mentioned in the GitLab Release Platform design document](https://gitlab.com/gitlab-com/content-sites/internal-handbook/-/merge_requests/7674/diffs#de7f639f645587dc201411c8c0d5a385112b83d2_0_193).

When a service is integrated in [the GitLab Release Platform's Release Manifest](https://gitlab.com/gitlab-com/content-sites/internal-handbook/-/merge_requests/7674/diffs#de7f639f645587dc201411c8c0d5a385112b83d2_0_191), the Release Manifest can be used
to find suitable CNG images for the currently checked-out working copy of the monolith.

When a service is not integrated in [the GitLab Release Platform's Release Manifest](https://gitlab.com/gitlab-com/content-sites/internal-handbook/-/merge_requests/7674/diffs#de7f639f645587dc201411c8c0d5a385112b83d2_0_191), we will not [not be able to retrieve the CNG image](https://docs.google.com/document/d/12TZ_z6f_h1dIf8REeUqamfKBzqyIrU7FABtwajyx4uA/edit?usp=sharing).
**In this case, the user will not be able to benefit from containerized services**

I assume that, as adoption of the proposed [Release Platform](https://gitlab.com/gitlab-com/content-sites/internal-handbook/-/merge_requests/7674/diffs#de7f639f645587dc201411c8c0d5a385112b83d2_0_191) grows, only few edge cases will remain, where
developers cannot use the container-based Service Runtime.

### Relation to UBT / TUBE

UBT and TUBE are key initiatives to unify the build process across the organization, which will enable Component Owners across the organization to build container images faster and more efficiently.
These initiatives are still [under active development](https://gitlab.com/groups/gitlab-org/distribution/build-architecture/-/work_items/4) and it's expected that we can evaluate their use in the latter half of FY27.
Ultimately these initiatives are expected to further drive the adoption of CNG across the organization, which will have synergistic effects with the proposed initiative.

## Alternative Solutions

We considered both, remote and local Kubernetes clusters as an alternative, but ultimately [decided against it](https://docs.google.com/document/d/1z2yufP5OTFjZ5lxiN10VRAEvbleGSXShsMKeGX7Xss4/edit?usp=sharing)

This seemed like a promising approach initially, because of the organizations desire to focus on CNG and because there are tools like [DevSpace](https://devspace.sh)
and [Tilt](https://tilt.dev), which we were hoping to use to craft a cloud-native Development Environment.

However, there are some conceptual issues related to managing a Kubernetes cluster from within GDK.

Providing only the option to use remote Kubernetes clusters is likely not inclusive of community members since most lack access to hosted infrastructure.

While team members could obtain budget approval, community support requires local Kubernetes options like MiniKube.
This, in turn, would either require manual setup by developers, increasing contribution friction, or adding maintenance overhead to GDK by providing automated management of a local Kubernetes cluster.

Furthermore, concerns were raised that Kubernetes itself could become a new source of instability in this context, especially if we had to support local and remote Kubernetes clusters.

> [!note]
> There are [architectural proposals](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/16581) underway which could change the variables on the long run. It is worth noting that this proposal enables us to implement a Service Runner deploying services on a local Kubernetes cluster, should the need therefor arise.

We have also considered using the new container runtime of MacOS briefly.
However, as this is still a relatively new feature with likely instabilities and missing features, we discarded this option for the moment.

As it does not provide a Docker API-compatible interface, it would require a seperate implementation just for MacOS, ultimately defying the point of providing isolation from the host operating system.

## Implementation and Rollout Plan

We intend to migrate a few components to the new Service Runner, while essentially leaving GDK's default configuration unchanged.

As there exists a [service gap](https://gitlab.com/gitlab-org/gitlab-development-kit/-/issues/3058#note_2838647818) between CNG and GDK,
we intend to start with an often-used service already integrated with CNG.
The GitLab Runner and the GitLab Registry would provide for suitable candidates.  

This would allow us to demonstrate value early on in the process and provide valuable examples to guide component-owning teams in the integration of new services, as migrations will follow a self-service approach.
We also intend to use tasks to implement new features for existing services to nudge teams into migrating existing services to the new runner.
Furthermore, once sufficient documentation is available, we intend to enforce integration of new services through the new self-service approach, while offering support when needed.
