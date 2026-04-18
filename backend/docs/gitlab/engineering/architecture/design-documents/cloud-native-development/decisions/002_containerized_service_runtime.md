---
title: 'Cloud-native Development ADR 002: Containerized Service Runtime'
toc_hide: true
---

## Context

As developers rarely contribute to more than one component at the same time,
it makes sense to fall back to some form of "production-like" deployment,
for services that are not to receive code changes in the current session.

Currently, GDK supports running precompiled binaries for some services, like Gitaly, to avoid time-consuming recompilation,
even when Gitaly is not the service a developer wants to contribute to in a session.
One drawback of this approach is that we have to maintain the compilation jobs and artifacts for each platform GDK is supported on, in particular, MacOS.

To reduce coupling between the services and the system of the host, we evaluated using a (local or remote) Kubernetes cluster as a deployment target.

The original idea was to reuse CNG Helm Charts and the underlying images to deploy services to Kubernetes when configured in production mode,
while using extended images, based on the CNG images, in development mode.

This should ensure smooth startup performance and a level of production alignment, while using established tooling, such as SSH Development integrations in IDEs, for cloud-enabled development.

However, our evaluation concluded that using the CNG Helm charts would [currently not be feasible](https://gitlab.com/groups/gitlab-org/-/epics/17447#note_2889556056)
for several reasons, detailed in the referenced closing statement.

## Decision

Services, which are not to receive code changes can be started in a containerized mode using a [container runtime](./../phases/1-modularization/#service-runtime-types).

## Consequences

This will allow us to use a lightweight approach to ensure isolation of configured services from the host system.
The decision will further ensure we harness stability and startup performance benefits, without introducing unnecessary
complexity and potential sources of instability.

Furthermore, it does not preclude us from adding a Kubernetes-based approach in future iterations.

## Alternatives

We considered both, remote and local Kubernetes clusters as an alternative.

This seemed like a promising approach because of the organizations desire to focus on CNG and because there are tools like [DevSpace](https://devspace.sh)
and [Tilt](https://tilt.dev), which we were hoping to use to craft a cloud-native Development Environment.

However, there are some conceptual issues related to managing a Kubernetes cluster from within GDK.

Providing only the option to use remote Kubernetes clusters is likely not inclusive of community members since most lack access to hosted infrastructure.

While team members could obtain budget approval, community support requires local Kubernetes options like MiniKube.
This, in turn, would either require manual setup by developers, increasing contribution friction, or adding maintenance overhead to GDK by providing automated management
of a local Kubernetes cluster.

Additionally, while an SSH-based development workflow is portable across major IDEs, ensuring that custom shell extensions
and configurations for editors like Emacs work, would require extra effort.

Furthermore, concerns were raised that Kubernetes itself could become a new source of instability in this context, especially if we had to support local and remote Kubernetes clusters.

### References

- [Decision on going with a containerized service runtime](https://docs.google.com/document/d/1z2yufP5OTFjZ5lxiN10VRAEvbleGSXShsMKeGX7Xss4/edit?usp=sharing)
- [Feasibility Matrix of Services](https://gitlab.com/gitlab-org/gitlab-development-kit/-/issues/3084#note_2864373908)
