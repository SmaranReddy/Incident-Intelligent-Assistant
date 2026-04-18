---
title: 'Cloud-native Development ADR 001: Component-level local-only editing mode'
toc_hide: true
---

## Context

Today, GDK currently orchestrates around a large number of services, which can be executed in several ways:

**Source-based services**: Originally, GDK managed a per-service working directory coupled to a git repository compiling enabled services on demand, whenever the GDK users executed commands like `gdk start` or `gdk update`.
The Rails application / monolith, the [HTTP Router](https://gitlab.com/gitlab-org/cells/http-router), and the Topology Service are prominent examples of services that still work this way.

**Binary-only services**: These services can be run from pre-compiled binaries.
Developers can configure the binary path from the source directory of the service.
The GitLab Runner, Workhorse, and Gitaly are all supported by GDK in this way, for example.

**Cloud-native Services:** These services can be run from pre-built container images. The GitLab Runner is one such example.

**Third-party Services:** These services are exclusively run from binaries. PostgreSQL, Elasticsearch, and Redis are examples of third-party services.
These services are generally not modified on a code level, and they are integrated into GDK mostly for the convenience of developers.

To improve the development workflow, especially the runtime of `gdk update`,
services with a comparatively low change rate, like the GitLab Runner,
were enabled to optionally run from container images or binaries, instead of only being source-based services.

However, we did this iteratively, largely without adapting the GDK architecture and without involving the owners of the services.

### Drawbacks of the current approach

The current approach does not scale well in relation to the frequent addition of new services to GDK.
This is true from an integration implementation perspective, as well as from a performance perspective.  

For example, [the GDK package registry](https://gitlab.com/gitlab-org/gitlab-development-kit/-/packages),
currently hosts MacOS native binaries for [Gitaly and other services](https://gitlab-org.gitlab.io/gitlab-development-kit/configuration/#skip-compile),
which are built within the GDK pipelines. The ownership of these jobs lies with Development Tooling and requires maintenance effort.
Secondly, as it stands, we are faced with a choice related to which deployment option to support,
whenever a new service is integrated with GDK.

<table>
  <tr>
    <th rowspan="2">Integration Option</th>
    <th colspan="3">Reduces...</th>
  </tr>
  <tr>
    <th>Build Tool Requirements</th>
    <th>Development Tooling Maintenance Effort</th>
    <th>Cross-org Duplicated Effort</th>
  </tr>
  <tr>
    <td>Source-based services</td>
    <td>❌</td>
    <td>❌</td>
    <td>❌</td>
  </tr>
  <tr>
    <td>Precompiled binary</td>
    <td>✅</td>
    <td>❌</td>
    <td>❌</td>
  </tr>
  <tr>
    <td>Containerized Service</td>
    <td>✅</td>
    <td>✅</td>
    <td>✅</td>
  </tr>
</table>

We conducted a comparison between the [different deployment modes](https://gitlab.com/gitlab-org/gitlab-development-kit/-/issues/3084),
as well as a [service-based feasibility analysis](https://gitlab.com/gitlab-org/gitlab-development-kit/-/issues/3084#note_2864373908) to identify a
suitable approach for the hybrid development environment as well as ideal candidate components for the first iteration.

Additionally, the GDK core has no standard interface with the services it manages. Therefore, we can't delegate the integration with GDK to the teams owning a service. Similarly, users have no consistent way to switch between binary-, container-, and source-based deployments across all services, due to the iterative and ad-hoc nature these services were integrated.

As an example, the default configuration in GDK today looks like this:

```yaml
gitaly:
  skip_compile: true
workhorse:
  skip_compile: true
runner:
  install_mode: 'binary'
```

Notice how there are multiple ways to toggle the same feature, deploy a service as a precompiled binary, and which one to choose can only be found by looking at the options for the service you are interested in.

## Alternatives

We considered several alternatives for running services where changes to the source code are reflected in the current instance.
This is described in [this issue](https://gitlab.com/gitlab-org/gitlab-development-kit/-/issues/3084).
One option considered was to continue deploying pre-compiled binaries, where possible, as we are doing today.
A drawback compared to the selected model is that we would have to continue being responsible for building these binaries, the same way we do now for Gitaly and other services.
With the organization having decided to aim for CNG (Cloud-native GitLab) being the default way of deployment, it makes more sense to rely on containers.

Furthermore, we conducted [spikes](https://gitlab.com/gitlab-org/quality/tooling/team/-/issues/121#note_2864497246) to use CNG Helm Charts
and/or the underlying container images directly to allow for cloud-native development workflows.
However, at this point, a solution harnessing CNG's artifacts would [not be feasible](https://gitlab.com/groups/gitlab-org/-/epics/17447#note_2889556056), due to
a significant gap between the services supported by CNG and those supported by GDK.

Consequently, [containerizing edit-mode services](https://gitlab.com/gitlab-org/gitlab-development-kit/-/issues/3084#2-dockerized-edit-mode-writing_hand) was rejected
because it adds various complexities around hot-reloading and debugging, due to additional filesystem and networking layers involved with containerized development environments.

Furthermore, while major IDEs support containerized workflows, it would likely disrupt the workflows of users
relying on shell extensions or other highly configurable local tooling, such as Emacs.

Ultimately, without being able to effectively harness CNG for the edit-mode, the additional complexity of working around these challenges does not provide sufficient benefits compared to a local "edit mode",
configurable on a per-service level.

## Decision

Follow a hybrid approach where **individual components** can be configured to start in one of two modes:

1. Run from a git working directory, compiling source code if necessary
2. Run from a pre-built containerized application or binary

Ideally, because the highest change-rate is in the monolith, the default configuration should provide for a result, where

1. All Third-party services (PostgreSQL, Elasticsearch) are not compiled from source.
2. The monolith is managed as a source-based service.
3. All optional services are managed as a non-source-based service.

### Configuration Layer

As detailed above, the current configuration layer requires that developers
changing their configuration consider

1. Inconsistent configuration keys to trigger different deployment modes.
1. Inconsistent support of binary vs. containerized deployments.

Instead, we will allow users to make certain services editable in a consistent way:

```yaml
runner:
  editable: true
```

Only when configured as being "editable", a component's services will be built on the user's machine from the source contained in the working directory.
The Component itself will [own the Service Runtimes it supports](/handbook/engineering/architecture/design-documents/cloud-native-development/phases/1-modularization/#service-runtime-types),
emphasizing the seperation of concerns between GDK as the orchestrator and the services themselves.

A developer wishing to make changes to Gitaly should be able to do so simply by

```shell
gdk config set gitaly.editable true
gdk reconfigure
```

### Separate GDK Core from Service Definition

As mentioned in the overall blueprint, we want to refactor GDK to separate the core functionality of orchestrating services from the service's definition itself. This will not entail any new tools to be used, but will rather entail changes to folder structure and existing `make`/`rake` tasks with the goal to define owners for components.

For the time being, component definitions will remain in the GDK repository but use `CODEOWNERS` to assign maintenance responsibility to component teams.
These centralized definitions ensure atomic updates and simplified dependency management.

We envision a documentation guide on how to integrate new services with GDK. Existing services would likely be ported to the new model by Development Tooling initially, with component teams taking over maintenance after the migration.

### Components in Development Mode

As explained in the [glossary](../#glossary), components such as Gitaly, can have multiple services (like `praefect-gitaly-1`, `praefect-gitaly-2`, ...).
In this case, configuring `gitaly.editable` as being `true`, should result in all instances of this component's services to be run from source code.
This will not increase the compilation cost, because it's only several logical nodes of the same service.
Should the need arise to run instances based on local source code together with already released versions of the same service,
for example to test backwards compatibility, we can follow up on this in future iterations by improving the service definition.

To reiterate: Only a component should know how it gets deployed and whether their configuration is valid.

## Consequences

The hybrid model decided upon will make it much easier and faster to integrate new services into GDK by allowing moving ownership to the respective teams,
as they know best how to build the necessary artifacts.

At the same time, this model will encourage teams to contribute to the respective service definitions, as they gain confidence that their changes would not have unintended consequences in GDK itself and errors would be compartmentalized.
Teams owning their service definitions inside GDK would also reduce the need for Development Tooling to approve these changes.
The rationale here is similar, teams know their services better than us.

Additionally, consolidating streamlining the configuration should encourage users to have higher confidence in optimizing it for their use case,
instead of simply relying on the default values of GDK.
It will also encourage developers to see the monolith and it's satellite services as decoupled components rather than a [distributed monolith](https://www.gremlin.com/blog/is-your-microservice-a-distributed-monolith).

### Use Cases

It is worth noting that a developer is likely to work on a service and its direct dependencies, which together constitute a **component**.
Team members are also unlikely to frequently switch between development activities on multiple services in a short period of time.

Additionally, the number of active developers and the number of incoming code changes, per such component, have a high variance.
There is a stark contrast between services, such as the HTTP Router, which have reached a mature feature-state, and actively developed services such as the [AI Gateway](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist) or the interconnected monolith.

Furthermore, the proposed approach considers use cases for newly onboarded and existing developers alike.

#### Onboarding targeting the monolith

A new user can [follow the existing documentation](https://gitlab-org.gitlab.io/gitlab-development-kit/#use-gdk-to-install-gitlab) on how to
set up a GDK. As the primary use case for most team members and contributors is to change the monolith, no configuration changes are necessary. The monolith will run in a source based way for the forseeable future and therefor, will always be `editable`.  

### Onboarding targeting Gitaly

A new user can [follow the existing documentation](https://gitlab-org.gitlab.io/gitlab-development-kit/#use-gdk-to-install-gitlab) on how to
set up a GDK. As the GDK will be optimized for the primary use-case, a developer wishing to make changes to Gitaly will be able to do so simply by executing

```shell
gdk config set gitaly.editable true
gdk reconfigure
```

The same would eventually apply for all other GitLab-developed services supported by the GDK.

### Switching from working on Gitaly to working on the monolith

When a developer has finished working on Gitaly they would need update the version of Gitaly to use.

Depending on the [outcome on the GitLab Release Platform](https://gitlab.com/gitlab-com/content-sites/internal-handbook/-/merge_requests/7674) there a few options here:

#### The relevant change is released

As noted in [the proposal for the Release Manifest](https://gitlab.com/gitlab-com/content-sites/internal-handbook/-/merge_requests/7674/diffs#de7f639f645587dc201411c8c0d5a385112b83d2_0_191),
the upcoming (monolith) release would then have a release manifest, which would have been [propagated to CNG](https://gitlab.com/gitlab-com/content-sites/internal-handbook/-/merge_requests/7674/diffs#de7f639f645587dc201411c8c0d5a385112b83d2_0_199)

If the above mentioned proposal is delivered upon,
a developer would indicate that they want to use the version in the release manifest.

```shell
gdk config set gitaly.editable false
gdk switch main
gdk reconfigure
```

#### The relevant change is not in the Release Manifest or not yet released

In the event that the satellite service in question is [not covered by the Release Manifest](https://gitlab.com/gitlab-com/content-sites/internal-handbook/-/merge_requests/7674/diffs#de7f639f645587dc201411c8c0d5a385112b83d2_0_203),
the developer will have to keep the satellite service `editable` and will not be able to use it in a containerized fashion.
This is consistent with today's behaviour. Efforts to enable local building of CNG-like images are not planned at the moment.
