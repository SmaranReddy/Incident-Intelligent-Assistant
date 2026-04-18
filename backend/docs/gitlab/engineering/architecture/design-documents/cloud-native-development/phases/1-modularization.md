---
title: Modularized development components
status: proposed
creation-date: "2025-11-05"
authors:
    - "@gitlab.mschoenlaub"
    - "@kkloss"
coaches:
    - "@splattael"
dris:
    - "@mgamea"
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

> [!note]
> This is a sub-blueprint for [Cloud-native development](../_index.md)

## Summary

The current architectural design of the GitLab Development Kit (GDK)
poses significant scalability and ownership concerns with a risk in
increasing maintenance cost for the GDK maintainers, stability concerns,
and adopting modern, cloud-native technologies in development.

By moving to a declarative, self-service model, we also align development
environment setups with the company-wide [Component Ownership Model](../../../../infrastructure-platforms/production/component-ownership-model.md).

## Motivation

We use the GitLab Development Kit (GDK) to set up GitLab development
instances for local development and reviews. It orchestrates required and
optional services, manages service configuration, and assists in
troubleshooting.

### Problem statement

The monolithic architecture of GDK creates significant pain points that
impact development velocity and system stability, such as:

- Tight coupling between component integration and orchestration code
- Monolithic nature of GDK prevents self-service ownership
- Absence of clear ownership boundaries
- Lack of component integration/smoke tests

Current component definitions are scattered across the GDK codebase in at
least seven different locations spanning multiple programming languages
(Ruby, Shell, ERB templates, Makefiles, ...). Due to the tight coupling
of existing services and the GDK core code, adding and managing
components as a component-owning team is cumbersome. Component definition
being scattered and using Makefiles has frequently led to support
requests from the Development Tooling team during component integration.

This absence of clear boundaries means that many components remain
untested and unmaintained. When services fail, it's unclear whether the
issue lies in GDK core orchestration code or the component itself,
leaving the Development Tooling team responsible for service-caused
instabilities they don't own.

### Current challenges

These architectural limitations manifest in concrete challenges:

**Reactive troubleshooting burden**: Developers spend significant time
troubleshooting GDK issues that are difficult to diagnose due to the lack
of component isolation.

**Slow component integration**: Setting up and integrating new services
takes 2–3 days for complex features such as AI services. This involves
navigating the GDK codebase to find all the places that need modification,
understanding implicit dependencies, and manually testing integration
points without automated verification.

**Unmaintained component integrations**: Without clear ownership
attribution, many component integrations deteriorate over time. Component
owners lack visibility into the GDK integration health of their services, and
the Development Tooling team lacks the domain expertise and capacity to
maintain all 40+ components effectively.

### Impact on business

This monolithic architecture directly impacts the GitLab development
velocity:

- **Engineering productivity**: The 2-3 day integration time for new
  services multiplied across teams significantly slows feature delivery,
  particularly for strategic initiatives like AI capabilities.
- **Increased maintenance costs**: Around [77% of the internal support requests through Slack](https://gitlab.com/gitlab-org/quality/tooling/team/-/work_items/137)
  and [33% of bug-reports](https://gitlab.com/gitlab-org/gitlab-development-kit/-/issues/3152) relating to GDK, which are handled by the Development Tooling team,
  are about component-specific issues instead of improving core GDK
  functionality. This friction reduces the innovation velocity of this
  team.
- **Quality and stability concerns**: [Integration bugs](https://gitlab.com/groups/gitlab-org/-/epics/20070#note_2942099171) escaping to
  production create customer-facing incidents and erode confidence in the
  development workflow.

### Goals

- Create a decoupled, declarative interface to register GDK components.
- Separate GDK components from GDK core code.
- Establish clear component ownership (and stability!) attribution.
- Ease the integration of future components into GDK with the declarative interface and documentation.

### Non-Goals

- Improve stability of components.
- Physically isolate component services.

### Success metrics

- **Service Integration Velocity**: 50% reduction in effort needed to integrate new services with GDK,
  including the first bug-fixing iterations.
- **Reduced Reactive Work for Developer Experience Team**: 50% reduction in number of bug-reports and support requests related to components not owned by Development Tooling.

## Proposal

By modularizing GDK, we transform the current monolithic development
environment architecture into a **component-based architecture** where
each GitLab component (such as Rails, Gitaly, Workhorse, PostgreSQL, and
Redis) becomes a self-contained module that allows for integration
through a standardized interface and logical isolation from the core GDK
framework.

### Standardized component interface

Each GDK component exposes a consistent contract that defines:

- **Dependencies**: What other components this service requires (for example, Rails requires PostgreSQL)
- **Configuration**: Environment variables, ports, and settings the component needs
- **Health checks**: How to verify the component is running correctly
- **Lifecycle hooks**: Service start commands, setup instructions, and initialization procedures

New components can be integrated by implementing this standard,
well-known interface without having to modify the core GDK code. This
makes it much easier for component owners to integrate their component
into GDK.

### Migration Path

The transition to modularized GDK happens incrementally.

1. **Foundation**: Establish the base component interface standard
1. **Smoke tests**: Introduce component ownership and smoke tests for each component
1. **Immediate results**: Refactor 3-5 core services (Rails, PostgreSQL, Redis)
1. **Default**: Make modular mode the default for new GDK installations
1. **Expansion**: Convert remaining services to modular components

This approach ensures that existing GDK installations continue to work
unchanged during the transition. It also provides immediate results by
adding smoke tests to yet untested components and refactoring a few
components as examples.

## Design and implementation details

We are proposing a component definition file that records the following
properties about a component in a single file per component:

- Name (inferred from directory)
- Owning product category
- Optional: Repository
- File templates
- Component configuration
- Automated installation commands
- CI verification steps (aka. integration smoke tests)
- Services
  - Enabled status
  - Runtime command
  - Environment variables
- Smoke tests
- Health checks

### Comparison

| Property | Before | After |
| --- | --- | --- |
| Ownership | Not defined | Defined in each `GDK.rb` |
| Repository | Defined in Makefiles and in global `config.rb` | Defined in each `GDK.rb` |
| File templates | In the global `support/templates/` directory | In `components/<name>/` directory |
| Component configuration | In global `config.rb` | Defined in each `GDK.rb` |
| Installation commands | In Makefiles | Defined in each `GDK.rb` |
| Service definition | In static `lib/gdk/services/*.rb` files | Dynamically in each `GDK.rb` |

### Component directory structure

Each component file and the associated file templates, such as
`gitaly.config.toml`, are stored in the `components` directory in
folder that's named in kebab-case after the component name, such as
`gitlab-http-router`. This allows GDK to infer the name from a single
source of truth.

```plaintext
<gdk_root>
└── components/
    ├── gitaly/
    │   ├── GDK.rb                    # Component definition
    │   └── gitaly.config.toml.erb    # Configuration template
    ├── workhorse/
    │   ├── GDK.rb
    │   └── config.toml.erb
    └── redis/
        ├── GDK.rb
        └── redis.conf.erb
```

This structure provides clear separation of concerns and makes component
ownership explicit through directory boundaries. It also enables us to
use the `CODEOWNERS` feature of GitLab to enforce component ownership boundaries.

### Code ownership

Each component directory is assigned to its **owning team** (never an
individual) through an entry in `.gitlab/CODEOWNERS`:

```plaintext
/components/gitaly/          @...
/components/workhorse/       @...
/components/redis/           @...
/components/gitlab-rails/    @...
```

This assignment allows:

- Automatic code review routing to the owning team, reducing load for the Development Tooling team
- Ownership being defined as code and clearly visible in the GitLab UI
- Teams to iterate on their component definitions without waiting for GDK maintainer availability

For third-party components (PostgreSQL, Redis, Elasticsearch), ownership
defaults to the Development Tooling team, unless component-specific
owners are identified.

### Service runtime types

Services can specify different runtime types:

- **Command** (default): Traditional process execution
- **Docker** (future): Future container-based execution for isolated services
- **Runway** (future): Support for Runway-based services

The runtime specification allows GDK to evolve toward containerized
development without breaking existing components.

### Component lifecycle

Components follow a standardized lifecycle:

1. **Discovery**: GDK scans the `components/` directory and loads all `GDK.rb` files
1. **Configuration**: Component configuration is merged with user-provided settings
1. **Dependency resolution**: Component dependencies are resolved to determine initialization order
1. **Service registration**: Services are registered with the service runtime based on the configuration

After these steps, GDK can enter the components into the following
workflows:

1. **Update**: GDK performs a dependency-aware update of all components, setting up those that aren't set up yet
1. **Reconfigure**: GDK generates all config files based on the registered templates
1. **Runtime**: Services are started, stopped, and monitored according to their definitions
1. **Verify**: In CI, components are set up based on the instructions and smoke-tested

### Ownership and stability attribution

Each component definition explicitly declares its owning product category
through the `feature_category` field. This allows us to delegate
ownership of components to their actual owners. This has several
benefits.

When a component service fails, GDK displays the owning team information,
which directs GDK users to the appropriate support channels instead of
the catch-all ones maintained by Development Tooling.

This also allows us to separately track bugs of GDK Core, which alongside
additions and improvements to the component definition schema and CLI
tool, is the only work the Development Tooling team should be responsible
for in GDK.

Likewise, we will establish error budgets—similar to production—, so that
GDK stability is attributed to component owners. To visualize this, we
will create a Grafana dashboard showing stability for each component,
which will be used by the owners to be aware of them and remediate these
themselves.

### Smoke tests and health checks

We first create smoke tests for every component. These tests run in CI,
where they set up a given component and verify that its service(s) are
healthy. This closes an existing test gap in GDK and ensures that we
don't break service installations while migrating to the new component
definitions.

Later, we will integrate health checks that are read-only and are used to
report whether a service is healthy or not to the user.

We can use health checks in smoke tests but some smoke tests, such as
creating a file in Gitaly and checking that it was created, are
read-write operations that aren't useful for users.

### Backward compatibility

During the migration period, GDK maintains backward compatibility by
supporting both the legacy service and new component registration methods
simultaneously. Thanks to the smoke tests, we ensure that the migration
doesn't break existing installations.

As soon as a feature is available in the component interface, it will
become the required default for new components, so we avoid a tailing
backlog problem, where new services being added follow the legacy service
registration.

## Alternative Solutions

### Assign ownership without modularization

- **Pros**: low upfront cost.
- **Cons**: stability stays near impossible to attribute and adding new services stays cumbersome.

### Do nothing

- **Pros**: no upfront cost.
- **Cons**: more reactive work for Development Tooling slowing planned efforts, worse developer experience due to inaction from other teams, and adding new services stays cumbersome.
