---
title: LabKit North Star Strategy
status: proposed
creation-date: 2025-10-24
coaches: [ "@andrewn" ]
authors: [ "@e_forbes" ]
dris: [ "@e_forbes" ]
owning-stage: "~devops::developer experience"
participating-stages: []
toc_hide: true
---

{{< engineering/design-document-header >}}

## Terminology/Glossary

- Platform Offering - functionality that is exposed to our engineers from our infrastructure platform teams.
Examples of this are: logging, and metrics.

## Summary

This document outlines the LabKit ecosystem needed to achieve
cohesive, standardised, and seamless operations across all GitLab platforms.

## Motivation

We want to ensure infrastructure teams are able to expose their platform offerings
in a consistent way in order to provide the best experience possible for our engineers.

Based on the results of the DX Surveys published in FY2026, Q3, our engineers struggled with
aspects such as general observability and their ability to debug production issues.

This is due to a myriad of reasons, one of the key ones being the fact that engineers lack an easy-to-adopt
observability setup that ties into the underlying GitLab observability infrastructure seamlessly.

This is where Developer Experience will step in and help raise the bar internally and provide
a cohesive and expressive setup that caters for the vast majority of use-cases.

By establishing consistent functionality in LabKit, we avoid engineers delivering systems that have
fragmented or non-existent observability setups and we directly reduce the time it takes
for teams to deliver high-quality business value.

### DX Survey Results

The [Aug 31, 2025 DX survey](https://app.getdx.com/snapshots/NDMzOA/drivers) provided a number of comments
from our engineers that this proposal looks to address. Below are a series of quotes taken from the
survey that we'd like folks to consider as they read through the rest of this proposal:

- 'Our logging lacks any standard of conventions.'
- 'consistency: Low consistency with logging.'
- 'There are many things we just don't log. Having a true observability layer would help a lot.'
- 'I find debugging production issues a bit challenging'
- 'I'd like to see a better logging and tracing for the AI features on production.'

## Goals

The overarching goal of this work is to provide a consistent set of libraries, across all of our supported runtimes,
that wrap critical platform functionality and reduce the cost for adoption.

Additionally, we will solve the following:

- The establishment of a pattern that teams, willing to expose platform-level functionality, can adhere to in order to
expose said platform-level functionality to all engineers within the company.
- To reduce the scattered, fragmented logic of interacting with these core platform functions.
- To enable platform teams to iterate quicker, and roll out sweeping changes across the entirety of our system.
- To reduce the number and impact of incidents and improve upon how engineers investigate issues in production.
- To improve the developer experience of engineers looking to utilise these core platform functions through the
adoption of these libraries.

## What is LabKit?

LabKit will act as the abstraction layer that exposes all of the enabled platform capabilities
for your services. This could be things like, but not limited to:

- service discovery
- scaling
- logging
- tracing
- health checks
- feature gates
- rate limiting

LabKit will be the conduit through which infrastructure teams expose new
platform-level functionality.

### Consistency Across All Services

By opting for this approach, we are looking to ensure consistency around how our internal engineers
set up their services to interact with these platform offerings.

This approach will help reduce the time to adoption for each of these features alongside allowing us
to centralise any GitLab-specific complexity in a single place that can be owned by the respective
infrastructure teams managing these offerings.

### Looser Architectural Coupling

How thick or thin these abstractions are will be dependent on the underlying platform functionality
being exposed, but these abstractions will be critical.

There currently exists a tight-coupling between the various engineering teams and the current offering
of core platform features. This is something that LabKit will be directly focused on removing.

By ensuring teams interact with these underlying platform offerings in a consistent way via labkit,
we unlock the ability for the core platform teams managing these underlying features to move far
quicker and safer than would otherwise be possible.

A hypothetical example of this could be "The Observability team wishes to standardise all logs across
all services on an OTEL format". In the current model, the onus would be on the observability team
making a series of complex, and potentially high-risk changes across all of our services.

With the new proposed model, they would have the freedom to deliver this change in a set of MRs across
the LabKit runtimes and then the underlying services would only have to bump their LabKit version in
order to be compliant with the new format.

### Improved Engineering Quality

All of the above factors directly play a role into improving the engineering quality going into,
not only existing, but all new services that make up the GitLab product.

We are empowering engineering teams to adopt industry-standard practices, in a timely fashion,
without them having to reinvent the wheel.

This approach also encourages more time spent reviewing and refining the abstractions held in LabKit
as the development of these abstractions is not directly tied to product delivery timelines.

We're not going to see engineers define a good-enough local abstractions for these core-platform
functions alongside their own product delivery timelines.

## The North Star

To sum up the above, LabKit will be a set of libraries (Initially focused on our main languages; Ruby and Go) that all
engineers must adopt when they need to interact with core platform and infrastructure features.

LabKit will abstract key platform capabilities exposed by infrastructure platform teams.

These abstractions will be owned by the infrastructure platform teams.

All new platform offerings must be delivered in conjunction with a corresponding LabKit package.

LabKit must be easy to plug-in and adopt in any system across GitLab.

LabKit should provide functional parity across the various runtimes where it makes sense to do so.

## Governance Model

### Ownership Model

The Developer Experience team owns LabKit as a product, including all language implementations. This ownership encompasses:

- Final approval authority on all changes
- Roadmap prioritisation and backlog management
- Ensuring consistency and quality across implementations
- Documentation and adoption support

However, LabKit operates under an open contribution model. Infrastructure platform teams, application teams, and individual engineers are encouraged to contribute. The Developer Experience team acts as steward rather than sole implementer.

### Convention Consistency

LabKit implementations across languages must maintain conceptual consistency. While idiomatic differences between languages are expected and encouraged, the following must remain consistent:

- Package/module structure and naming
- Configuration patterns and option naming
- Field names and log formats (per the fields package contract)
- Behavioural semantics for equivalent functionality
- Error handling philosophy

Departures from established conventions require documented justification and approval from the Developer Experience team. Significant departures—those affecting interoperability or operational consistency—require broader agreement per the architectural review process.

### Architectural Review

LabKit occupies a privileged position in GitLab's architecture. Changes to LabKit propagate across all adopting services, making certain changes de facto architectural decisions for the organisation.

Changes requiring architectural review include:

- Addition of new platform capabilities (new packages or major features)
- Changes that mandate or strongly encourage adoption of specific technologies (tracing backends, service mesh implementations, discovery mechanisms)
- Breaking changes to public APIs, even when providing migration paths
- Changes affecting operational behaviour across services (log formats, metric cardinality, default timeouts)
- Introduction of new external dependencies, particularly those with licensing or security implications

**Review authority:**

Until the Architectural Council is established, changes in this category require approval from a Distinguished Engineer or Engineering Fellow with relevant domain expertise. The approving party should be documented in the merge request.

Once the Architectural Council is operational, that body will assume review authority for architectural changes to LabKit. The Developer Experience team will maintain a standing agenda item for LabKit changes requiring council input.

**Process:**

1. Proposer opens a blueprint MR describing the change and its architectural implications
2. Developer Experience team triages and identifies required reviewers
3. Architectural review occurs asynchronously or via scheduled discussion as appropriate
4. Approval is recorded by blueprint merger before implementation proceeds

### Versioning and Compatibility

LabKit's existing architectural guidelines mandate backward-compatible, forward-extensible APIs. Configuration injection interfaces must use functional options to allow LabKit to provide defaults without breaking existing service configurations. New platform capabilities added to LabKit must not require changes to services already using LabKit.

LabKit follows semantic versioning. Given its position as foundational infrastructure:

- Major version bumps (breaking changes) should be rare and well-communicated
- Services should be able to adopt new minor versions without code changes
- Platform teams introducing new capabilities via LabKit must provide adoption paths that don't force immediate changes on consuming services

## LabKit Roadmap

The development roadmap for LabKit will be evaluated on a regular basis by the Development tooling
team and communicated via this page: [LabKit](../../../infrastructure-platforms/developer-experience/labkit.md)

## Requesting Functionality or Roadmap Additions

Engineers can request new functionality via the RFH process for the Development Tooling team. These
issues will be evaluated against our current roadmap and we'll work with the respective
underlying infrastructure platform teams in order to deliver on said requests.

All engineers must go through this consultation process to ensure that the development tooling
team is aware of ongoing work. This is going to be critical to avoiding any duplication of efforts.

### Contributing New Language Support

At the time of writing, we officially support Ruby and Go as the main official languages within GitLab.

However, when a team requires LabKit support for a language not currently maintained, the following applies:

**Initiating team responsibilities:**

- Building the initial implementation with feature parity appropriate to the use case
- Providing test coverage and documentation consistent with existing LabKit libraries
- Ongoing maintenance contributions as the primary users of the new library
- Ensuring the implementation follows established LabKit conventions

**Developer Experience team responsibilities:**

- Reviewing all contributions for consistency with LabKit standards
- Approving and merging changes
- Assuming long-term ownership of the library once accepted
- Integrating the new library into the broader LabKit ecosystem and documentation

New language implementations should not be undertaken lightly. Teams considering this path should engage with Developer Experience early to discuss whether the use case justifies the ongoing maintenance burden, and whether alternative approaches exist.

## Alternative Solutions

**Do Nothing:** we've already determined through interviews with corresponding staff+ engineers across the company
and through the DX survey that the current setup is too fragmented and doesn't sufficiently address our debugging and
production investigation needs.

**A Less Opinionated Approach:** through conversations with senior leadership, we've determined that this is unfortunately
not an option. As a maturing engineering company, there has to be a certain level of standardisation defined so that
we can continue to scale quickly and deliver features in a timely fashion.

Fragmented setups and the lack of standardisation around observability has so far led us to a situation where, in some situations,
we have little to no true understanding as to how our systems are behaving in production.

Doing nothing also further proliferates bad anti-patterns, such as the propagation of feature flag state through the system.
A standardised setup exposed via LabKit could help to reduce the temptation to rely on these anti-patterns by providing
a smoother adoption experience. If we're able to provide easy and architecturally better solutions, our engineers are more
likely to do the right thing.

## Opportunity Cost

We'd like to highlight that a number of the comments referenced in the DX survey results section
were focused upon the AI features being delivered.

By adopting this proposal, we'll be setting up a system that will effectively act as a force multiplier for internal engineers
looking to deliver critical business value.

We'll be providing guardrails to ensure that these engineers who are under pressure to deliver quickly, will be
delivering high-quality software that will automatically meet some of our baseline feature readiness standards.
