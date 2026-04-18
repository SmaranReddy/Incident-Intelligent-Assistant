---
title: "LabKit"
description: "LabKit is GitLab's internal tooling library which aims to bring consistency and improved developer velocity to internal teams."
---

<!-- This document is intended to surface at a high level the LabKit project and the ongoing work.
  Discussions around software governance, and some of the in-depth goals for LabKit should be
  added to the North Star decision record.
-->

## What is LabKit?

LabKit is a set of libraries that are intended to provide core-development functionalities such as logging, metrics, tracing, and more.

## Pre-Reads

- [LabKit North Star Strategy](../../architecture/design-documents/labkit_north_star_strategy/)

## Vision

LabKit provides a consistent, intuitive interface for engineers to interact with application infrastructure,
accelerating development and reducing time-to-production.

- **Consistency**
  - Standardized patterns for common infrastructure interactions
  - Enforced conventions across the GitLab ecosystem

- **Ease of Use**
  - Idiomatic APIs following language-specific best practices
  - Comprehensive documentation enabling rapid onboarding
  - Reduced Cognitive Load - engineers focus on business logic, not infrastructure boilerplate

- **Accelerated Delivery**
  - Production-ready defaults out of the box
  - Reduced setup time for new services and components
  - Focus on business logic, not infrastructure plumbing

- **Observability by Default**
  - Built-in instrumentation for metrics, traces, and structured logging
  - Consistent signal correlation across services
  - Improved incident response times and lower mean-time-to-recovery

## LabKit Libraries

Development Tooling currently supports the following libraries:

- [Go Library](https://gitlab.com/gitlab-org/labkit)
- [Ruby Library](https://gitlab.com/gitlab-org/ruby/gems/labkit-ruby)

**New language adoption** - please adhere to the - [LabKit North Star Strategy](../../architecture/design-documents/labkit_north_star_strategy/) to understand what is expected of teams adopting new languages.

## LabKit Roadmap

This section lays out a rough roadmap describing some of the sub-projects we are intending to deliver over the coming quarters.

If you have any questions or suggestions about projects you'd like to see in this list, please reach out to the [Development Tooling team](development-tooling/_index.md).

### Now

**Focus: Best-In-Class Logging For Internal Services** (FY27-Q1 to FY27-Q2)

- Ship production-ready logging packages for supported languages
- Drive adoption through [PREP](https://gitlab.com/gitlab-org/architecture/readiness/) and enforcement mechanisms
- Consolidate fragmented log schemas into consistent structure for cross-service debugging

### Next

**Focus: Metrics and Tracing** (FY27-Q2 to FY27-Q4)

- Extend observability foundations with standardized metrics and tracing packages
- Deliver composite abstractions that reduce boilerplate (e.g., auto-instrumented handlers)
- Enable correlation across logs, metrics, and traces by default

### Later

**Expanded Infrastructure Abstractions** (FY27-Q4+)

- Identify high-impact DevEx opportunities based on adoption feedback
- Candidate areas: HTTP/gRPC clients, queuing, caching, alert templates
- Prioritize based on pain points and time-to-production impact

## In-Flight Projects

This section encapsulates the current, in-flight projects that we're focused on delivering:

### Field Standardization

Status: **Under Development**

Epic Links:

- [Define Field Logging Standards for Developers](https://gitlab.com/groups/gitlab-org/quality/-/epics/235)

Handbook link: [Field Standardization in Observability](../../../engineering/architecture/design-documents/observability_field_standardisation/)

The goal here is to define a schema for cross-domain fields and roll out this schema across our internal services.

- We’ve already helped to **improve the upgrade experience for self-managed** by reducing the number of incidents caused by high-dimensionality field names.
- Our next aim is to define a consistent field schema that is used across all services at GitLab in order to reduce toil when dealing with incidents. This ties directly into **strengthening our monitoring and observability capabilities for managing GitLab** for both self-managed and SaaS.
- The secondary outcome from this work is that we’re reducing the size of log messages being ingested by our logging infrastructure - this **reduces our cost of goods sold (COGS)** and indirectly helps to protect our, as well as our customer's, observability infrastructure.

### Logging Standardization

Status: **Requirements Gathering**

Epic Links:

- [Migrate Core GitLab Systems to Labkit-based slog](https://gitlab.com/groups/gitlab-org/quality/-/epics/266)

Discussion issue: [Discussion: Deprecating Logrus in favour of log/slog in Go Systems](https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks/-/issues/4103)

The goal is to standardise how logging is configured within our GitLab services.

- This will enable the Observability team to move faster in their efforts to adopt OTeL logging and again ties into **strengthening our monitoring and observability capabilities for managing GitLab** for both self-managed and SaaS.
- Initial investigations have also highlighted that we can improve the startup and runtime performance of our systems by migrating away from legacy logging libraries.
- This will also **accelerate customer value** as we’ll be leaning on modern industry standards for logging and subsequent feature development work should be faster.
