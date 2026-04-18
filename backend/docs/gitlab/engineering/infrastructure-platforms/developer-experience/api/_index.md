---
title: "API Platform team"
description: "The API Platform team enables GitLab contributors to efficiently build and maintain APIs that are discoverable and stable for customers"
---

## Team scope

The API Platform team owns GitLab's API architectures, standards, and platform-wide capabilities. Contact us for cross-cutting API concerns, design patterns, or foundational improvements. For feature-specific endpoints or functionality, [reach out to the team that owns that feature area](/handbook/product/categories/features/).

Please direct all API Platform team requests to our [request for help process](/handbook/engineering/infrastructure-platforms/developer-experience/api/#working-with-us). Alternatively, GitLab team members are welcome to attend our [office hours](https://docs.google.com/document/d/1MuN71IU-Y_EhMRKouiANMHdz7yG5znhO4lhT0Lyfdpk/edit?tab=t.0#heading=h.k0ifed17isg9) held at **11am UK time** on the last Tuesday of the month.

## Mission

The API Platform team owns GitLab's REST and GraphQL APIs as a product, and enables API-first development across GitLab. As part of Developer Experience, the API Platform team provides tooling, processes, and practices to help contributors efficiently build and maintain APIs for customers.

As a newly formed team in FY26-Q2, we're building our expertise in `#f_rest_api` and `#f_graphql` by working on platform-level projects and working closely with existing domain experts. Over time, the API Platform team's expertise will evolve into driving steering committee for API standards and decisions across GitLab.

## Vision

The API Platform team transforms GitLab into an API-first platform where APIs are a core part of how customers and agents integrate with GitLab. We achieve this through the 4Ds framework, where each investment reinforces the others in a continuous improvement cycle:

- **Documentation**
  - Automated OpenAPI specifications for REST API endpoints
  - Interactive documentation and practical workflow examples

- **Deprecations**
  - Clear lifecycle progression from experimental to GA
  - Predictable deprecation timelines that never break customer integrations

- **Data-driven**
  - Collect aggregated product usage metrics to guide API improvements and investment decisions
  - Improved observability for GraphQL API

- **Development**
  - Contributors ship APIs by default using automated tooling and paved paths
  - Architecture that facilitates consistency across REST/GraphQL APIs

These initiatives create a flywheel effect: better documentation drives adoption, deprecation management builds trust, data insights guide improvements, and development tools make it all sustainable.

As the owners of APIs at GitLab, the API Platform team provides the stewardship and platform-level investment needed to transform our APIs into a product customers can rely on. Through scalable automation and tooling rather than one-off fixes, we achieve economies of scale that benefit both contributors and customers.

## Team structure

The API Platform team consists of 4 engineers (2 Staff, 2 Intermediate) and 1 Engineering Manager.

### Members

{{< team-by-manager-slug manager="pjphillips" team="Developer Experience:API(.*)" >}}

## Roadmap

As a lean team, we balance high confidence in our "Now" commitments while keeping "Next" and "Later" commitments directionally aligned and flexible. For more information on technical roadmap, refer to [Infrastructure Platforms roadmap](https://infra-roadmap-c6d14f.gitlab.io/).

### Now

**Focus: Documentation - API strategy and discoverability** (FY26-Q3 to FY26-Q4)

- Autogenerate OpenAPI v3.0 specification for REST API
- Automate style guide enforcement using linters for REST API consistency
- Deploy interactive documentation for OpenAPI v3.0 specification
- Improve frontend GraphQL query observability by extending endpoint product category attribution
- Set the strategic direction for GitLab APIs, including planning for feature parity across GraphQL and REST, and introducing lifecycle management (e.g. experimental/beta/GA)

For more information on current projects, refer to our [top-level epic](https://gitlab.com/groups/gitlab-org/quality/-/epics/218).

### Next

**Focus: Deprecations & Data-driven - Build trust and visibility** (FY27-Q1 to FY27-Q2)

- Build visibility into API usage patterns and customer needs for REST and GraphQL APIs
- Establish lifecycle management policies, tooling, and notification systems for REST and GraphQL APIs
- Improve API observability and monitoring capabilities for GraphQL API

### Later

**Focus: Development - Enable API-first by default** (FY27-Q3 and beyond)

- Create tooling that makes API development efficient for REST and GraphQL APIs
- Explore architectural improvements for parity and consistency across REST and GraphQL APIs
- Enable teams to self-serve their entire API needs from development to deprecation

### Keeping The Lights On (KTLO)

In addition to planned work, the API Platform team will also be responsible for ongoing maintenance that impacts shared functionality across API surface areas, such as security vulnerabilities, critical bug fixes, etc.

## Working with us

**During team formation period:**

- For REST API questions: Start in `#f_rest_api` where domain experts can help
- For GraphQL questions: Start in `#f_graphql` where domain experts can help
- For platform-level API improvements: Please **[create an issue](https://gitlab.com/gitlab-org/quality/request-for-help/-/issues/new?issuable_template=api_team_request)** in our [RFH repo](https://gitlab.com/gitlab-org/quality/request-for-help#developer-experience---request-for-help). Or you can Engage us in `#g_api-platform`
- We're not yet requiring reviews by API Platform team until we've built up toward that capability

For individual questions, please mention team members directly on GitLab.com or reach out to the team through our Slack channels.

### Slack Channels

| Channel | Purpose |
| :---: | :--- |
| [#g_api-platform](https://gitlab.enterprise.slack.com/archives/C095E77BLHJ) | Engage directly with API Platform Team |
| [#f_rest_api](https://gitlab.enterprise.slack.com/archives/C08LQBMLXPF) | Engage with domain experts on REST API |
| [#f_graphql](https://gitlab.enterprise.slack.com/archives/C6MLS3XEU) | Engage with domain experts on GraphQL API |

## How we work

The API Platform team is geographically distributed across AMER and EMEA regions and works asynchronously by default.

### Meetings

The API Platform meets synchronously once a week. For more information on sync meeting, refer to [agenda notes](https://docs.google.com/document/d/1GeZ47_EIHVYw1KB4fCn-VUGe-fRNrs5ezTjm2s51YZE/edit?tab=t.0#heading=h.1n5rp8nv4ncb).

### Project Management

The API Platform team follows project management process for [Infrastructure Platforms department](/handbook/engineering/infrastructure-platforms/project-management/).

For more information on current projects, refer to [parent epic](https://gitlab.com/groups/gitlab-org/quality/-/epics/200).
