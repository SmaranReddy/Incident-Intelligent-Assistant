---
title: "Test Governance Group"
description: "Test Governance Group under Developer Experience section"
---

## Common Links

| **Category**            | **Handle**                                                                                     |
|-------------------------|------------------------------------------------------------------------------------------------|
| **GitLab Group Handle** | [`@gl-dx/test-governance`](https://gitlab.com/gl-dx/test-governance)                           |
| **Slack Channel**       | [`#g_test-governance`](https://gitlab.enterprise.slack.com/archives/C064M4S0FU5)               |
| **Slack Handle**        | `@dx-test-governance`                                                                     |
| **Team Boards**         |                                                                                                |
| **Issue Tracker**       | [`tracker`](https://gitlab.com/groups/gitlab-org/developer-experience/test-governance/-/issues) |
| **GitLab Repositories** | [test-governance](https://gitlab.com/gitlab-org/developer-experience/test-governance)          |

## Mission

Ensure highly effective testing across all teams by providing test frameworks and tools, optimizing configurations, and partnering with development teams to create and maintain comprehensive functional tests that prevent bugs from reaching customers.

## Vision

* Providing stable, repeatable, and fast test frameworks and configurations to catch bugs as early as possible
* Training to upskill teams on functional testing and quality. Every engineering team should know what to test, when, and how to contribute valuable tests to our test suites to maintain test coverage as the application grows
* Incident and bug analysis for GitLab.com and Dedicated - identifying test gaps and working with development teams to improve testing
* Effective quarantine process to quickly identify and isolate flaky tests for stage groups to fix or remove

## Team members

{{< team-by-manager-slug "ndenisenko" >}}

## Core Responsibilities

```mermaid
graph LR
    A[Test Governance Team]

    A --> B[Provide test expertise for critical Product releases]
    B --> B1[Define test strategy for the critical feature releases]
    B --> B2[Ensure adequate and robust test coverage for critical features]
    B --> B3[Stay informed about the stage roadmap]

    A --> C[Stage-level test tools and infrastructure]
    C --> C1[Design, Build and continuously improve tests, test frameworks and tools]
    C --> C2[Influence, advise and increase the testing capabilities for product teams]
    C --> C3[Monitor and address test flakiness]

    A --> D[Test guidance. Provide guidance and coach engineering teams in the areas:]
    D --> D1[Writing e2e tests and feature specs]
    D --> D2[Debugging and fixing test failures]
    D --> D3[Planning testing early in the development process]
    D --> D4[Anticipating test infrastructure needs and requesting change in advance]
    D --> D5[Shift left and maintaining appropriate ratio between unit, integration and e2e tests]
    D --> D7[Post-incident action]

    A --> E[E2E Pipeline triage. Shared responsibility among all DevX sub-department]
```

## Roadmap and themes

Test Governance Roadmap is based on commitments we do withing the current quarter and aligned with company goals. All Test Governance commitments are roughly divided into the following themes and sub-themes:

* Test Resilience
  * Stability. We ensure that quality processes are intuitive and test outcomes are predictable.
  * Speed. We make sure that functional testing frameworks are performing at desired speed and new tooling don't add to execution time.
* Test Observability
  * Track test levels. We ensure that functional test levels are tracked and respected.
  * Flakiness. We ensure that flaky tests detected and taken care of earlier.
  * Quarantine. We ensure that our test suite is healthy and consists of a high-value tests.
  * Coverage. We improve quality coverage by providing new and extending existing test automation frameworks to enable diverse testing types.
* Testing Knowledge base
  * We proved a comprehensive set of documentation, guidelines, how-tos and trainings to coach engineers on quality.
* Test Governance
  * We guarantee the quality of the end product by providing the right balance in test coverage; notify about feature changes to upstream teams; making sure that critical journeys are always tested thoroughly.
  * Strategy and tools. We evolve quality strategy to support organisational growth.
* Development Enablement
  * We provide helper tools to engineers to empower engineers with quality ownership.

Use [DevEx: Test Governance Issue](https://gitlab.com/groups/gitlab-org/quality/-/epics/116) to see the current Test Governance effort.

### Short-term commitments

Focus: Optimize Test Pipeline Performance Without Compromising Quality (FY27Q1)

* Reduce pipeline execution time
* Reduce CI costs
* Maintain or improve quality (no increase in escaped defects)

### Mid-term commitments

Focus: Continue reducing runner costs, defect escape rate and improving pipeline stability (FY27Q2 - FY27Q3)

* Improve Predictive tests system by leveraging AI to reduce runner costs
* Build system that adapts automated tests to changes in the code to reduce defect escape rate
* Integrate self-healing tests with flaky tests and quarantined tests processes to improve pipeline stability

### Working with us through request for help

The Test Governance group aims to better enable teams to apply the principle that [quality is everyone's responsibility](/handbook/engineering/development/principles/#quality).
Please request all support via the RFH process below. This will allow us to prioritize requests against our planned project roadmap.
Please use the following Request for Help process for all support requests.

#### Request for Help Process

1. Creates an issue in the [Request for Help](https://gitlab.com/gitlab-org/quality/test-governance/request-for-help#step-1-create-a-new-issue) project. Please complete all sections of the template so we can quickly triage your request
1. The Test Governance team will triage the request within a week, adding appropriate labels and assigning team members based on the request type and priority. You will recieve details about the prioritization and next steps on the request for help issue.

For more detailed guidance on E2E test coverage, consider these approaches:

* Engage with key DRIs to define [persona](/handbook/product/personas) use cases that illustrate how different customers will use new features
* Evaluate which parts of use cases can be covered by lower-level tests versus E2E tests, keeping the entire [testing pyramid](https://docs.gitlab.com/ee/development/testing_guide/testing_levels.html) in mind
* Refer to our documentation on [Testing Best Practices](https://docs.gitlab.com/development/testing_guide/end_to_end/best_practices) before submitting your request

<!--
## How we work
### Work related rituals
### Work management
#### Planning
-->

## Team meetings

Please refer to [Group Test Governance's processes](/handbook/engineering/infrastructure-platforms/developer-experience/test-governance/workflows/#processes-and-their-frequencies) for more details on the purposes of these meetings.

* **Team sync, knowledge sharing**: bi-weekly (2 sessions to cover all timezones)
* **Issues/Epics refinement**: weekly, alternating between sync (2 sessions to cover all timezones) and async
* **Quarterly planning**: once per quarter
* **Retrospective**: monthly but grouped by quarter
