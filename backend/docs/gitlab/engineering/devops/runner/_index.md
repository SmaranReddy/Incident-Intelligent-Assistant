---
title: "Runner"
description: "The GitLab Runner team page."
---

## Vision

By 2025, our vision for GitLab Runner is that the runner's setup and day-to-day operations at scale be an almost zero-friction experience.

## Mission

Our mission is to enable organizations to efficiently run GitLab CI/CD jobs on any computing platform and do so in an operationally efficient and highly secure way at any scale.

## Who We Are

Verify:Runner is made up of two teams:

- [CI Functions Platform](/handbook/engineering/devops/runner/ci-functions-platform/_index.md)
- [Runner Core](/handbook/engineering/devops/runner/runner-core/_index.md)

### CI Functions Platform

{{< team-by-manager-slug manager="nicole-williams" team="(Principal|Functions)" >}}

### Runner Core

{{< team-by-manager-slug manager="adebayo_a" >}}

### Runner Leadership

{{< team-by-manager-slug manager="nicole-williams" team="(Principal|Manager)" >}}

## How To Contact Us

Slack

- [`#runner_help`](https://gitlab.slack.com/archives/CBQ76ND6W)
- [`#g_runner_core`](https://gitlab.slack.com/archives/C09EDQXBMPH)
- [`#g_ci_functions_platform`](https://gitlab.slack.com/archives/C09HJ2UL9L4)

Requests for Help:
For customer support requests please [open an issue](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-Runner) in the Request for Help project.

## Stable Counterparts

For a comprehensive list of counterparts, look at the [runner product categories](/handbook/product/categories/#runner-group).

## Technical Vision

The Runner Core and CI Functions Platform teams' priorities and technical roadmap are guided by the [Runner technical vision](/handbook/engineering/architecture/design-documents/runner_technical_vision/), an architectural blueprint defining how GitLab Runner will evolve to support effortless installation, configuration, and execution of CI/CD workloads at any scale.

## How We Work

### Prioritization Framework

Our team maintains a clear commitment to prioritizing and scheduling work based on several critical factors (in priority order):

- **Forced prioritization issues** (infradev, security, availability) receive immediate resource allocation as our highest priority
- **Critical Bug resolution** to maintain quality and reliability
- **Strategic direction features** that advance our long-term vision
- **Technical debt reduction** to ensure sustainable development velocity
- **Maintenance activities** that keep our systems healthy
- **Community contributions** that deliver value to our users and support contributor engagement

This framework guides our resource allocation decisions and ensures we maintain focus on what matters most to GitLab Runner's reliability, security, and continued evolution.

## Shared Responsibilities

The Runner Core and CI Functions Platform groups operate autonomously but collaborate on specific responsibilities to distribute maintenance overhead.

### Releases

At the end of each iteration, we release Runner and associated projects. The release process is documented in the [releases project README](https://gitlab.com/gitlab-org/ci-cd/runner-tools/releases/-/blob/main/README.md).

We maintain a [list of released Runner versions](https://gitlab.com/gitlab-org/gitlab-runner/-/wikis/Released-runner-versions) in the Runner project wiki.

#### Release Manager Rotation

Each milestone, an automated issue is created in the [team-tasks project](https://gitlab.com/gitlab-com/runner-group/team-tasks/-/issues?sort=created_date&state=opened&first_page_size=100) assigning one engineer as the release manager for that iteration. The automation is managed through the [team-tasks-release-rotation project](https://gitlab.com/gitlab-org/ci-cd/runner-tools/team-tasks-release-rotation).

Each generated issue contains detailed instructions for completing the release process. If the assigned team member will be out of office during their assigned release period, they are responsible for finding a replacement.

### Maintenance Rotations

The Runner teams have weekly rotating maintenance assignments to ensure consistent coverage for support, triage, and community engagement activities.

#### How It Works

Each week, an automated issue is created in the [team-tasks project](https://gitlab.com/gitlab-com/runner-group/team-tasks/-/issues) with role assignments for the week. Team members are automatically rotated through different responsibilities to balance the workload and ensure everyone maintains familiarity with all aspects of team operations.

#### Responsibilities

Detailed instructions for each section can be found in the team task generator [issue template](https://gitlab.com/gitlab-org/ci-cd/runner-tools/runner-team-task-issue-generator/-/blob/main/template.md?ref_type=heads).

| Task | Description |
| ------ | ------ |
|    **🐛 Bug Wrangler 🤠**     |   Responsible for triaging incoming bugs to ensure they have appropriate severity and priority labels.     |
|   **🛟 Support & Security Responder 🚒**     |    Responsible for monitoring, labeling and responding to incoming security vulnerabilities and support requests. You can find more details on our security review process on the team resources page.    |
|     **🥷 Merge Marauder 🏴‍☠️**    |    Responsible for ensuring community contributor merge requests receive timely attention.    |
|     **💬 Community Contribution Triager 📌**    |    Responsible for initial triage of new community contributions.    |

#### Finding Your Assignment

Check the current week's issue in the [team-tasks project](https://gitlab.com/gitlab-com/runner-group/team-tasks/-/issues) to see your assignment. Issues are automatically created each Monday with the week's rotation schedule and include checkboxes for tracking completion of duties throughout the week.

## Team Resources

See [dedicated page](/handbook/engineering/devops/runner/team-resources/#overview).
