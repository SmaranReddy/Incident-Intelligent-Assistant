---
title: "GitLab Delivery: Operate"
description: "The Operate team enables operators to deploy and maintain GitLab environments at any scale through standardized tooling, reference architectures, and operational excellence frameworks. We deliver the deployment infrastructure and operational tooling for GitLab deployments."
---

## Overview

The Operate team builds and maintains the deployment tooling and operational infrastructure that enable GitLab to be deployed across various environments and scales. We develop the tools, automation, and guidance that make GitLab deployments viable and maintainable, while development teams retain ownership of GitLab's application features and functionality.

## Mission

The Operate team builds and maintains the deployment automation, infrastructure tooling, and operational frameworks that enable GitLab to be installed and operated reliably across different environments. We develop the tools that bridge the gap between GitLab's packaged components and successful deployments in diverse infrastructure environments.

Our focus is on creating standardized deployment tooling, infrastructure automation, and operational guidance that simplify GitLab deployments and reduce operational complexity.

## Strategic Vision

We develop and maintain deployment tools that support multiple deployment patterns:

### Single Node Deployments

- Tooling: Co-ownership of Omnibus deployment mechanics with the [Build](../build/_index.md) team
- Focus: Simple installation automation, upgrade processes, operational guidance
- Support: Documentation and troubleshooting for single-node configurations

### Multi-Node and Cloud Native Deployments

- Tooling: Omnibus, GitLab Charts, GitLab Operator, GitLab Environment Toolkit (GET), Reference Architectures
- Focus: Infrastructure automation, orchestration at scale, zero-downtime operations
- Support: Sizing guidance, architectural reviews, and operational runbooks.

## Team members

The following people are members of the team:

{{< team-by-manager-slug manager="cjwilburn" team="Operate">}}

Product Manager: [Martin Brümmer](/handbook/company/team/#mbruemmer)

## Core Responsibilities

### Deployment Tooling Development

- **Omnibus Integration:** Co-development of deployment mechanics with the Build team
- **GitLab Charts (Helm):** Primary Kubernetes deployment method with comprehensive configuration options
- **GitLab Operator:** Kubernetes operator for deploying GitLab on OpenShift and vanilla Kubernetes (with a new version in development)
- **GitLab Environment Toolkit (GET):** Infrastructure provisioning and configuration automation
- **Reference Architectures:** Deployment patterns, sizing guidance, and architectural specifications

### Infrastructure Automation

- **Provisioning Automation:** Tools for consistent infrastructure setup across cloud providers
- **Configuration Management:** Automated configuration of GitLab deployments
- **Scaling Automation:** Tools and processes for horizontal and vertical scaling

### Operational Guidance and Standards

- **Documentation:** Installation procedures, operational runbooks, and troubleshooting guides  
- **Architectural Reviews:** Environment design consultations and recommendations
- **Best Practices:** Operational standards and deployment guidance

## Primary Projects and Tools

### GitLab Charts (Helm) - Primary Kubernetes Deployment

**Repository:** [gitlab-org/charts/gitlab](https://gitlab.com/gitlab-org/charts/gitlab)
**Documentation:** [docs.gitlab.com/charts/](https://docs.gitlab.com/charts/)

The primary method for deploying GitLab on Kubernetes, providing:

- Comprehensive Helm charts for all GitLab components
- Flexible configuration options for various deployment scenarios
- Integration with cloud-native ecosystem (Ingress, cert-manager, etc.)
- Production-ready defaults with extensive customization capabilities
- Regular updates aligned with GitLab releases

### GitLab Operator (Limited Availability)

**Repository:** [gitlab-org/cloud-native/gitlab-operator](https://gitlab.com/gitlab-org/cloud-native/gitlab-operator)

Currently in limited availability with plans for replacement:

- Kubernetes-native lifecycle management (limited scope)
- Being replaced with [new operator version](https://gitlab.com/gitlab-org/cloud-native/operator)
- Current focus on GitLab Charts for production Kubernetes deployments

### GitLab Environment Toolkit (GET)

**Repository:** [gitlab-org/gitlab-environment-toolkit](https://gitlab.com/gitlab-org/gitlab-environment-toolkit)

Opinionated Terraform and Ansible scripts for deploying scaled GitLab environments following Reference Architectures. GET provides:

- Standardized infrastructure provisioning across cloud providers
- Integration with GitLab Operator for Cloud Native environments  
- Automated configuration management
- Clear migration paths between deployment models

### Reference Architectures

**Repository:** [gitlab-org/quality/reference-architectures](https://gitlab.com/gitlab-org/quality/reference-architectures)

Scaled deployment patterns that provide:

- Real-world sizing guidance based on actual usage metrics
- Validated configurations for various scales and requirements
- Architecture reviews and environment design consultations
- Integration patterns for new services

## Working with the Operate Team

### Slack Channels

- **#g_operate** - Primary team channel for discussions and requests
- **#gitlab_environment_toolkit** - GET-specific discussions and questions  
- **#reference-architectures** - Reference Architecture discussions and requests
- Team handle: `@gitlab-org/software-delivery/operate`

### Support Requests for Help

GitLab provides a unified process to open a request for help (RFH) to support customers. This process is in place to ensure we have a single source of truth for those, so that we better collaborate cross-functionally as, many times, the requests actually require expertize of multiple areas of the product, or is not initially clear which area is more suitable to support the customer. When sharing the information with multiple relevant groups, within the same support request process, we’re able to get to a solution much more efficiently.

To open an RFH, refer to the procedures of our [how to get help](/handbook/support/workflows/how-to-get-help.md) handbook page.

This process allows us to track time involved and ensure that the right parties are involved at the correct time.

### Environment Build Requests

**The Operate team cannot fulfil bespoke environment build requests due to capacity constraints.** Also, building large sandbox environments with GET should be generally discouraged due to cost implications unless strictly required. However, here are some options to self-serve in building these types of environments:

- Self-service using GET documentation
  - Teams can follow the comprehensive [GET documentation](https://gitlab.com/gitlab-org/gitlab-environment-toolkit#documentation) to build their own environments. This is the primary recommended approach for teams that need custom environments.
  - Using the automated [Sandbox Cloud](/handbook/company/infrastructure-standards/realms/sandbox/) offering is recommended for setting up a cloud account for testing and development purposes.
- Run GitLab locally: For simpler needs, running GDK or Docker locally is recommended instead of building large environments to reduce costs and complexity.
- Use shared environments from Infra: Utilizing existing shared environments provided by the Infrastructure team.

## Working with the community

The installation and upgrade process is the first feature that all system administrators experience when working with GitLab.
As a result, the projects managed by the Operate team have a high level of engagement by the user-base. The GitLab
community is made up of more than just code contributors; users logging issues and feature requests are constantly pushing
us forward and helping create a better experience.

We strive for the following in our public projects:

1. Uphold our [Community Code of Conduct](https://about.gitlab.com/community/contribute/code-of-conduct/).
1. Enable [GitLab's mission that everyone can contribute.](/handbook/company/mission/#mission).
1. Show our work in [public](#public-by-default).
1. [Recognize and thank](/handbook/marketing/developer-relations/engineering/community-contributors-workflows/#recognition-for-contributors) contributors for their work.
1. Respect contributors donated time by providing [a timely review turnaround time](/handbook/engineering/workflow/code-review/#review-turnaround-time).

### Working with Open Source communities

The [open core of GitLab](/handbook/company/stewardship) is built on top of thousands of open source
dependencies. These dependencies and their communities are important to the GitLab strategy,
and working with these dependencies is an essential part of the projects the team maintains.

We strive to:

1. Consider the impact of our work on the open source communities that we benefit from.
2. Promote the importance of these open source communities within GitLab.
3. Raise issue with any decision that goes against our [stewardship promises](/handbook/company/stewardship/#promises).
4. Find opportunities to [contribute back the changes we make](/handbook/engineering/open-source/#using-forks-in-your-code).

## Public by default

All work carried out by the team is public. Some exceptions apply:

- Work has possible security implications - If during the course of work security concerns are no longer valid, it is expected for this work to become public.
- Work is done with a third party - Only when a third party requests that the work is not public.
- Work has financial implications - Unless financial details can be omitted from the work.
- Work has legal implications - Unless legal details can be omitted from the work.

Some of the team work is carried out on our development server at `dev.gitlab.org`.
[Infrastructure overview document](https://docs.gitlab.com/omnibus/release/#infrastructure) lists the reasons.

Unless your work is related to the security, all other work is carried out in projects on `GitLab.com`.
If you need to submit a sensitive issue, please use confidential issues.

If you are unsure whether something needs to remain private, check with the team Engineering Manager.

## Work/life harmony

Working [all-remote](/handbook/company/culture/all-remote/) and [asynchronous first](/handbook/company/culture/all-remote/asynchronous/)
offer flexibility in how team members approach their workday. Team members must make choices on how best to balance work time with other areas of life.

For new team members, the following resources provide examples on how to focus their time:

- [How team members approach their day](https://gitlab.com/gitlab-org/distribution/team-tasks/-/issues/907)
- Blog post: [A day in life of a remote worker](https://about.gitlab.com/blog/2019/06/18/day-in-the-life-remote-worker/)
- The option of a [non-linear workday](/handbook/company/culture/all-remote/non-linear-workday/)
- GitLab handbook: [Work/life Harmony](/handbook/company/culture/all-remote/)

The following GitLab Handbook areas are key in maintaining a healthy work/life balance.

- [Family and Friends First, work second](/handbook/values/#family-and-friends-first-work-second)
- [Combating burnout, isolation, and anxiety in the remote workplace](/handbook/company/culture/all-remote/mental-health/)
- [Recognizing Burnout](/handbook/people-group/time-off-and-absence/time-off-types/)
