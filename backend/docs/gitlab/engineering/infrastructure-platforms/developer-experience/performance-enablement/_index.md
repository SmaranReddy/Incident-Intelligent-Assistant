---
title: "Performance Enablement"
description: "Performance Enablement group within Developer Experience section"
---

![DevPerfOps Logo](/images/engineering/infrastructure-platforms/developer-experience/performance-enablement/devperfops-diagram.svg)

## Mission

Enable GitLab's R&D teams to deliver fast, reliable customer experiences by making performance testing accessible, actionable, and embedded throughout the development lifecycle.

## Vision

A GitLab where every engineer confidently owns performance, catches issues before customers do, and takes pride in building fast, efficient software.

## Goals

1. Enable development teams to prevent performance-related incidents by providing tools to detect and resolve issues before they reach customers
2. Accelerate organization-wide performance improvements by making trends visible, measurable, and actionable for all teams
3. Establish a performance-centric culture where every development team owns performance throughout the SDLC

## Team Members

{{< team-by-manager-slug manager="mkomor1" team="Performance Enablement(.*)" >}}

## Roadmap

This roadmap translates our team's strategic pillars — building platforms over bespoke solutions, shifting performance ownership to engineering teams, and enabling self-service capabilities — into quarterly execution phases.

The direction outlined here reflects what we know today. As we execute, we'll learn what works, what team adoption actually looks like, and where our assumptions need adjustment. This is intentional: we're building organizational capabilities in a space where patterns aren't yet established. Discovery and course correction are part of the work.

When we uncover new information through team feedback, incident trends, or adoption challenges, we'll adapt while staying anchored to our strategic intent. The roadmap provides direction and helps us make consistent decisions, not rigid commitments.

We'll review this roadmap quarterly, updating it as we validate or refine our approach.

### Now

**Focus:** Build foundations and early adoption (FY26Q4)

- Build Git SSH performance testing platform with Gitaly as early adopter, creating feedback loops and stress testing capabilities for common developer use cases
- Expose direct k6 access within GPT to transform it from Reference Architecture-specific testing to a workflow-agnostic performance tool
- Consolidate GitLab's k6 implementations and prevent tooling duplication by creating a flexible, standardized testing platform that serves as a building block for broader performance testing
- Explore internal test environments within GitLab to understand usage and gaps, enabling informed decision-making for future testing environment unification initiatives

### Next

**Focus:** Expand platform adoption and build organizational influence (FY27Q1/Q2)

- Apply learnings from Gitaly pilot to refine tooling, documentation, and support models for broader team adoption
- Expand performance testing capabilities to 2-3 additional engineering teams, using established feedback loops and adoption patterns
- Begin tracking performance-related incident trends to identify testing gaps and prioritize tooling investments
- Develop initial opinions on performance testing approaches based on Gitaly collaboration and cross-team patterns
- Create self-service enablement materials (guides, examples, templates) that allow teams to adopt performance testing with minimal support

### Later

**Focus:** Establish unified performance testing strategy and organizational standards (FY27Q3+)

- Define and socialize opinionated testing guidelines: methodologies, environment usage, metrics that matter, and when/where to test
- Build organizational consensus around standardized performance testing approaches, establishing Performance Enablement as the center of expertise
- Create prescriptive patterns and best practices for performance testing across different service types and deployment models
- Establish cross-organizational relationships and trust through consistent delivery, support, and thought leadership

### Keeping The Lights On (KTLO)

In addition to planned work, the Performance Enablement team will be responsible for ongoing support of existing performance testing infrastructure, responding to critical performance incidents, and maintaining current testing frameworks and tooling.

## Common Links

| What     | Links                                                                                                             |
|------    |---------------------------------------------------------------------------------------------------------------------|
| **GitLab Team Handle** | [`@gl-dx/performance-enablement`](https://gitlab.com/gl-dx/performance-enablement)                               |
| **Team Boards** | [Epic Board](https://gitlab.com/groups/gitlab-org/-/epic_boards/2079993) \| [Issue Board (by assignee)](https://gitlab.com/groups/gitlab-org/-/boards/8955771)  \| [Issue Board (by status)](https://gitlab.com/groups/gitlab-org/-/boards/9759214) |
| **Recently created** | [Issues](https://gitlab.com/groups/gitlab-org/-/issues/?sort=created_date&state=opened&label_name%5B%5D=group%3A%3Aperformance%20enablement&first_page_size=100) \| [Epics](https://gitlab.com/groups/gitlab-org/-/epics?sort=created_date&state=opened&label_name%5B%5D=group%3A%3Aperformance%20enablement&first_page_size=100)|
| **Work roll up Epic** | [DevEx: Performance Enablement](https://gitlab.com/groups/gitlab-org/quality/-/epics/96)|
| **Business as usual epic** | [Performance Enablement: Business as usual](https://gitlab.com/groups/gitlab-org/quality/-/epics/99)|
| **Services and component ownership** | [Performance Enablement Ownership](https://docs.google.com/spreadsheets/d/1529zsP-rdml_pI7pfCHwBTzHzXjV38sun4F8N57GTLw/edit?gid=0#gid=0)|

## Primary Projects

| Name | Description |
| :---: | :--- |
| [GitLab Browser Performance Tool](https://gitlab.com/gitlab-org/quality/performance-sitespeed)| Tool that is a [SiteSpeed](https://www.sitespeed.io/) wrapper which measures frontend performance in browsers, providing insights into web page performance across GitLab environments. |
| [GitLab Component Performance Tool](https://gitlab.com/gitlab-org/quality/component-performance-testing)| Tool which leverages containerization and automated testing to provide insights on individual component performance. |
| [GitLab Performance Tool](https://gitlab.com/gitlab-org/quality/performance)| Tool to provide performance testing of any GitLab instance. |
| [GitLab Verify Playbook](https://gitlab.com/gitlab-org/quality/quality-engineering/gitlab-verify-playbook)| Experimental Tool to verify that a GitLab instance is up and functional after deployment or reconfiguration. |

## All Projects

| Name | Description |
| :---: | :--- |
| [Test Data Generator](https://gitlab.com/gitlab-org/quality/test-data-generator)| Tool designed to populate a GitLab instance with simulated data that can be used to simulate a larger production instance. |
| [Performance Test Data](https://gitlab.com/gitlab-org/quality/performance-data)| This Project serves as an LFS data repository for the GitLab Performance Tool |
| [Performance Docker Images](https://gitlab.com/gitlab-org/quality/performance-images)| Docker builder and registry for GitLab Performance testing |
| [AI Gateway Latency Baseline Executor](https://gitlab.com/gitlab-org/quality/gitlab-environment-toolkit-configs/aigw-latency-baseline-executor)| Gets the latency baseline for AI Gateway in a specific region |

## Working with us

To request for help with performance testing of a new feature, please create a new issue within the GPT project with the request for help template.

For individual questions please reach out to the team through our Slack channels.

### Slack Channels

| Channel | Purpose |
| :---: | :--- |
| [#g_performance_enablement](https://gitlab.slack.com/archives/C081476PPAM) | Channel to engage with the Performance Enablement Team |

## How we work

### Meetings and Scheduled Calls

Our preference is to work asynchronously, within our projects issues trackers.

The team does have a set of regular synchronous calls:

- Performance Enablement Team meeting
- 1-1s between the Individual Contributors and Engineering Manager

### Project Management

The majority of our [project management process is described at the Platforms level](/handbook/engineering/infrastructure-platforms/project-management/) and is shared between all Infrastructure Platform teams.

Project management links

- Team [Project Status epic](https://gitlab.com/groups/gitlab-org/quality/-/epics/96)
- Team [Roadmap epic](https://gitlab.com/groups/gitlab-org/quality/quality-engineering/-/epics/117)

## Our Strategy

### Observations

**Performance is treated as a specialized function, rather than a shared responsibility**
Currently, performance testing and optimization is viewed as the Performance Enablement team's function. Dev teams see performance as something that happens later, and by another team, and not something that is a fundamental part of feature development and ownership. This means that performance issues go unaddressed and are discovered in production, rather than prevented as part of development. Without a shared responsibility mindset, teams lack accountability for performance outcomes.

**Our team currently operates at a hands-on support level rather than as a force multiplier across R&D**
We've seen successes building tooling and providing bespoke environment setup and testing for a small fraction of R&D teams. However, these tools require significant manual involvement from our team, such as environment setup, custom configurations, and ongoing testing support. This limits our ability to scale across R&D and creates a bottleneck.

**We lack holistic visibility and governance of performance**
While we trust development teams, we have no systematic way to track performance trends across features or releases. We miss opportunities to celebrate wins, identify degradation patterns early, and make data-driven decisions around staffing and investment strategies relating to performance. Without this visibility, we're limited in our ability to prevent incidents and drive global performance improvements.

**The gap between test and production environments limits effectiveness**
Many performance issues manifest at production scale and data volumes. Our current approach to performance is primarily limited to small scale testing environments and doesn't adequately utilize production-like environments. Additionally, we do not leverage existing observability platforms or incident data to inform our performance testing guidelines. This means customers often discover performance issues before we do.

### Our guiding policies

**Transform from service providers to platform builders**
Our team will shift from providing hands-on support and focus on building self-serve platforms and onboarding teams to use them. We will lean heavily into software development that enables all R&D teams to own performance within their area of responsibility independently. We build leverage through technology, and not by being on a team's critical path

Tradeoffs: Teams that rely on our direct involvement will need to adopt self-serve tooling and platforms. We will give up direct involvement with individual teams to focus on providing a greater impact to the entire R&D organization.

Resource allocation: Allocate 80% of team capacity to platform building, enhancements, and guidance, and 20% to providing direct services and support. We will measure this quarterly and adjust our goal as needed.

**Shift performance ownership to development teams**
Performance becomes a fundamental part of each team's responsibility throughout the SDLC, from local development, through MRs, CI/CD pipelines, and production monitoring. Development teams will leverage our tooling, platform, and general guidance to test, interpret, and remediate issues.

Tradeoffs: Development teams take on additional responsibilities for performance testing and monitoring, but gain faster feedback loops, catch issues early, and prevent leaked performance issues to customers. They gain autonomy, but accept accountability for performance outcomes.

How performance decisions are made: Teams choose their performance thresholds. This includes input from their PM or other representatives close to customers. Teams are welcome to escalate Performance Enablement platform and tooling issues, as well as add or request additional functionality from the Performance Enablement team. The Performance Enablement team will prioritize work with the largest potential customer impact.

**Establish organization-wide performance visibility and governance**
We will systematically track, measure, and report on performance trends across all features, teams, and releases. This allows us to provide an early warning indicator that identifies trends before they become incidents, as well as enables data-driven investment decisions, and allows us to identify and celebrate wins.

How this will work: We will publish monthly performance trends to engineering leadership to highlight top performance improvements and successes, degradations, and their associated teams or areas of ownership. Each team will have access to dashboards showing their performance metrics and trends.

**Test across all deployment models with real data, and leverage existing observability**
We prioritize enabling testing for all deployment models with real data to reflect a real customer experience. We integrate with existing observability tools and platforms to surface additional performance insights and identify performance challenges.

Tradeoffs: This will require teams to understand additional deployment models (.com, self-managed, dedicated), environments, and tooling, which require additional time. In exchange, we catch issues early and prevent escape defects.

How this will work: Teams will need to validate performance across .com, self-managed, and dedicated environments as part of the SDLC. The performance enablement team will provide platforms, environments, and tooling for the deployment models to enable teams to perform testing with as little friction as possible.

**Self-service first, escalations only for critical issues or requests**
R&D Teams must use self-service platforms, tooling, and documentation to test, identify, and remediate performance challenges. Escalations and manual work should be rare and reserved for business-critical reasons.

Tradeoff: R&D teams will invest more time learning tooling and platforms. Requests for Help (RFH) will not receive immediate help, except for critical escalations. As a result, teams will build additional autonomy.

Escalation criteria & RFH guidelines:

- Critical (immediate response): customer impacting production incident or complete platform outage preventing releases
- High priority (2 business days): General platform bugs or missing required functionality
- Standard priority (1 week): Usage questions/shadowing, interpreting results, or other general platform interactions or teachings

Teams should consult documentation and attend office hours (to be booked) for general questions and concerns before escalating standard priority issues.
