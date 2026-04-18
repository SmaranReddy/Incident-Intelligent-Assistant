---
title: Environment Automation Ecosystem Team
---

## Summary

Ecosystem is part of [Environment Automation](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/environment-automation/) team, which is within the [Dedicated Group](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/). Our mission is to maintain ecosystem SLA at the same level as customer SLA by applying identical standards and automation to GitLab Dedicated and all external dependencies.

We follow the same processes as listed on the [Environment Automation](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/environment-automation/) and [Dedicated Group](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/) pages, unless a difference exists which is explicitly noted on this page.

## Team Mission

Maintain ecosystem SLA at the same level as customer SLA by applying identical standards and automation to GitLab Dedicated and all external dependencies. We measure success through end-to-end customer journey performance, including integrations like AI tools, and external products like ClickHouse.

## Roles & Responsibilities

### Architecture & Integration

- Design blueprints for external system integrations
- Define integration interfaces (Auth, APIs)
- Set SLA requirements for integrations (exclusions, response times)
- Establish deployment standards and operational requirements

### Evaluation & Strategy

- Execute proof-of-concepts for new technologies
- Define evaluation criteria for external services
- Lead build vs. buy decisions (managed service vs. in-house)
- Define and implement disaster recovery strategies

### Operations & Automation

- Enforce automated rollouts across ecosystem components
- Eliminate manual operations through automation-first approach
- Collaborate with teams early in development cycles
- Align change, dependency, and response processes with product standards

*This list is not exhaustive and will evolve as the team matures.*

## Operating Principles

- **Automation First**: Eliminate human operations by automating repetitive tasks
- **End-to-End Observability**: Instrument complete customer journey across all touchpoints
- **Ecosystem Parity**: Maintain external service availability on par with GitLab Dedicated SLA
- **Integrated Processes**: Align all ecosystem processes with whole product standards
- **Zero Toil**: Ensure ecosystem processes are fully automated and don't create operational burden

## Success Metrics

- **SLA Alignment**: Ecosystem SLA matches GitLab Dedicated application SLA
- **Incident Response**: Ecosystem MTTR meets whole product targets
- **Operational Efficiency**: All ecosystem processes are automated with zero operational toil

## Key Relationships

- **Reports to**: Director, Dedicated Infrastructure
- **Collaborates with**: Switchboard, FedRAMP, EA1, EA2, Security, Platform Engineering, GitLab Infra, Cells teams
- **Integration Dependencies**: Work within Switchboard framework and meet FedRAMP requirements where feasible

## Scope Boundaries

### What We Focus On

Operational reliability, automation, and integration patterns for external services that extend GitLab Dedicated capabilities

### What Others Own

Core platform development (Runners Platform team), product features, direct vendor relationships

---

**DRI**: @nitinduttsharma  
**Team**: {{< team-by-manager-slug "nitinduttsharma" >}}
**Approval**: @fviegas - Director, Dedicated Infrastructure
