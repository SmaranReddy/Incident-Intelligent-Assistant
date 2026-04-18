---
title: "Production Engineering Networking and Incident Management Team"
description: "We manage both the networking platform that controls traffic into our systems, and GitLab's incident response process"
---

## Mission

We provide protection for GitLab from two vectors:

- We provide a networking platform to provide teams with the first line of defense in how traffic is allowed into our systems
- We manage the response system through our incident management process and tooling for when GitLab needs to respond to any incidents.

We seek to build and evolve the networking infrastructure that powers GitLab with focus on developing innovative networking solutions that scale with GitLab's growth. We empower teams at GitLab to feel confident in responding to incidents involving their services.

## Vision

1. **Excellence in networking infrastructure** We will drive GitLab's networking capabilities for GitLab forward by building scalable, secure, and efficient solutions. This includes evolving our edge services, load balancing, rate limiting, and network security to meet the growing demands of all GitLab platforms. Through centralized networking tooling and infrastructure, we create a foundation that supports GitLab's continued growth and innovation.
1. **Service ownership and standardized incident response** We will enable teams to operate their own services confidently by providing teams with the frameworks and tooling to confidently respond to any problems as they arise.

## Ownership and Responsibilities

The Networking and Incident Management team focuses on:

1. Incident Management - we are responsible for improving the processes GitLab uses for incident management.
1. Disaster Recovery - we are responsible for managing our disaster recovery processes with a particular focus on reducing our [Recovery time objective](../../../gitlab-com/policies/disaster-recovery/) (RTO).
1. Networking infrastructure - we actively work to improve and expand our services that manage traffic from the edge of our network to the application layer.

## Getting Assistance

- Slack: [#g_networking_and_incident_management](https://gitlab.enterprise.slack.com/archives/C09BM5XCPBP)
- Request for help project: [New issue](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?description_template=SupportRequestTemplate-ProductionEngineering-NetworkingAndIncidentManagement)

## Team Members

{{< team-by-manager-slug "steve-abrams" >}}

## How We Work

We follow [Infrastructure Platforms Project Management practices](/handbook/engineering/infrastructure-platforms/project-management/).

As a new team within Production Engineering, we are currently establishing our workflows and processes. We'll continue to update this page as our team evolves.

### Labels

- For incoming requests use `~"NIM::Requests"`. These are requests coming from outside the team.
- For keeping the lights on (KTLO) issues use `~"NIM::KTLO"`. These are issues related to maintaining our areas of ownership that may not be a full project.
- For project work use `~"NIM::Project Work"`. This should be applied to any issues that are part of epics being surfaced in the [Networking epic](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/1676) or [Incident Management epic](https://gitlab.com/groups/gitlab-com/gl-infra/-/work_items/1873).
- For issues related to team processes (retros, planning, NIM team process changes) use `~"NIM::Meta"`.
- For access requests:
  - `~"NIM::Todo"` - This is applied automatically on all of the baseline entitlement templates for Cloudflare access.
  - `~"NIM::Doing"` - [Optional] Use this if it is going to take some time to action and you want to signal to others it's already been picked up.
  - `~"NIM::Done"` - Once an access request is actioned, change it to this label.

Many issues templates already apply these labels.

### Recurring Task Delegation

The team manages several recurring tasks that require regular attention with a small time commitment (typically around an hour per week). Individual team members own these tasks and are responsible for finding coverage when they are unavailable.

#### Current Recurring Tasks

- **Reminding folks to do overdue post-incident tasks in incident.io** - weekly - Following up on incomplete post-incident action items. **DRI: [Alex](https://gitlab.com/ahanselka)**
- **Reliability reports** - monthly - Published in https://gitlab.com/gitlab-com/gl-infra/reliability-reports/-/issues. **DRI: [Devin](https://gitlab.com/devin) / [Sarah](https://gitlab.com/sarahwalker)**
- **Dealing with incident followup issues** - weekly - Managing and tracking resolution of issues identified during incidents. **DRI: [Steve](https://gitlab.com/sabrams)**
- **EOC coordinator** - ongoing - Rotation lead for the Tier 1 SRE oncall rotation. **DRI: [Sarah](https://gitlab.com/sarahwalker)**
- **IM coordinator** - ongoing - Rotation lead for the Incident Manager oncall rotation. **DRI: [Devin](https://gitlab.com/devin)**
- **Issue triage** - weekly - Currently owned by the Engineering Manager, involves triaging, delegating, and scheduling incoming issues. **DRI: [Steve](https://gitlab.com/sabrams)**

#### Task Ownership Expectations

- Each task has a [DRI](/handbook/people-group/directly-responsible-individuals/) who is accountable for its completion on a regular cadence.
- Task owners must arrange coverage when they will be out of office.
- Coverage arrangements should be communicated in a PTO coverage issue.

## Common Links

- [#g_networking_and_incident_management](https://gitlab.enterprise.slack.com/archives/C09BM5XCPBP)
- [Networking project epic](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/1676)
- [Incident Management project epic](https://gitlab.com/groups/gitlab-com/gl-infra/-/work_items/1873)
- [Issue tracker](https://gitlab.com/gitlab-com/gl-infra/production-engineering/-/issues/?sort=created_asc&state=opened&label_name%5B%5D=group%3A%3ANetworking%20%26%20Incident%20Management&first_page_size=100)
- [Disaster Recovery Practice](/handbook/engineering/infrastructure-platforms/production-engineering/networking-and-incident-management/dr-practice/)
- [AI Prompt library](ai-prompts.md)
