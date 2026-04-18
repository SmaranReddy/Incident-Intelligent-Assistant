---
title: "Production Engineering"
description: "Responsible for operating our multi-tenant SaaS offering - GitLab.com"
---

## Teams

Production Engineering consists of:

1. [Cloud Cost Utilization](./cloud-cost-utilization/)
1. [Networking & Incident Management](./networking-and-incident-management/)
1. [Observability](./observability/)
1. [Runners Platform](./runners-platform/)
1. [Runway and Fleet Management](/handbook/engineering/infrastructure-platforms/gitlab-delivery/runway/)

## How We Work

We default to working inline with the GitLab [values](/handbook/values/) and by following the [processes of the wider Infrastructure Platforms section](/handbook/engineering/infrastructure-platforms/project-management/). In addition to this, listed below are some processes that are specific, or particularly important, to how we work in Production Engineering.

### Operating Model

Every quarter, Infrastructure Platforms engineering and product leaders set up our [Operating Model Epics](https://gitlab.com/groups/gitlab-operating-model/-/epics?label_name%5B%5D=Owner%3A%3AInfrastructure%20Platforms). These represent the goals that they need us to accomplish in that quarter.

The Directors and Senior EM's contribute the epics that we commit to for that quarter in support of those goals. The team's epics are [linked](https://docs.gitlab.com/user/group/epics/linked_epics/) to these Operating Model Epics.

By looking at the linked epics on your project, you should be able to see how this tracks back to the department and company goals.

### Roadmaps

In order to know what we can contribute to the quarterly goals, we prepare roadmaps by team in advance. Towards the end of each quarter, the EM arranges a roadmap review session where we agree on what projects are important.

The Senior EM proposes these projects to the Infrastructure Platforms engineering and product leaders who help to clarify any work that should be added or removed before the quarter begins.

### Epic structure

Production Engineering has a [top-level-epic](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/1443). This epic references all of the team top-level-epics.

Each engineering manager maintains their top-level epic so that it can be used for the group review each week. The team's top-level-epic shows all of the work the team is doing.

#### Project Epics

1. Every project in progress is an epic that is linked to the team's top-level-epic.
2. A project epic is assigned to a DRI
3. The DRI is responsible for making sure that the project status is updated in time for the group review each week.

#### Non-Project Epics - KTLO and Incoming Requests

Every quarter, each team has an epic for KTLO and an epic for Incoming Requests. These epics are gathered to the matching Production Engineering Epics for that quarter, which are in turn linked to the Operating Model epics.

At the end of each quarter, the EM summarizes the non-project work in the closing summary.

**What Is KTLO?**

KTLO means "keeping the lights on". This is work we need to do to keep the systems available, performant, reliable and secure.
When we see large KTLO coming, we plan this and turn it into projects. We use the `KTLO` label for these projects.
Smaller KTLO items that we work on should be linked to the quarterly KTLO epic for that team.

**What are incoming requests?**

Incoming requests are things that other teams ask us to do. They ask us because we own certain services, or because we have specialised knowledge to help them out.
Incoming requests that we work on should be linked to the quarterly Incoming Requests epic for that team.

**But I'm working on something else...**

It's likely that you can add this to the KTLO epic, but please ask your manager how to allocate the work if you aren't sure.

### Engagement with Incidents

The Production Engineering team members often have specialized knowledge that is helpful in resolving incidents. Some team members are also SREs who are part of the on-call rota. We follow the guidelines below when contributing to incidents.

For an on-call SRE:

* follow the [ordinary incident management procedures for EOC](/handbook/engineering/infrastructure-platforms/incident-management/#engineer-on-call-eoc-responsibilities)
* at the end of your shift, if there are active incidents or corrective actions in progress, inform the EM who will help you to prioritize the remaining work

For an Incident Manager:

* follow the [ordinary incident management procedures for IM](/handbook/engineering/infrastructure-platforms/incident-management/#incident-manager-on-call-imoc-responsibilities)
* handover should occur as normal, but inform the EM if there are active incidents you are still working on

If you are not EOC or an Incident Manager when an incident occurs:

* For S1 incidents
  * the priority is to get GitLab.com up and running and getting back to a stable state takes priority over project work
  * when the system is stable, contribute to determining the root cause and writing up the corrective actions
  * the IM or Infrastructure EM will delegate corrective actions
  * work with the Production Engineering EM to prioritize any work that arises from an S1
* For all other incidents
  * if you are called into an incident, the priority is to enable others to resolve the problem
  * the expectation is to be hands-off, giving guidance where necessary, and returning to project work as soon as possible

The reason for this position is that our project work prevents future large S1 incidents from occurring.
If we try to participate in and resolve many incidents, our project work is delayed and the risk of future S1 incidents increases.
