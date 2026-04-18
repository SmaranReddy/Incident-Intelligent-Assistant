---
title: Incident Roles - Communications Lead
---
 
## Role Assignment

A separate Communications Lead is assigned based on need, and not automatically paged. Incident Leads may play this role by default.

| Platform | Role |
|----------|------|
| **GitLab.com** | Communications Managers On Call (CMOC) |
| **Dedicated** | GitLab Dedicated Communications Manager On Call (GDCMOC) |

## Core Responsibilities

The Communications Lead manages three main areas of communication during incidents.

### Customer-Facing Communications

This includes incident status updates, required customer actions, and legal notices when applicable.
**Note**: Public RCAs are out of scope for this role and are covered [through our RCA process](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/blob/main/runbooks/on-call.md?ref_type=heads#external-rcas).

### Internal Communications

The Communications Lead also owns internal communications during incidents, serving as the central coordination point for information flow within GitLab. This role ensures that tactical stakeholders have clear channels for questions, updates, and coordination.

**Note**: The exception here is e-group communications, which are overseen by the [Infrastructure Liaison](../#tier-2)

**Information Coordination**: The Communications Lead maintains awareness of what information has been shared with customers and ensures internal teams understand the current status. This includes sharing details about impacted customers and coordinating messaging across different internal groups.

**Channel Management**: When incidents involve multiple Slack channels or communication threads, the Communications Lead ensures proper organization and flow of information. This includes directing stakeholders to the appropriate channels, consolidating duplicate discussions, and making sure incident responders know where to provide updates.

**Stakeholder Guidance**: Customer Success Managers (CSMs), Support teams, and other stakeholders need to know where to ask questions and get updates. The Communications Lead establishes and communicates these pathways, ensuring questions reach the right people and responses flow back effectively.

**Cross-Team Coordination**: The Communications Lead bridges communication gaps between technical incident responders and customer-facing teams, translating technical updates into information or operating procedures (for example, "tag related issues with this tag") that CSMs and Support can use when interacting with customers.

### Status Page Management

For **GitLab.com** incidents, this involves regular updates to [status.gitlab.com](https://status.gitlab.com/) via Status.io. For **Dedicated** incidents, updates are made to the Switchboard Communications page.

## Detailed Workflows

### Public Incident Communications Cadence

The Communications Lead maintains regular update schedules that vary by platform.

#### GitLab.com

Communications follow the [established frequency guidelines](/handbook/support/workflows/cmoc_workflows/#frequency-of-updates) and coordinate with the Incident Manager for update content. To create incidents on Status.io, use `/woodhouse incident post-statuspage` in Slack. Manual updates follow [these instructions](/handbook/support/workflows/cmoc_workflows/#stage-2-manage).

#### Dedicated

For **S1/S2 incidents**, updates are provided every 60 minutes. If the customer didn't raise an emergency ticket through which to centralize communications, [create an outreach ticket](/handbook/support/workflows/dedicated_cmoc/#keep-the-customer-informed). Follow-up with ongoing updates messages every 60 minutes, unless otherwise arranged with the customer. Coordination happens through 5-10 minute sync calls with the Incident Manager to gather update information.

### Status Page Management Process

**Incident Creation** begins immediately upon confirmation of customer impact. The Communications Lead creates an incident on the appropriate status page, but always coordinates with Incident Responder(s) and Incident Manager before making public communications. It's important to avoid rushing communications before understanding the full impact scope.

**Regular Updates** maintain a consistent cadence based on severity, focusing on providing immediate notification and actionable information for customers. Detailed technical information can be provided in follow-up communications.

**Resolution Communications** clearly communicate when the incident is resolved, provide any necessary follow-up actions for customers, and reference where customers can find additional information.

### One-Time Customer Communications

For special communications like bug notices or external communications, the process involves drafting, approval, and delivery phases.

During the **Draft Process**, customer-facing communications are drafted based on incident details, focusing on customer impact and required actions.

The **Approval Process** requires seeking review and approval via the [Customer Communications Approval Process](https://internal.gitlab.com/handbook/legal-and-corporate-affairs/legal-and-compliance/comms-approval-process/) and coordinating with Legal, Corporate Communications, and other stakeholders as needed.

**Delivery and Follow-up** ensures delivery coordination with CS Ops, Marketing, and relevant teams. Field teams are notified of sent communications, and delivery and customer responses are tracked.

### Internal Communications Management

**Question Collection** involves monitoring incident channels for internal questions, adding questions as private comments on incident issues, and collecting and organizing frequently asked questions.

**Information Sharing** includes creating internal FAQ documents shared as private comments on public issues. For **Dedicated** incidents, this also means managing customer channel threads and directing questions to the Communications issue, while coordinating with support teams to ensure consistent messaging.

**Stakeholder Coordination** requires notifying GitLab stakeholders (Customer Success, Community team, Support) of incident status, providing regular updates on customer communications sent, and coordinating with field teams on customer-specific concerns.

### Customer Call Management

There are times when there's an individual customer requesting synchronous communication during incidents. Depending on the specific circumstances, the Communications Lead (in consultation with customer support, account teams and SREs) will decide if synchronous communication is the best option to resolve the customer's concern.

#### When Customer Calls Are Needed

For **Dedicated** incidents, calls may be helpful when incidents are highly impactful, long lasting, or both, as they give customers what they need to keep their own stakeholders and leadership informed. For **GitLab.com** incidents, calls are evaluated case-by-case, typically when there's a significant impact on only a single customer. The general trigger is when direct customer interaction would help resolve the incident or address a specific customer's concerns.

#### Communications Lead Role in Customer Calls

The Communications Lead attends customer calls for a pre-agreed time period to share information, gather feedback, and coordinate internally.

The primary purpose is **Sharing Information with the Customer** involves explaining current incident status and mitigation efforts, responding to previously collected customer questions, and providing clear, non-technical updates appropriate for the customer audience.

During calls, it may also be helpful to conduct **Information Gathering** from the customer. This includes collecting new customer questions, documenting customer-specific impacts or concerns, and identifying any required customer actions that would help with the incident (for example, gathering logs, identifying specific projects, or contacting individual users).

Following a call, the role of **Coordination** means propagating call information within GitLab via the incident channel, internal FAQs, or other means as appropriate to the situation, and ensuring decisions and action items from calls are properly tracked.

#### Incident Lead Participation

Incident Leads should be available to help with customer calls if the Communication Lead believes they would be helpful as a technical resource. This ensures that technical expertise is available as needed, while allowing incident responders the space required to solve the incident.

For **Dedicated** incidents, the Incident Lead is prepared to attend time-limited customer calls (maximum 10 minutes) at the Communications Lead's discretion. For **GitLab.com** incidents, coordination with the Incident Lead for technical expertise happens when needed.

## Key Principles

### Communication Timing

Customers expect prompt notification of incidents affecting them, so **immediate notification** is critical. **Regular updates** with consistent cadence are more important than detailed technical information. When balancing **quality vs. speed**, focus on synchronous signals and timely updates; detailed analysis can follow.

### Coordination Requirements

- **Always coordinate** with Incident Responder(s) and Incident Lead before public communications.
- **Avoid premature communication** that could create inaccurate perception of reliability impacts.
- **Remember** that GitLab often declares incidents early and may downgrade severity as impact becomes clearer.

### Scope Boundaries

| Scope | Activities |
|-------|------------|
| **In Scope** | Status updates, customer actions, immediate incident communications |
| **Out of Scope** | Public RCAs, detailed technical analysis, long-term architectural discussions |

**Escalate** to Infrastructure Leadership when communications needs exceed role scope.

## Resources

### Key Resources

- [CMOC Workflows](/handbook/support/workflows/cmoc_workflows/) - Support team guidance
- [Customer Communications Approval Process](https://internal.gitlab.com/handbook/legal-and-corporate-affairs/legal-and-compliance/comms-approval-process/) - For special communications
- [Infrastructure Leadership Escalation](/handbook/engineering/infrastructure-platforms/incident-management/#infrastructure-leadership-responsibilities) - For role delegation

### Getting Started

New Communications Leads should review the [support handbook section](/handbook/support/workflows/cmoc_workflows/) for detailed operational guidance and onboarding information.
