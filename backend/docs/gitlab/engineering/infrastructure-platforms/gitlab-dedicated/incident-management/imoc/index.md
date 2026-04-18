---
title: "GitLab Dedicated IMOC Response Team"
description: "Workflows, responsibilities, and processes for the GitLab Dedicated Incident Manager On Call (IMOC) response team"
---

## Overview

The GitLab Dedicated Incident Manager On Call (IMOC) response team staffs the **Incident Lead role** for incidents affecting GitLab Dedicated customer tenants. As part of the GitLab Dedicated Platform Leadership Escalation rotation, Dedicated IMOC provides leadership and coordination during high-severity incidents, serving as the escalation point for GitLab Dedicated Engineers On Call (GDEOC).

This page documents Dedicated-specific workflows, tools, and procedures for Engineering Managers serving in the Dedicated IMOC rotation.

**Key Context:**

- **Role Staffed**: [Incident Lead](/handbook/engineering/infrastructure-platforms/incident-management/roles/incident-lead/)
- **Platform**: AWS infrastructure with customer-specific single-tenant
- **Customer Relations**: Tenants in Dedicated have CSMs (Customer Success Managers) and ASE (Assigned Support Engineers). These roles add an extra communication venue. They are kept up to date in Switchboard.
- **Rotation**: [Dedicated Platform Leadership escalation](https://gitlab.pagerduty.com/schedules/PBA2PDS) providing 24/7 coverage
- **Tooling**: incident.io for incident management, Switchboard for customer communications

## When to Engage the Dedicated IMOC

| Scenario                                                     | Engagement Method              | Urgency   |
| ------------------------------------------------------------ | ------------------------------ | --------- |
| **S1/S2 incidents**                                          | Automatic or manual escalation | Immediate |
| **GDEOC non-responsive** (30 min exceeded)                   | Automatic PagerDuty escalation | Immediate |
| **Critical decisions** (Geo failover, emergency maintenance) | Manual escalation by GDEOC     | Immediate |
| **Complex coordination needed**                              | Manual escalation by GDEOC     | As needed |
| **Security vulnerabilities** (high/critical)                 | SIRT engages IMOC              | Immediate |

**How to Page:**

1. In PagerDuty incident, click **"Escalate"** → Select **Level 2**
2. Pages **GitLab Dedicated Platform Leadership Escalation** schedule

3. IMOC acknowledges within 15 minutes, hands on keyboard within 30 minutes

## Incident Lead Responsibilities (Dedicated Context)

As Incident Lead for Dedicated incidents, the IMOC provides coordination and decision-making during incidents. This section describes how these responsibilities are executed in the Dedicated environment.

### 1. Incident Leadership & Coordination

**IMOC leads, does not solve.** Your role is coordination and decision-making, not technical troubleshooting.

**Key Actions:**

- Provide strategic oversight for S1/S2 incidents
- Make critical decisions when GDEOC needs escalation support
- Coordinate resources across teams and external partners
- Determine when to escalate to Director/VP leadership

**Example Questions:**

- "What's your mitigation plan? What do you need from me?"
- "Who else should we bring into this incident?"
- "Based on customer impact, let's proceed with [option X]"

### 2. Customer Communication Oversight

**Communication Flow:** GDEOC → IMOC → GDCMOC → CSM/ASE → Customer

**Critical Timing:**

- **S1**: IMOC provides incident channel updates every 30 minutes (aligned with .com). GDCMOC sends customer updates every 60 minutes. Incident.io has a reminder for incident channel updates.
- **S2**: Regular updates based on impact level
- **Emergency maintenance**: Customer **informed** (not approval-gated) before changes

**Your Actions:**

- Ensure GDCMOC paged immediately for S1/S2 incidents
- Provide clear status updates GDCMOC can relay to customers
- Review external RCAs before customer delivery
- Engage with customers as needed, as specified in [Communications Lead Role in Customer Calls](/handbook/engineering/infrastructure-platforms/incident-management/roles/communications-lead/#communications-lead-role-in-customer-calls)
- Approve cost-impacting mitigations for S1 incidents when the action will restore service (follow [Incident Mitigation process](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/incident-management/-/blob/main/procedures/incident-mitigation-approvals.md))

### 3. Technical Decision-Making

| Decision Type                  | When Required                      | Authority                                                                                                                                                    |
| ------------------------------ | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Geo Failover**               | S1 outage, Geo-enabled tenant      | IMOC decision                                                                                                                                                |
| **Emergency Maintenance**      | Changes outside maintenance window | EM+ (IMOC)                                                                                                                                                   |
| **Cost-Impacting Mitigations** | Infrastructure scaling             | Follow [Incident Mitigation process](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/incident-management/-/blob/main/procedures/incident-mitigation-approvals.md) |
| **Severity Adjustment**        | Impact level changes               | IMOC + GDEOC                                                                                                                                                 |

**Emergency Maintenance Protocol:**

- Always **"customer informed"** - never wait for approval
- Engage GDCMOC immediately
- Proceed with changes while notification is in progress

### 4. Escalation Management

As IMOC, you have several escalation paths available to resolve incidents quickly. This section outlines when and how to engage each escalation point for prompt issue resolution.

#### GDCMOC (GitLab Dedicated Communications Manager On Call)

- **When**: S1 incidents, customer-impacting S2 incidents, emergency maintenance, any customer notification needed
- **How**: `/pd trigger` → Service: "Incident Management - GDCMOC"

#### Dedicated Group Technical Escalation

- **When**: Need Dedicated-specific technical expertise
- **How**: PagerDuty escalation: "Dedicated Group Technical Escalation"

#### AWS Enterprise Support

- **When**: AWS infrastructure issues
- **How**: Follow [AWS Enterprise Support escalation process](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/blob/main/runbooks/on-call.md#escalating-to-aws-enterprise-support)
- **Reference**: [AWS Enterprise Support Sheet](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/blob/main/runbooks/img/GitLab-AWS_Enterprise_Support_Sheet.pdf)

#### Tier 2 / Dev Escalation

- **When**: GitLab app bugs (Database, Gitaly, Rails)
- **How**: [`/inc escalate`](https://gitlab-com.gitlab.io/gl-infra/gitlab-dedicated/team/runbooks/on-call.html#escalating-internally-gitlab-teams) → tier2: [team] in incident Slack channel

#### SIRT (Security Incident Response Team)

- **When**: High/critical security vulnerabilities
- **How**: SIRT engages you; support with Dedicated-specific mitigation

### 5. Post-Incident Oversight

**Required:**

- **S1**: Internal RCA + External RCA (within 1 week)
- **S2/S3/S4**: Internal RCA only if requested by leadership

**Checklist:**

- [ ] Internal RCA completed within 1 week
- [ ] External RCA reviewed and approved (S1)
- [ ] Corrective actions created (`/inc follow-up`)
- [ ] Incident follow-ups documented and linked

## GitLab Dedicated Platform Context

### Dedicated Platform Characteristics

**Single-Tenant:**

- Dedicated AWS infrastructure in customer's account
- Customer-specific maintenance windows
- Isolated from other customers

**Geo-Enabled:**

- Secondary region for disaster recovery
- Geo failover: **30-45 min** (5K Reference Architecture)
- Post-failover requires manual failback

**US Dedicated for Gov (PubSec):**

- FedRAMP-authorized, strict compliance
- **Cannot use incident.io** - PagerDuty only
- No customer protected data in PagerDuty

### Maintenance Windows

**Standard:** Weekly customer-specific windows for planned changes. See [Dedicated Maintenance Windows](https://docs.gitlab.com/administration/dedicated/maintenance/) for details.

**Emergency:**

- Critical security patches, S1 mitigations, infrastructure failures
- Customer informed (not approval-gated)
- Requires EM+ approval (IMOC has authority)

### Key Tools

#### Switchboard (Dedicated-specific)

- Customer notification tool
- GDCMOC sends notifications (IMOC provides status updates)
- Templates: Investigation start, Update, Escalated response, Mitigation in progress, Resolved
- Contains: Customer contacts (ASE and CSM emails), Maintenance Window information, GitLab versions in Tenant information

#### incident.io (same as .com)

- Used by both .com and Dedicated
- Not used by Dedicated for Government (PubSec) due to FedRAMP compliance       
  - PagerDuty only
- Automatic Slack channel and GitLab issue creation
- Works alongside PagerDuty for paging and escalations

## Key Differences: Dedicated vs. .com IMOC

| Aspect                 | GitLab.com                 | Dedicated                                       |
| ---------------------- | -------------------------- | ----------------------------------------------- |
| **Communication**      | StatusPage (public)        | Switchboard (direct to customer)                |
| **Comm Manager**       | CMOC (company-wide)        | GDCMOC (Dedicated-specific)                     |
| **Scope**              | Platform-wide outages      | Customer-specific incidents                     |
| **Infrastructure**     | GCP, continuous deployment | AWS, maintenance windows                        |
| **Emergency Approval** | [.com process]             | IMOC approval                                   |
| **DR/Failover**        | [.com approach]            | Geo failover (30-45 min)                        |
| **Compliance**         | Standard                   | Dedicated for Gov (PubSec) FedRAMP restrictions |
| **Infra Escalation**   | [.com path]                | AWS Enterprise Support                          |

## Critical Decision Scenarios

**Note:** Geo failover decisions require prior training to understand the procedure, tooling, and operational impact. Review the following materials before initiating or approving a failover:

- **Geo Failover Runbook:** https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/blob/main/runbooks/geo-failover.md
- **Geo Failover Fire Drill Training:** https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/blob/main/engineering/disaster_recovery/fire_drills/drill-failover-operator.md

### Scenario 1: S1 Complete Outage - Geo Failover

**Situation:** Customer's GitLab instance completely unavailable.

**Actions (First 15 minutes):**

1. Acknowledge PagerDuty, confirm with GDEOC: "Is GitLab completely unavailable?"
2. Page GDCMOC: `/pd trigger` → "S1 complete outage, investigating mitigation"
3. Ask GDEOC: "Is tenant Geo-enabled? [Preconditions](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/blob/main/engineering/disaster_recovery/README.md#dr-plan-eligibility) met?"
4. **Decision:** Geo failover vs. other recovery?
5. If yes: "Approve Geo failover. GDEOC, proceed. GDCMOC, notify customer."
6. Monitor progress, ensure 60-min updates to GDCMOC, escalate to Director if >1 hour without mitigation

**Key Lesson:** Geo failover is a complex procedure. Evaluate pros and cons with GDEOC and make a balanced decision.

### Scenario 2: Emergency Maintenance - Security Patch

**Situation:** SIRT identifies critical vulnerability requiring patch outside maintenance window.

**Actions:**

1. Confirm with SIRT: "Risk if we wait?" If high: proceed with emergency maintenance
2. Page GDCMOC: One-line summary + estimated downtime
3. Approve as IMOC: "What's deployment + rollback plan?"
4. **Critical**: Customer informed while deployment in progress (not approval-gated)
5. Monitor, provide updates to GDCMOC, verify success
6. Ensure RCA within 1 week

**Key Lesson:** Emergency maintenance needs to be evaluated against risk. It requires resources and coordination against normal maintenance windows. Use the communication tools abundantly to inform customers and next oncallers in the week.

### Scenario 3: GDEOC Non-Responsive

**Situation:** PagerDuty escalates to you. GDEOC hasn't acknowledged (30 min exceeded).

**Actions:**

1. Acknowledge alert, check who's current [GDEOC](https://gitlab.pagerduty.com/schedules/PE57MNA)
2. Slack DM: "Hi [name], you have a PagerDuty alert. Can you respond?"
3. If no response (10 min): Reassign to another Dedicated SRE in current timezone and post in #gitlab-dedicated-team
4. If still stuck: Page all Dedicated Management (@fviegas, @o-lluch, @denhams, @nitinduttsharma)
5. Document, let GDEOC's manager follow up on missed page

**Key Lesson:** Find coverage first, address missed page later.

## Communication Expectations

### S1 Incidents

- IMOC provides incident channel updates every 30 minutes. GDCMOC sends customer updates every 60 minutes. Updates may be sent through GDCMOC zendesk ticket or Switchboard Incident Comms.
- Include: Current status, actions, ETA if available
- Follow template comms style in Switchboard for updates. For manual updates to GDCMOC avoid internal details, give an impact directed statement.
- Escalate to GitLab Dedicated Director if: >1 hour without a mitigation in sight, or significant coordination needed, or customer escalation

### S2 Incidents

- Regular updates based on impact
- IMOC engagement lighter than S1
- Monitor for escalation to S1

### S3/S4 Incidents

- IMOC typically not required
- May engage for complex coordination

## Quick Reference

### S1 Incident Checklist

```plaintext
☐ Acknowledge PagerDuty within 15 min (hands on keyboard within 30 min)
☐ Engage GDCMOC immediately

☐ Confirm severity and impact with GDEOC
☐ Ask: "What is the mitigation plan? What do you need?"
☐ For outage: "Geo-enabled? Consider failover?"
☐ Make critical decisions (failover, emergency maintenance, costs)
☐ Provide incident channel updates every 30 min
☐ Approve emergency changes (EM+ authority)
☐ If >1 hour without mitigation: Escalate to Dedicated Director
☐ Post-incident: RCA within 1 week + external RCA
```

### Who to Page

| Need                   | Who                            | How                                            |
| ---------------------- | ------------------------------ | ---------------------------------------------- |
| Customer communication | GDCMOC                         | `/pd trigger` → "Incident Management - GDCMOC" |
| Dedicated tech help    | Dedicated Technical Escalation | PagerDuty escalation                           |
| AWS infrastructure     | AWS Enterprise Support         | AWS Support Sheet                              |
| GitLab app bug         | Tier 2 / Dev Escalation        | [`/inc escalate`](https://gitlab-com.gitlab.io/gl-infra/gitlab-dedicated/team/runbooks/on-call.html#escalating-internally-gitlab-teams) → tier2: [team] in incident Slack channel                |
| Security vulnerability | SIRT                           | They engage you                                |
| Management support     | Dedicated Management           | @fviegas, @o-lluch, @denhams, @nitinduttsharma |

## Leading vs. Doing

**IMOC SHOULD:**

- Ask: "What's your mitigation plan?" "What is our next step?" "What options do we have?" "Who do we need?"
- Focus: Keep team focused on mitigation, not fixes or root causing
- Decide: "Based on impact, proceed with Geo failover"
- Coordinate: "I'll engage AWS Enterprise Support"
- Ensure process: "Has GDCMOC been updated?"
- Communicate: Ensure updates are ready and sent to customers
- Remove blockers: "I'll approve the cost increase"

**IMOC SHOULD NOT:**

- Jump into debugging: "Let me SSH and check logs"
- Take over: "I'll handle this"
- Contact customers directly: "I'll email the customer"
- Get lost in details: Spending 30 min reading logs

**Exception:** Step in tactically if no GDEOC available or incident escalating rapidly.

## Common Pitfalls

**Not engaging GDCMOC early:** Engage within 10 min for S1 and customer-impacting S2 incidents

**Waiting for perfect info:** Make decisions with 70% information - speed matters

**Taking over vs. leading:** Ask, don't solve. Trust GDEOC as technical expert

**Forgetting post-incident:** Set reminders for RCA (1 week), use `/inc follow-up`

**Not documenting decisions:** Update Slack with rationale: "Decision: Geo failover based on 30-45 min recovery vs. 2-4 hour backup restore"

## Glossary

- **GDEOC**: GitLab Dedicated Engineer On Call
- **GDCMOC**: GitLab Dedicated Communications Manager On Call
- **CSM**: Customer Success Manager
- **ASE**: Account Solution Engineer (also known as Assigned Support Engineers)
- **Switchboard**: Customer notification tool (Dedicated)
- **Geo-enabled tenant**: Instance with secondary region for DR
- **EM+**: Engineering Manager or above
- **SIRT**: Security Incident Response Team
- **FedRAMP**: Federal Risk and Authorization Management Program

## Resources

**Required:**

- [Dedicated On-Call Runbook](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/blob/main/runbooks/on-call.md)
- [Incident Management Handbook](/handbook/engineering/infrastructure-platforms/incident-management/)
- [Incident Lead Role](/handbook/engineering/infrastructure-platforms/incident-management/roles/incident-lead/)

**Dedicated-Specific:**

- [Switchboard Guide](https://console.gitlab-dedicated.com/)
- [Geo Failover Runbook](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/blob/main/runbooks/geo-failover.md)
- [DR Preconditions](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/blob/main/engineering/disaster_recovery/README.md#dr-plan-eligibility)

**Training:**

- **Video: Incident Response Roles vs Response Teams** (Lyle Kozloff): https://youtu.be/vmK9-7roDFM
- **Geo Failover Runbook:** https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/blob/main/runbooks/geo-failover.md
- **Geo Failover Fire Drill Training:** https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/blob/main/engineering/disaster_recovery/fire_drills/drill-failover-operator.md
- LevelUp course: "GitLab Dedicated IMOC Training" (20-25 min) - to be created
- Video walkthrough: "Switchboard Customer Notifications" (5 min) - to be created

**Help:**

- Current IMOC: PagerDuty "GitLab Dedicated Platform Leadership Escalation" schedule
- EMs: @fviegas, @o-lluch, @denhams, @nitinduttsharma
- Slack: #gitlab-dedicated-team, #incident-management

---

**Maintained by:** GitLab Dedicated Infrastructure Team
