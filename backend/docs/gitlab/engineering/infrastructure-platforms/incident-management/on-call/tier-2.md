---
title: On-Call Processes and Policies - Tier 2
aliases:
  - /handbook/engineering/infrastructure-platforms/incident-management/tier2-escalations/
---

Tier 2 Rotations refer to on-call rotations that respond to pages where a human makes a decision to page a team member for support.

## Subject Matter Experts On-Call (Tier 2 SME)

The Tier-2 SME On-Call program enhances incident response by establishing a second tier of specialized support. Subject Matter Experts (SMEs) provide domain-specific knowledge to help resolve complex incidents faster, improve MTTR (Mean Time To Recover), and increase ownership and accountability for service reliability.

This program was introduced at GitLab in 2025 with a target of providing 24x7 coverage for areas where specialised domain knowledge will improve incident response. In practise, many teams are not set up to provide this level of cover. As such, we began with a Pilot Program to understand these gaps and learn how to support these teams to achieve this level of cover.

## When to escalate to Tier 2

Escalate to a Tier 2 team when:

- The incident requires deep domain expertise in a specific service
- The EOC has identified the problem area but needs specialized assistance
- Performance issues or outages are isolated to a specific subsystem

## How to escalate

To page a Tier 2 team:

1. Use the `/inc escalate` command in Slack or click to escalate in the right sidebar of the incident UI
2. Select the appropriate team from the "Oncall team" dropdown menu
3. Provide a clear message describing the issue and what assistance is needed

## Active Tier 2 rotations

A summary of currently active Tier 2 rotations is listed below.

### Gitaly

- Rotation Leader: John Cai
- Coverage: 24x5 partial (2-hour gap)
- Schedule: [schedule](https://app.incident.io/gitlab/on-call/schedules/01JJWAE08T9WDE8T6D4VZPBNXE)
- Escalation History Link: [escalations](https://app.incident.io/gitlab/on-call/escalations?escalation_path%5Bone_of%5D=01JJWB07RXAG02RXYR4QR47J9E)
- [More Information](/handbook/engineering/infrastructure-platforms/tenant-scale/gitaly/#on-call-rotation)

**Expertise Areas:**

- Git repository storage, access, and replication issues
- Gitaly service performance and node failures
- Repository corruption or data integrity concerns
- Git operations (clone, fetch, push) failures

**When to Escalate:**

- High error rates on Git operations
- Repository access failures affecting multiple projects
- Gitaly node or cluster issues

---

### Database Operations (DBO)

**Expertise Areas:**

- PostgreSQL performance, replication, and failover
- Query performance issues, deadlocks, and connection pool problems
- Database migrations blocking deployments
- PgBouncer and database capacity issues

**When to Escalate:**

- Database performance degradation or replication lag
- Failed migrations blocking deployments
- Connection pool saturation
- Slow queries impacting application performance

**Coverage:** Best Effort - 24x5 (Monday-Friday)

---

### AI Powered

- Rotation Leader: Martin Wortschack
- Coverage: 24x5
- Schedule: [schedule](https://app.incident.io/gitlab/on-call/schedules/01K22BJ3V6C41NW8RJ881B08XZ)
- Escalation History Link: [escalation](https://app.incident.io/gitlab/on-call/escalations?escalation_path%5Bone_of%5D=01K22CAST6CK8Y4DVN7ET8YQZX)

**Expertise Areas:**

- AI Gateway and Duo feature availability
- Model serving infrastructure and AI feature performance
- Token usage, rate limiting, and AI provider integrations

**When to Escalate:**

- AI features unavailable or degraded
- High error rates from AI services
- Model serving or AI Gateway connectivity issues

---

### DevOps

- Rotation Leader: [see who is on call](https://app.incident.io/gitlab/on-call/schedules/01K611ZT9YX2PSA8WAMEP6A66G) (falls back to Michelle Gill)
- Coverage: 24x5
- Schedule: [schedule](https://app.incident.io/gitlab/on-call/schedules/01K611MG8T5CW874Q5JZER3H0Z)
- Escalation History Link: [escalation](https://app.incident.io/gitlab/on-call/escalations?escalation_path%5Bone_of%5D=01K6P0Q3D6B7AYV0JT41DP0VME)
- Slack Channel for Rotation Swaps: [`#tier-2-devops-rotation-swaps`](https://gitlab.enterprise.slack.com/archives/C09LLF79AK0)
- Escalation upon non-response: `@mention` the EM or SEM/Director for the on-call team member who did not respond, using the slack channel [`#tier-2-devops-rotation-swaps`](https://gitlab.enterprise.slack.com/archives/C09LLF79AK0) to ask for additional support. In the event that leadership does not respond, use `@here + msg` in [`#tier-2-devops-rotation-swaps`](https://gitlab.enterprise.slack.com/archives/C09LLF79AK0) requesting help from another available engineer.

DevOps is the name given to a group of features that are part of the Rails monolith.
They should be contacted when assistance is needed with one of the features below.

**Teams represented in DevOps Tier 2 on Call:**

CI Platform, Code Review, Container Registry, Environments, Import, Knowledge, Package Registry, Pipeline Authoring, Pipeline Execution, Product Planning, Project Management, Source Code

**Categories/Services represented in DevOps Tier 2 oncall:**

Fleet Visibility, Design Management, Environments, Deployments, Release Management, Importers, Migration, Direct Transfer, Package Registry, Virtual Registry, Dependency Proxy for Containers, Product Planning, Portfolio Management, Requirements Management, Project Management, Issue Tracking, Work Items, Boards, Workspaces, Source Code Management, Repository Management, Protected Branches, Workspaces Rails code, Container Registry Rails Code

**When to Escalate:**

Please do not escalate for general Rails concerns.

- Application-level errors (500s, 422s) with cause inside of one of these features.
- Sidekiq queue backlogs or processing failures where the worker is the responsibility of this group.
- Memory issues in Rails processes originating from this group.
- Application deployment failures requiring rollback where the failure is linked to a feature in this group.

*Note: APAC coverage utilizes IMOC rotation during APAC hours*

---

### Runners Platform

- Rotation Leader: Kam Kyrala
- Coverage: Best Effort - 24x5 (Monday-Friday)
- Schedule: [schedule](https://app.incident.io/gitlab/on-call/schedules/01K7HNBCW9EN2MMS4SHAJ5B2WF)
- Escalation History Link: [escalations](https://app.incident.io/gitlab/on-call/escalations?escalation_path%5Bone_of%5D=01K7HSQ433CMD61V4RNS70BJ47)
- Primary Slack Channel: #g_runners_platform

**Expertise Areas:**

- Runner platform infrastructure and SaaS runner managers
- Job execution issues related to runners (provisioning, startup, teardown)
- Runner registration, capacity, and scheduling concerns
- Runner manager service performance and connectivity

**When to Escalate:**

- Incidents impacting job execution attributable to runners or runner managers
- Widespread runner provisioning failures, hangs, or unexpected timeouts
- Capacity shortfalls or saturation in runner managers affecting customers
- Repeated job failures suspected to be caused by runner platform issues

---

### Fulfillment

- Rotation Leader: James Lopez
- Coverage: 24x5 (Monday-Friday, business hours)
- Schedule: [schedule](https://app.incident.io/gitlab/on-call/schedules/01K99JAT82M1D5HB1MVXX79WHR)
- Escalation History Link: [escalations](https://app.incident.io/gitlab/on-call/escalations?escalation_path%5Bone_of%5D=01K99K4HEXYB7Z7P21BTCY44BF)
- Primary Slack Channel: #s_fulfillment_engineering
- [More Information](/handbook/engineering/development/fulfillment/#escalation-process-for-incidents-or-outages)

**Expertise Areas:**

- CustomersDot application and purchasing infrastructure
- Subscription management, billing, and provisioning systems
- Usage billing flows and consumption-based pricing
- License generation and validation
- Zuora integration and order processing
- Customer portal and self-service workflows

**When to Escalate:**

- CustomersDot outages or critical errors affecting purchases
- Subscription provisioning or license generation failures
- Billing system integration issues impacting customers
- High error rates in purchase or subscription workflows

---

### Authn/Authz/Pipeline Security

- Rotation Leader: Adil Farrukh
- Coverage: 24x5 (Monday-Friday, with APAC being best effort)
- Schedule: [schedule](https://app.incident.io/gitlab/on-call/schedules/01KBH1JNFC4M00T7KDJ4BCFRDD)
- Escalation History Link: [escalations](https://app.incident.io/gitlab/on-call/escalations?escalation_path%5Bone_of%5D=01KBH1JNFC4M00T7KDJ4BCFRDD)
- Primary Slack Channel: ##s_software-supply-chain-security (or #g_sscs_authentication, #g_sscs_authorization, #g_sscs_pipeline_security)
- [More Information](/handbook/engineering/development/sec/software-supply-chain-security/oncall/)

**Expertise Areas:**

- Authentication (SAML, LDAP, OAuth login, Access tokens such as PATs/PrAT/GrATs/CI_JOB_TOKENS)
- Authentication (Enterprise users, Service accounts and Cloud Connector authentication)
- Authorization (Custom roles, Granular permissions on CI_JOB_TOKENS/PATs, ProjectAuthorizationWorker)
- Pipeline Security (OIDC with ID tokens, Secrets manager, External Secrets integrations, Build attestations and Cosign integration)

**When to Escalate:**

- Incidents impacting login or authentication to GitLab.com
- Incidents causing severe disruption due to sidekiq overload on permission update workers
- SIRT issues S2 and above that require immediate action from the engineering team to remediate the problem.
- Recent feature additions for secrets manager, granular permissions or authentication services that are degrading availability of GitLab.com

---

### Dev escalation

- This on-call process is designed for GitLab.com operational issues that are escalated by the Infrastructure team.
- Development team currently does NOT use PagerDuty or incident.io for scheduling and paging.
- No weekday schedule is maintained, the **pagerslack** app will ping available engineers for help.
- A weekend on-call schedule is maintained in this [schedule sheet](https://docs.google.com/spreadsheets/d/10uI2GzqSvITdxC5djBo3RN34p8zFfxNASVnFlSh8faU/edit#gid=0).
- Issues are escalated in the Slack channel [#dev-escalation](https://gitlab.slack.com/messages/CLKLMSUR4) by the Infrastructure team.
- On weekdays, the pagerslack app pings a new engineer every 2 minutes, and if none respond after 6 attempts, it pings all engineering managers with backend engineers to find someone.
- On weekends, first response SLO is 15 minutes. If no response within the first 5 minutes, the requesting team will call the engineer's phone number on the schedule sheet.
- Development engineers do 4-hour shifts.
- Engineering managers do monthly shifts as scheduling coordinators.
- Check out [process description and on-call workflow](/handbook/engineering/workflow/development-processes/infra-dev-escalation/process/) when escalating GitLab.com operational issue(s).
- Check out more detail for [general information](/handbook/engineering/workflow/development-processes/infra-dev-escalation/) of the escalation process.

## Coverage expectations

- **24x5 Coverage**: Monday 00:00 UTC through Friday 23:59 UTC
- **Response SLA**: 15 minutes during coverage hours
- **Weekend/Holiday Coverage**: Critical escalations go to IMOC and Infrastructure Leadership

## Pilot program

The Pilot Program aims to cover ordinary working hours with 24x5 coverage. The Pilot was viewed as an acceptable first iteration towards full coverage because 90% of S1 and S2 incidents take place during ordinary working hours.

For the purpose of this program, ordinary working hours means:

1. *As close as possible to the 8 hours that you would ordinarily work*
2. *Not public holidays or weekends*

As described on the main on-call page, rotation leaders can choose an 8-hour cycle that meets their needs. The recommendation is (UTC):

1. APAC 23:00 - 07:00
2. EMEA 07:00 - 15:00
3. AMER 15:00 - 23:00

If you have team members that don't naturally align to these times, it is at the rotations leader's discretion for how to manage this situation. It is important to provide coverage, and to enable team members to contribute to on-call in a meaningful way. There will always be circumstances where we need to be flexible - and this flexibility goes both ways.

### Public holidays

It is very difficult for the rotation leader to know the public holidays for every team member in their rotation. It is the team member's responsibility to find coverage if they are scheduled for on-call on a public holiday.

If a team members would like to work on a public holiday, they may switch their public holiday to a different day in accordance with our [Public Holiday Policy](/handbook/people-group/time-off-and-absence/time-off-types/#overview-1). This is a voluntary action.

Exceptions:

1. The Netherlands - If a team member cannot take an assigned shift, they must notify their rotation leader with at least 2 working days and the rotation leader (not the team member) is reponsible for finding cover. (As agreed with the Works Council).

## How to determine if we need a specific set of Subject Matter Experts to form a Tier 2 rotation

1. Any new feature or service arriving through the Component Owner Model should have a 24x5 Tier 2 SME for at least the first six months.
2. Any feature or service that generates many incidents regardless of severity.
3. Any feature or service where frequent incidents would cause reputational damage.
4. Any feature or service where the mean time to recover from incidents is consistently high.

### Metrics that can be reviewed to assist in this decision

1. Trend analysis of incidents. This can be done by searching the [Production Incident tracker](https://gitlab.com/gitlab-com/gl-infra/production-engineering/-/issues?sort=created_date&state=opened&first_page_size=100) for the `~incident` label and narrowing your search by group.
   1. Look at how often incidents occur and see if there are any patterns.
   2. Consider incident severity.
2. Incident resolution time.

## How to create a Tier 2 rotation

Tier 2 rotations are for subject matter experts. On average, they should know more about their subject matter than engineers outside of the group. Consider providing training material to confirm that participants are subject matter experts.

See the [general information about how to set up an on-call rotation](_index.md#how-to-set-up-an-on-call-rotation).

Use the [team onboarding template](https://gitlab.com/gitlab-com/gl-infra/production-engineering/-/issues/new?description_template=onboarding-tier2-oncall) to set up a new Tier 2 rotation.

Rotations in the process of being created and onboarded can be viewed in the [On Call Rotation Onboarding board](https://gitlab.com/groups/gitlab-com/-/boards/9981508?label_name%5B%5D=On%20Call%20Rotation).

## When does Tier 2 get paged?

### Tier 1 EOC or IM requests

#### Escalation criteria

The Tier-1 Engineer On-Call (EOC) will perform initial triage and use available documentation before escalating to Tier-2 SMEs. Pages may also be initiated by the Incident Manager (IM) supporting the incident.

##### Before escalating to Tier-2

Tier-1 must:

1. Follow all recommendations in runbooks and playbooks for the affected area
2. Document attempted solutions and outcomes in the incident issue

###### Resource locations

- [Runbooks](https://gitlab.com/gitlab-com/runbooks/-/tree/master/docs)
- [Playbooks](https://internal.gitlab.com/handbook/engineering/tier2-oncall/playbooks/)

#### By severity level

- **S1/S2 Incidents**: When the Tier-1 team cannot resolve them independently using runbooks, documentation or other sources. Due to their critical nature, Tier-2 SMEs should expect to be paged for these incidents when domain-specific expertise is needed.

- **S3/S4 Incidents**: These typically do not require escalation to Tier-2 SMEs during weekends. However, Tier-1 may escalate S3/S4 incidents in specific circumstances:
  - When the customer impact is unclear and requires domain expertise to assess
  - When there's uncertainty about whether the issue might develop into a higher severity incident
  - When multiple lower-severity incidents combined create a potentially broader impact

- If Tier-1 needs help determining whether errors or unusual behavior in a service will affect customers, they may consult with Tier-2 SMEs
