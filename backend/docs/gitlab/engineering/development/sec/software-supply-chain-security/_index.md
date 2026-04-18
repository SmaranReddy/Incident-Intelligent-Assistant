---
title: Software Supply Chain Security Sub-department
---

The Software Supply Chain Security sub-department teams are the engineering teams in the [Software Supply Chain Security Stage](https://about.gitlab.com/direction/software_supply_chain_security/) of the product.

## Vision

To support GitLab's product vision through alignment with the [Software Supply Chain Security stage](https://about.gitlab.com/direction/software_supply_chain_security/) product direction.

## SSCS Charter

**Software Supply Chain Security (SSCS) Stage Team Charter**

### Our Mission

Protect GitLab customers from supply chain attacks while maintaining developer velocity.

We secure the entire software development lifecycle - from code commit to production deployment - by ensuring every artifact is verified, every access is authorized, and every risk is visible.

### What We Do

SSCS secures three critical surfaces:

#### 1. Build & Pipeline Security

- Ensuring artifacts are verifiably built from trusted source code
- SLSA compliance and artifact attestation
- Runner security and isolation
- Secrets management
- Pipeline integrity

#### 2. Identity & Access Management

- Controlling who can do what across the platform
- Authentication (how users prove identity)
- Authorization (what authenticated users can do)
- Zero-trust architecture

#### 3. Compliance & Policy

- Making security measurable and auditable
- Security policies as code
- Compliance evidence generation
- Audit trails and visibility

### How Our Priorities Align with GitLab Executive Priorities

We maintain a 30% capacity buffer for unplanned cross-functional work that supports company-level priorities. This allows us to respond to company-level priorities such as:

- **GitLab Duo & AI** - Anything blocking DAP or AI security
- **Protocells** - Authentication/authorization foundations for Cells architecture

### Our Top 3 Priorities

#### Priority 1: Build a strong security foundation for AuthN/AuthZ

**Objective:** Reduce security incidents
**Why it matters:** Every incident erodes customer trust and drains engineering capacity.

**What we're doing:**

- Authentication consolidation (CYCP)
- Token security improvements
- Critical auditing gaps

#### Priority 2: Engineering Excellence

**Objective:** Become a high-performing, predictable engineering organization
**Why it matters:** We can't deliver strategic value if we're drowning in firefighting and support escalations.

**What we're doing:**

- Reduce support escalation rate from 30% to <15%
- Improve delivery confidence (currently 60-70% → 85%+)

#### Priority 3: Supply Chain Leadership

**Objective:** Establish GitLab as the most trusted supply chain security platform
**Why it matters:** Supply chain attacks are increasing YoY. Customers like GovTech and enterprises demand verifiable security.

**What we're doing:**

- SLSA Level 3 compliance
- Runner identity and verification
- End-to-end artifacts and containers attestation
- Supply chain visibility dashboard

### How Our Top 3 Priorities Enable Strategic Cross-Functional Work

Our priorities aren't just about internal SSCS goals—they're designed to enable the strategic cross-functional work that supports company objectives.

#### Priority 1: Build a strong foundation → Enables Platform Capabilities

Code Yellow/Purple (CYCP) authentication consolidation and security architecture enables:

- Workload Identity federation
- OAuth for ProtoCells - Authentication foundation that ProtoCells depends on
- Fine grained access controls for customers
- Agentic authentication built on secure machine identities

#### Priority 2: Engineering Excellence → Creates Capacity for Unplanned Work

Reducing support burden, improving delivery confidence, protecting capacity for quality enables:

- Faster response time - Can pivot to urgent requests without breaking commitments
- Higher quality - Strategic work doesn't create technical debt that haunts us
- Better estimation - Know true capacity available for unplanned work
- Sustainable pace - Team doesn't burn out from constant context switching

#### Priority 3: Supply Chain Leadership → Differentiates in Enterprise Deals

SLSA compliance, runner security, artifacts attestation, threat analysis, advanced compliance features enables:

- Enterprise credibility - SLSA certification enables deals that fund strategic work
- Compliance foundation - Security architecture that DAP and ProtoCells inherit

### AI Security: Our Newest Frontier

#### Why AI Security Matters

AI features (GitLab Duo, AI agents, code generation) are fundamentally changing our security model:

- AI agents act on behalf of users - requiring composite identity
- AI generates code and modifies repositories - requiring attribution and auditability
- AI accesses customer data - requiring privacy and compliance controls
- AI crosses organizational boundaries - requiring robust authorization

The challenge: We're securing AI features while simultaneously building the platform capabilities they need. This creates unavoidable dependencies and unplanned work.

### How We Work

#### Resource Allocation Model

- **70%** - Planned work
- **30%** - Unplanned work

Unplanned work is not the exception—it's part of the job. Cross-functional dependencies, urgent customer needs, security issues, and tech debt from re-orgs consume 30% of capacity in typical milestones. We budget for this reality rather than pretending it doesn't exist.

#### Managing Unplanned Work

The 30% unplanned budget includes:

- Cross-functional dependencies (Protocells, Dedicated, DAP, etc.)
- Security incidents and vulnerability fixes
- Customer escalations and support
- Tech debt from re-orgs and domain transfers
- Infrastructure issues requiring urgent attention

#### Visibility & Accountability

**Unplanned Work Log** (updated weekly):

- What unplanned work arrived
- Capacity consumed (eng-weeks)
- What got delayed as a result
- Link to the relevant request

This visibility creates accountability and helps justify pushback on future requests.

## Groups

- [Authentication](authentication/)
- [Authorization](authorization/)
- [Compliance](compliance/)
- [Pipeline Security](pipeline-security/)

### Product Documentation Links

- [Authentication and Authorization](https://docs.gitlab.com/administration/auth/)
- [Compliance Center](https://docs.gitlab.com/user/compliance/compliance_center/)
- [Security glossary](https://docs.gitlab.com/ee/user/application_security/terminology/)
- [Pipeline Security](https://docs.gitlab.com/ee/ci/pipelines/pipeline_security.html)

## All Team Members

### Authentication

{{% team-by-manager-slug manager="adil.farrukh" team="Engineer(.*)Software Supply Chain Security:Authentication" %}}

### Authorization

{{% team-by-manager-slug manager="ajaythomasinc" team="Engineer(.*)Software Supply Chain Security:Authorization" %}}

### Compliance

{{% team-by-manager-slug manager="nrosandich" team="Engineer(.*)Software Supply Chain Security:Compliance" %}}

### Pipeline Security

{{% team-by-manager-slug manager="ken.mcdonald" team="Engineer(.+)Software Supply Chain Security:Pipeline Security" %}}

## Stable Counterparts

The following members of other functional teams are our stable counterparts:

{{% engineering/stable-counterparts role="Software Supply Chain Security" other-manager-roles="Engineering Manager(.*)Software Supply Chain Security:(.*)|Director of Engineering(.*)Software Supply Chain Security" %}}

## Software Supply Chain Security staff meeting

The Software Supply Chain Security stage engineering department leaders meet weekly to discuss stage and group topics in the `Software Supply Chain Security staff meeting`. This meeting is open to all team members and is published on the Software Supply Chain Security stage calendar.

Meetings have an agenda and are async-first, where the aim is to resolve discussions async and leave time in the meeting to deep dive into topics that require more discussion.

We use the [Software Supply Chain Security Sub-department Board](https://gitlab.com/gitlab-com/software_supply_chain_security-sub-department/-/boards/4833026) to better organize our discussions.

### Weekly updates

The Software Supply Chain Security development teams provide [weekly status updates](https://gitlab.com/groups/gitlab-com/-/epics/2126) using an issue template and CI scheduled job.
As priorities change, engineering managers update the [template](https://gitlab.com/gitlab-com/software_supply_chain_security-sub-department/-/blob/main/.gitlab/issue_templates/sscs_stage_weekly_update.md) to include areas of interest such as priorities, opportunities, risks, and security and availability concerns. The updates are GitLab internal.

### Quarterly review updates

Every quarter, an engineering manager for each group in the Software Supply Chain Security Sub-department prepares the quarterly review update using the issue template and records approximately 5 minutes to summarize the last quarter from the engineering perspective and present a high-level plan for the group for the next one to respond to quarterly Product strategy and explain our goals for next quarter.

We aim to foster collaboration and communication between engineering managers in the Software Supply Chain Security Sub-department, align groups on product priorities for the next quarter, and celebrate our successes together.

Quarterly review update template can be found [here](https://gitlab.com/gitlab-com/software_supply_chain_security-sub-department/-/blob/main/.gitlab/issue_templates/sscs_stage_quarterly_review.md)).

### PTO

We follow the [Engineering process for taking time off](/handbook/engineering/#taking-time-off) and [GitLab team members Guide to Time Off](/handbook/people-group/time-off-and-absence/time-off-types/).

#### Engineering Leadership - PTO or unavailable

Team members should contact any Software Supply Chain Security Engineering Manager by mentioning in `#sd_sscs_engineering` or `#sscs-development-people-leaders` if they need management support for a problem that arises, such as a production incident or feature change lock, when their direct manager is not available. The Software Supply Chain Security manager can provide guidance and coordination to ensure that the team member receives the appropriate help.

Some people management tasks, including [Workday](https://theloop.gitlab.com/site/4455aa7f-24d9-41f2-b940-467b54962e4d/page/0fa19bf4-fd6a-41b9-9316-c2dcf3add854) and [Navan Expense](/handbook/business-technology/enterprise-applications/guides/navan-expense-guide), may require for escalation or delegation.

## Skills

Because we have a wide range of domains to cover, it requires a lot of different expertise and skills:

| Technology skills | Areas of interest    |
|-------------------|----------------------|
| Ruby on Rails     | Backend development  |
| Go                | Backend development  |
| Vue, Vuex         | Frontend development |
| GraphQL           | *Various*            |
| SQL (PostgreSQL)  | *Various*            |
| Docker/Kubernetes | Threat Detection     |
| [New Auth Claude Expert](https://claude.ai/project/019a0ff4-0efe-7373-af96-82a23aaac734) | New Auth Design |

## Everyone can contribute

At GitLab our goal is that [everyone can contribute](/handbook/company/mission/#contribute-to-gitlab-application). This applies to GitLab team members and the wider community through community contributions. We welcome contributions to any and all features, but recognize that first time contributors may prefer to start with smaller features. To support this we maintain a list of `quick wins` that may be more suitable for first time contributors, and contributors new to the domains in Software Supply Chain Security.

If the contributor needs an EE license, we can point towards the [Contributing to the GitLab Enterprise Edition (EE)](/handbook/marketing/developer-relations/engineering/community-contributors-workflows/#contributing-to-the-gitlab-enterprise-edition-ee) section on the Community contributors workflows page.

## Testing

During the planning phase of a milestone, the EM for each group will create a new issue using the template in [epic](https://gitlab.com/groups/gitlab-org/quality/quality-engineering/-/epics/70), for any major new features and tag Software Engineer in Test from Software Supply Chain Security. SETs from Test Engineering and EMs can periodically review/discuss the list of open issues, and add appropriate priority labels.

The intent of [shifting left and testing at the right level](https://docs.gitlab.com/ee/development/testing_guide/testing_levels.html#how-to-test-at-the-correct-level) is that teams are responsible for testing and to have engineers doing the feature coverage reviews and adding specs or E2E test as needed. The reason for including the SET is to give oversight across the groups and provide guidance/support. If the SET has capacity then they can contribute as needed, using the priority labels, but this is not the expectation.

## Links and resources

{{% include "includes/engineering/software_supply_chain_security-shared-links.md" %}}

### AI and Learning Resources

- [New Auth Expert Claude Project](https://claude.ai/project/019a0ff4-0efe-7373-af96-82a23aaac734) - AI expert for getting answers and information about the New Auth and Code Purple initiative, design, and progress
  - **Note:** Access to this project requires organizational Claude access. Team members need to be part of the GitLab organization in Claude to access this project.

#### Example Prompts for New Auth Expert

**Architecture & Design**

- "What is the GATE architecture and its L0/L1/L2 layers?"
- "Explain the difference between TS (Topology Service) and IAM services"
- "What are the key architectural decisions made for Code Purple?"
- "Show me the 3-level architecture design"

**Timeline & Deliverables**

- "What's the delivery timeline for Code Yellow vs Code Purple?"
- "What are the Q2/Q3 FY27 deliverables?"
- "When will GATE be in production?"
- "What's the roadmap for token consolidation?"

**Token & Permissions**

- "What's the plan for granular Personal Access Token (PAT) permissions?"
- "How will OAuth token permissions work?"
- "Explain Workload Identity Federation timeline"
- "What are the requirements for CI/CD job tokens?"

**Implementation Status**

- "What POCs are currently in progress?"
- "Which features are in scope vs out of scope for Code Purple?"
- "What are the current blockers?"
- "Show me the latest weekly status notes"

**Dependencies & Infrastructure**

- "What infrastructure dependencies exist for GATE deployment?"
- "How does this relate to the Cells architecture?"
- "What database operations are needed?"
- "What's required for self-managed vs GitLab.com deployment?"

**Service Accounts & Machine Identity**

- "How are service accounts being consolidated?"
- "What's the plan for machine identities?"
- "When will service accounts be available on Free tier?"

**Specific Issues & Epics**

- "Find GitLab issues related to granular PAT permissions"
- "What epics are tracking Code Purple delivery?"
- "Show me recent discussions about token scopes"

**Quick Status Checks**

- "What's the latest Code Purple status?"
- "Are there any blockers this week?"
- "What was decided in the most recent sync meeting?"

### Technical Documentation Links

- [End-to-end tests](https://gitlab.com/gitlab-org/gitlab/-/tree/master/qa/qa/specs/features/ee/browser_ui/10_govern)
