---
title: US Public Sector Services team
---

## Mission

The GitLab US Public Sector Services team mission is to create a fully managed, single-tenant GitLab environment, served through a GitLab Dedicated platform that is purpose-built to help address specific regulatory and compliance requirements of US government agencies at the federal, state, and local level, as well as contractors, educational institutions, and other U.S. customers that run sensitive workloads. It is developed to remove any manual interactions with customer tenant installations, and to ensure that the customer tenants are fully focused on unlocking the power of The One DevOps Platform.

## Vision

The US Public Sector Services group is a customer facing team, with team members focused on a high level of infrastructure automation, and enabling customer interactions with the GitLab Dedicated for US Government platform.

Team mission is to:

- Build cloud infrastructure that meets or exceeds the requirements of the [Federal Risk and Authorization Management Program (FedRAMP)](https://www.fedramp.gov/)
- Develop a 100% automated system for provisioning a large number of single tenant GitLab sites
- Automate maintenance tasks without human interaction
- Create and manage central observability stack, as well as observability stack per customer tenant
- Provide for customer portal (Switchboard), exposing administrative operations to customer tenants

## Performance Indicators

Team performance indicators are not fully defined. We are going to consider a **Provisioning SLO** to start with, possibly followed by [DORA 4 metrics](https://cloud.google.com/blog/products/devops-sre/using-the-four-keys-to-measure-your-devops-performance).

## Team Members

{{< team-by-manager-slug "mckgl" >}}

## Working with us

To engage with the GitLab US Public Sector Services team:

- Create an [in-boundary RFH](https://compsecgov.gitlab-dedicated.us/gitlab-dedicated-us-public-sector/customer-support/-/issues/new?issuable_template=request-for-help) if it is customer supporting specific, and assign it to the GDGEOC.
- [Create a general issue](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/work_items/new?issuable_template=uspubsec_operations) in the GitLab Dedicated team issue tracker. Label the issue with `group::US PubSec` label
- When creating an issue, it is not necessary to `@`mention anyone
- In case you want to get attention, use a specific team handle as defined in [group hierarchy below](#gitlab-group-hierarchy)
- Slack channels
  - For GitLab US Public Sector specific questions, you can find us in [#g_dedicated-us-pubsec](https://gitlab.slack.com/archives/C03R5837WCV)
    - The `@dedicated-uspubsec-team` Slack group can be used in any internal channel to tag the team.
  - Issues relevant to the wider Dedicated Stage may be raised in [#g_dedicated-team](https://gitlab.slack.com/archives/C025LECQY0M)
  - Other teams in Dedicated group have their own work channels for team work discussions:
    - [#g_dedicated-environment-automation-team](https://gitlab.slack.com/archives/C074L0W77V0)
    - [#g_dedicated-switchboard-team](https://gitlab.slack.com/archives/C04DG7DR1LG)

## How we work

### Meetings and Scheduled Calls

Our preference is to work asynchronously, within our project issue tracker as described in [the project management section](#project-management).

The team does have a set of regular synchronous calls:

- `Team call` - During this call, we are sharing important information for team-members day-to-day, as well as project items requiring a sync discussion
- 1-1s between the Individual Contributors and Engineering Manager

Impromptu Zoom meetings for discussing GitLab Dedicated work between individuals are created as needed.
It is expected that these meetings are private streamed, or recorded[^1] and then uploaded to [GitLab Unfiltered playlist](https://www.youtube.com/playlist?list=PL05JrBw4t0KqC5FfUVPyndvLvTWifWbfB) as relevant and allowed per compliance. The outcome of the call is shared in a persistent location (Slack is not persistent). This is especially important as the team grows, because any decisions that are made in the early stage have will be questioned in the later stages when the team is larger.

[^1]: Exceptions to the recording rule are: 1-1 calls, discussions around non-project work, cases where parties do not feel comfortable with recording, or we cannot record due to FedRAMP Compliance. However, even with the exceptions, outcome of project related discussions need to be logged in a persistent location, such as the main issue tracker.

### GitLab Group Hierarchy

We use [GitLab Groups](https://docs.gitlab.com/ee/user/group/#groups) to logically organize team-members working on GitLab Dedicated projects.
The groups cover the following use-cases:

1. GitLab US Public Sector Services group membership: `@gitlab-dedicated/uspubsec`
    - All permanent team-members in the GitLab Dedicated US PubSec team should gain access to this GitLab group as part of onboarding
    - Group mention should only be used in circumstances where the information shared is pertinent for all team members of the GitLab Dedicated US PubSec team
2. Individual team GitLab Dedicated groups have two additional subgroups `maintainers` and `reviewers`, for example: `@gitlab-dedicated/uspubsec/maintainers`
    - `reviewers` GitLab group access is granted to permanent team-members, external contractors, team-members on borrows and similar. This GitLab group type is used to distinguish users without merge rights. Initial reviews should be requested from this group, using the quick action, for example. `/assign_reviewer @gitlab-dedicated/uspubsec/reviewers`
    - `maintainers` GitLab group is granted to permanent team-members only. This group has merge rights, and the group is granted access through [CODEOWNERS approval rules](https://docs.gitlab.com/ee/user/project/codeowners/#code-owners)
      - The transition between `reviewers` and `maintainers` groups is defined by the succesful completion of [Dedicated Maintainer Training](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/#maintainer-training)

### Collaboration Guide

Our teams collaborate using two official issue trackers to track our work in GitLab US Public Sector Services.

### 1. CompSecGov (Secure Tenant)

**Link:** https://compsecgov.gitlab-dedicated.us/gitlab-dedicated-us-public-sector/
**Label:** `group::US PubSec`

| Attribute | Details |
|-----------|---------|
| **Purpose** | Customer-sensitive data and FedRAMP-compliant information |
| **Access** | US PubSec eng team, US PubSec Product Management, Security Compliance, Security Teams, Support, Public Sector field team members (no customer access) |
| **Access Requirement** | [FedRAMP onboarding and Citizen Validation](https://gitlab.com/gitlab-com/gl-security/security-assurance/fedramp/fedramp-certification/-/blob/main/.gitlab/issue_templates/FedRAMP_Onboarding-Current_Employees.md) |
| **Owner** | Engineering / Product Management |
| **Response SLA** | 24 hours (excluding holidays/weekends) to **respond** to account team on issues/comments (excluding incidents or RFHs) |

**Use Cases:**

- Security-sensitive customer information
- Compliance documentation
- Internal planning with customer-specific details
- RFH (Requests for Help)
- Incidents
- Trials
- Customer Onboarding
- Self-service access point for account teams

### 2. Dedicated Team Tracker

**Link:** https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/issues  
**Label:** `group::US PubSec`

| Attribute | Details |
|-----------|---------|
| **Purpose** | Internal planning and feature development |
| **Access** | GitLab internal team only (no customer access). Shared with other Dedicated teams |
| **Owner** | Product Management / Engineering |

**Use Cases:**

- Sprint planning and roadmaps
- Feature development (using codenames where needed)
- Grand Review items
- Self-service access point for account teams
- Feature requests
- Migration planning

> NOTE: Additional collaboration issue trackers for specific customers may be owned and operated by field teams (CSMs/SAs). Only the two official issue trackers serve as the SSOT for US Pubsec PM/Engineering teams

### Slack Channels

We collaborate over slack. Here is a guide to understand the purpose of each channel

| Channel Name | Purpose |
|--------------|---------|
| **dedicated-for-gov-field** | Private channel to discuss dedicated for gov. This channel is private to protect public sector information of our customers. Membership is restricted to US Pubsec members of the Field, Dedicated for Gov R&D staff, Security, Compliance & Support among others to answer product questions |
| **g_dedicated-us-pubsec** | Public channel. Use this for any general engineering discussions. |
| **dedicated-for-gov-stable** | Private channel. For members of the dedicated for gov stable working group. |

### Project Management

We use [epics](https://docs.gitlab.com/ee/user/group/epics/), [issues](https://docs.gitlab.com/ee/user/project/issues/), and [issue boards](https://docs.gitlab.com/ee/user/project/issue_board.html) to organize our work, as they complement each other.

The single source of truth for _all_ GitLab US Public Sector Services work across different functions is the top-level [GitLab US Public Sector Services epic](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/876). Please view that epic for more details on active and upcoming work. To view the specific issues the team is working on, see the overall [issue board](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/boards/4964764?label_name[]=team%3A%3AUS%20PubSec) for the team's work.

#### Epic Hierarchy

[Sub-epics](https://docs.gitlab.com/ee/topics/plan_and_track.html#hierarchies-with-epics) are created under the top-level epic to logically segment work into an organized list of issues that are targeted towards a specific initiative or project milestone.

When applicable, additional sub-epics may be created within the existing epic hierarchy to further segment issues for project tracking purposes.

1. Sub-epics group tasks required to deliver an item mentioned
2. Sub-epics represent an item from the roadmap and are delivered in a specific phase
3. Sub-epics can span multiple months, but their end date should match the 'anticipated completion date' of the roadmap phase they are added to.

In addition to epics, [milestones](https://docs.gitlab.com/ee/user/project/milestones/) can be used as a project tracking tool that allows issues across multiple epics to be associated with a target milestone date. Milestones can be filtered on an issue board for a single-pane view of current priority issues to be actioned by the team.

#### Epic Owners

Each epic has a single DRI who is responsible for delivering the project. DRIs for each epic are listed at the top of the description of each epic per Epic Structure.

#### Epic Owner Responsibilities

The DRI needs to:

1. Work with others to move issues through the boards
2. Ensure epic meets criteria outlined in [Epic Structure](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/#epic-structure)
3. Provide updates on DRI's epic in epic description according to process outlined in [Status Update Process](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/#status-update-process) below.

#### Epic structure

Each epic and child sub-epics must include the following:

**Description**

1. **DRI** who is responsible for this epic.
2. **Background**, including a problem statement, to provide context for people looking to understand the epic.
3. **Exit criteria** for the specific goals of the epic.
4. **Status yyyy-mm-dd** should be the final heading in the description.
    1. This enables others who are interested in the epic to see the latest status without having to read through all comments or issues attached to the epic.
    2. This heading is used to auto-generate the status information on the top-level epic.

**Epic meta data**

1. **Start date** is set to the expected start date, and updated to be the actual start date when the project begins.
2. **Due date** is set to be the expected end date.

Labels are described in the [epic label section](#epic-structure).

#### Issue boards

[Issue boards](https://docs.gitlab.com/ee/user/project/issue_board.html) are used to track the overall status of epics and/or milestones.

##### Accessing US Public Sector Services Issue Boards

1. Navigate to the [Dedicated Issue Tracker](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team) project.
2. Using the navigation menu on the left, hover over the **Issues** section and select **Boards**
3. Using the drop-down menu located in the upper-left (next to the **Search Filter**) select the specific board associated with the epic or milestone that you want to view. If you know the name of the board you want to view, you can type it in the drop-down search box.
4. Once selected, the issue board will contain a kanban layout that contains a list of issues based on the filter provided (ex. milestone, epic, label, etc.), as well as various columns (also referred to as **Lists**) to reflect the desired organization of the issues. The US Public Sector Services team uses labels as the primary List filter. See the section below for how these issue boards are created and filtered.

> Note: Issue boards used for time-blocking project work should be deprecated once complete to maintain simplicity.

### Execution

The team operates in a kanban fashion. Issues are prioritized in the kanban board and self-assigned. We leverage scoped [workflow labels](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/#workflow-labels) to track different stage of work.

### Status Updates/Process

See the [Stage level Status Updates page section](/handbook/engineering/infrastructure-platforms/gitlab-dedicated/#status-updates) for details on status updates and status update process

The status for all work for this team is maintained in the description of the top-level [GitLab US Public Sector Services epic](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/876) so that it is visible at a glance.

#### Reporting

We provide reports on status of GitLab Dedicated to meet Top Cross-Functional Initiative requirements.
