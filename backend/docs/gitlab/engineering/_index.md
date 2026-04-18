---
title: Engineering
---

[The GitLab Product team](/handbook/product/) looks ahead for expanding the platform “What" (customer needs) and "Why” (business strategy) and Engineering determines the “How" (technical implementation) and "When” (scheduling) of the platform releases. The content on this page talks about how we do engineering at GitLab.

## Engineering Direction

GitLab has a Three-Year Strategy, and we're excited to see every member of the Engineering division contribute to achieving it. Whether you're creating something new or improving something that already exists, we want you to feel empowered to bring your best ideas for influencing the product direction through improved scalability, usability, resilience, and system architectures. And when you feel like you need to expand your knowledge in a particular area, know that you're supported in having the resources to learn and improve your skills.

Our focus is to make sure that GitLab is enterprise grade in all its abilities and to support the AI efforts required to successfully launch AI features to General Availability.

Making sure that GitLab is enterprise grade involves several teams collaborating on improving our disaster recovery and support offerings through ongoing work with GitLab Dedicated and Cells infrastructure. Our goal here is improved availability and service recovery.

## Engineering Culture

Engineering culture at GitLab encompasses the processes, workflows, principles
and priorities that all stem from our [GitLab Values](/handbook/values/).
All these things continuously strengthen our engineering craftsmanship and
allow engineers to achieve engineering excellence, while growing and having a
significant, positive impact on the product, people, and the company as a whole.
Our engineering culture is primarily being carried and evolves through
knowledge sharing and collaboration. Everyone can be part of this process
because at GitLab everyone can contribute.

### Engineering Excellence

Engineering excellence can be defined as an intrinsic motivation to improve
engineering efficiency, software quality, and deliver better results while
building software products. Engineering excellence is being fueled by a strong
engineering culture combined with a mission: to build better software that
allows everyone to contribute.

### Engineering Initiatives

Engineering is the primary advocate for the performance, availability, and security of the GitLab project. Product Management prioritizes 60% of engineering time, so everyone in the engineering function should participate in the Product Management [prioritization process](/handbook/product/product-processes/cross-functional-prioritization/) to ensure that our project stays ahead in these areas. Engineering prioritizes 40% of time on initiatives that improve the product, underlying platform, and foundational technologies we use.

Work in the 40% time budget should be coordinated and prioritized by the Engineering Manager of a team. Use the label `Engineering Time` for issues and MRs that are done as part of it so we can follow the work and the results across the engineering division.

- Contributing to broad engineering initiatives and participating in working group-related tasks.
- Review fixes from our support team. These merge requests are tagged with the `Support Team Contributions` label. You can [filter on open MRs](https://gitlab.com/gitlab-org/gitlab/-/merge_requests?label_name%5B%5D=Support%20Team%20Contributions).
- Working on high priority issues as a result of [issue triaging](/handbook/product-development/how-we-work/issue-triage/). This is our commitment to the community and we need to include some capacity to review MRs or work on defects raised by the community.
- Improvements to the performance, stability and scalability of a feature or dependency including underlying infrastructure. Again, the Product team should be involved in the definition of these issues but Engineering may lead here by planning, prioritizing, and coordinating the recommended improvements.
- Improvements and upgrades to our toolchain in order to boost efficiency.
- Codebase improvements: Removing technical debt, updating or replacing outdated dependencies, and enhancing logging and monitoring capabilities.
- Constructing Proof-of-Concept models for thorough exploration of new technologies, enhancements and new possibilites.
- Work on improvements and feature enhancements to the product, in the sense of internal community contributions, that would increase our internal engineering productivity by focusing on ready-to-go items that are currently assigned a low priority in the backlog.

### Engineering Innovation

Engineering Innovation is a new process geared toward individual or small-team collaboration that encourages engineers to explore new ideas and Proof-of-Concepts. These projects are typically lean, time-boxed, iterative, and designed to validate whether an idea has the potential to evolve into a viable experimental feature or product.  See the [Innovation at GitLab Guide](./workflow/engineering-innovation.md).

### Technical Roadmaps

Some of the above examples for the 40% time budget can help in forming a long-term technical roadmap for your group, and determine how best to prioritize your technical work to support overall business goals. In addition to the examples above:

- Ask yourself these questions
  - What are your most frequent sources of delays? (Could be long-standing tech debt you have to work past while developing, could be lack of reviewers for your domain, could be external to your team like with pipeline duration)
  - Do you have any consistently similar bugs or security issues that come in due to a certain area?
  - Has your team been talking about potentially refactoring any areas?
  - Is your team struggling with certain processes?
  - Have you had recent incidents that allude to a larger problem?
  - Are you getting frequent requests for help in some area?
  - Is your team frequently missing their deliverable commitments? What would help?
  - Does your area have performance (slow endpoints, inconsistent responses, intermittent errors) or scalability (the feature or area as-is will not scale) concerns?
  - Where do you see the biggest instability? Have you talked to operations and support about feedback for  your area?
  - Do you have application or rate limits in the right places?
  - Have you burned down your security, corrective action, and infradev issues?
  - Is your error budget green?
  - Have your feature flags been removed from the codebase yet?
  - Do you have adequate unit test, integration test and E2E coverage?
  - Do you have adequate documentation for your features?
  - Do you have adequate telemetry , logging, monitoring of your features?
  - Do you have adequate error handling and error codes that allows fast and easy diagnostics?
- Gather data like this
  - Master:Broken issues
  - ~"severity::1" and ~"severity::2"  bugs
  - Missed-Slo issues
  - Flaky test issues
  - ~"type::maintenance" issues
- Think about the future state of your product
  - Where do you want your product to be this time next year?
  - What are the technical requirements to achieve that?
  - What are technical topics that would benefit from research/POCs?
  - What would make it easier for you to achieve that if it was no longer a factor?
  - What would be the performance and/or business impact once you address these issues?
  - How would you evolve your team processes to regularly review your technical roadmap?

#### Technical roadmap process

Engineering Managers (EMs) are responsible for collaboratively developing their team's technical roadmap backlog. All items should be documented as epics and issues using the "Technical Roadmap" label.

Global initiatives will be defined and must be incorporated into each group's roadmap and prioritization (e.g., allocating 40% of front-end capacity for Vue upgrade, completing all Cells issues for a specific area by milestone XYZ).

Prioritization of items should align with:

1. General business goals
2. Engineering vision
3. Team capacity and expertise

Planning Guidelines:

- Allocate 40% of the overall time budget for technical roadmap items in the normal milestone planning process.
- Use the "Technical roadmap" label for all related issues to facilitate tracking and coordination.

Key Steps:

1. Identify and document technical debt and improvement opportunities
2. Assess impact and effort for each item
3. Prioritize based on business value and strategic alignment
4. Integrate with existing iteration/milestone planning
5. Regularly review and adjust the roadmap

This process ensures a balanced approach between feature development and technical improvements, promoting long-term sustainability and efficiency of the engineering organization.

### Community Contributions

We have a 3-year goal of [reaching 1,000 monthly contributors](/handbook/company/strategy/#2-build-on-our-open-core-strength) as a way to mature new stages, add customer-desired features that aren't on our roadmap, and even translate our product into multiple languages.

### Diversity

[Diverse teams perform better](https://www.cio.com/article/189194/5-ways-diversity-and-inclusion-help-teams-perform-better.html). They provide a sense of belonging that leads to higher levels of trust, better decision making, and a larger talent pool. [They also focus more on facts, process facts more carefully, and are more innovative](https://hbr.org/2016/11/why-diverse-teams-are-smarter). By hiring globally and increasing the numbers of women and under represented groups (URGs) in the Engineering division, we're helping everyone bring their best selves to work.

### Growing our team

Strategic hiring is a top priority, and we're excited to continue hiring people who are passionate about our product and have the skills to make it the best DevSecOps tool in the market. Our current focus areas include reducing the amount of time between offer and start dates and hiring a diverse team (see [above](#diversity)). We're also implementing industry-standard approaches like structured, behavioral, and situational interviewing to help ensure a consistent interview process that helps to identify the best candidate for every role. We're excited to have a recruiting org to partner with as we balance the time that managers spend recruiting against the time they spend investing in their current team members.

### Expand customer focus through depth and stability

As expected, a large part of our focus is on improving our product.

For **Enterprise customers**, we're refining our product to meet the levels of security and reliability that customers rightfully demand from SaaS platforms _(SaaS Reliability)_. We're also providing more robust utilization metrics to help them discover features relevant to their own DevOps transformations _(Usage Reporting)_ and offering the ability to purchase and manage licenses without spending time contacting Sales or Support _(E-Commerce and Cloud Licensing)_. Lastly, in response to Enterprise customer requests, we're adding features to support Suggested Reviewers, better portfolio management through Work Items, and Audit Events that provide additional visibility into user passive actions.

For **Free Users**, we're becoming more efficient with our open core offering, so that we can continue to support and give back to students, startups, educational institutions, open source projects, GitLab contributors, and nonprofits.

For **Federal Agencies**, we're obtaining FedRAMP certification to strengthen confidence in the security standards required on our SaaS offering. This is a mandated prerequisite for United States federal agencies to use our product.

For **Hosted Customers**, we're supporting feature parity between Self-Managed and GitLab Hosted environments through the Workspace initiative. We're also launching GitLab Dedicated for customers who want the flexibility of cloud with the security and performance of a single-tenant environment.

For customers using **CI/CD**, we're expanding the available types of Runners to include macOS, Linux/Docker, and Windows, and we're autoscaling build agents.

### Taking time off

{{% note %}}
This process is expected for PTO that is five consecutive days or more, inclusive of adjacent public holidays (excluding weekend days). For PTO that is fewer than five consecutive days, including the cases where there are multiple PTO blocks with fewer than 5 consecutive days and a few working days in-between, a coverage issue is not required but a coverage issue can be filed for PTO of any length, especially if it'd be helpful to balance team continuity and individual flexibility.
{{% /note %}}

In order to ensure business continuity, and deliver on commitments; the Engineering Division is adopting a PTO Coverage Issue Process. Processes like this are already formalized in GitLab (e.g. [PM Coverage Issue](/handbook/product/product-management/product-manager-role/#creating-a-pm-coverage-issue)) and some team's within Engineering have practiced this regularly at the Management+ level. This allows us to continue to support team member well-being through time away without negatively impacting the rest of the team.

A PTO Coverage issue is required for [job grades 8 and up](/handbook/total-rewards/compensation/compensation-calculator/#job-grades). For job grades 7 and below a PTO Coverage issue is recommended as there is value in going through the process of creating the PTO Coverage issue even if there are minimal items to include (for all levels) in that it forces you to think about what you have on your plate and what impact your PTO will have on those items. So whether the result is that the work waits or there is someone designated as a replacement DRI, it makes the decision explicit and documented.

Once planning for a milestone has been completed (see [**Monday, 5 days before the milestone begins**](/handbook/engineering/workflow/#product-development-timeline)) PTO for periods longer than 5 consecutive days, inclusive of adjacent public holidays (excluding weekend days), cannot be requested. This is to prevent disrupting plans for that milestone. There are exceptions to this, but all need to be discussed with your manager. Examples include:

- urgent scenarios
- a team/individual hits targets earlier in the milestone ([we measure impact, not activity](/handbook/values/#measure-impact-not-activity))
- a strong need for a team member to take PTO of this length

These issues will help inform teams as they plan their milestones to ensure the work teams are committed to can be achieved with the staff available, or if there will be a lack of staff to achieve those commitments, to work with team members to see what can be done to achieve the results for our customers.

The process below helps to clarify and expand upon the [Flexible PTO Policy](/handbook/people-group/time-off-and-absence/time-off-types/) by making the coordination with the team members manager explicit.

#### 1. Creating an Engineering coverage issue

You should use [this issue template](https://gitlab.com/gitlab-com/engineering-division/pto-coverage/-/issues/new) to define handshake responsibilities. For extended leave, it is important to find one or more Directly Responsible Individuals (DRIs) that will be able to make decisions while you are away. This may be your manager, another engineer, or maybe the Product Manager for your team. The coverage issue should contain all the necessary information for the DRIs to make good decisions in your absence, so please make sure to include as much detail as needed. The coverage issue should highlight work impact estimates, mitigations identified, and coverage alternatives.

If additional context needs to be shared to provide color to the coverage issue, you can consider a specific handover meeting to cover further details.

It is recommended to work with your manager and other stakeholders when considering cross-functional teammate capacity for a coverage task assignment. For example, while it’s optimal for PM, EM, and PDs to assist in covering for each other given their shared knowledge of their product area including customers and users, PM teammates may or may not have the bandwidth or expertise to take on covering engineering specific responsibilities. Alternatively, it may be better for the manager of the engineer or another engineer in the same stage to aid in coverage. Plan to have the necessary conversations across teams and managers.

#### 2. Sharing your Engineering coverage issue with your manager

Once you’ve filed your engineering coverage issue, share this with your manager prior to milestone planning so they can review and approve. Check the [latest guidance in our PTO policy](/handbook/people-group/time-off-and-absence/time-off-types/) on how much notice is required.

Consider whether any new commitments would be affected by your planned PTO. If a team member falls behind on something, they will need to make sure they have a coverage plan in place to ensure success of their commitments.

#### 3. Manager reviews coverage issue

Once the team member has shared their coverage issue with their manager, the manager will review the coverage issue and validate assumptions with stakeholders or impacted project DRIs as needed.

The manager will make a decision on approval or discuss different arrangements or other contingency plans. Once the manager ticks their box on the coverage issue approving the leave, enter the time off into Workday.

#### 4. Communicate your time off

After team members' coverage issue is approved, team members will [communicate their time off](/handbook/people-group/time-off-and-absence/time-off-types/) and enter the PTO into Deel/Workday including a link to their coverage issue. Team members will share their coverage issue with their relevant colleagues via Slack channels, GitLab status, etc. ahead of the milestone planning.

#### 5. Take your time off

Please disconnect and take the time off that you need!

#### 6. Returning from Time Off

Returning from time off can be overwhelming and daunting. You should work with your DRIs to understand what has changed during your absence and what the current priorities are. Also, communicate transparently that your response time may be slower because you are catching up. Here are some additional tips on [how to return back to work after time off](/handbook/people-group/time-off-and-absence/time-off-types/).

## Engineering Departments

There are five departments within the Engineering Division:

- [DevOps Engineering Department](/handbook/engineering/devops/)
- [AI Engineering Department](/handbook/engineering/ai/)
- [Sec Department](/handbook/engineering/development/sec/)
- [Infrastructure Platforms](/handbook/engineering/infrastructure-platforms/)
- [Support Engineering Department](/handbook/support/)

## Other Related Pages

- [CTO Leadership Team](/handbook/engineering/cto-leadership-team/)
- [Database Engineering](/handbook/engineering/development/database/)
- [Development Principles](/handbook/engineering/development/principles/)
- [Engineering Metrics](/handbook/product/groups/product-analysis/engineering/dashboards/)
- [Engineering READMEs](/handbook/engineering/readmes/)
- [Frequently Used Projects](/handbook/engineering/projects/)
- [GitLab Innovation Program](/handbook/legal/patent-program/), managed by the GitLab Legal Team
- [Mentorship](/handbook/people-group/learning-and-development/mentor/)
- [Pajamas Design System](/handbook/product/ux/pajamas-design-system/)
- [R&D Tax Credit Applications](/handbook/engineering/tax-credits)

### Workflows

- [Engineering Workflow](/handbook/engineering/workflow/)
  - [Code Review](/handbook/engineering/workflow/code-review/)
  - [Security Issues](/handbook/engineering/workflow/#security-issues)
  - [Architecture Design](/handbook/engineering/architecture/workflow/)
  - [Root Cause Analysis](/handbook/engineering/workflow/root-cause-analysis/)
  - [Strategic Priority Codes](/handbook/engineering/workflow/strategic-priority-codes/)
  - [Automation](/handbook/engineering/workflow/automation/)
  - [GitLab Repositories](/handbook/engineering/workflow/gitlab-repositories/)
  - [Hiring](/handbook/engineering/workflow/hiring/)
  - [Demos](/handbook/engineering/workflow/demos/)
  - [Cross-Functional Prioritization](/handbook/engineering/workflow/cross-functional-prioritization/)
  - [Developer Onboarding](/handbook/engineering/workflow/developer-onboarding/)
  - [Engineering Communication](/handbook/engineering/workflow/engineering-comms/)
  - [Development Processes](/handbook/engineering/workflow/development-processes/)
  - [Development Onboarding](/handbook/engineering/workflow/development-onboarding/)
- [Issue Triage Policies](/handbook/product-development/how-we-work/issue-triage/)
- [Contributing to Go projects](https://docs.gitlab.com/ee/development/go_guide/index.html)
- [Wider Community Merge Request Triage Policies](/handbook/engineering/infrastructure-platforms/developer-experience/merge-request-triage/)
- [Unplanned Critical Patch releases](/handbook/engineering/releases/patch-releases/#patch-release-types)
- [Incident Management](/handbook/engineering/infrastructure-platforms/incident-management/)

### GitLab in Production

- [Workflow Diagram](/handbook/engineering/workflow/related-workflows/)
- [Error Budgets](/handbook/engineering/error-budgets/)
- [Performance of GitLab](/handbook/engineering/performance/)
- [Monitoring of GitLab.com](/handbook/engineering/monitoring/)
- [Production Readiness Guide](https://gitlab.com/gitlab-com/gl-infra/readiness/-/blob/master/.gitlab/issue_templates/production_readiness.md)

### People Management

- [Engineering Career Development](/handbook/engineering/careers/)
- [Engineering Career Mobility Principles](/handbook/engineering/careers/#mobility-principles)
- [Emerging Talent @ GitLab](/handbook/hiring/)
- [Engineering Management](/handbook/engineering/careers/management/)

### Cross-Functional Prioritization

See the [Cross-Functional Prioritization page](/handbook/engineering/workflow/cross-functional-prioritization) for more information.

### SaaS Availability Weekly Standup

To maintain high availability, Engineering runs a weekly SaaS Availability standup to:

- Review high severity (S1/S2) public facing incidents
- Review important SaaS metrics
- Track progress of Corrective Actions
- Track progress of [Feature Change Locks](#feature-change-locks)

Similar to Rounds run at a hospital, this meeting is intended to provide engineering leadership with a review of the reliability of all GitLab Platforms and enable coordinating cross-department discussion and decision making.

- [SaaS Availability Meeting Agenda](https://docs.google.com/document/d/1vCPb5VsF0nldiVVAfQXWegeNrvejvxn07TxorTn8E-Y/edit?usp=sharing)

Each week teams from the engineering division report on incidents and key metrics:

1. Incident Review
   - Page Volume - indicates the stress on our responding teams
   - High severity incidents (S1/S2) - raises visibility of the challenges faced by all GitLab platforms
1. [Capacity Planning](/handbook/engineering/infrastructure-platforms/capacity-planning/)
   - P1 and P2 issues and other highlights from each GitLab platform are raised.
1. Action Overview
   - Overdue S1/S2 [infradev issues](/handbook/engineering/workflow/#infradev) from each department
    Include screenshots of the following graphs in the [agenda](https://docs.google.com/document/d/164hNObllaLWosG110-A0UouYlcaqOxbPpHATFD38_Gw/edit#heading=h.59wtcja0o8t7).
   - [Feature Change Locks](#feature-change-locks) - each team in an FCL provides an update
1. Discussions and Deep Dives - specific cross-department discussions

### Feature Change Locks

A Feature Change Lock (FCL) is a process to improve the reliability and availability of the GitLab platform. FCLs are enacted for all S1 and S2 severity incidents on any GitLab Platform (GitLab.com or Dedicated) including the License App, Customers Dot, and Versions.

All exceptions must be approved by a VP of Engineering. Reasons for an FCL exception may include:

- The incident had no public facing impact
- The incident was not caused by an engineering division change

The [team](/handbook/company/structure/#organizational-structure) involved is the owner of the service or feature. The team is responsible for both the coordination and completion of the FCL. The manager of the team is responsible for:

- Form the group of engineers working under the FCL. By default, it will be the owning team, but it could be a reduced group if there is not enough work for everyone.
- Plan and execute the FCL.
- Inform their manager (e.g. Senior Manager / Director) and Product counterpart that the team will focus efforts towards an FCL which may impact capacity planning.
- Provides updates at the [SaaS Health Review](/handbook/engineering/infrastructure-platforms/saas-health-review).

Direct reports involved in an active [borrow](/handbook/product/product-processes/pm-procedures/#borrow) should be included if they were involved in the authorship or review of the change.

The purpose is to foster a sense of ownership and accountability amongst our teams, but this should not challenge our no-blame culture.

#### Timeline

Rough guidance on timeline is provided here to set expectations and urgency for an FCL.  We want to balance moving urgently with doing thoughtful important work to improve reliability.  Note that as times shift we can adjust accordingly. The DRI of an FCL should pull in the timeline where possible.

The following bulleted list provides a suggested timeline starting from incident to completion of the FCL.  "Business day x" in this case refers to the x business day after the incident.

- Day 0: Incident
- Business day 1: open the FCL issue and begin planning. Request approval from VP of engineering if an FCL is not believed to be necessary.
- Business day 2-3: planning time
- Business days 2-9:  complete planned work
- Business days 10-11:  closing ceremony, retrospective and report back to the SaaS Health Review

#### Activities

During the FCL, all in-flight feature work is paused on the impacted service or feature category. Team members involved in the FCL are exclusively focused on [reliability work](#scope-of-work-during-fcl). Maintainer duties can still be done during this period and should keep other teams moving forward. Explicitly higher priority work such as security and data loss prevention should continue as well.

While an FCL generally will include the team that owns the feature category or service, other team members who contribute to the development of the feature or service may be included. As part of FCL setup, the team should:

1. Identify all services and feature categories the team is responsible for that are under FCL
2. Identify closely coupled or dependent services and teams who may also make changes to those services or feature categories
3. Notify those teams about the FCL and coordinate to ensure changes are appropriate within the [scope](#determining-fcl-scope)
4. Consider applying a [Change Lock](https://gitlab.com/gitlab-com/gl-infra/change-lock/-/blob/master/README.md) to your teams services to prevent unintended deployments of your service.

Teams making changes to services or feature categories owned by teams in an FCL should coordinate with the FCL team and should be included in the FCL issue for visibility. The [Feature Change Locks project](https://gitlab.com/gitlab-com/feature-change-locks/-/work_items) tracks all open FCLs.

The team(s) must:

- Create a public slack channel called `#fcl-incident-[number]`, with members
  - The Team's Manager
  - The Author and their teammates
  - The Product Manager, the stage's Product leader, and the section's Product leader
  - All reviewer(s)
  - All maintainers(s)
  - The chain-of-command from the manager to the VP (Sr Manager, Sr/Director, VP, etc)
- Create an [FCL issue](https://gitlab.com/gitlab-com/feature-change-locks/-/issues/new?issuable_template=feature-change-lock) in the [FCL Project](https://gitlab.com/gitlab-com/feature-change-locks/) with the information below in the description:
  - Name the issue: `[Group Name] FCL for Incident(s) ####`
  - Links to the incident, original change, and slack channel
  - FCL Timeline
  - List of work items
- Complete the written Incident Review as the first priority after the incident is resolved.  The Incident Review must include completing all fields in the Incident Review section of the issue template. The incident issue should serve as the single source of truth for this information, unless a linked confidential issue is required. Completing it should create a common understanding of the problem space and set a shared direction for the work that needs to be completed.
- See that not only all procedures were followed but also how improvements to procedures could have prevented it
- A work plan referencing all the Issues, Epics, and/or involved MRs must be created and used to identify the scope of work for the FCL. The work plan itself should be an Issue or Epic.
- Daily - add an update comment in your FCL issue or epic using the template:
  - Exec-level summary
    - Target End Date
    - Highlights/lowlights
- Add an agenda item in the [SaaS Availability weekly standup](/handbook/engineering/#saas-availability-weekly-standup) and summarize status each week that the FCL remains open.
- Hold an asynchronous `closing ceremony` in the issue and/or slack channel upon completing the FCL to review the retrospectives and celebrate the learnings. Document all learnings in the issue.
  - All FCL stakeholders and participants shall participate async. Managers of the groups participating in the FCL, including Sr. EMs and Directors should be invited.
  - Outcome includes [handbook](/handbook/) and [GitLab Docs](https://docs.gitlab.com/ee/) updates where applicable.

##### Determining FCL Scope

**What's In Scope**

The **team** that owns the service or feature category identified as causal or contributing to the incident goes into FCL. The team pauses all in-flight feature work on the services and feature categories they are responsible for.

For teams that maintain shared service infrastructure (e.g., the team that maintains Sidekiq infrastructure), if they go into FCL, they may not make changes to that infrastructure. Other teams may continue to use the service normally - for example, adding new Sidekiq jobs or running database migrations. As part of FCL setup, the team should notify other teams who may make changes to the infrastructure about the FCL

**Side-Effects vs Related Causes**

When determining FCL scope, it's important to distinguish between side-effects and related causes:

**Side-Effects (Team NOT included in FCL scope):**

These are incidents where a change to one feature or service impacts another feature or service unexpectedly:

- _Example_: A Topology Service configuration change causes a 404 error in the Repository tree page, but the repository code itself did not contribute to causing the 404. The repository team would not be subject to an FCL since their code was not a contributing factor. Both teams should contribute to the post-incident review to better understand and improve the dependency or coupling that caused the incident.

**Related Causes (Team included in FCL scope):**

These are incidents where external changes occur, but the team's code, configuration, or service compounds or contributes to the effect:

- _Example_: A shared service configuration change occurs, and a sidekiq job from feature category X compounds the effect due to a slow query, contributing to the incident. The team owning feature category X would be subject to an FCL because their code contributed to the incident's impact.

The key distinction is whether the team's code, service, or configuration actively contributed to the incident beyond being a passive recipient of external changes.

##### Scope of work during FCL

After the Incident Review is completed, the team(s) focus is on preventing similar problems from recurring and improving detection. This should include, but is not limited to:

- Address immediate corrective actions to prevent incident reoccurrence in the short term
- Introduce changes to reduce incident detection time (improve collected metrics, service level monitoring, which users are impacted)
- Introduce changes to reduce mitigation time (improve rollout process through feature flags, and clean rollbacks)
- Ensure that the incident is reproducible in environments outside of production (Detect issues in staging, increase end-to-end integration test coverage)
- Improve development test coverage to detect problems (Harden unit testing, make it simpler to detect problems during reviews)
- Create issues with general process improvements or asks for other teams

Examples of this work include, but are not limited to:

- Fixing items from the Incident Review which are identified as causal or contributing to the incident.
- Improving observability
- Improving unit test coverage
- Adding integration tests
- Improving service level monitoring
- Improving symmetry of pre-production environments
- Improving the [GitLab Performance Tool](https://gitlab.com/gitlab-org/quality/performance)
- Adding mock data to tests or environments
- Making process improvements
- Populating their backlog with further reliability work
- Security work
- Improve communication and workflows with other teams or counterparts

Any work for the specific team kicked off during this period must be completed, even if it takes longer than the duration of the FCL. Any work directly related to the incident should be kicked off and completed even if the FCL is over. Work paused due to the FCL should be the priority to resume after the FCL is over. Items created for other teams or on a global level don't affect the end of the FCL.

## Engineering Performance Indicator process

The [Product Analytics team](/handbook/product/groups/product-analysis/engineering/metrics/) is responsible for maintaining Engineering Performance Indicators. Work regarding KPI / RPI is tracked using the [Product Analytics task intake tracker](https://gitlab.com/gitlab-data/product-analytics/-/issues/new?issuable_template=PI%20Chart%20Help).

## Manual verification

We manually verify that our code works as expected.
Automated test coverage is essential,
but manual verification provides a higher level of confidence that features behave as intended and bugs are fixed.

We manually verify issues when they are in the `workflow::verification` state.
Generally, after you have manually verified something, you can close the associated issue.
See the [Product Development Flow](/handbook/product-development/how-we-work/product-development-flow/) to learn more about this issue state.

We manually verify in the staging environment whenever possible.
In certain cases we may need to manually verify in the production environment.

If you need to test features that are built for GitLab Ultimate then you can get added to the [issue-reproduce](https://gitlab.com/issue-reproduce)
group on production and staging environments by asking in the [#development](https://gitlab.slack.com/archives/C02PF508L) Slack channel.
These groups are on an Ultimate plan.

## Critical Customer Escalations

We follow the below process when existing [critical customer escalations](/handbook/customer-success/csm/escalations)
requires immediate scheduling of bug fixes or development effort.

### Requirements for critical escalation

- Customer is in [critical escalation](/handbook/customer-success/csm/escalations/#escalation-for-non-professional-services-projects) state
- The issues escalated have critical business impact to the customer, determined by Customer Success and Support Engineering leadership
  - Failure to expedite scheduling may have cascading business impact to GitLab
- Approval from a VP from Customer Success AND a Director of Support Engineering are required to expedite scheduling
  - Customer Success: approval from VP, Customer Success Management - [Sherrod Patching](https://gitlab.com/spatching)
  - Support Engineering: approval from VP, Support - [Johnny Scarborough](https://gitlab.com/jscarborough)

### Process

- The issue priority is set to `~"priority::1"` regardless of severity
- The label `~"critical-customer-escalation"` is applied to the issue
- The issue is scheduled within 1 business day
  - For issues of type features, approval from the Product DRI is needed.
- The DRI or their delegate provides daily process updates in the escalated customer slack channel

### DRI

- If issue is type bug DRI is the Director of Development
- If issue is type feature DRI is the Director of Product
- If issue requires Infrastructure work the DRI is the Engineering Manager in Infrastructure

The DRI can use the [customer critical merge requests](https://docs.gitlab.com/ee/development/code_review.html#customer-critical-merge-requests) process to expedite code review & merge.

## Pairing Engineers on priority::1/severity::1 Issues

In most cases, a single engineer and maintainer review are adequate to handle a priority::1/severity::1 issue. However, some issues are highly difficult or complicated. Engineers should treat these issues with a high sense of urgency. For a complicated priority::1/severity::1 issue, multiple engineers should be assigned based on the level of complexity. The issue description should include the team member and their responsibilities.

| Team Member | Responsibility |
| ----------- | -------------- |
| `Team Member 1` | `Reproduce the Problem` |
| `Team Member 2` | `Audit Code Base for other places where this may occur` |

If we have cases where three or five or X people are needed, Engineering Managers should feel the freedom to execute on a plan quickly.

Following this procedure will:

- Decrease the time it takes to resolve priority::1/severity::1 issues
- Allow for a smooth handover of the issue in case of OOO or End of the Work Day
- Provide support for Engineers if they are stuck on a problem
- Provide another set of eyes on topics with high urgency or securing security-related fixes

## Internal Engineering handbook

There are some engineering handbook topics that are [internal only](/handbook/communication/confidentiality-levels/#internal). These topics can be viewed by GitLab team members in the [engineering section of the internal handbook](https://internal.gitlab.com/handbook/engineering/).
