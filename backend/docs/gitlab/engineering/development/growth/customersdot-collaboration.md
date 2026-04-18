---
title: CustomersDot Trials Collaboration Framework
description: "Decision framework for Growth-Provision collaboration on CustomersDot trials work"
---

## Overview

This framework defines how the Growth team collaborates with the Fulfillment/Provision team when working on CustomersDot trials functionality. It was established as part of the initiative to enable Growth's independence for trials work in CustomersDot (see [Epic #20063](https://gitlab.com/groups/gitlab-org/-/epics/20063)).

The goal is to enable Growth to move quickly on trial-specific improvements while maintaining clear boundaries around core provisioning, billing, and subscription infrastructure owned by the Provision team. This framework provides clarity on when Growth can contribute independently versus when to engage the Provision team, along with escalation paths for priority conflicts.

## Decision Framework

### Growth Contributes Independently

Growth team members can contribute directly to CustomersDot for changes that are isolated to trial-specific logic and do not impact core provisioning or billing systems.

**Examples of Growth-owned changes:**

- **Trial eligibility logic**: Modifying rules for who qualifies for a trial (e.g., domain restrictions, user attributes, geographic limitations)
- **Trial duration modifications**: Adjusting trial length, extending trials for specific cohorts, or implementing trial extension experiments
- **Trial UX improvements**: Enhancing the trial signup flow, onboarding experience, or trial-to-paid conversion flows
- **Trial tracking and analytics**: Adding instrumentation, analytics events, or experiment tracking specific to trial user behavior
- **Trial-specific messaging**: Copy changes, email templates, or in-app messaging related to trial experiences
- **Trial feature flags**: Managing feature flags that control trial-specific functionality or experiments

**Process for independent contributions:**

1. Follow standard GitLab [contribution guidelines](/handbook/engineering/workflow/code-review/)
2. Apply appropriate labels: `devops::growth`, `group::activation` (or relevant Growth group), `CustomersDot`
3. Notify the Provision team in [#s_fulfillment](https://gitlab.slack.com/channels/s_fulfillment) for awareness
4. Ensure code review includes a Provision team maintainer when touching CustomersDot codebase

### Engage Provision Team

Growth must engage the Provision team for changes that affect billing, subscriptions, provisioning infrastructure, or shared systems that impact paying customers.

**Examples of Provision-owned changes:**

- **Billing integration**: Any changes to Zuora integration, payment processing, or invoice generation
- **Subscription provisioning**: Modifications to how subscriptions are created, activated, or managed
- **License generation**: Changes to license key generation, validation, or distribution
- **Zuora integration**: Any work involving Zuora API calls, data synchronization, or billing workflows
- **Shared infrastructure**: Database schema changes, API endpoints used by multiple systems, or core CustomersDot services
- **Compliance and security**: Changes affecting PCI compliance, data privacy, or security controls

**Process for engaging Provision:**

1. Create an issue in the [fulfillment-meta](https://gitlab.com/gitlab-org/fulfillment-meta) project using the appropriate intake template
2. Reach out in [#s_fulfillment](https://gitlab.slack.com/channels/s_fulfillment) to discuss the request
3. Schedule a sync meeting if needed to align on approach and timeline
4. Follow the Provision team's [project management process](/handbook/engineering/development/fulfillment/provision/#project-management-process)
5. Collaborate on implementation with Provision team members as DRIs

### Judgment Call / Consultation

When uncertain about whether a change falls into Growth's domain or requires Provision involvement, err on the side of consultation. Consider the following factors:

**Consult with Provision if:**

- The change could impact billing accuracy or revenue recognition
- You're unsure about the "blast radius" of the change (how many users/systems it affects)
- The change involves domain knowledge specific to provisioning, licensing, or billing
- There's potential for the change to affect paying customers or production systems
- The change modifies shared code paths or infrastructure

**How to consult:**

1. Post a brief description in [#s_fulfillment](https://gitlab.slack.com/channels/s_fulfillment) with the issue link
2. Tag relevant Provision team members (PM, EM, or engineers familiar with the area)
3. Request a quick sync if needed to discuss approach
4. Document the decision in the issue for future reference

## Escalation Paths

When priority conflicts arise or Growth needs to move faster than Provision's current capacity allows, use the following escalation paths:

### Level 1: Direct Contribution

**When to use:** Growth has the capability to implement the change but needs Provision's review and approval.

**Process:**

1. Growth implements the change following CustomersDot contribution guidelines
2. Request review from Provision team maintainers
3. Address feedback and iterate as needed
4. Provision team reviews for correctness and potential impact
5. Merge with Provision team approval

**Timeline:** Typically 1-2 milestones depending on complexity

### Level 2: Leadership-Level Reprioritization

**When to use:** The work requires Provision team implementation but is blocked by capacity or competing priorities.

**Process:**

1. Growth PM escalates to Growth Product Director
2. Growth Director engages with Fulfillment Product Director to discuss priorities
3. Joint assessment of business impact and urgency
4. Reprioritization decision made at Director level
5. Provision team adjusts milestone planning accordingly

**Timeline:** Typically requires 1 milestone for planning + implementation time

### Level 3: Resource Allocation Adjustments

**When to use:** Persistent capacity constraints require structural changes to team composition or responsibilities.

**Process:**

1. Directors identify the ongoing capacity gap
2. Evaluate options: temporary team member loans, hiring, or reorganization
3. Escalate to VP level if cross-department resource allocation needed
4. Implement agreed-upon resource adjustments
5. Document new responsibilities and handoffs

**Timeline:** Typically 1-2 quarters for structural changes

## Collaboration Process

### Communication Channels

- **Primary Slack channels:**
  - [#s_growth](https://gitlab.slack.com/channels/s_growth) - Growth team channel
  - [#s_fulfillment](https://gitlab.slack.com/channels/s_fulfillment) - Fulfillment/Provision team channel
  - Use these channels for questions, collaboration requests, and status updates

- **Issue tracking:**
  - Growth issues: [gitlab-org/gitlab](https://gitlab.com/gitlab-org/gitlab) with `devops::growth` label
  - Provision issues: [gitlab-org/fulfillment-meta](https://gitlab.com/gitlab-org/fulfillment-meta)
  - CustomersDot issues: [gitlab-org/customers-gitlab-com](https://gitlab.com/gitlab-org/customers-gitlab-com)

### Regular Touchpoints

- Growth and Provision teams should maintain regular communication through:
  - Async updates in shared Slack channels
  - Mentions on relevant issues and merge requests
  - Quarterly planning alignment sessions
  - Ad-hoc sync meetings as needed for complex initiatives

### Code Review Process

- All CustomersDot changes require review from a Provision team maintainer
- Growth engineers should request review from Provision team members familiar with the affected area
- Provision team commits to timely reviews (within 2 business days for standard changes)
- For urgent changes, coordinate in [#s_fulfillment](https://gitlab.slack.com/channels/s_fulfillment) to expedite review

## Examples and Scenarios

### Scenario 1: Trial Duration Experiment

**Situation:** Growth wants to test extending trial duration from 30 to 60 days for a specific user segment.

**Decision:** Growth contributes independently

**Rationale:** This is trial-specific logic that doesn't affect billing or provisioning. Growth can implement the change, add appropriate tracking, and notify Provision for awareness.

**Process:**

1. Create issue with experiment plan
2. Implement trial duration logic change
3. Add analytics tracking
4. Post in #s_fulfillment for awareness
5. Request Provision maintainer review
6. Deploy and monitor experiment

### Scenario 2: Trial-to-Paid Conversion Flow

**Situation:** Growth wants to streamline the conversion flow when a trial user upgrades to a paid plan.

**Decision:** Engage Provision team

**Rationale:** This touches subscription creation and billing integration, which are Provision-owned. The change could impact revenue recognition and requires Provision domain expertise.

**Process:**

1. Create issue in fulfillment-meta
2. Schedule sync with Provision PM and engineers
3. Collaborate on design and implementation approach
4. Provision team implements or closely reviews implementation
5. Joint testing and validation
6. Coordinated deployment

### Scenario 3: Trial Eligibility Based on Company Size

**Situation:** Growth wants to restrict trials for companies over a certain size to encourage direct sales engagement.

**Decision:** Judgment call - consult with Provision

**Rationale:** While this is trial eligibility logic (Growth domain), it could impact sales workflows and may have implications for how leads are routed. Consultation ensures alignment.

**Process:**

1. Post in #s_fulfillment describing the proposed change
2. Tag Provision PM for input
3. Quick sync to discuss potential impacts
4. Document decision and proceed based on consultation outcome

### Scenario 4: Adding Trial Analytics Events

**Situation:** Growth wants to add new analytics events to track trial user behavior in CustomersDot.

**Decision:** Growth contributes independently

**Rationale:** This is purely additive analytics instrumentation that doesn't change business logic or affect other systems.

**Process:**

1. Implement analytics events following GitLab instrumentation guidelines
2. Request standard code review from Provision maintainer
3. Deploy and validate events are firing correctly

## Resources

### Related Documentation

- [Provision team handbook](/handbook/engineering/development/fulfillment/provision/) - Provision team processes and project management
- [Growth team handbook](/handbook/engineering/development/growth/) - Growth team processes and experimentation guidelines
- [Growth experimentation guidelines](/handbook/engineering/development/growth/experimentation/) - How Growth runs experiments
- [Fulfillment project management process](/handbook/engineering/development/fulfillment/#project-management-process) - Fulfillment intake and planning
- [Growth-Feature Product Owner collaboration process](/handbook/product/groups/growth/#collaboration-process-with-other-product-teams) - General Growth collaboration model

### Technical Resources

- [CustomersDot repository](https://gitlab.com/gitlab-org/customers-gitlab-com) - Main CustomersDot codebase
- [CustomersDot architecture documentation](https://gitlab.com/gitlab-org/customers-gitlab-com/-/tree/main/doc) - Technical documentation
- [Provision Tracking System monitoring](https://gitlab.com/gitlab-org/customers-gitlab-com/-/blob/main/doc/provision_tracking_system/failure_monitoring.md) - PTS monitoring guide

### Key Contacts

- **Growth Product Director:** See [Growth team handbook](/handbook/engineering/development/growth/#leadership)
- **Growth Engineering Director:** See [Growth team handbook](/handbook/engineering/development/growth/#leadership)
- **Provision Engineering Manager:** See [Provision team handbook](/handbook/engineering/development/fulfillment/provision/#team-members)
- **Fulfillment Product Management:** See [Provision stable counterparts](/handbook/engineering/development/fulfillment/provision/#stable-counterparts)

## Feedback and Iteration

This framework is a living document. As we learn from our collaboration, we'll iterate and improve these guidelines. Suggestions for improvements can be proposed through:

- Issues in [gitlab-org/fulfillment-meta](https://gitlab.com/gitlab-org/fulfillment-meta)
- Discussion in [#s_growth](https://gitlab.slack.com/channels/s_growth) or [#s_fulfillment](https://gitlab.slack.com/channels/s_fulfillment)
- Direct feedback to Growth or Provision leadership

Following GitLab's value of [iteration](/handbook/values/#iteration), we encourage teams to try this framework, gather feedback, and propose improvements based on real-world experience.
