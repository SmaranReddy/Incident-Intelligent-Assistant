---
title: On-Call Processes and Policies
aliases:
  - /handbook/engineering/on-call/
---

{{% alert color="warning" %}}
If you're a GitLab team member working on an incident, please view the [Incident Management Handbook Pages](../_index.md)
{{% /alert %}}

## On-Call at GitLab

When production incidents occur, we must resolve them quickly and safely. To ensure that there are team-members available to respond to incidents, we maintain a set of on-call rotations.

This section of the handbook looks at how we do this and what the responsibilities are for team members involved.

The specifics of how incidents are managed are kept in the [Incident Management Handbook Pages](../_index.md).

## What is a rotation?

A rotation is a set of team members with specific knowledge and skills who are contactable to help resolve incidents.

A rotation has a leader who coordinates the rotation. They ensure that they have enough team members to provide the right level of coverage, they ensure that team members understand the requirements of the rotation, and they perform regular reviews of the rotation.

A rotation also has team members who respond to requests for help through a paging system.

A rotation has a tier and a coverage target.

## Tier and Coverage System

### Tiers

There are three tiers of rotations:

1. [Tier 1](./tier-1.md) - where a team member is paged by a system (through automated alerting)
2. [Tier 2](./tier-2.md) - where a team member is paged by a human (a human involved in an incident makes a specific decision to request more help)
3. Escalations - where leadership intervention is required or when a page is missed

### Coverage

There are several different levels of cover:

1. 24x7 - At any time there is always a named team member available to respond to a page. Tier 1 rotations are always 24x7. Tier 2 rotations can be 24x7.
2. 24x5 (full coverage) - At any time during the working week (M/T/W/T/F) there is always a named team member available to respond to a page.
3. 24x5 (partial coverage) - At most times during the working week (M/T/W/T/F) there is a named team member available to respond to a page. There are documented gaps in coverage.
4. [Best Effort Rotation](./best-effort-rotations.md) - Pages are sent to a group of team members instead of a named team member. There is no single team member with the responsibility of responding to the page at any given time.

It is important to note that team members participating in best effort rotations are not considered to be "on-call".

When someone is on-call, we are giving them a specific duty to perform. They will hold the pager and they are expected to respond in a timely fashion when it pages them. You are expected to set aside your current item and respond to the pager.

When on a best-effort rotation, there is no single team member with role responsibility for responding at any given time. If you are working you should respond to a page and this page should take priority over other work items.

The distinction is important because there are legal and HR implications for on-call cover (which differs by level) but there are no legal or HR implications for best effort rotations.

#### Coverage Periods

The 24 hour period of one day is broken up into three periods of 8 hours each. We are standardizing on 8 hour coverage periods that happen 3 times per day for rotations that are not escalation rotations.

Team members are able to choose which of the three slots best matches their ordinary working hours and they are able to move between slots with three months notice.

The recommendation is (UTC):

1. APAC 23:00 - 07:00
2. EMEA 07:00 - 15:00
3. AMER 15:00 - 23:00

Rotation leaders may choose different 8-hour periods to suit their needs.

#### Coverage on weekends and public holidays

If a rotation is 24x7, then coverage on weekends and public holidays is required. Please see the tier pages for more specific information.

##### Country specific requirements

1. New Zealand - Public holiday shifts are voluntary, if a team member chooses to work a shift on a public holiday, this requires manager approval because responding to pages on a public holiday requires compensation. It is the team member's responsibility to arrange cover or approval for public holidays. [Please see the New Zealand guide for full details](/handbook/total-rewards/benefits/general-and-entity-benefits/pty-benefits-australia/#on-call-support-engineering-only).

## Responsibilities for Rotation Leaders

Slack Channel for Rotation Leaders: [#oncall-rotation-leaders](https://gitlab.enterprise.slack.com/archives/C09M62ECHHS)

1. Define and maintain:
   1. Escalation rules
   1. Onboarding materials to help team members join the rotation
   1. Documentation and runbooks.
1. Maintain your rotation at the agreed level of coverage paying careful consideration to timezones.
1. When new team members join or existing team members depart, be sure to update the schedules accordingly. We recommend making schedule changes 60-days out so that rotations in the short term are not disrupted. Changes required inside of the 60-day periods can be done with overrides.
1. Prepare and publish a [holiday coverage plan](./holiday-coverage-planning.md) before the 4th Thursday in November each year.
   1. There are higher levels of PTO between American Thanksgiving and the end of the year than any other time of year, so we request special coverage plans.
1. Maintain an escalation path to handle unacknowledged pages.

## On-Call Responsibilities and Expectations for both Tier 1 and Tier 2

If you are on-call, you are expected to:

1. Regularly check your on-call schedule and know when you are on-call. You can use our incident management system to alert you before your on-call shifts start.
1. Check that your [notification preferences are set](https://app.incident.io/gitlab/user-preferences/on-call-notifications) and that you know how to acknowledge a page with these settings.
1. If you are assigned a shift that you cannot take due to external commitments, it is your responsibility to find cover. Use your team’s slack channel and notify your rotation leader that you are looking for cover. You can look for partial cover (a few hours), day cover, or full cover.
1. If you are sick and will not be able to work, you should not be on-call. When you inform your manager of your sick day, let them know that you are on-call and they will arrange cover.
1. If a personal emergency happens during your shift, in your message to your manager that you will be away you should let them know that you are on-call for this rotation. The manager will take responsibility for finding you cover.
1. Acknowledge a page within 15 minutes and be at your place of work shortly thereafter to assist with incident resolution.
1. If you are interviewing or part of an active interviewer pool, keeping your ModernLoop preferences up to date so you are not scheduled an interview while you are on call. If you happen to have an interview scheduled over your on-call period you should look for partial cover for that time.
1. If you will be on extended leave, (for example, parental leave), you need to contact your rotation leader to adjust the on-call schedule.

**Exceptions to the above**

1. For team members in The Netherlands, if they cannot take an assigned shift, they must notify their rotation leader with at least 2 working days notice and the rotation leader (not the team member) is responsible for finding cover. (As agreed with the Works Council).

### General Expectations for On-Call

- If you are on call, then you are expected to be available and ready to respond to PagerDuty pages or incident.io Escalations as soon as possible, and within any response times set by our [Service Level Agreements](https://about.gitlab.com/support/#priority-support) in the case of Customer Emergencies. If you have plans outside of your workspace during your on-call shift, this may require that you bring a laptop and reliable internet connection with you.
- We take on-call seriously. There are escalation policies in place so that if a first responder does not respond in time, another team member is alerted. Such policies are not expected to be triggered under normal operations, and are intended to cover extreme and unforeseeable circumstances.
- Because GitLab is an asynchronous workflow company, @mentions of On-Call individuals in Slack will be treated like normal messages, and no SLA for response will be associated with them.
- Provide support to the release managers in the release process.
- As noted in the [main handbook](/handbook/people-group/time-off-and-absence/time-off-types/), after being on-call for Tier 1 or Tier 2 rotations, make sure that you take time off if you need to. If you have been involved in many pages and incidents during your shift, and you feel that you need to rest, please do. Resting after a stressful on-call shift is critical for preventing burnout. Be sure to inform your team of the time you plan to take for time off.
  - The expectation is that you take 1-2 days off as "time off in lieu" if you need to recover from your shift (where a shift is 5+ days).
  - The expectation is that you will communicate with your manager about the pressures of that shift so that improvements to alerts, processes, or other aspects about incident resolution can be addressed.
  - Team members in Australia should review the [Australia time in lieu policy](/handbook/total-rewards/benefits/general-and-entity-benefits/pty-benefits-australia/).
- During on-call duties, it is the team member's responsibility to act in compliance with local rules and regulations. If ever in doubt, please reach out to your manager and/or [aligned People Business Partner](/handbook/people-group/).

## Practical aspects of being on-call

1. You don’t need to install anything specific on your phone. The paging system can be set to notify you by email, phonecall, sms or (if you choose to install the app) in-app notification.
1. You can take breaks during your shift. You need to ensure that you are able to acknowledge a page within 15 minutes and be back at your desk shortly thereafter.

## How to set up an on-call rotation

### Identifying participants

Identify which team members will participate in the new rotation.

### Create training and onboarding

Use LevelUp to construct and issue training for team members joining this rotation.

### How many people are needed to set up a rotation?

To provide coverage across all timezones, you should aim for a minimum of 4 people per region. This will enable a 1-in-4 week rotation pattern (where a person is on-call 1 week every 4 weeks).

If you have more than 12 team members per region, you risk having team members forgetting that they are on-call or not knowing what to do when paged. It is encouraged to have a rotation of fewer than 12 team members per region.

If you have fewer team members than this in some timezones, you can use a 24x5 and declare where you have gaps in coverage. If you have a very small team, you can look at a [Best Effort rotation](./best-effort-rotations.md).

### Defining a rotation pattern

The recommendation is to create three groups of team members: AMER, EMEA and APAC. See [coverage periods](#coverage-periods) above.

3 shifts of equal length are recommended to make it easier to manage, but the rotation leader should consider the ordinary working hours of team members and incorporate this as far as possible.

### Defining escalation paths

Define the escalation procedures for when an on-call team member doesn't respond to a page. These escalation paths will be encoded in Incident.io for each rotation. Some options are below:

Options include:

1. Setting the page to round-robin all team members after 15 minutes
2. Escalating to the rotation of the non-responding on-call team member. Each rotation should list "Escalation upon non-response: `@here + msg` on appropriate slack channel `#tier-2-(team-name)-rotation-swaps`" under [Tier 2 handbook page](/handbook/engineering/infrastructure-platforms/incident-management/on-call/tier-2.md#active-tier-2-rotations) — or any other similar action.
3. Creating a custom escalation schedule if it is a special case that IMOC can't handle

Escalations are rare, but it's still important to have a plan to respond to requests for help.

### Updating the list of participants

When you add a team member to an existing rotation, the schedule recalculates automatically and may change who is currently on call. To keep the current shift intact, add the new team member(s) to the top of the list and choose one of these options:

1. Click the deferral button that appears when adding the user
1. Select the "delay changes" option and set the effective date to the end of the current shift
1. Reorder the user list so the current on-call person remains in their position and new members are placed where their shifts should begin

This applies your changes at the next scheduled handover, with new team members taking the first shift in the updated rotation.

If a team member needs to be temporarily or permanently removed from a rotation (e.g., due to leave, team transfer, or changed responsibilities), please reach out to your rotation leader. The rotation leader will arrange coverage for any upcoming shifts and update the schedule accordingly.

### Slack Channels

Recognizing that some teams may already have effective communication channels, setting up Slack channels for each rotation is optional.  It is up to the discretion of the rotation owner whether to use an existing channel, or create one channel for the whole schedule, or create one channel per region.

The Slack channel that will be used should be added to the [Tier 2 handbook page](/handbook/engineering/infrastructure-platforms/incident-management/on-call/tier-2.md#active-tier-2-rotations).

However, if Slack channels are created, ensure that they follow this format:

- `tier-2-(team-name)-rotation-swaps`
- `tier-2-(team-name)-rotatoin-swaps-apac`
- `tier-2-(team-name)-rotatoin-swaps-emea`
- `tier-2-(team-name)-rotatoin-swaps-amer`

## incident.io

We use [incident.io](https://app.incident.io/) to set the on-call schedules, and to route notifications to the appropriate individual(s).

### Swapping On-Call Duty

Team members covering a shift for someone else are responsible for adding the override in incident.io. This can be arranged in the [#eoc-general](https://gitlab.enterprise.slack.com/archives/C07G9CP5XRR) Slack channel or via the Request Coverage feature of incident.io. They can delegate this task back to the requestor, but only after explicitly confirming they will cover the requested shift(s). To set an override, click the "Create Override" button in the upper right of the page, or click the relevant block of time on the schedule view. This action defaults the person in the override to *you* &mdash; incident.io assumes that you're the person volunteering an override. If you're processing this for another team member, you'll need to select their name from the drop-down list. Also see [this article](https://help.incident.io/articles/2815264840-cover-me%2C-overrides-and-schedules#overrides-38) for reference.

### Adding and removing people from the roster

When adding a new team member to the on-call roster, it's inevitable that the rotation schedule will shift. The manager adding a new team member will add the individual towards the end of the current rotation to avoid changing the current schedule, if possible. When adding a new team member to the rotation, the manager will raise the topic to their team(s) to make sure everyone has ample time to review the changes.

## Slack

In order to facilitate informal conversations around the on-call process and quality of life, as well as coordination of shifts and communication of broader announcements, we have the [#eoc-general](https://gitlab.enterprise.slack.com/archives/C07G9CP5XRR) channel.

## Other Engineering On-Call Rotations

The following on-call rotations exist outside of Infrastructure Platforms but are documented here for a single reference point.

### Customer Emergency On-Call Rotation

- We do 7 days of 8-hour shifts in a follow-the-sun style, based on your location.
- After 10 minutes, if the alert has not been acknowledged, the support manager on call will be alerted. After a further 5 minutes, senior support leadership from all 3 regions will be alerted.
- All tickets that are raised as emergencies will receive [the emergency SLA](https://about.gitlab.com/support/#priority-support). The on-call engineer's first action will be to [triage the emergency request](/handbook/support/workflows/customer_emergencies_workflows/#triage-the-emergency-request) and work with the customer to find the best path forward.
- After 30 minutes, if the customer has not responded to our initial contact with them, let them know that the emergency ticket will be closed and that you are opening a normal priority ticket on their behalf. Also let them know that they are welcome to open a new emergency ticket if necessary.
- You can view the [schedule](https://gitlab.pagerduty.com/schedules#PIQ317K) and the [escalation policy](https://gitlab.pagerduty.com/escalation_policies#PKV6GCH) on PagerDuty. You can also opt to [subscribe to your on-call schedule](https://support.pagerduty.com/main/docs/schedules-in-apps#export-only-your-on-call-shifts), which is updated daily.
- After each shift, *if* there was an alert / incident, the on call person will send a hand off email to the next on call explaining what happened and what's ongoing, pointing at the right issues with the progress.
- If you need to reach the current on-call engineer and they're not accessible on Slack (e.g. it's a weekend, or the end of a shift), you can [manually trigger a PagerDuty incident](https://support.pagerduty.com/main/docs/incidents#trigger-an-incident) to get their attention, selecting **Customer Support** as the Impacted Service and assigning it to the relevant Support Engineer.
- See the [GitLab Support On-Call Guide](/handbook/support/on-call) for a more comprehensive guide to handling customer emergencies.

### Security Team On-Call Rotation

#### Security Operations (SecOps)

- SecOps on-call rotation is 7 days of 24-hour shifts.
- After 15 minutes, if the alert has not been acknowledged, the Security Manager on-call is alerted.
- You can view the [Security Operations schedule](https://gitlab.pagerduty.com/schedules#PYZC2CG) on PagerDuty.
- When on-call, prioritize work that will make the on-call better (that includes building projects, systems, adding metrics, removing noisy alerts). Much like the Production team, we strive to have nothing to do when being on-call, and to have meaningful alerts and pages. The only way of achieving this is by investing time in trying to automate ourselves out of a job.
- The main expectation when on-call is triaging the urgency of a page - if the security of GitLab is at risk, do your best to understand the issue and coordinate an adequate response. If you don't know what to do, engage the Security manager on-call to help you out.
- More information is available in the [Security Operations On-Call Guide](/handbook/security/security-operations/secops-oncall/) and the [Security Incident Response Guide](/handbook/security/security-operations/sirt/sec-incident-response/).

#### Security Managers

- Security Manager on-call rotation is 7 days of 12-hour shifts.
- Alerts are sent to the Security Manager on-call if the SecOps on-call page isn't answered within 15 minutes.
- You can view the [Security Manager schedule](https://gitlab.pagerduty.com/schedules#PJL6CVA) on PagerDuty.
- The Security Manager on-call is responsible to engage alternative/backup SecOps Engineers in the event the primary is unavailable.
- In the event of a high-impact security incident to GitLab, the Security Manager on-call will be engaged to assist with cross-team/department coordination.

### Developer Experience Stage On-Call Rotation

- Developer Experience's on-call do not include work outside GitLab's normal business hours. The process is defined on our [pipeline on-call rotation](/handbook/engineering/testing/oncall-rotation/) page.
- The rotation is on a weekly basis across 3 timezones (APAC, EMEA, AMER) and triage activities happen during each team member's working hours.
- This on-call rotation is to ensure accurate and stable test pipeline results that directly affects our continuous release process.
- The list of pipelines which are monitored are defined on our [pipeline](/handbook/engineering/testing/end-to-end-pipeline-monitoring/) page.
- The schedule and roster is defined on our [schedule](https://gitlab.com/gitlab-org/quality/pipeline-triage#dri-weekly-rotation-schedule) page.
