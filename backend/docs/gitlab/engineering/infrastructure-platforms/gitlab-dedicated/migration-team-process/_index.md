---
title: Dedicated Migration Team - Process
---

### Dedicated Migration Team Time Off

The Dedicated Migration team follows [GitLab's paid time off policy](/handbook/people-group/time-off-and-absence/).
We balance that with the need to deliver support to our customers every day.
This page is intended to provide an understanding of what we need to do to achieve that balance, making it possible for all to take time off as needed and desired, while we as a team continue to deliver for our customers.

### One-time setup actions

#### Dedicated Migration Team calendar

1. Be sure you have access to the
[**Dedicated Migrations Team Calendar**](https://calendar.google.com/calendar/embed?src=c_fb53e24f590edfb8f313253126123e48d57254dd73266fae6547a2a4890b0c82%40group.calendar.google.com&ctz=America%2FVancouver)
team calendar.
   1. If you don't have it, verify whether you are part of the Dedicated Migrations Team (`dedicated-migrations-team@`) if you are not, ask to be added via an Access Request.

1. Set up the Google Calendar integration with Time Off by Deel, so that you do not need to populate your personal and "Dedicated Migrations Team" calendars manually.
   1. In Slack, click the `+` sign next to 'Apps' at the bottom of the left sidebar
   1. Search for 'Time Off by Deel' and click 'View'
   1. Under 'Home', click on 'Your Events' to show a dropdown
   1. Click on 'Calendar Sync' under the Settings break
   1. Click on 'Connect your Calendar' and complete the actions to sync your calendar to Time Off by Deel
      - You will see a 'Success! Your calendar has been connected.' message and
        your calendar listed under 'Your synced calendar' in Time Off by Deel on Slack
   1. After your personal calendar is linked, click 'Add calendar' under
   'Additional calendars to include?'. The 'Dedicated Migrations Team Calendar' calendar ID is
   `c_fb53e24f590edfb8f313253126123e48d57254dd73266fae6547a2a4890b0c82@group.calendar.google.com`

## Choosing and recording time off

Please take time off whenever you need it.

A little thoughtfulness on everybody's part will go a long way toward making it
possible for everyone to take their desired days off. As you look to plan your
time off, please:

- ensure you know the [company wide paid time off guidelines](/handbook/people-group/time-off-and-absence/time-away-philosophy/)
- check the calendar and coordinate with the team to ensure we have sufficient cover for days with low availability.
- schedule your time off as far in advance as you can
- don't lock yourself into nonrefundable travel itineraries before you've taken steps such as the above to be sure you can get the planned days off
- if possible, plan your time off to avoid disruptions

### Communicate through a coverage issue

Please put together a [coverage issue](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/dedicated-migrations-group/pto-coverage/-/issues/new?description_template=pto_coverage) for any PTO longer than 5 days. Some Divisions may have other guidance (for example [Engineering PTO Coverage](/handbook/engineering/#taking-time-off)). Please follow your Division procedures if there is any conflict, and circulate with the team.

### After PTO

See [returning from pto](/handbook/people-group/time-off-and-absence/time-away-philosophy/#transitioning-back-mindfully).

### Geo Engagement Model For Dedicated Migrations

**Core Support Structure:**
Starting Q4 FY26 (2025-11-01), a Geo DRI is assigned for a 3-week window around each migration: 1 week before, the migration weekend, and 1 week after. The DRI is fully committed to migration issues via Slack during pre/post weeks and available via PagerDuty during business hours (9am-5pm in their timezone) on the migration weekend itself. If no issues arise, the DRI can resume normal Geo duties.

**Migration Weekend Responsibilities:**

The Geo team must have someone prepared and aware of the cutover, coordinate timing with the Project Manager, and be ready to troubleshoot any issues that arise during the customer instance cutover following full data sync.

#### Outside the 3-Week Window

Support is primarily asynchronous via Request for Help (RFH) tickets. Synchronous customer calls are handled adhoc on a case-by-case basis, requiring approval from Engineering Manager Lucie Zhao based on team availability.

#### RFH Process

For sync issues outside the dedicated window, teams submit a [Geo Support Request](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-Geo) indicating priority (P1-P4), sync start time, and cutover deadline. These are triaged by the weekly customer support DRI. Geo engineers need access to Opensearch, Grafana, and ideally break glass access to support effectively.
