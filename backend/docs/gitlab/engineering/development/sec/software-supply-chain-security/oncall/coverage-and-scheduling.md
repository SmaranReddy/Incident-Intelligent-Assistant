---
title: Coverage and Scheduling
---

## Where Do I Find My Schedule?

Your on-call schedule lives in **Incident.io**. This is our central source of truth.

### How to Access It

1. Log into Incident.io
2. Go to **Rotations** or **Schedule**
3. Find the SSCS team rotation

You can also:

- Filter to see only your shifts (look for the filter option)
- Set [calendar notifications](https://help.incident.io/articles/1494747929-how-do-i-sync-on-call-schedules-to-google-calendar)
- See who's on-call this week, next week, and beyond

## Reading the Schedule

A typical schedule shows:

- **Your name or initials** when you're on-call
- **The dates and times** of your shift
- **Your rotation "slot"** (which week you're covering)
- **Who's on-call before and after you** (helpful for handoffs)
- **Your timezone coverage** (APAC, EMEA, or AMER)

## Understanding the SSCS 24x5 Coverage Model

### 24x5 (24 hours, 5 days)

The SSCS rotation provides coverage Monday-Friday, 24 hours each day. Coverage is split into three 8-hour regional blocks:

- **APAC** (Best effort basis): UTC 23:00 - 07:00
- **EMEA**: UTC 07:00 - 15:00
- **AMER**: UTC 15:00 - 23:00

You're expected to respond during your assigned 8-hour block, which should align closely with your normal work hours.
Based on your residence (APAC, EMEA or AMER), you'll be assigned to the region that matches your timezone.
This ensures someone knowledgeable is always available during their normal working hours.

## Domain Coverage

The SSCS rotation covers three domains:

- **Authentication** (group::authentication)
- **Authorization** (group::authorization)
- **Pipeline Security** (group::pipeline security)

While you may specialize in one domain, you're expected to provide initial triage for all SSCS issues and escalate to domain specialists as needed.

## Multiple Rotations

There are multiple types of rotations:

- **Tier 2 On Call** — Tier 2 SME on Call specific to SSCS
- **IMOC** — [Incident Manager On Call](/handbook/engineering/infrastructure-platforms/incident-management/incident-manager-onboarding/)
- **Dev OnCall** — [Dev On Call](/handbook/engineering/workflow/development-processes/infra-dev-escalation/process/)

These should NOT be **simultaneous**. If you are in the IMOC rotation you should not be in the Tier 2 rotation and the reverse is also true.

### Are You in Multiple Rotations?

If you are currently in the IMOC rotation and you have been selected for Tier 2 do the following:

- Confirm with the [tier2-sme-rollout](https://gitlab.enterprise.slack.com/archives/C089VUTQLV6) slack channel that your offboarding will not negatively impact the IMOC rotation, if it does not then proceed with Offboarding from the IMOC rotation
- Onboard into the Tier 2 rotation

## How Shifts Are Distributed Fairly

Your rotation leader considers:

- **Equal frequency** — Everyone gets roughly the same number of shifts
- **Predictability** — Publishing schedules far in advance so you can plan
- **Historical load** — Not putting the same person on-call too many times in a row
- **Regional balance** — Ensuring fair distribution within each timezone

## What If My Shift Gets Reassigned?

Occasionally, circumstances require a change:

- Someone calls out sick and needs coverage
- Business priorities shift
- There's an error in the schedule

**If this happens:**

1. You'll get notified (usually by Slack or your rotation leader)
2. You might swap with someone else
3. Or an [override](https://help.incident.io/articles/2815264840-cover-me%2C-overrides-and-schedules) might be created to reassign that period
4. You should receive reasonable notice unless it's an emergency

## Viewing Future Coverage

You can typically see your on-call schedule:

- **3-6 months in advance** in Incident.io
- This helps you plan vacation, major projects, etc.
- If you have conflicts, let your rotation leader know early

### Related Pages

- [Joining and Leaving the Rotation](/handbook/engineering/development/sec/software-supply-chain-security/oncall/joining-and-leaving-rotation) — Understand rotation frequency by region
- [Time Off and Holidays](/handbook/engineering/development/sec/software-supply-chain-security/oncall/time-off-and-holidays) — Manage your scheduled time off
- [Rotation Leader](/handbook/engineering/development/sec/software-supply-chain-security/oncall/rotation-leader) — Contact your rotation leader about schedule changes
