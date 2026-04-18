---
title: Coverage and Scheduling
---

## Where Do I Find My Schedule?

Your on-call schedule lives in **Incident.io**. This is our central source of truth.

### The Schedule

**To publish, manage, or view the schedule which includes AMER, EMEA, and APAC DevOps Rotations within it:**

- [See all team members on a rotation here](https://app.incident.io/gitlab/catalog/Schedule/01K611MG8T5CW874Q5JZER3H0Z)
- [Access the on-call schedule here](https://app.incident.io/gitlab/on-call/schedules/01K611MG8T5CW874Q5JZER3H0Z)
- [See all management included in escalations here](https://app.incident.io/gitlab/catalog/Schedule/01K611ZT9YX2PSA8WAMEP6A66G)
- [Access the escalation schedule here](https://app.incident.io/gitlab/on-call/schedules/01K611ZT9YX2PSA8WAMEP6A66G)

You can also:

- Filter to see only your shifts (look for the filter option)
- Set [calendar notifications](https://help.incident.io/articles/1494747929-how-do-i-sync-on-call-schedules-to-google-calendar)
- See who's on-call this week, next week, and beyond

## Changing the Schedule

A typical schedule shows:

- **Your name or initials** when you're on-call
- **The dates and times** of your shift
- **Your rotation "slot"** (which week you're covering)
- **Who's on-call before and after you** (helpful for handoffs)

Occasionally, circumstances require a change:

- Public holidays or planned / unplanned time off
- Someone calls out sick and needs coverage
- Business priorities shift
- There's an error in the schedule

**If this happens:**

1. You'll get notified (usually by Slack or your rotation leader)
2. You might swap with someone else [by using an override](#using-overrides)
3. Or an [override](https://help.incident.io/articles/2815264840-cover-me%2C-overrides-and-schedules) might be created to reassign that period
4. You should receive reasonable notice unless it's an emergency; Ideally, you will be able to provide 2-4 weeks notice when taking time off to reschedule shifts.

You can typically see your on-call schedule:

- **3-6 months in advance** in Incident.io
- This helps you plan vacation, major projects, etc.
- If you have conflicts, let your rotation leader know early

### Using Overrides

An **override** is a way to reassign a shift when someone can't make it. If you need assistance finding someone to take over your shift, ask for swaps in the swaps slack channels ( [#tier-2-devops-rotation-swaps](https://gitlab.enterprise.slack.com/archives/C09LLF79AK0) )

**To remove yourself from the shift and add someone else:**

1. [Access our schedule](https://app.incident.io/gitlab/on-call/schedules/01K611MG8T5CW874Q5JZER3H0Z) (or the schedule you are interested in) from incident.io
1. Click the Create override button in the top right corner. It will auto-populate assuming you are adding yourself as an override starting "right now." You can Adjust the details in the bottom section or type a descriptive example at the top for the specific override and it will update automatically. Once you’re happy with the override, click the Save override button.
1. You can also create an override for a specific existing shift by finding the shift in the calendar view and clicking on it. If you click someone else’s shift, the override view will pop up adding you as the override. If you click your own shift, it will pop up with a Request cover view. Click the Create an override link at the bottom of the view.
1. Remove yourself and add the person who is going to be covering for you. Then click the Save override button. NOTE: Requesting cover **does not** create an override, it just requests coverage! It is still up to you to ensure coverage has been secured and an override has been added.

## What Happens If You're Assigned During PTO?

This is a mistake. Here's what to do:

1. **Don't just miss it.** Tell someone immediately.
2. Contact your [rotation leader](https://app.incident.io/gitlab/on-call/schedules/01K611ZT9YX2PSA8WAMEP6A66G?startTime=2025-10-13T00%3A00%3A00.000%2B00%3A00&timePeriodOption=two_weeks&calendarToggle=timeline) or manager
3. They'll create an override ASAP

## Multiple Rotations

There are multiple types of rotations:

- **Tier 2 On Call** — Tier 2 SME on Call specific to DevOps (Rails)
- **IMOC** — Incident Manager On Call

These should NOT be **simultaneous**. If you are in the IMOC rotation you should not be in the Tier 2 rotation and the reverse is also true.

### Are You in Multiple Rotations?

If you are currently in the IMOC rotation and you have been selected for Tier 2 do the following:

- Confirm with the [tier2-sme-rollout](https://gitlab.enterprise.slack.com/archives/C089VUTQLV6) slack channel that your offboarding will not negatively impact the IMOC rotation, if it does not then proceed with Offboarding from the IMOC rotation
- Onboard into the Tier 2 rotation

### Related Pages

- [Joining and Leaving the Rotation](/handbook/engineering/devops/oncall/joining-and-leaving-rotation) — Understand rotation frequency by region
- [DevOps Rotation Leader](/handbook/engineering/devops/oncall/rotation-leader) — Contact your rotation leader about schedule changes
