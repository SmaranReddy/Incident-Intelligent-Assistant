---
title: Joining and Leaving the Rotation
---

## How Do I Get Added to the Rotation?

### When You Join the SSCS Team

Your manager will discuss on-call expectations with you if there is a need to add people to a rotation. Not all team members are on a rotation. If we need additional team members or product coverage for a rotation, your manager will nominate team members for the rotation.

These are questions we consider when building out the SSCS Tier 2 On Call rotation:

- Do we have adequate coverage for each region (APAC, EMEA, AMER)?
- Do we have coverage for all three domains (Authentication, Authorization, Pipeline Security)?
- Is the rotation frequency sustainable for team members?

**What happens:**

1. You and your manager agree you're ready
2. Your rotation leader adds you to Incident.io (our scheduling system)
3. You'll see your first shift appear in your calendar
4. You go through the [first shift preparation](/handbook/engineering/development/sec/software-supply-chain-security/oncall/your-first-shift) checklist

### When You Switch Teams or Domains

If you're moving to a different team or taking on a new service area, you may be added to a different rotation. This works similarly — your new manager/rotation leader handles the addition once you're ready.

## How Do I Get Removed from the Rotation?

### Planned Removal

If you're leaving a team or no longer supporting a service:

1. Tell your manager and rotation leader
2. They'll start the offboarding process
3. You'll receive your final shifts and then be removed from the schedule
4. Hand off any active responsibilities to your replacement

### During Your Rotation

If you have a legitimate reason to step back temporarily (extended leave, major project, etc.), talk to your rotation leader. You might:

- Take a break between rotations
- Swap with a colleague
- Reduce your on-call frequency

## Tracking On-Call Load

Your rotation leader keeps track of:

- How many times each engineer has been on-call
- How many alerts each person handled
- Who's been carrying more than their fair share

**Why?** To ensure fairness and prevent burnout. If someone is getting paged constantly, that's valuable data for the team to use in tuning alerts and distributing workload.

## How Often Will I Be On-Call?

The frequency depends on your region and team size. Here's the current distribution for the SSCS 24x5 rotation:

### By Region (24x5 Weekday Coverage)

**APAC (Asia-Pacific)**

- Coverage: UTC 23:00 - 07:00 (Best effort basis)
- Approximately 6-7 weeks per person per year
- Team size: 7 engineers (Authentication: 2, Authorization: 3, Pipeline Security: 2)
- Higher frequency due to smallest regional team size

**EMEA (Europe/Middle East/Africa)**

- Coverage: UTC 07:00 - 15:00
- Approximately 5-6 weeks per person per year
- Team size: 10 engineers (Authentication: 6, Authorization: 1, Pipeline Security: 3)
- Medium frequency

**AMER (Americas)**

- Coverage: UTC 15:00 - 23:00
- Approximately 4-5 weeks per person per year
- Team size: 12 engineers (Authentication: 3, Authorization: 5, Pipeline Security: 4)
- Lower frequency due to largest regional team size

The differences exist because we're staffing for 24/5 coverage across three regions, and regions have different team sizes. APAC has the most frequent rotation because it has the smallest team; AMER has the least frequent because it has the largest team.

### Our Goal: Balance and Growth

We recognize the disparity in shift frequency and are actively working on:

- Adding new APAC team members to the rotation as they join
- Exploring cross-stage coverage options to better balance the load
- Monitoring rotation frequency to ensure sustainability

### Maximum Rotation Frequency

We cap on-call duty to ensure sustainability. You should never be on-call more than once every 4 weeks. If you find yourself exceeding this, tell your manager immediately—it's a signal that we need to adjust staffing or reduce alert volume.

Your rotation leader will tell you your specific frequency. This information should be published in Incident.io well in advance (at least a month out).

## What's the Minimum and Maximum Team Size?

Your rotation leader has calculated:

- **Minimum number of people needed** to provide coverage without burning anyone out
- **Maximum number who can be added** before the rotation becomes too infrequent

If someone gets added or removed, these numbers might shift. Your rotation leader is responsible for keeping this balanced.

## Understanding Your Rotation Frequency

**Example:** If you have 8 engineers in AMER and you rotate weekly:

- Each engineer is on-call roughly 1 week every 8 weeks
- Coverage is continuous with someone always available during AMER hours
- Burnout is minimized because no one is on-call too frequently

**Example:** If you have 7 engineers in APAC and you rotate weekly:

- Each engineer is on-call roughly 1 week every 7 weeks
- Coverage is continuous but the load is slightly higher
- We're actively working to add more APAC engineers to reduce frequency

## Staffing Model and Fairness

Our SSCS rotation uses a structured staffing model designed to be fair and sustainable across all regions.

### Target Team Sizes

Each region has a target number of engineers on the Tier 2 rotation:

- **Minimum:** 6 people per region (ensures coverage even with absences)
- **Target:** 8-10 people per region (balanced workload and flexibility)
- **Current:** APAC: 7, EMEA: 10, AMER: 12

If your region is understaffed (below 6), we work on hiring or redistributing. If overstaffed, we may adjust to ensure everyone gets a fair frequency.

### Fairness Principles

Our staffing model is built on these principles:

- **Everyone carries an equal share within their region** — Your workload percentage should match your regional teammates'
- **Regional consideration** — We account for timezone differences and team size variations
- **Predictability** — Your rotation frequency won't suddenly jump without notice
- **Sustainability** — No one is on-call more than once every 4 weeks
- **Domain balance** — We ensure coverage across Authentication, Authorization, and Pipeline Security

### Monitoring Fairness

Every quarter, we review:

- **How many times was each person on-call?** — Is it even within regions?
- **Were shifts fairly distributed?** — Did some people get lucky and miss busy weeks?
- **Did anyone burn out?** — Are people satisfied with their frequency?
- **Did we meet our coverage goals?** — Did we have the right number of people in each region?
- **Is domain coverage balanced?** — Are all three domains adequately covered?

If the distribution becomes unfair, we rebalance.

### If You're Concerned About Fairness

If you notice:

- You're being on-call significantly more or less than teammates in your region
- Your region seems understaffed or overstaffed
- The rotation frequency doesn't match what you were told
- One domain is carrying more load than others

**Speak up.** Talk to your rotation leader or manager. We actively monitor this, but human error happens. Your feedback helps us catch problems early.

## Domain-Specific Considerations

While you may specialize in one domain (Authentication, Authorization, or Pipeline Security), the rotation expects:

- **Initial triage** for all SSCS issues
- **Deep expertise** in your primary domain
- **Basic familiarity** with other SSCS domains
- **Willingness to escalate** to domain specialists when needed

We're building runbooks to help with cross-domain triage.

### Related Pages

- [Coverage and Scheduling](/handbook/engineering/development/sec/software-supply-chain-security/oncall/coverage-and-scheduling) — Understand your rotation frequency
- [Time Off and Holidays](/handbook/engineering/development/sec/software-supply-chain-security/oncall/time-off-and-holidays) — Plan PTO within your rotation
- [Rotation Leader](/handbook/engineering/development/sec/software-supply-chain-security/oncall/rotation-leader) — Contact your rotation leader for add/removal
