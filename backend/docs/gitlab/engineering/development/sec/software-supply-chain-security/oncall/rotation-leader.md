---
title: SSCS Tier 2 On-Call Rotation Leader
---

Welcome to the Rotation Leader role for the Software Supply Chain Security (SSCS) Tier 2 On-Call program. The starting rotation leaders for this rotation are @adil.farrukh, @ajaythomasinc and @ken.mcdonald. As a Rotation Leader, you are responsible for the health, fairness, and effectiveness of your team's on-call rotation. This guide outlines your core responsibilities and how to execute them.

## Your Role and Responsibilities

As a Rotation Leader, you serve as the primary point of contact for the SSCS rotation. Your responsibilities include managing the schedule across three regions and three domains, onboarding new team members, tracking workload distribution, maintaining team health, and ensuring continuous improvement of the program.

## Managing the Schedule

### Building and Maintaining the Rotation

You are responsible for the overall structure and composition of your rotation:

- Ensure adequate coverage across all three regions (APAC, EMEA, AMER)
- Ensure coverage across all three domains (Authentication, Authorization, Pipeline Security)
- Target 6-10 people per region for balanced workload
- Current distribution: APAC: 7, EMEA: 10, AMER: 12
- Publish schedules at least one month in advance so team members can plan

### Setting Coverage Hours

The SSCS rotation operates on a 24x5 model with three 8-hour regional blocks:

- **APAC** (Best effort basis): UTC 23:00 - 07:00
- **EMEA**: UTC 07:00 - 15:00
- **AMER**: UTC 15:00 - 23:00

Coverage is Monday-Friday only (no weekends in the initial iteration).

### Publishing and Managing the Schedule

- Use Incident.io as your single source of truth for all scheduling
- Make schedules visible to your team and update them regularly
- Create overrides when someone needs coverage swapped
- Communicate schedule changes promptly to affected team members
- Track future scheduling 3-6 months in advance when possible

## Staffing and Fairness

### Rotation Frequency by Region

Your rotation targets specific frequency based on team size and regional coverage needs:

**APAC (Asia-Pacific)**

- Coverage: UTC 23:00 - 07:00 (Best effort basis)
- Approximately 6-7 weeks per person per year
- Current team size: 7 engineers
- Higher frequency due to smallest team size

**EMEA (Europe/Middle East/Africa)**

- Coverage: UTC 07:00 - 15:00
- Approximately 5-6 weeks per person per year
- Current team size: 10 engineers
- Medium frequency

**AMER (Americas)**

- Coverage: UTC 15:00 - 23:00
- Approximately 4-5 weeks per person per year
- Current team size: 12 engineers
- Lower frequency due to largest team size

### Ensuring Fair Distribution

Monitor workload distribution to maintain equity:

- Ensure every engineer gets roughly the same number of shifts within their region
- Track how many times each person has been on-call
- Review who handled the most alerts during their shifts
- Identify anyone carrying more than their fair share
- Watch for patterns where someone consistently gets busier weeks
- Cap on-call duty at no more than once every 4 weeks maximum

### Domain Balance

Ensure fair distribution across domains:

- Track incidents by domain (Authentication, Authorization, Pipeline Security)
- Monitor if one domain is generating significantly more incidents
- Adjust staffing or alert tuning if one domain is overloaded
- Ensure cross-domain knowledge sharing

### Quarterly Fairness Reviews

Every quarter, conduct a review of:

- How many times was each person on-call?
- Were shifts fairly distributed within regions?
- Did anyone burn out or report unsustainable load?
- Did you meet your coverage goals for each region?
- Is domain coverage balanced?
- Rebalance if the distribution became unfair

## Onboarding New Team Members

### Adding Someone to the Rotation

When adding a new person to your rotation:

1. Work with the team member and their manager to confirm readiness
2. Identify their region (APAC, EMEA, AMER) and primary domain
3. Add them to Incident.io with their contact information (phone number, email)
4. Ensure they complete the [first shift preparation](/handbook/engineering/development/sec/software-supply-chain-security/oncall/your-first-shift) checklist
5. Provide them with access to all necessary tools and documentation
6. Schedule their first shift and communicate it clearly
7. Be available to support them during their initial shifts

### Required Onboarding Resources

Ensure new team members have access to and understand:

- [On-Call Process & Policies](/handbook/engineering/infrastructure-platforms/incident-management/on-call/)
- [Tier 2 On Call Level Up Channel](https://levelup.edcast.com/pathways/ECL-123259b5-e469-485e-a0fe-4be27ee118b3)
- SSCS-specific runbooks and playbooks (as they're developed)
- Incident.io training and setup
- Your team's escalation criteria
- Critical dashboards and monitoring platforms for their domain
- Communication protocols and Slack channels

## Managing Public Holidays

### Your Responsibilities

It is very difficult for rotation leaders to know every team member's public holidays across all regions. Clarify expectations:

- It is the team member's responsibility to find coverage if they are scheduled on a public holiday
- Team members may voluntarily switch their public holiday to a different day according to company policy
- Exception: For The Netherlands, team members must notify you with at least 2 working days notice, and you (not the team member) are responsible for finding cover, as agreed with the Works Council

### Handling Holiday Conflicts

When a team member identifies a public holiday conflict:

- Help them swap shifts with a willing colleague in the same region, or
- Create an override to reassign that shift to someone else

Communicate holiday schedules clearly to your team in advance.

## Tracking Workload and Health Metrics

### Monitoring Alert Volume and Pages

Track metrics that indicate rotation health:

- How many times does each engineer get paged per 8-hour shift?
- Are pages distributed fairly across the team and regions?
- Are there domains generating excessive pages (alert fatigue)?
- Are there domains generating too few pages (potential coverage gaps)?

**Actions based on volume:**

- If too high: Work on alert tuning, reduce false alarms, improve runbooks
- If too low: Verify real issues aren't being missed; check alert coverage

### Tracking Incident Response Quality

Review metrics to understand incident response effectiveness:

- Time to Declare: How quickly are incidents recognized and formally declared?
- Time to Fix: How long from declaration to resolution?
- Total Incident Duration: Full impact window including detection, declaration, fix, and verification
- Target: Aim for declaration within 5-10 minutes and fix times of 30 minutes or less

### Monitoring Escalation Patterns

- Track which teams are escalating to your specialists and how often
- Track cross-domain escalations (e.g., Authentication → Authorization)
- Ensure escalations are going to the right teams/domains on first try (target 90%+)
- Identify patterns in what gets escalated and why
- Use this data to improve runbooks or training

### Domain-Specific Metrics

Track metrics by domain:

- Incidents per domain (Authentication, Authorization, Pipeline Security)
- Average resolution time per domain
- Cross-domain incident frequency
- Runbook coverage per domain

### Burnout Prevention

Conduct quarterly surveys asking:

- "How sustainable is your on-call workload?"
- "Do you feel supported when on-call?"
- "Have you experienced on-call-related stress or fatigue?"
- "Would you recommend this program to new team members?"
- "Do you feel comfortable handling incidents outside your primary domain?"

Target 80%+ of engineers rating on-call as sustainable. If scores are low, investigate immediately and take corrective action.

## Supporting Team Members

### Handling Conflicts and Schedule Changes

When team members need to step back from on-call:

- Accommodate legitimate reasons (extended leave, major projects, etc.)
- Offer breaks between rotations, shift swaps, or reduced frequency as needed
- Document temporary changes clearly in Incident.io
- Reintegrate smoothly when the person is ready

### Managing Removals

When someone leaves the rotation permanently:

1. Start the offboarding process with them and their manager
2. Determine their final shifts
3. Remove them from the schedule in Incident.io
4. Ensure they hand off active responsibilities to their replacement
5. Assess impact on regional and domain coverage

### Addressing Unsustainable Load

If someone is on-call more frequently than intended or handling excessive alerts:

- Investigate why this is happening (staffing, alert volume, domain-specific issues)
- Work with your team and leadership to resolve it
- Document the issue and action taken
- Communicate changes to the affected person

## Program Improvement and Learning

### Documenting and Improving Runbooks

As escalations come in, identify gaps in documentation:

- When Tier 2 engineers escalate, ask what information would have helped them resolve it faster
- Use post-incident retrospectives to identify runbook gaps
- Prioritize creating or updating runbooks for frequent escalation patterns
- Aim for 15-20 core runbooks covering 80% of common incidents
- Target 5-7 runbooks per domain (Authentication, Authorization, Pipeline Security)
- Create 3-5 cross-domain runbooks for common scenarios

### Conducting Effective Retrospectives

For S1/S2 incidents (or significant S3/S4 incidents):

- Ensure 100% of escalated S1/S2 incidents have a formal retrospective or write-up
- Lead retrospectives in a blameless manner, focusing on system improvements
- Document what was learned and what can be improved
- Track action items and follow up on completion
- Share learnings across domains

### Tracking Runbook Usage

- Monitor whether incidents reference runbooks
- Ask your team which runbooks are most helpful
- Identify runbooks that aren't being used and update or remove them
- Create new runbooks based on escalation patterns
- Track runbook coverage by domain

## Communication and Escalation

### Supporting Your Team During Incidents

When your team is on-call:

- Be available for questions and escalation decisions
- Help them determine when to escalate beyond Tier 2
- Provide context on customer impact and business priorities
- Debrief after significant incidents
- Facilitate cross-domain coordination when needed

### Communicating with Leadership

Keep leadership informed about:

- Rotation health and any sustainability concerns
- Trends in incident volume and resolution times by domain
- Staffing needs (do you need to hire or redistribute?)
- Alert tuning and runbook improvements
- Cultural health of the rotation
- Regional balance and coverage gaps

### Coordinating with Other Teams

Maintain relationships with:

- Infrastructure and platform teams who may receive escalations
- Incident Managers who page your team
- Other rotation leaders to share best practices
- Engineering Managers for the three SSCS domains
- Your team's managers for support with team members

## When to Escalate to Leadership

Bring issues to your manager or leadership when:

- Someone is unsustainably on-call more than once every 4 weeks
- Burnout or turnover is increasing
- Alert volume is unmanageable in a specific domain
- Staffing levels are insufficient in a region
- The rotation structure isn't working
- Cultural issues emerge (blaming, lack of escalation, etc.)
- Cross-domain coordination is breaking down
- One domain is significantly overloaded

## Success Indicators

You'll know your rotation is healthy when:

- Engineers don't burn out from on-call
- Shifts are fairly distributed across regions and team members
- Incident response times are improving (TTDec and TTFix trending down)
- Retrospectives are blameless and focus on system improvements
- Runbooks are being used and updated regularly across all domains
- Escalations are going to the right teams/domains
- New engineers feel supported during their first shifts
- Team members view on-call as manageable and educational
- Cultural health surveys score 80%+ on sustainability
- Cross-domain collaboration is smooth
- All three regions have adequate coverage

## Quick Reference: Key Responsibilities

**Schedule Management:**

- Maintain adequate coverage across 3 regions and 3 domains
- Publish schedules 1+ month in advance
- Track fairness quarterly
- Cap individual rotation frequency at once per 4 weeks maximum

**Onboarding:**

- Add new members to Incident.io
- Provide tool access and documentation
- Support first shifts
- Ensure training completion

**Workload Tracking:**

- Monitor pages per shift by region and domain
- Track incident response metrics
- Watch for burnout indicators
- Conduct quarterly fairness reviews

**Team Support:**

- Help with schedule conflicts and swaps
- Create overrides for absences
- Address unsustainable load
- Support escalations during incidents
- Facilitate cross-domain coordination

**Improvement:**

- Identify runbook gaps by domain
- Lead blameless retrospectives
- Track metrics and trends
- Share learnings across domains

### Related Pages

- [Joining and Leaving the Rotation](/handbook/engineering/development/sec/software-supply-chain-security/oncall/joining-and-leaving-rotation) — Manage team member add/removal
- [Coverage and Scheduling](/handbook/engineering/development/sec/software-supply-chain-security/oncall/coverage-and-scheduling) — Manage and publish schedules
- [Time Off and Holidays](/handbook/engineering/development/sec/software-supply-chain-security/oncall/time-off-and-holidays) — Manage holiday coverage for your team
- [Measuring Success](/handbook/engineering/development/sec/software-supply-chain-security/oncall/measuring-success) — Track rotation health metrics
- [Communication and Culture](/handbook/engineering/development/sec/software-supply-chain-security/oncall/communication-and-culture) — Foster blameless culture in your rotation
