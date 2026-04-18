---
title: Measuring Success of the SSCS Tier 2 On-Call Program
---

How do we know if the SSCS Tier 2 on-call program is working? We measure it through specific metrics that reflect both operational excellence and engineer well-being. This page explains what we intend to track/implement and why it matters.

## Core Success Metrics

Our program focuses on three interconnected metrics that together tell the story of incident response quality:

### Time to Declare (TTDec)

**What it measures:** How quickly do we recognize a problem and formally declare it as an incident?

**Why it matters:** The faster we declare an incident, the faster we activate our full response machinery. A long time between when something breaks and when we declare it means we're already losing ground.

**What it shows:** Good TTDec indicates strong observability, clear ownership, and alert systems that work. Poor TTDec suggests blind spots in our monitoring.

**Target:** Declare incidents within 5-10 minutes of detection for critical services

**How Tier 2 helps:** You validate early alert signals, confirm severity, and lead the decision to formally declare incidents promptly.

### Time to Fix (TTFix)

**What it measures:** How long from when we declare an incident to when it's actually resolved?

**Why it matters:** This is what customers care about most. Shorter fix times mean less business impact and better reliability.

**What it shows:** Good TTFix indicates effective troubleshooting, strong runbooks, and skilled on-call engineers. Poor TTFix suggests we need better tools, documentation, or training.

**Target:** 30 minutes or less for most incidents; under 5 minutes for critical issues

**How Tier 2 helps:** You own the technical resolution, execute runbooks, coordinate with other teams, and validate the fix through testing.

### Total Incident Duration

**What it measures:** Total elapsed time from when an incident starts to when it's completely resolved (including verification and monitoring).

**Why it matters:** This captures the full window of customer impact. It includes detection time, declaration time, fix time, and verification.

**What it shows:** Trends in incident duration over time reveal whether we're getting better at preventing and responding to issues.

**Target:** Reduce overall incident duration by 20-30% within the first year of the program

**How Tier 2 helps:** You coordinate the response, update status milestones in real time, and ensure proper verification before declaring resolution.

## Composite Metrics: Reading the Patterns

When you look at all three metrics together, they tell you what's really happening in the system. Here's how to interpret patterns:

### ↓ All Three Metrics Improving

**What it means:** Excellent incident management. We're detecting problems quickly, declaring them promptly, fixing them fast, and minimizing total impact.

**What's working:** Strong observability, clear ownership, effective runbooks, skilled team.

**Action:** Maintain and continue improving. This is the goal state.

### Fast Declaration but Slow Fix

**What it means:** TTDec ↓ but TTFix ↑ — We're detecting problems quickly, but taking too long to resolve them.

**What's wrong:** Our runbooks might be incomplete, team lacks expertise in specific domains, or we need better tools and access.

**Action:** Invest in runbook quality for Authentication, Authorization, and Pipeline Security; provide training; audit escalation paths.

### Slow Declaration but Quick Fix

**What it means:** TTDec ↑ but TTFix ↓ — We're slow to detect problems, but once we do, we fix them fast.

**What's wrong:** Gaps in our monitoring and alerting. We're not seeing problems until they're severe.

**Action:** Audit observability for SSCS services, add missing metrics and dashboards, improve alert thresholds.

### Both Slow (TTDec ↑ and TTFix ↑)

**What it means:** We're detecting problems late and taking too long to fix them.

**What's wrong:** This requires comprehensive improvement—both monitoring and response capabilities need work.

**Action:** Phase 1: Improve observability. Phase 2: Improve troubleshooting. Both are critical.

## On-Call Health Metrics

Beyond incident response metrics, we also measure the health and sustainability of the on-call program itself.

### Rotation Frequency

**What we measure:** How often is each engineer on-call?

**Current state:**

- APAC: ~6-7 weeks per year
- EMEA: ~5-6 weeks per year
- AMER: ~4-5 weeks per year

**Target:** Maximum 1 week per month per engineer (no more than once every 4 weeks)

**Why it matters:** Sustainable on-call. If someone is on-call too frequently, they'll burn out.

**What to do if high:** Add team members to the rotation, improve alerting to reduce pages, or distribute coverage differently.

### Alert Volume (Pages Per Shift)

**What we measure:** How many times does a Tier 2 engineer get paged during their 8-hour shift?

**Target:** 1-2 pages per shift week is expected; Note that currently this is very rare at around <5 pages per team, per year.

**Why it matters:** Too many pages = alert fatigue. Too few = maybe we're not detecting real issues.

**What to do if too high:** Tune alert thresholds, fix noisy monitoring, remove false alarms.

**What to do if too low:** Verify we're not missing real issues; check alert coverage.

### Escalation Accuracy

**What we measure:** When Tier 2 escalates an incident, are they escalating to the right team or domain?

**Why it matters:** Escalating to the wrong person wastes time. Escalating to the right person fixes the issue faster.

**Target:** 90%+ of escalations go to the correct team/domain on first try

**What to do if low:** Improve escalation decision tree, clarify when to escalate, provide more context in runbooks, improve cross-domain knowledge.

## Domain-Specific Metrics

Since SSCS covers three domains, we track metrics by domain:

### Incidents by Domain

**What we measure:** How many incidents affect each domain?

- Authentication incidents
- Authorization incidents
- Pipeline Security incidents
- Cross-domain incidents

**Why it matters:** Helps us understand where to focus improvement efforts and runbook development.

### Domain Escalation Patterns

**What we measure:** How often do incidents get escalated from one domain to another?

**Why it matters:** Frequent cross-domain escalations might indicate:

- Need for better initial triage
- Complex dependencies between services
- Opportunities for cross-domain training

## Burnout Prevention Metrics

### Quarterly Surveys

**What we ask:**

- "How sustainable is your on-call workload?"
- "Do you feel supported when on-call?"
- "Have you experienced on-call-related stress or fatigue?"
- "Would you recommend this program to new team members?"
- "Do you feel comfortable handling incidents outside your primary domain?"

**Target:** 80%+ of engineers rate on-call as sustainable

**What to do if scores are low:** Investigate immediately. On-call burnout is serious and requires action.

## Learning and Improvement Metrics

### Incident Knowledge Capture

**What we measure:** For critical incidents (S1/S2), do we create documented learning?

**Target:** 100% of S1/S2 incidents have a retrospective or formal write-up

**Why it matters:** Incidents are learning opportunities. If we don't capture what we learned, we'll repeat the same mistakes.

### Runbook Usage

**What we measure:**  Are new runbooks created after incidents?

**Target:** Overtime 80%+ of incident reports reference a runbook

**Why it matters:** Runbooks save time and reduce errors. If they're not being used, we're missing an opportunity.

### Runbook Coverage

**What we measure:** How many documented runbooks/playbooks exist for common scenarios?

**Target:** Minimum 15-20 core runbooks covering 80% of common incidents across all three domains

**Current state:** Building initial runbook library

**Why it matters:** Having documented procedures speeds up response and reduces toil.

### Domain-Specific Runbooks

**What we track:**

- Authentication runbooks: Target 5-7 core scenarios
- Authorization runbooks: Target 5-7 core scenarios
- Pipeline Security runbooks: Target 5-7 core scenarios
- Cross-domain runbooks: Target 3-5 scenarios

## Fair Distribution Metrics

### Regional Balance

**What we measure:** Is on-call load distributed fairly within each region?

**Why it matters:** Engineers in the same region should have similar on-call frequency.

**Action:** Monitor patterns, rebalance quarterly if needed.

### Domain Balance

**What we measure:** Are incidents distributed across domains, or is one domain carrying more load?

**Why it matters:** If one domain has significantly more incidents, we may need to:

- Add more engineers to that domain's rotation
- Improve monitoring to reduce false alarms
- Investigate systemic issues in that domain

## Baseline vs. Target Metrics

Your rotation leader has established metrics for your program. Understanding the difference matters:

### Baseline Metrics (Current State)

These are measured **before or at the start** of Tier 2:

- "Currently, SSCS issues escalate to EMs an average of X times per week"
- "Average time to fix is Y minutes"
- "Incidents last an average of Z hours from start to resolution"

### Target Metrics (Goal State)

These are what we're aiming for:

- "With Tier 2, on-call specialists will be paged an average of 2-5 times per shift"
- "Average time to fix will be 20 minutes"
- "Incidents will last an average of 30 minutes"

### Progress Dashboard

Your team should have a dashboard showing:

- Current metrics vs. targets
- Trend over time (are we improving?)
- Metrics by domain (Authentication, Authorization, Pipeline Security)
- Metrics by region (APAC, EMEA, AMER)
- Areas where we're ahead of target vs. behind

## How to Read the Metrics

### Trending the Right Direction

If metrics are improving (TTDec down, TTFix down, total duration down), you're winning. Celebrate this. But also:

- Ask what's working and keep doing it
- Identify what changed and document it
- Share best practices across domains

### Metrics Plateau or Get Worse

If metrics stop improving or get worse:

- Don't panic. Investigate why.
- Did something change (new service, team change, alert tuning)?
- Is it a temporary spike or a trend?
- Is it specific to one domain?
- What action is needed?

### Using Metrics to Improve

Metrics aren't about blame. They're about identifying where to focus effort:

- "Alert volume is 3x what we expected for Pipeline Security; we should tune thresholds"
- "TTFix is high for authorization issues; we need better authorization runbooks"
- "Authentication incidents resolve quickly but authorization takes longer; let's learn why"

## SSCS Program Success Criteria

Beyond individual metrics, the program itself succeeds when:

### Foundation & Standardization (Phase 1)

- All three domains (Authentication, Authorization, Pipeline Security) have clear ownership
- Escalation paths are documented and accessible
- Incident taxonomy is standardized
- Team members are trained on Tier 2 processes
- Structured retrospectives are completed for escalated incidents
- 24x5 coverage is operational across all three regions

### Enhancement & Integration (Phase 2)

- Process audit completed with identified improvements
- 15-20 core runbooks/playbooks documented and in use across all domains
- At least 80% of team members aware of and using runbooks
- Baseline and target metrics defined
- Evidence that incidents reference runbooks and learnings
- At least 30% reduction in time spent on manual toil
- Cross-domain knowledge sharing is effective

## Cultural Success Indicators

Beyond metrics, we measure success by culture:

### Blameless Incident Response

- Retrospectives focus on systems, not people
- Engineers feel safe escalating when needed
- Learning is celebrated, not punishment

### Knowledge Sharing

- Runbooks are continuously updated
- Team members teach each other across domains
- Patterns are recognized and prevented
- Cross-domain collaboration is smooth

### Sustainability

- Engineers don't burn out from on-call
- People feel on-call is manageable and educational
- On-call experience is valued in career growth
- Regional balance is maintained

## Your Role in Measuring Success

As an on-call engineer, you contribute by:

- Providing honest feedback on your experience
- Referencing runbooks and noting when they help or fail
- Participating in retrospectives authentically
- Suggesting improvements based on what you see
- Celebrating wins and learning from failures
- Sharing knowledge across domains
- Contributing to runbook development

### Related Pages

- [Rotation Leader](/handbook/engineering/development/sec/software-supply-chain-security/oncall/rotation-leader) — Rotation leaders track these metrics
- [Communication and Culture](/handbook/engineering/development/sec/software-supply-chain-security/oncall/communication-and-culture) — Blameless culture supports these goals
- [Joining and Leaving the Rotation](/handbook/engineering/development/sec/software-supply-chain-security/oncall/joining-and-leaving-rotation) — Understand fairness metrics in your rotation
