---
title: Measuring Success
---

How do we know if the Tier 2 on-call program is working? We measure it through specific metrics that reflect both operational excellence and engineer well-being. This page explains what we track and why it matters.

## Core Success Metrics

1. Reduction in time to resolve: The primary purpose behind expanding to Tier 2 is to provide Subject Matter Expertise to engineers on call in order to solve incidents faster. This is a primary metric in our overall incident response when it comes to Tier 2.
1. Escalation accuracy: 90%+ of escalations go to the correct team on first try because of the usability in our error messages, stack trace, observability categorization, etc
1. [Zero pages to Tier 2](https://app.incident.io/gitlab/insights/dashboards/core/pager_load_native?date_aggregation=months&schedule%5Bone_of%5D=01K611MG8T5CW874Q5JZER3H0Z) because of the resiliency of our system, and/or the effectiveness of our runbooks
1. [No escalations past Tier 2](https://app.incident.io/gitlab/insights/dashboards/core/pager_load_native?date_aggregation=months&schedule%5Bone_of%5D=01K611ZT9YX2PSA8WAMEP6A66G) because we always respond in < 15 minutes
1. [Sustainable on call schedules](https://app.incident.io/gitlab/on-call/schedules/01K611MG8T5CW874Q5JZER3H0Z?timePeriodOption=one_month&calendarToggle=calendar): Engineers are not on call more than 1 week per month

### Related Pages

- [DevOps Rotation Leader](/handbook/engineering/devops/oncall/rotation-leader) — Rotation leaders track these metrics
- [Communication and Culture](/handbook/engineering/devops/oncall/communication-and-culture) — Blameless culture supports these goals
- [Joining and Leaving the Rotation](/handbook/engineering/devops/oncall/joining-and-leaving-rotation) — Understand fairness metrics in your rotation
