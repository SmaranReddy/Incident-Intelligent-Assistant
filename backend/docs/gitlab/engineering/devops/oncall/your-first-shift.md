---
title: Your First Shift
---

Getting ready for your first on-call rotation? Here's what you need to do.

## Before Your Shift Starts

### 1. Get Your Tools Set Up

Make sure you have everything installed and configured:

- **Incident.io** Install on your phone (iOS or Android), test that pages come through, ensure phone number is current, working, and reachable
- **Email** Make sure your work email works and you can receive incident notifications.
- **Slack** You should already have this, but make sure you are in the following Slack channels and have notifications on so you can see mentions:
  - [tier2-sme-rollout](https://gitlab.enterprise.slack.com/archives/C089VUTQLV6)
  - [#tier-2-devops-rotation-swaps](https://gitlab.enterprise.slack.com/archives/C09LLF79AK0)

### 2. Know Where to Look

- **Grafana** — [Dashboards and metrics](/handbook/engineering/monitoring/#monitoring-infrastructure)
- **Logs** — [Logs](/handbook/engineering/monitoring/#logs)
- **Sentry** — [Error Tracking](/handbook/engineering/monitoring/#sentry)
- [Runbooks](https://gitlab.com/gitlab-com/runbooks/-/tree/master/docs)
- [Playbooks](https://internal.gitlab.com/handbook/engineering/tier2-oncall/playbooks/)

### 3. Learn the Basics

Read through these quickly:

- [Understanding the Basics](/handbook/engineering/devops/oncall/understanding-the-basics)
- [On Call Process & Policies](/handbook/engineering/infrastructure-platforms/incident-management/on-call/)
- [Tier 2 On Call Level Up Channel](https://levelup.edcast.com/pathways/ECL-123259b5-e469-485e-a0fe-4be27ee118b3)
- [Incident.io - Get started as an On-call responder](https://help.incident.io/articles/3472064049-get-started-as-an-on-call-responder)
- [Walk-throughs and FAQs](https://docs.google.com/document/d/1shCf1FuskKrzrTMb60hq0YP0ftbiqvqvHytJK-GlYnA/edit?tab=t.0)

This doesn't need to be perfect. You're building familiarity, not becoming an expert overnight.

### 4. Know Who to Contact

Who do I contact if I get paged and don't know what to do?
  
- Contact the [Incident Rotation Leader on Call](https://app.incident.io/gitlab/on-call/schedules/01K611ZT9YX2PSA8WAMEP6A66G?startTime=2025-10-13T00%3A00%3A00.000%2B00%3A00&timePeriodOption=two_weeks&calendarToggle=timeline)

## The Night Before Your Shift

- **Check your schedule** in [Incident.io](https://app.incident.io/gitlab/on-call/schedules/01K611MG8T5CW874Q5JZER3H0Z?startTime=2025-10-13T00%3A00%3A00.000%2B00%3A00&timePeriodOption=two_weeks&calendarToggle=timeline) so you know exactly when you start
- **Make sure your phone is charged** and notifications are on
- **Test Incident.io** on your phone — Verify the app works

## Your First Page

When you get paged, don't panic. Here's what to do:

### Immediately

1. **Acknowledge the alert** in Incident.io (usually within 5 minutes)
2. **Read the alert details** — What service? What metric? What's the threshold?
3. **Join the incident channel** on Slack (you'll usually get a link in the page)

### Next Steps

1. **Say hello** — Let people know you're investigating: "I'm on this"
2. **Look at dashboards** — Check your monitoring platform to understand what's happening
3. **Read the runbook** if one exists for this alert
4. **If you're unsure** — Ask questions. "What have we tried?" "Is this customer-facing?"
5. **Keep talking** — Update the Incident Slack Channel every 5-10 minutes with what you're finding

### If You're Stuck

- Ask in Slack for help
- Page domain specific Slack Channels if you are not familiar with the domain  
- Don't sit in silence debugging for 30 minutes — escalate earlier rather than later

## Common Questions New On-Call Engineers Ask

**"What if I break something while investigating?"**

- You won't. Your investigation tools (looking at logs, checking dashboards) don't change anything. And if you need to make a change, you'll have procedures documented (runbooks) to follow safely.

**"What if I can't figure it out?"**

- Escalate. Your job isn't to be a superhero; it's to be available and engaged. Escalating to someone more experienced is exactly what you should do.

**"What if I get multiple pages at once?"**

- Work through them one at a time. Update your team on what you're working on and what order you're prioritizing them in. Contact the DevOps rotation leader to inform them in case they can gather additional support.

**"What if an incident is still going when my shift ends?"**

- Hand it off to the next on-call engineer with clear notes about what you've tried and what you've learned. We'll cover handoffs in detail later.

### Related Pages

- [Understanding the Basics](/handbook/engineering/devops/oncall/understanding-the-basics) — Start here if you're new to on-call
- [Communication and Culture](/handbook/engineering/devops/oncall/communication-and-culture) — Know how to communicate when paged
