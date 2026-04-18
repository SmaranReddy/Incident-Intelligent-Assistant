---
title: Tools and Access
---

Before you go on-call, you need the right tools and permissions. This page covers what you need and how to get it set up.

## Core Tools You'll Need

### 1. Incident.io (On-Call Platform)

**What it is:** Our central system for managing on-call schedules, receiving pages, and coordinating incident response.

**What to do:**

- You should already have been invited
- Install on your phone (iOS or Android)
- Test that pages come through

### 2. Monitoring Platform Access

Your team uses one or more of these:

- **Grafana** — [Dashboards and metrics](/handbook/engineering/monitoring/#monitoring-infrastructure)
- **Logs** — [Logs](/handbook/engineering/monitoring/#logs)
- **Sentry** — [Error Tracking](/handbook/engineering/monitoring/#sentry)

**What to do:**

- Ask your manager which ones your team uses
- Get added to each platform
- Bookmark the critical dashboards for Authentication, Authorization, and Pipeline Security
- Test accessing them during work hours

### 3. Access to Logs and Observability

**What it is:** The ability to search logs, traces, and metrics for your services.

**What to do:**

- Ask your team which log system you use
- Ensure you have access to production logs for SSCS services
- Familiarize yourself with common log queries for your domain

### 4. Runbooks and Documentation

- [Runbooks](https://gitlab.com/gitlab-com/runbooks/-/tree/master/docs)
- SSCS-specific runbooks (to be developed as the rotation matures)

**What to do:**

- Bookmark the main Runbook pages
- Familiarize yourself with 3-5 core runbooks for your domain
- Contribute to runbook creation based on incidents you handle

## Communication Tools

### Slack

You should already have this, but make sure you are in the following Slack channels:

- [tier2-sme-rollout](https://gitlab.enterprise.slack.com/archives/C089VUTQLV6)
- [#sscs-tier2-rotation-coordination](https://gitlab.enterprise.slack.com/archives/C09R509V25V)
- Your team-specific channels:
  - Authentication team channel: #g_sscs_authentication/#authentication_lounge
  - Authorization team channel: #g_sscs_authorization
  - Pipeline Security team channel:#g_sscs_pipeline-security

Additionally make sure the following is true:

- You're in incident channels (will be added dynamically)
- You have notifications on so you see mentions

### Phone Number

Your phone number needs to be:

- In Incident.io (for pages)
- Current and working
- Reachable during your shift

### Email

Make sure your work email works and you can receive incident notifications.

## SSCS-Specific Access

Depending on your domain, you may need:

### Authentication Team

- Access to authentication service logs and metrics
- SAML/OAuth debugging tools
- Session management dashboards

### Authorization Team

- Access to authorization service logs and metrics
- Permission debugging tools
- Role and policy management dashboards

### Pipeline Security Team

- Access to CI/CD pipeline logs and metrics
- Security scanning service dashboards
- Artifact and dependency scanning tools

## Creating a Pre-Shift Checklist

Before going on-call, test:

- [ ] Incident.io app on phone sends notifications
- [ ] I can access production logs for my domain
- [ ] I can read my team's documentation and runbooks
- [ ] My Slack notifications are working
- [ ] My phone number is current
- [ ] I know how to access monitoring dashboards

### Related Pages

- [Your First Shift](/handbook/engineering/development/sec/software-supply-chain-security/oncall/your-first-shift) — Set up tools before your first shift
- [Handoffs and Continuity](/handbook/engineering/development/sec/software-supply-chain-security/oncall/handoff-and-continuity) — Use these tools during handoffs
- [Communication and Culture](/handbook/engineering/development/sec/software-supply-chain-security/oncall/communication-and-culture) — Communicate through tools like Slack
