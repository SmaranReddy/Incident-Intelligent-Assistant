---
title: Handoffs and Continuity
---

The end of your on-call shift is as important as the beginning. A good handoff means the next engineer is set up to succeed.

## What is a Handoff?

A **handoff** is a formal transfer of on-call responsibility from you to the next person.

For the SSCS 24x5 rotation, handoffs happen:

- **Every 8 hours** as coverage moves between regions (APAC → EMEA → AMER → APAC)
- **At any time** if an incident is ongoing when your shift ends

## Why Handoffs Matter

Poor handoffs cause:

- Lost context and confusion
- Issues being forgotten
- Repeated work
- Missed critical details
- Stressed out engineers

Good handoffs:

- Give the next person what they need to succeed
- Prevent duplication of effort
- Ensure nothing falls through the cracks
- Build trust in the rotation
- Maintain continuity across timezones

## What to Include in a Handoff

### 1. Current Incidents (Active Issues)

If you have any ongoing issues:

**Details needed:**

- **What is the issue?** — Clear description
- **What domain?** — Authentication, Authorization, or Pipeline Security
- **What's the impact?** — How many users/services affected?
- **What have you tried?** — What investigation steps?
- **Where are you stuck?** — What's the blocker?
- **What should they do next?** — Your recommendation

**Example:**

"High authentication failure rate on SAML login. I restarted the authentication service twice; failures dropped to normal, but came back up within 30 minutes. Latest deployment was 4 hours ago. I think there's a session state issue in the new SAML handler. Look at the deployment details and consider rolling back. If it gets worse, escalate to the Authentication EM."

### 2. Recent Issues You Resolved

What did you fix during your shift?

- **What was the issue?**
- **What domain was affected?**
- **What was the root cause?**
- **How did you fix it?**
- **Is there follow-up work needed?** (ticket, monitoring change, etc.)

**Example:**

"Resolved: Authorization policy evaluation errors around 3 PM. The policy cache wasn't being refreshed due to a Redis connection timeout. I manually flushed the cache and fixed the Redis connection pool settings. Created issue #12345 to add monitoring for cache refresh failures."

### 3. Alerts That Fired (Noise)

Which alerts went off but weren't real issues?

- **What alert fired?**
- **Why wasn't it a real issue?**
- **What should be done about it?**

**Example:**

"PipelineSecurityScanTimeout fired 6 times but was always a false alarm. Scans completed successfully, just took longer than the alert threshold. We should either increase the threshold or adjust for expected scan duration. Not urgent, but worth noting."

### 4. Pending Changes or Deployments

Did anything get deployed or changed during your shift?

- **What changed?**
- **When?**
- **Why?** (hotfix, planned change, etc.)
- **What should they watch for?**

**Example:**

"Deployed authentication service version 2.1.3 at 8 PM to fix OAuth token refresh issues. Seems stable so far but monitor AuthTokenRefreshRate and SessionCreationErrors metrics closely for the next hour."

### 5. Cross-Domain Context

Things the next person should know about interactions between domains:

- **Are multiple domains affected?**
- **Are there dependencies between services?**
- **Did you coordinate with other domain teams?**

**Example:**

"The authentication issue earlier also caused authorization timeouts. I coordinated with @AuthorizationEngineer. Both services are stable now, but watch for cascading effects if authentication has issues again."

### 6. Context You Have

Things they might not know:

- **Is the customer aware of any ongoing issues?**
- **Is leadership tracking something?**
- **Are there open incidents you escalated?**
- **Was there a meeting or decision they should know about?**

**Example:**

"The big pipeline security scanner update is tentatively going out tomorrow at 10 AM. Leadership is watching for issues, so if anything unusual happens overnight, don't wait to escalate."

## How to Do a Handoff

### Before Your Shift Ends

**30 minutes before:**

- Start writing your handoff notes
- Summarize any active issues
- Gather details and links

### Synchronous Handoff (Preferred)

If possible, talk to the next person:

1. **Reach out:** "Hey, I'm ending my shift now. Do you have 10 minutes for a quick handoff?"
2. **Walk through:** Go through each active issue
3. **Answer questions:** Let them ask clarifying questions
4. **Verify understanding:** Make sure they get it
5. **Leave your contact:** "You can Slack me if issues come up"

This takes 15-20 minutes but is worth it, especially for cross-timezone handoffs.

### Asynchronous Handoff (When Needed)

If you can't connect directly (common for cross-timezone handoffs):

1. **Write detailed notes** in Slack or a shared doc
2. **Post them in the on-call channel** so the next person sees them
3. **Send a direct message** to the next person: "@NextEngineer, check the on-call channel for my handoff notes"
4. **Be specific** so they don't have to guess or search for context

### Example Handoff Message

```markdown
@Sarah, I'm handing off from AMER to APAC. Here's what you need to know:

**Active Issue:**

- HighAuthFailureRate on SAML login. Restarted service twice; failures are back up.
  Likely session state issue in new SAML handler from deployment 4 hours ago.
  Recommend checking the deploy; consider rolling back if it keeps happening.
  Domain: Authentication

**Resolved During My Shift:**

- Fixed authorization policy cache refresh (Redis connection timeout)
- Manually flushed cache at 3 PM
- Created issue #12345 for monitoring
  Domain: Authorization

**False Alarms:**

- PipelineSecurityScanTimeout fired 6 times. Not real, just longer scan times.
  Domain: Pipeline Security

**Deployments:**

- Authentication service v2.1.3 went out at 8 PM for OAuth fix. Stable so far but monitor metrics closely.

**FYI:**

- Big pipeline security scanner update tentatively going out tomorrow at 10 AM
- Leadership is tracking issues, so escalate early if anything seems wrong

Let me know if you have questions. I'm around for a bit if you need clarification.
```

## What If You're Still on an Incident?

If an issue isn't resolved when your shift ends:

1. **Tell the next person immediately**
2. **Don't just disappear** and leave it for them
3. **Stay for a quick handoff conversation** to explain context
4. **Offer to help** if they get stuck: "Let me know if it gets worse, I can jump in"

The next person shouldn't inherit a mystery.

## After the Handoff

- The next person acknowledges they have the information
- You're officially off-call (sort of — stay near your laptop for 30 min)
- If they have questions, they can reach out
- You're available if they need help (but not required to jump back in)

## What Happens If You Don't Get a Good Handoff?

If you inherit a shift with no context:

1. **Ask Slack** — "Can someone give me context on this issue?"
2. **Check recent deployments** — What changed?
3. **Look at Incident.io** — Is there a ticket or log?
4. **Reach out to whoever was on-call** — Even if they're off, they can help quickly

This is frustrating, so don't do this to others.

## Common Handoff Mistakes to Avoid

### ❌ Disappearing Immediately

Don't just go offline at the end of your shift. Give context first.

### ❌ Being Too Vague

"Everything was fine" is not helpful. Be specific about what you checked and what happened.

### ❌ Assuming They Know the Context

The next person might be in a different timezone and domain. Explain everything.

### ❌ Leaving Active Issues Unaddressed

If something is still breaking, don't ignore it in the handoff. Explain what you've tried and what you recommend.

### ❌ Not Specifying the Domain

Always clarify whether the issue is Authentication, Authorization, or Pipeline Security.

### ❌ Being Defensive

If someone asks questions, answer them clearly. "I didn't debug that far" is fine. "It's probably not the issue" is not helpful.

### Related Pages

- [Communication and Culture](/handbook/engineering/development/sec/software-supply-chain-security/oncall/communication-and-culture) — Learn clear communication during handoffs
- [Your First Shift](/handbook/engineering/development/sec/software-supply-chain-security/oncall/your-first-shift) — Receive good handoffs when you take over
- [Coverage and Scheduling](/handbook/engineering/development/sec/software-supply-chain-security/oncall/coverage-and-scheduling) — Understand when handoffs occur
