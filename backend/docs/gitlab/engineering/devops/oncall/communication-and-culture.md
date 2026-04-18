---
title: Communication and Culture
---

Being on-call isn't just about technical skills. It's also about working well with others during stressful situations. This page covers how we communicate and the culture we build.

## Blameless Culture

The single most important thing about incident response: **No one gets blamed.**

### What This Means

When an incident happens:

- We ask "what happened?" not "who messed up?"
- We look for system problems, not person problems
- We learn from incidents, not punish people
- Everyone contributes to postmortems without fear

### Why It Matters

If people are afraid of being blamed:

- They won't escalate when they should
- They'll hide mistakes instead of fixing them
- Incidents take longer to resolve
- We don't learn from failures

### Your Role

When an incident happens:

- Focus on fixing it, not assigning blame
- If someone made a mistake, that's okay—it's a learning opportunity
- In postmortems, discuss what the system could do better
- Avoid phrases like "they should have known" or "that was careless"

### Example

**Bad:** "John deployed broken code; he should have tested it"

**Better:** "The deployment process didn't catch this issue. How can we improve testing or CI/CD?"

## Communicating During Incidents

### Being Clear

When you find something during an investigation, communicate it clearly:

- ✅ "I found the memory leak in the transaction handler, it started after the 4 PM deployment"
- ❌ "Things look bad"

### Frequency

Update your team regularly:

- At the start: "I'm investigating this"
- Every 5-10 minutes: "Here's what I've found"
- When making changes: "Rolling back version 2.1.3"
- When resolved: "Issue is fixed, monitoring for stability"

### Escalation Communication

When escalating:

- ✅ "I've investigated for 20 minutes, looked at X, Y, Z. This is beyond my expertise. Escalating to the database team."
- ❌ "I don't know what to do"

Give context. Help the next person understand what you've tried.

### Over-Communication is Okay

It's better to update too often than not often enough. People would rather see regular updates than sit wondering if you're still looking at the issue.

## Slack During Incidents

### Using Slack Effectively

- Post in the incident channel, not DMs
- Use threads to keep discussions organized
- Pin important information
- Use @channel or @here only for urgency (not spam)

### Incident Channel Norms

Most teams have standards like:

- Updates at least every 10 minutes
- Clear status (Investigating / Mitigating / Resolved / Monitoring)
- Action items with owners
- Links to dashboards or logs when helpful

### What NOT to Do

- ❌ Start side conversations in DMs (keep context in the channel)
- ❌ Go silent for 30+ minutes (always update progress)
- ❌ Use vague language ("it looks like..." without evidence)
- ❌ Blame others

## Before You Get Paged: Building Relationships

### Being a Good Teammate

- Answer questions from newer engineers
- Share what you've learned from incidents
- Update runbooks so others can learn
- Acknowledge good work from others

## Escalation Communication

### When to Escalate

- You've investigated for 15+ minutes and are stuck
- It's beyond your domain
- It's too urgent for your pace
- You need help making a decision

### How to Escalate

In Slack and/or Incident.io:

- **Why:** "This is a database issue, I need database expertise"
- **What you've tried:** "I checked dashboards, logs, and recent deployments. Nothing obvious."
- **What's needed:** "Need someone to check DB replication status"

### Getting Escalated

When someone escalates to you:

- Respond quickly
- Thank them for doing the groundwork
- Take the investigation forward
- Keep them in the loop

## Post-Incident Communication

### Postmortems

After significant incidents, your team holds a postmortem:

- **What happened?** — The sequence of events
- **Why did it happen?** — Root causes
- **What did we learn?** — Takeaways
- **What can we improve?** — Action items

### Participation

- Everyone involved participates
- Be honest about mistakes (without blame)
- Contribute ideas for improvement
- Follow up on action items

### Blameless Postmortem Language

**During the postmortem:**

- ✅ "The deployment process didn't catch this issue"
- ✅ "We didn't have monitoring for this condition"
- ✅ "The runbook didn't have a step for this scenario"

**Not:**

- ❌ "Person X made a mistake"
- ❌ "They didn't follow the process"

## Team Norms and Expectations

### Response Times

Acknowledge a page within 15 minutes and be at your place of work shortly thereafter to assist with incident resolution

### Staying Engaged

Don't disappear while investigating. Even if you're stuck:

- "Still investigating, haven't found the root cause yet"
- "Escalating because this is beyond my expertise"
- "Waiting for the next level to respond"

Silence creates anxiety.

### Professional Behavior

During incidents:

- Stay calm
- Be respectful
- Admit when you don't know something
- Ask for help
- Don't let stress turn into rudeness

We're all on the same team.

## When Things Go Wrong

### You Make a Mistake During an Incident

1. Acknowledge it: "I made an error, here's what I'm doing to fix it"
2. Fix it: Focus on resolving the impact
3. Learn from it: How will you prevent this next time?

You won't be blamed. We all make mistakes.

### Someone Else Makes a Mistake

- Don't call them out publicly
- Focus on fixing the issue
- In postmortem, discuss what the system could do better
- Privately, offer to help them learn

### Blaming Happens (But Shouldn't)

If you hear blame-focused language in postmortems or Slack:

- Gently redirect: "Let's focus on system improvements rather than individual mistakes"
- Bring it up with your manager: "I think we're using language that isn't aligned with blameless culture"

## Cultural Observations

### Good Signs

- Incidents are discussed openly
- People escalate without fear
- Mistakes are treated as learning opportunities
- Postmortems focus on systems, not people
- People thank each other during incidents

### Warning Signs

- People are afraid to escalate
- Blame is assigned in postmortems
- Mistakes are hidden
- People are defensive
- Burnout is common

If you see warning signs, mention it to your manager or rotation leader.

## Building Psychological Safety

**Psychological safety** means you feel safe taking risks, admitting mistakes, and asking questions.

Ways we build it:

- Blameless culture in incidents and postmortems
- Experienced engineers mentoring newer ones
- Questions are encouraged
- Escalation is valued, not punished
- Time for learning and improvement

### Related Pages

- [Handoffs and Continuity](/handbook/engineering/devops/oncall/handoff-and-continuity) — Apply blameless culture to handoffs
- [Your First Shift](/handbook/engineering/devops/oncall/your-first-shift) — Use these principles when you get paged
- [Measuring Success](/handbook/engineering/devops/oncall/measuring-success) — See how escalation communication impacts metrics
