---
title: "Production Engineering Networking and Incident Management Team AI prompts "
description: Common prompts for use with Duo Chat
---

## Networking and Incident Management Common AI Prompts

### Announcement authoring

Networking and Incident Management builds platforms and manages tools that other teams throughout GitLab use. For example, we manage the incident management tooling and processes around incident management. We also build frameworks for adopting standardized networking infrastructure throughout engineering.

Whenever we roll out changes to a process or framework, it is important to communicate those changes and ensure they are discoverable by everyone who will need to use and know about them.

This prompt helps write a slack announcement for a given issue or merge request.

#### Prompt

```text
I need help drafting an announcement for this [MR/ISSUE/EPIC] on slack.

Please write a summary with the audience of [SRE/ENGINEERING/PRODUCT/SUPPORT/ETC] in mind.

Be sure to include [SPECIFIC DETAILS TO INCLUDE].
```

### Fix linting errors

In the handbook and other markdown-based projects and files, it's easy to make linting mistakes, especially when working in the Web IDE. Use this prompt to have Duo Agent fix the changes for you.

#### Prompt

```text
Can you fix the linting errors in [link-to-failing-job] for this MR?
```

### Triage incoming requests

Being a Production Engineering team, we often have issues being opened from a variety of departments requesting changes to our Networking architecture. When triaging these issues, it is helpful to know if similar work has been completed in the past.

#### Prompt

```text
I'm triaging this issue for the Networking and Incident Management team. I want to know if there have been similar requests in the past from this or other teams. 
Please search in [PROJECT] and provide me with any relevant issues or specific comments.

These changes may take place in [PROJECT(S)], please find any MRs that may have solved similar problems in the past.
```

### Find a discussion

The history of our infrastructure and production environments is long and combined with the changing of teams and projects over time, it can be difficult to understand why some decisions were made or why certain things are the way they are. Use this prompt to find details about a historic discussion or decision.

#### Prompt

```text
I need to find a specific discussion about [TOPIC/DECISION]. Here's what I know about it:

What was discussed:
[DESCRIBE THE DISCUSSION TOPIC OR DECISION MADE]

Where it might be:
[EPIC/ISSUE/MR NUMBERS/PROJECT OR GENERAL AREA]

Keywords or phrases:
[SPECIFIC TERMS THAT WERE LIKELY USED]

Who might have said it:
[TEAM MEMBERS WHO LIKELY PARTICIPATED]

When (approximately):
[TIME FRAME IF KNOWN]

Please help me find by:
1. Searching through the epic and all linked MRs/issues
2. Looking for specific keywords in comments
3. Checking resolved/collapsed threads
4. Searching in commit messages
5. Looking in MR descriptions and updates
6. Checking system notes for mentions
7. Searching related epics or issues
8. Looking for similar discussions in related work
9. Checking design documents linked
10. Searching team member's recent comments
11. Looking for decision records or ADRs
12. Checking for screenshots or code snippets

Context: This decision is needed for [WHY YOU NEED IT]. The discussion was about [PROVIDE MORE CONTEXT]. It's blocking [WHAT IT'S BLOCKING].
```

### Map Service Dependencies for Incident Response

A prompt template for GitLab SREs to map service dependencies across GitLab's distributed architecture and understand failure cascades for effective incident response.

#### Prompt

```text
I'm a GitLab SRE mapping dependencies for [GITLAB SERVICE] (e.g., Rails API, Gitaly, Registry, Pages, CI Runner infrastructure). Here's the code/configuration:

[PASTE SERVICE CODE, CONFIGURATION, OR HELM CHARTS]

GitLab architecture context:
- Cell/Shard: [Main cell/CI cell/Pages cell]
- Communication method: [gRPC/HTTP API/Redis pub-sub/PostgreSQL]
- Load balancer: [HAProxy/GCP LB/Cloudflare]

Please analyze:
1. What GitLab services does this depend on (Gitaly, Redis, PostgreSQL, Elasticsearch)?
2. Which Redis instance does it use (cache, shared_state, queues, rate_limiting)?
3. How does it handle Gitaly node failures or latency spikes?
4. What happens during PostgreSQL replica lag or connection pool exhaustion?
5. Are there dependencies on external services (GCS, Container Registry, Cloudflare)?
6. How do feature flags affect service dependencies and fallback behavior?
7. What Consul health checks or readiness probes validate dependencies?
8. How does this interact with GitLab's rate limiting and abuse detection?
9. What services would be affected if THIS component fails (reverse dependencies)?
10. What are the manual bypass options (feature flags, admin commands)?

Context: I'm updating [RUNBOOK/CELL ARCHITECTURE DOCS/INCIDENT RESPONSE PLAN] for [SPECIFIC INCIDENT TYPE/DEPENDENCY FAILURE SCENARIO] that could affect [GitLab.com/Dedicated instances].
```

### Prepare MRs for Review

Before submitting MRs for code review, use this prompt to have Duo give it a prereview, helping to reduce the back-and-forth with reviewers.

#### Prompt

```text
I'm about to submit [MR TITLE] for review and want to ensure a smooth review process. Here's what I have:

Changes made:
[SUMMARIZE THE KEY CHANGES]

Files modified:
[NUMBER OF FILES AND GENERAL AREAS TOUCHED]

Complexity areas:
[IDENTIFY COMPLEX OR RISKY CHANGES]

Context needed:
[BACKGROUND INFO REVIEWERS MIGHT NEED]

Please help me prepare by:
1. Writing a clear MR description with problem/solution
2. Creating a review checklist for reviewers
3. Identifying which parts need extra explanation
4. Adding inline comments for complex code
5. Suggesting the right reviewers for different parts
6. Breaking down into smaller MRs if too large
7. Highlighting areas that need specific attention
8. Providing testing instructions
9. Linking related issues/documentation
10. Anticipating likely reviewer questions
11. Adding before/after comparisons if relevant
12. Ensuring CI/CD passes before requesting review

Context: This MR addresses [ISSUE/FEATURE]. Team size is [NUMBER]. Typical review turnaround is [TIMEFRAME]. This is [PRIORITY LEVEL].
```
