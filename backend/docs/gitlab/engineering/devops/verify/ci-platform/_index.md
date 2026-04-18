---
title: "Verify:CI Platform Group"
description: "The GitLab Verify:CI Platform Group Handbook page"
---

## Vision

To provide the foundational platform that makes GitLab CI scalable, reliable, and performant for millions of developers worldwide.
Our team is responsible for the following categories:

- [CI Scaling](https://about.gitlab.com/direction/verify/#vision)
- [Fleet Visibility](https://about.gitlab.com/direction/verify/#fleet-visibility)

## Mission

- Ensure that GitLab Continuous Integration is scalable, performant and reliable by improving product resiliency and future-proofing the CI platform as we grow.
- Provide fleet visibility and CI/CD analytics to enable proactive management of CI/CD efficiency and infrastructure costs.
- Support database and infrastructure initiatives that our CI Platform requires to remain operational. [Operational aspects outlined in our technical roadmap](/handbook/engineering/devops/verify/#fy25-to-fy26).

## Team Members

The following people are permanent members of the Verify:CI Platform group:

{{< team-by-manager-role role="Engineering Manager(.*)Verify:CI Platform" >}}

## Useful Links

- [Issue tracker: `~group::ci platform`](https://gitlab.com/groups/gitlab-org/-/issues?label_name%5B%5D=group%3A%3Aci+platform&scope=all)
- [Slack channel: `#g_ci-platform`](https://gitlab.enterprise.slack.com/archives/C06URHZUTBP)
- [Issue board: `CI Platform Workflow`](https://gitlab.com/groups/gitlab-org/-/boards/7604546)

## Dashboards

See [internal handbook page](https://internal.gitlab.com/handbook/engineering/core-development/ci/verify/ci-platform)

## How we work

### Milestone Planning Process

CI Platform does not have a dedicated Product counterpart for CI Scaling, but works with a Product Manager for [Fleet Visibility](https://about.gitlab.com/direction/verify/#fleet-visibility). We use [planning work items](https://gitlab.com/gitlab-org/ci-cd/ci-platform-group/-/work_items?sort=created_date&state=all&first_page_size=50) to centralize milestone commitments. Our planning process:

- Starts mid-milestone with goal-setting
- Includes async discussions in the planning issue
- Finalizes all issues including [weight](#issue-weighting-guidelines) assignment before the milestone begins

Engineering Manager responsibilities:

- Reviews carryover work from previous milestone
- Evaluates realistic delivery capacity

Team member responsibilities:

- Selects next set of issues from their projects to meet roadmap timelines
- Surfaces risks and timeline impacts early

Non-roadmap items that may be planned include:

- Security issues (to meet SLAs)
- Customer bugs and requests for help (RFH)
- Minor reliability/performance fixes

Instrumentation is a key requirement for all features to measure impact and effectiveness. This helps us validate adoption and make data-driven decisions.

### Issue Management

#### Workflow

We use the issue status field to note the current state of the issue.

- **New/Start**: issues that have just been created
- **Problem Validation**: applies to features and bugs we want to work on but have not a clear proposal or implementation details ([need refinement](#problem-validation--refinement))
- **Blocked**: issues that require other issues to be completed first
- **Ready for development**: issues that have a clear proposal and have [weight](#issue-weighting-guidelines) set
- **In dev**: work has started and is currently in progress
- **In review**: all related MRs are now in the review process
- **Verification**: all MRs have been merged and are waiting to be verified in pre/staging/production
- **Complete**: all has been verified, the issue can be closed

#### Issue Weighting Guidelines

| Weight | Description | Confidence Level |
| ------ | ----------- | -----------------|
| 1: Trivial | Well understood, no investigation needed, exact solution known | ≥90% |
| 2: Small | Well understood, minimal investigation needed, few surprises expected | ≥75% |
| 3: Medium | Well understood but requires investigation, some surprises expected | ≥60% |
| 4: Large | Basic understanding, requires further investigation, can potentially be broken down as the work progresses | ≥50% |
| 5: Extra Large | Should be broken down into smaller issues before starting work | ≥40% |

An issue with weight 1 should take no more than 2 days to complete.

Weight measures complexity and effort. When estimating issue weight, consider these technical factors:

- Classes or files affected
- Lines of code to change
- Tests requiring updates
- Number of MRs needed
- Existing patterns to follow vs. new approaches to learn
- Database migrations required?
- Gradual rollout strategy needed?
- Performance implications
- Documentation updates needed? etc

**Getting Help:**
When lacking domain knowledge, ask team members with relevant expertise. For high-complexity issues, always seek a second opinion.

#### Problem Validation / Refinement

When an issue is in the `Problem Validation` state and is part of the current milestone, the aim is to have it ready for development
in the following milestone. For example, if issue #123 state is `Problem Validation` in milestone 18.4, refinement should have been completed
before the end of the milestone. It should be `Ready for development` in 18.5.

Follow these steps:

1. Review description for accuracy and verify if proposal/solution is ready for development
   1. If issue it's not ready, schedule time during the milestone to clarify the requirements
   1. If necessary, break down the issue into smaller sub issues or open a Spike issue to investigate the problem further
1. Assign appropriate [weight](#issue-weighting-guidelines)
1. Set status to `Ready for development`

#### Async Status Updates

**Why we do this:** In our globally distributed team, async updates are our superpower - they help teammates across timezones stay connected to your work, prepare for handoffs, and jump in to help when needed.

**When:** Please update your assigned issues by Friday midday GMT each week at the latest (or more frequently when significant changes occur). This ensures everyone has visibility and sufficient time to adjust plans for next week if necessary

**Where:** Add updates directly in the issue comments rather than merge requests - this keeps the full story in one place.

##### Update Template

```markdown
## Status Update

**Progress this week:** <!-- What got done? Any wins or learnings? -->
- ... 

**What's next:** <!-- Planned work for next week -->
- ...

**Blockers:** <!-- Any blockers or questions? Tag specific people if needed -->
- None <!-- (if no blockers) -->

**Confidence for current milestone:**
- [ ] :red_circle: At risk - may not make it
- [ ] :yellow_circle: Some concerns - watching closely
- [ ] :green_circle: On track - confident we'll deliver

/health_status <on_track|needs_attention|at_risk>
/cc @golnazs
```

You can also leverage Duo Agent inside the issue to pre-fill the template for you with the following prompt:

````markdown
Succinctly fill the weekly status template below using the state of this issue and related MRs, and output it as markdown. Follow these guidelines:
- When outputting GitLab user names, wrap them in backticks, so that they are not mentioned.
- When writing feature flag names or about epics, try to link to the respective issue/epic.
- Do not remove items from the list in the Confidence section.

```
## Status Update

**Progress this week:**
<!-- What got done? Any wins or learnings? --->
-

**What's next:**
<!-- Planned work for next week -->
- ...

**Blockers:**
<!-- Any blockers or questions? Tag specific people if needed -->
- None <!-- (if no blockers) -->

**Confidence for current milestone:**
- [ ] :red_circle: At risk - may not make it
- [ ] :yellow_circle: Some concerns - watching closely
- [ ] :green_circle: On track - confident we'll deliver

/health_status <on_track|needs_attention|at_risk>
/label ~"workflow::in dev" 
/status "In dev" 
/cc @golnazs

<!--
/label ~"workflow::in review" 
/status "In review" 

/label ~"workflow::verification" 
/status "Verification" 
-->
```

### Async Collaboration

#### Weekly Priority Update

We use [Geekbot](https://geekbot.com/) integrated with Slack to share our weekly priorities with the rest of the team.

#### Weekly Team Sync with Async Agenda

We hold weekly sync meetings for available team members. Given our distributed nature, we strongly encourage async collaboration through our [Google Doc Agenda (internal)](https://docs.google.com/document/d/1JsS4kVu8X8LtFva35StlNfabWfgZTd0tl3I8-w7hJwE/edit#heading=h.kvc0p7nyngz5). Slackbot sends weekly reminders to review and contribute to the agenda.

#### Monthly Retrospectives

We use a GitLab issue in [the (internal) async retrospectives project](https://gitlab.com/gl-retrospectives/verify-stage/ci-scaling/-/issues/) for our monthly retrospective. The purpose of the monthly retrospective issue is to collaborate async and reflect on how the milestone went, in terms of:

- what went well
- what didn't go so well
- what we can do improve upon
- what praise we have for one another.

We encourage team members to add retrospective feedback throughout the month rather than waiting until milestone end. Issue due dates, tags, and Slack reminders prompt ongoing contributions before closure.

### Duo Agentic Chat Prompts

We use Duo's Agentic chat capabilities to help the team manage continuous tasks that improve our efficiency.
This list contains prompts we use in our day-to-day activities.

1. Help me triage this bug to apply severity and priority labels. Verify that this report belongs to "~group::ci platform".

### Labels

#### Category Labels

The CI Scaling group supports the feature categories described below:

| Label                 | |  | | |
| ----------------------| -------| ----|------------| ---|
| `Category:Continuous Integration (CI) Scaling` | [Issues](https://gitlab.com/groups/gitlab-org/-/issues?sort=created_date&state=opened&label_name[]=Category:Continuous+Integration+%28CI%29+Scaling) | [MRs](https://gitlab.com/gitlab-org/gitlab/-/merge_requests?scope=all&state=opened&label_name[]=Category%3AContinuous%20Integration%20%28CI%29%20Scaling) | [Direction](https://about.gitlab.com/direction/verify/#continuous-integration-ci-scaling) | Documentation - TBD |

## Developer Onboarding

Refer to the [Developer Onboarding in Verify](/handbook/engineering/devops/verify/#developer-onboarding-in-verify) section.
