---
title: "Infrastructure Platforms Department Request for Help Process"
description: "Infrastructure Platforms standardized approach to manage RFH issues."
---

## Background

The Infrastructure Platforms department is restructuring how inbound support requests are handled to help us triage and respond to requests within in a reasonable amount of time.

We are trying to create a process that is broad enough that it can be accepted by as many teams as possible.  We are working on rolling it out starting with the Production-Engineering teams, so that we can iterate and improve as we move forward.

Our current model of team members responding to Slack questions in generalized channels is no longer scaling. As an example, the #infrastructure-lounge Slack channel has 800 members and generates approximately 200 messages per month. Without a process we're creating constant interrupts for team members and missing urgent requests in the noise.

## Current Challenges

- Scale Problem: Every message in #infrastructure-lounge becomes an interrupt for nearly all Infrastructure Platform engineers
- Inconsistent Response: Not every team member gets the help they need due to irregular message handling
- Bystander Effect: Questions without clear ownership often go unanswered
- Knowledge Gaps: Requesters often don't know which team to ask

## New Process Overview

### Philosophy

This process should be as generic as possible, leaving as much of the decision making in the individual teams and groups as possible, while still offering guidance on the best way to solve this common problem.

It should enable teams with a process and tools that can help, but each team is different and should use the approach that works best for them.

It should limit duplication of data and issue templates.

It should have a single source of truth on how to request help that we direct people to for all group level questions.

### Structure

1. Group Responsibilities

- Designate a DRI (Directly Responsible Individual) per group to coordinate request handling
- Implement a triage process for general questions
- Establish SLA expectations (typically 1 business day response time)
- Create a Slack workflow automation for general questions
- Implement a continuous improvement process

1. For General Infrastructure Questions:

- Route to centralized Request for Help (RFH) repository: gitlab-com/saas-platforms/saas-platforms-request-for-help
- Categorize and redistribute to appropriate teams as needed

1. Team Responsibilities

- Use Slack workflow automation with emoji reactions to route requests to appropriate team trackers. See further down for guidance on how to do this.
- Each team maintains their own issue templates and triage processes
- Each team confirms that the single source of truth is updated to route people correctly
- Implement a continous improvement process

1. Continuous Improvement

- Regular Analysis: Weekly/monthly review of request patterns and categories
- Documentation Updates: Improve self-service documentation based on common questions
- Process Iteration: Adjust routing and handling based on data and feedback
- Toil Reduction: Use analytics to identify and address systemic issues causing repeated requests

## Process Rollout

1. Create a RFH Template with all the required information in your team's or group's repository.
1. Create a Slack Workflow to route issues to your request tracker.
1. Implement a triage process to deal with the incoming issues.
1. Create a regular process to analyze the issues by fixing bugs, prioritizing projects, or creating documentation.
1. Announce this broadly and encourage the team to use the emojis to route questions to the right place.

## Automation and Tools

### Slack Workflow Emojis

The simplest way to route from Slack to issues is an emoji workflow. [Slack has documentation](https://slack.com/help/articles/17542172840595-Build-a-workflow--Create-a-workflow-in-Slack) on how to create one.

We suggest picking an emoji and naming it :team-create-issue:, and assigning the workflow to all relevant channels.

Here's an example from the Observability team.

![Example Slack Workflow](/images/engineering/infrastructure-platforms/production-engineering/example-slack-workflow.png)

### Issue Templates

Each team should create a RFH issue template in their own issue tracker.  Within the template, please use the following labels:  TeamName: Requests.  See [the Observability RFH template](https://gitlab.com/gitlab-com/gl-infra/observability/team/-/blob/master/.gitlab/issue_templates/Request_for_Help.md?ref_type=heads) for an example.

### Triage Bot Integration

The [GitLab Triage Bot](https://gitlab.com/gitlab-org/quality/triage-ops) is one of the tools you can use to manage incoming issues.

### Analytics Dashboard

Using the default [Issues Dashboard](https://docs.gitlab.com/user/group/issues_analytics/) in your repository, you can get a good deal of information on the requests for help.

We are also working on a suggested [Insights Dashboard](https://docs.gitlab.com/user/project/insights/) that teams can use to get more information.
