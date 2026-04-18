---
title: "Verify:Runner Core"
description: "The Runner Core team page."
---

## Product Strategy and Roadmap

The product strategy and roadmap for the runner product categories are covered on the following direction pages.

- [Runner Core](https://about.gitlab.com/direction/verify/runner_core/)

## Team Members

The following people are permanent members of the Verify:Runner Core group:

{{< team-by-manager-slug "adebayo_a" >}}

## Stable Counterparts

{{< engineering/stable-counterparts role="Verify:Runner Core" manager-role="Engineering Manager(.*)Verify:Runner Core" >}}

## Development Process

### Planning and Tracking

The issues in progress or scheduled for the current milestone can be tracked at [Milestone Board](https://gitlab.com/groups/gitlab-org/-/boards/9726167).

This board contains all the necessary columns to track the workflow of the team, showing:

- All issues in the current milestone
- Issues organized by their GitLab status ("Ready for Development", "In dev", "In review", "Closed", etc.)

Once a team member self-assigns an issue on the Milestone Board, the issue status should be updated according to GitLab's native issue statuses to reflect the current state of work.

#### Upcoming Work

Issues planned for upcoming releases can be viewed in:

- [Upcoming Milestone](https://gitlab.com/groups/gitlab-org/-/issues?sort=updated_desc&state=opened&milestone_title=Upcoming&label_name%5B%5D=group%3A%3Arunner%20core&first_page_size=50)
- [Next 1-3 releases](https://gitlab.com/groups/gitlab-org/-/issues?sort=updated_desc&state=opened&milestone_title=Next%201-3%20releases&label_name%5B%5D=group%3A%3Arunner%20core&first_page_size=50) - for issues of interest which haven't been scheduled yet

### Features We Maintain

Runner Core maintains other feature categories apart from developing and advancing GitLab Runners.

The following categories are in **maintenance mode**:

- Auto DevOps
- Deployment Management
- Environment Management
- Infrastructure as Code

What does this mean? It means we typically only prioritize issues that met at least one criteria below:

- All Security Vulnerabilities
- S1 Bugs Only
- InfraDev Issues
- Error Budget related issues
- Support Escalations for Strategic Customers

### Handling Support Requests

For questions that aren't a quick fix, we route incoming requests through the Request for Help process rather than triaging in Slack. This ensures questions don't get lost in Slack and allows us to loop in subject-matter experts more easily.

A Slack workflow facilitates this. Reacting to a message with :RFH: in #g_runner_core or #runner_help prompts the requester to create an RFH issue.

The Support & Security Responder on rotation is responsible for checking new RFH issues daily and ensuring we meet our target response times based on severity.

#### When to Use RFH vs. Slack

- **Use Slack directly** for quick clarifications, pointers to documentation, or questions answerable in a few minutes.
- **Use RFH** when the question requires investigation, involves debugging a specific customer environment, needs input from multiple team members, or would benefit from being documented for future reference.

### 🔍 Spike Issues for Complex Work

If a task is too large, has too many unknowns, or requires proof of concept (POC),
it should be broken down into smaller investigation tasks or POC issues.
These tasks help clarify the scope, reduce risks, and identify the necessary steps to proceed with implementation.

#### Create an Investigation Issue

- **Purpose:** Research, investigate, and document or breakdown the necessary work.
  Clearly define the core question or problem you're investigating.
- **Label:** Assign the ~spike label to the issue.
- **Blocking Relationship:**
  - For issue investigations: Mark the investigation as blocking the original issue.
  - For epic investigations: Add the investigation issue to the epic.

#### Close the Investigation

- Align with key stakeholders on findings and next steps. Consider using a sync meeting to make decisions together.
- Depending on the feedback, we can decide to allocate more time or settle for something that works based on the information we have.
- **For Issue Investigations:** Summarize findings in the investigation issue. The outcome should inform the implementation of the blocked issue.
  Update the blocked issues to now include a clear direction and acceptance criteria.
- **For Epic Investigations:** Break down and refine the epic into actionable issues. The outcome could include:
  - **Architecture Plan:** High-level technical direction, quality goals (like performance, security), and supporting approaches.
  - **Iteration Plan:** A breakdown of work into clearly scoped, refined issues.
  - Add the plans to the epic description and close the investigation issue.
