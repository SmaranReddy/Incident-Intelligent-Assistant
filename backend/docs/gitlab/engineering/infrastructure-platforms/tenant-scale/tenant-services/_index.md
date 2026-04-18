---
title: Tenant Services Team
description: "Information about the Tenant Services Team"
---

## Vision

The Tenant Services team is responsible for providing SRE support to other teams within Tenant Scale.
We are the service owners for Sidekiq and Redis for GitLab.com, the team is composed mostly of SREs with some backend engineering support for Sidekiq.

## Team Members

{{< team-by-manager-slug "jarv" >}}

## How We Work

### Project Management

#### Issue Tracking

The Tenant Services team adheres to these core principles:

1. **Assign only actively in-progress issues**. Issues that aren't being worked on should remain unassigned.
1. **Apply mandatory labels**. Every issue must include:
   - `group::tenant services`
   - `devops::runtime`
1. **Add service-specific labels as needed**. If an issue pertains to a particular service (e.g., the Topology Service), tag it accordingly—e.g., `Service::Topology Service`.

By default, you should open issues that will be owned by the team under the [Tenant Services Team Issue Tracker](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/tenant-services/team/-/issues) and apply the `group::tenant services` label.

For issues tracked in the team's issue tracker or other `gitlab-com/gl-infra` projects, we use GitLab's built-in issue status feature to track an issue's workflow status. Ensure that all issues that are in progress, ready, or need triage have the relevant status applied.

##### When to Use Issue Status

Issue status should be used to track the current state of work items throughout their lifecycle. Update issue status when:

- **Creating new issues**: All new issues should start with an appropriate status (typically "New" or "Planning breakdown")
- **Transitioning between work phases**: Update the status whenever work moves from one phase to another (e.g., from "Ready for development" to "In dev")
- **Indicating blockers or delays**: Use "Blocked" status to communicate when work cannot proceed. Alternatively use `/blocked_by #item, group/project#item, url` quick action. If an issue is blocked by work required from another team please make sure that team is aware and the appropriate labels are applied.
- **When work needs validation**: Use appropriate validation statuses ("Problem validation", "Solution validation") when research or validation is needed
- **Showing completion**: Apply "Complete" status when work is finished and verified, closing the issue will also mark an issue as complete
- **Closing Issues when not complete**: Use the "Won't Do" status when work is not applicable

#### Issue Descriptions

All issues need to include a description of the work, any identified `action items` or `exit criteria`. When decisions are made regarding implementation details that change the initial description, update the description (and issue title if necessary) so that it always accurately reflects what the issue is implementing or achieving.

#### Epic Tracking

A single [team epic](https://gitlab.com/groups/gitlab-com/gl-infra/tenant-scale/tenant-services/-/epics/1) is used for all team owned project work where a team member is the DRI.
Epics follow the [Tenant Scale project management guidelines](/handbook/engineering/infrastructure-platforms/tenant-scale/operating-system/project-management/).

### Ways of Working

#### Team Meetings

Each week Tenant Services meets for two one hour meetings on Thursdays - alternating between APAC/EMEA and AMER friendly time zones. The intent of this meeting is to:

- Demos
- Review any technical decisions, blockers, etc. as a team to get feedback.
- Share information that the team should know
- Team discussions (process, roadmaps/upcoming work, company items, etc.)

#### Manager Responsibilities

Each week the EM will review the issue boards and/or epics

- Ensure that issues have the appropriate status set.
- Check on the status of in progress, blocked, and in review issues.
- Ensure that there are refined issues ready to be worked on. Refined issues are weighted, have sufficient context in the description, and have the **Ready for development** status indicating that it is ready to be worked on.
- Check that issues are aligned to our roadmap and status updates on relevant epics.

## Services

Tenant Services is responsible for infrastructure that supports the following GitLab application services:

1. Sidekiq
2. Redis
3. Gitaly

## Getting Assistance

Should you require assistance from the Tenant Services team, please open a request for help in the [Request For Help Tracker](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-TenantServices.md)
