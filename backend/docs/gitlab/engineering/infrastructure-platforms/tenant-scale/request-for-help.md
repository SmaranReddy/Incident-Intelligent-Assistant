---
title: Tenant Scale Group
description: Requesting For Help from the Tenant Scale Group 
---

## How to Get Support from the Group Tenant Scale (GTS) Teams

The Group Tenant Scale (GTS) organization includes multiple specialized teams with the mission of ensuring the viability of GitLab's multi-tenant platforms. This section outlines the process and expectations when requesting support from GTS teams for customer support issues or technical assistance.

### Our Teams

GTS includes the following teams, each with their own areas of expertise:

- **Geo and Disaster Recovery** - [Data redundancy and disaster recovery solutions](geo/_index.md)
- **Cells** - [Horizontal scaling and tenant isolation](cells-infrastructure/_index.md)
- **Organizations** - [Organization entity development to support Cells and Groups and Projects](organizations/_index.md)

### Before Requesting Support

Before submitting a request for support, please:

1. **Review relevant documentation** - Check the specific team's documentation, handbook pages, and any related GitLab docs
2. **Search previous issues** - Look through existing customer issues in the relevant team's project
3. **Check team-specific channels** - Reach out in the appropriate team's Slack channel first for quick questions:
   - Geo: `#spt_pod_geo` for general questions and documentation clarifications, `#g_geo` for anything else.
   - Cells: `#g_cells_infrastructure`
   - Organizations: `#g_organizations`

### Getting Quick Help

#### General Questions

If you have a general question that you can't find an answer to in documentation, feel free to ask in the relevant team's Slack channel. Please keep in mind that engineers will do their best to answer questions quickly, but for more complex scenarios requiring research, you'll need to create a support issue.

#### When to Create a Support Issue

Create a support issue when:

- Slack discussions indicate more investigation is needed
- The question involves customer-specific scenarios
- You need documentation or context preserved beyond Slack retention
- The issue requires coordination across team members
- Priority assignment and tracking is needed

### Creating Support Requests

Please create issues in the [Request For Help Repo](https://gitlab.com/gitlab-com/request-for-help) using the respective team's templates.

- Geo: [Geo](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-Geo), [GEO Migrations](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-GeoDedicatedMigrations), [Collaboration](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-Collaboration)
- Cells - To be created
- [Organizations](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-Organizations)

#### Request for Help (RFH) Requirements

- RFHs must generally be opened by Support Engineers once a support ticket exists, this allows for alignment across multiple organizations incase of customer escalations. Support engineers typically will be the first to help with emergencies/escalations and we want to make sure all stakeholders are aligned on the context available.
- Include all relevant context and previous troubleshooting steps
- Issues with no updates for 2 weeks will be auto-closed by the team

### Priority Levels and Response Times

You may assign a priority label to your request. Team members will review and adjust priority during triage as needed.
Please refer to the various team pages for response times.

### Escalation for Urgent Issues

For outages and other urgent matters that require immediate attention:

- Use GitLab's [incident management](../incident-management/) processes
- Contact the appropriate on-call engineer if your organization has established escalation procedures
- For P1 emergencies, you may also reach out directly in team Slack channels while creating the formal issue

### Team-Specific Resources

Each team maintains their own additional resources and processes:

- **Geo Team**: See [Geo support process](geo/#how-to-ask-for-support-from-geo) for detailed Geo-specific guidance
- **Cells and Organizations**: Check the [Cells handbook page](cells_and_organizations/_index.md) for cells-specific information
- **Organizations Team**: See [Organizations Process page](organizations/process/#engineering-customersupport-rotation-process) for organization-specific guidance

### Best Practices

- **Be specific**: Include exact error messages, steps to reproduce, and environment details
- **Check documentation first**: Teams maintain comprehensive docs - please review them before requesting support  
- **Provide context**: Explain what you're trying to achieve, not just what's not working
- **Use clear titles**: Make issue titles descriptive and actionable
- **Follow up appropriately**: Respond to questions from team members promptly to keep issues moving

This process helps us provide the best possible support while managing our workload effectively. Thank you for following these guidelines!
