---
title: Incident Roles - Incident Lead
---

## Responsibilities of the Incident Lead

### During the incident

1. Responsible for posting regular status updates using the `/incident update` in the incident Slack channel. These updates should summarize the current customer impact of the incident and actions we are taking to mitigate the incident. This is the most important section of the incident timeline. It will be referenced to status page updates and should provide a summary of the incident and impact that can be understood by the wider community.
2. Ensure that the incident issue has all of the required fields applied. If not set them using `/incident field` command from the incident slack channel
3. Ensure that the incident issue is appropriately restricted based on [data classification](/handbook/engineering/infrastructure-platforms/incident-management/#incident-data-classification), to mark the issue as confidential use `/incident field` and set the `Keep GitLab Issue Confidential` to `true`
4. The Incident Lead should not consider immediate work on an incident completed until the Incident Summary is filled out with useful information to describe all the key aspects of the Incident.
5. Ensuring that the Timeline section of the incident in the `post-incident` tab is accurate and complete with the start and end of the customer impact.
6. Ensuring that the root cause is stated clearly and plainly in the incident description by updating the `causes` section in the `/incident summary`, or can be alternatively shared as an internal status update using emoji reactions. Reacting with the `:pushpin:` will post a public comment on the GitLab incident issue, reacting with a `:star:` will add an internal comment to the GitLab incident issue.
7. Ensure all follow-up items are properly documented and assign initial owners when possible.
8. Be available for customer interactions when requested by the Communications Lead. See [Communications Lead - Customer Call Management](communications-lead.html#customer-call-management).

### After the incident

1. Review automatically created follow-up issues within one business day. Issues pasted into the incident channel are automatically linked as follow-ups, so it is possible some of these are not valid follow-up items.
2. Verify each follow-up issue has appropriate context from the incident.
3. Move follow-up issues from the [follow-up issues project](https://gitlab.com/gitlab-com/gl-infra/incident-follow-ups/-/issues) to the correct project for the responsible team (typically this will be `gitlab-org/gitlab` or `production-engineering`.
4. Apply appropriate labels such as team and group to follow-up issues.
5. The Incident Lead should review the comments and ensure that the [corrective actions](/handbook/engineering/infrastructure-platforms/incident-management/#corrective-actions) are added to the issue description, regardless of the incident severity.
6. For all Severity 1 and Severity 2 incidents, [initiate an async incident review](/handbook/engineering/infrastructure-platforms/incident-review/#incident-review-process) and inform the Engineering Manager of the team owning the root cause that they may need to initiate [the Feature Change Lock process](/handbook/engineering/#feature-change-locks).

## Special Handling for S1 / S2 Incidents

When paged, the IMOC has the following responsibilities during a Sev1 or Sev2 incident and should be engaged on these tasks immediately when an incident is declared:

1. **In the event of an incident which has been triaged and confirmed as a clear Severity 1 impact:**
    1. Notify Infrastructure Leadership by typing `/incident escalate` in Slack. In the `On-call teams` drop-down menu, select `dotcom leadership escalation` with the appropriate message in the `Notification Message`. This notification should happen 24/7.
    2. In the case of a large scale outage where there is a serious disruption of service, the IMOC should check in with Infrastructure Leadership whether a senior member should be brought into the incident to coordinate and manage recovery efforts. This is to ensure that the person in charge of coordinating multiple parallel recovery efforts has a deeper understanding of what is required to bring services back online.
2. Consider engaging the release-management team if a code change related issue is identified as a potential cause and we need to explore rollbacks or expedited deployment. This can be done by using their slack handle `release-managers`
3. Ensure that necessary public communications are made accurately and in a timely fashion by the [Communications Lead](/handbook/engineering/infrastructure-platforms/incident-management/#communications-lead-responsibilities). Be mindful that, due to the directive to [err on the side of declaring incidents early and often](/handbook/engineering/infrastructure-platforms/incident-management/#report-an-incident-via-slack), we should first confirm customer impact with the Incident Responder prior to approving customer status updates.
4. If necessary, help the Incident Responder to engage development using the [InfraDev escalation process](/handbook/engineering/workflow/development-processes/infra-dev-escalation/process/).
5. If applicable, coordinate the incident response with [business contingency activities](/handbook/business-technology/entapps-documentation/policies/gitlab-business-continuity-plan/).
6. Following the first significant Severity 1 or 2 incident for a new member of the IMOC, schedule a feedback coffee chat with the Engineer On Call, Communications Manager On Call, and (optionally) any other key participants to receive actionable feedback on your engagement.

The IMOC is the DRI for all of the items listed above, but it is expected that they will do it with the support of the Incident Lead, Incident Responder, or others who are involved with the incident. If an incident runs beyond a scheduled shift, the IMOC is responsible for handing over to the incoming IMOC member.

The IMOC won't be engaged on these tasks unless they are paged, which is why the default is to page them for all Sev1 and Sev2 incidents. In other situations, [page the IMOC](/handbook/engineering/infrastructure-platforms/incident-management/#how-to-engage-response-teams) to engage them.

### Paging Infrastructure Leadership

To page the Infrastructure Leadership directly, run `/inc escalate` and choose the `dotcom leadership escalation` from the `Oncall Teams` drop-down menu

### Paging the Infrastructure Liaison

During a verified Severity 1 Incident the IMOC will page the Infrastructure Liaison.

To page the Infrastructure Liaison directly, run `/pd trigger` and choose the `Infrastructure Liaison` as the impacted service.
