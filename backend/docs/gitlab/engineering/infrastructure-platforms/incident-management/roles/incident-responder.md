---
title: Incident Roles - Incident Responder
---

1. **As an Incident Responder, your highest priority for the duration of your shift is the stability of GitLab.com.**
2. When there is uncertainty of the cause of a degradation or outage, the **first action of the Incident Responder** is to evaluate whether any changes can be reverted. It is always appropriate to toggle (to previous state) any recently changed application feature flags without asking for permission and without hesitation. The next step is to review Change Requests and validate the eligibility criteria for application rollbacks.
3. The SSOT for who is the current EOC is the [GitLab.com Production EOC](https://app.incident.io/gitlab/on-call/schedules/01K5YWAGZ7YCQGAG7ATQ9XQWHW) schedule in incident.io.
    1. SREs are responsible for arranging coverage if they will be unavailable for a scheduled shift.  To make a request, send a message indicating the days and times for which coverage is requested to the `#eoc-general` Slack channel. Alternatively, use incident.io "request cover" feature. If you are unable to find coverage reach out to the [EOC coordinator](/handbook/engineering/infrastructure-platforms/incident-management/#engineer-on-call-coordinator) for assistance.
4. Alerts that are routed to incident.io require acknowledgment within 15 minutes, otherwise they will be escalated to the oncall Incident Manager.
    1. Alerts that create an escalation in incident.io will automatically create a triage incident in [`#incidents-dotcom-triage`](https://gitlab.slack.com/archives/alerts).
       1. If it is determined to be a true incident, the triage incident should be accepted by joining the channel and choosing "Accept it".
       1. If there are multiple pages/triage incidents created for the same incident, merge them into the primary incident. However, resolving the incident takes precedence. It is fine if a related triage incident auto-closes instead of getting merged while you're working an incident.
       1. The triage incident will automatically declined if no action is taken and the generating alert clears.
    1. Alert-manager alerts in [`#alerts`](https://gitlab.slack.com/archives/alerts) and [`#feed_alerts-general`](https://gitlab.slack.com/archives/feed_alerts-general) are an important source of information about the health of the environment and should be monitored during working hours.
    1. If the incident.io alert noise is too high, your task as an EOC is clearing out that noise by either fixing the system or changing the alert.
    1. If you are changing the alert, it is your responsibility to explain the reasons behind it and inform the next EOC that the change occurred.
    1. Each event (may be multiple related pages) should result in an issue in the `production` tracker. See [production queue usage](/handbook/engineering/infrastructure-platforms/production/#implementation) for more details.
5. If sources outside of our alerting are reporting a problem, and you have not received any alerts, it is still your responsibility to investigate. [Declare a low severity incident](/handbook/engineering/infrastructure-platforms/incident-management/#reporting-an-incident) and investigate from there.
    1. Low severity ([S3/S4](/handbook/engineering/infrastructure-platforms/incident-management/#severities)) incidents (and issues) are cheap, and will allow others a means to communicate their experience if they are also experiencing the issue.
    2. **"No alerts" is not the same as "no problem"**
6. GitLab.com is a complex system. It is ok to not fully understand the underlying issue or its causes. However, if this is the case, as Incident Responder you should [page the IMOC](/handbook/engineering/infrastructure-platforms/incident-management/#how-to-engage-response-teams) to find a team member with the appropriate expertise. Requesting assistance does not mean relinquishing your responsibility.
7. As soon as an [S1/S2](/handbook/engineering/infrastructure-platforms/incident-management/#severities) [incident is declared](/handbook/engineering/infrastructure-platforms/incident-management/#report-an-incident-via-slack), join the Zoom room for the incident. The Zoom link is in the bookmarks of the relevant incident channel.
    1. GitLab works in an asynchronous manner, but incidents require a synchronous response. Our collective goal is high availability of 99.95% and beyond, which means that the timescales over which communication needs to occur during an incident is measured in seconds and minutes, not hours.
8. Keep in mind that a GitLab.com incident is not an "infrastructure problem". It is a company-wide issue, and as Incident Responder, you are leading the response on behalf of the company.
    1. If you need information or assistance, engage with Engineering teams. If you do not get the response you require within a reasonable period, escalate through the IMOC.
    2. As Incident Responder, require that those who may be able to assist to join the Zoom call and ensure that they post their findings in Slack and pin (📌) the message to the incident timeline.
    3. The EOC is **not expected to join customer calls** during an incident. Your focus should remain on resolving the technical issue.
    4. The EOC can escalate through the Infrastructure Leadership pathway at any time for any reason without needing to justify the escalation.
9. By acknowledging an escalation in incident.io, you are implying that you are working on it. To further reinforce this acknowledgement, post a note in Slack that you are joining the incident Zoom as soon as possible.
10. _Be inquisitive_. _Be vigilant_. If you notice that something doesn't seem right, investigate further.
