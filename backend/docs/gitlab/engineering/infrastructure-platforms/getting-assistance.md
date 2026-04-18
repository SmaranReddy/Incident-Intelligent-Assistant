---
title: "Getting Assistance on Infrastructure Platforms"
description: "How to get assistance for problems on Production Platforms"
---

## GitLab.com

If you need to report an incident - follow the instructions on the [Report An Incident page](/handbook/engineering/infrastructure-platforms/incident-management/#reporting-an-incident).

If you are looking for help, and you know what service you need help with - find the owner in the [tech stack](https://gitlab.com/gitlab-com/www-gitlab-com/-/blob/master/data/tech_stack.yml). If they are listed below, then please create a Request For Help issue. If they aren't listed, please contact the owners through the Slack channel listed in the tech stack file.

If you need help, but you aren't sure who to ask, look through the teams below to see which team is the best fit for your question.

If you have read this whole page and are unsure how to proceed, please ask in the [#saas-platforms-help](https://gitlab.enterprise.slack.com/archives/C07DU5Z7V6V) channel. You will be redirected to the appropriate team for help.

We aim to respond to your request within 24 hours. If you raise your request on a Friday, it may only be responded to on Monday.

### Production Engineering

#### Teleport Requests

Teleport access requests will appear in the #teleport-requests Slack channel. Engineering and Security division managers approve all read-only teleport requests for team members. SRE or infrastructure managers are responsible for approving any write requests.

Teleport is owned by Infrastructure Security team, see [Teleport runbook](https://gitlab.com/gitlab-com/runbooks/-/blob/master/docs/teleport/README.md) for further details on how to get access or assistance.

#### Observability

Open a request for help in the [Request For Help Tracker](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-ProductionEngineering-Observability.md)

We can help with:

1. Observability
1. Logging
1. Metrics
1. Grafana / Kibana / Mimir / Prometheus
1. Error Budgets
1. Capacity Planning

Our Slack channel is: [#g_observability](https://gitlab.enterprise.slack.com/archives/C065RLJB8HK)

#### Runway

Open a request for help in the [Request For Help Tracker](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-ProductionEngineering-Runway.md)

We can help with:

1. Runway
1. Fleet Management

Our Slack channel is: [#f_runway](https://gitlab.enterprise.slack.com/archives/C05G970PHSA)

#### Networking and Incident Management

Open a request for help in the [Request For Help Tracker](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-ProductionEngineering-NetworkingAndIncidentManagement.md)

We can help with:

1. Networking and traffic management
   1. Rate Limiting: create an issue with the [rate limiting request template](https://gitlab.com/gitlab-com/gl-infra/production-engineering/-/issues/new?issuable_template=request-rate-limiting)
   1. Cloudflare: create an issue with the [Cloudflare Troubleshooting template](https://gitlab.com/gitlab-com/gl-infra/production-engineering/-/issues/new?issuable_template=Cloudflare%20Troubleshooting)
1. Incident Management
1. Disaster Recovery

Our Slack channel is: [#g_networking_and_incident_management](https://gitlab.enterprise.slack.com/archives/C09BM5XCPBP)

#### Runners Platform

Open a request for help in the [Request For Help Tracker](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-ProductionEngineering-RunnersPlatform.md)

We can help with:

1. Hosted Runners questions (.com/Dedicated)
2. Pipelines and jobs troubleshooting
3. Runners related incident support

Our Slack channel is: [#g_runners_platform](https://gitlab.enterprise.slack.com/archives/C08TJEKF0JZ)

### Software Delivery

#### Release and Deploy

Open a request for help in the [Request For Help Tracker](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-Delivery.md))

-We can help with:

1. Deployments to GitLab.com
1. Post Deployment Migrations (in relation to deployments)
1. Auto-Deploy
1. Hot Patching Process
1. Mean Time To Production
1. Release Management
1. Release Processes
1. Maintenance Policy
1. Patch Releases
1. Deployments
1. Monthly and Patch Releases
1. Backports

Our Slack channel is: [#g_release_and_deploy](https://gitlab.enterprise.slack.com/archives/CCFV016SV)

## Dedicated

In order to deal with RFHs as efficiently as possible we have a number of issue templates. Please use the appropriate issue template for your request.

1. For a Private Link Config Request raise an issue in the Request For Help Tracker using the [Private Link Request template](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-GitLabDedicatedPrivateLinkRequest)
1. For a SAML Config Request raise an issue in the Request For Help Tracker using the [SAML Config Request template](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-GitLabDedicatedSAMLConfigRequest)
1. For a Switchboard Request for Help raise an issue in the Request For Help Tracker using the standard [Dedicated Request template](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-GitLabDedicatedRequest)
1. For a standard request for help raise an issue in the Request For Help Tracker using the [Switchboard Request template](https://gitlab.com/gitlab-com/request-for-help/-/issues/new?issuable_template=SupportRequestTemplate-Switchboard)

We can help with:

1. Questions and support for GitLab Dedicated

Our Slack channel is: [#f_gitlab_dedicated](https://gitlab.enterprise.slack.com/archives/C01S0QNSYJ2)
