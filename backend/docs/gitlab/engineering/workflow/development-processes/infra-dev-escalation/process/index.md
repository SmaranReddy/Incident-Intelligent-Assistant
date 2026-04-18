---
aliases:
  - /handbook/engineering/workflow/development-processes/infra-dev-escalation/process/
title: Development Escalation Process (Discontinued)
---

{{% alert title="Process Discontinued - Effective January 5th" color="warning" %}}
The Development Escalation process has been discontinued as of January 5th. Please use the **Tier 2 On-Call Program** for subject matter expert support.
{{% /alert %}}

## Process Change

The Development Escalation process is no longer active. This process was initially set up to provide Rails Monolith support to the Infrastructure department. Over time, responsibilities expanded to providing general application support to both Infrastructure and Support teams. However, the application has become so broad that it's unlikely a single on-call engineer will have expertise in any specific area that needs attention.

## What to Use Instead

For escalations requiring subject matter expert support, please use the **[Tier 2 On-Call Program](/handbook/engineering/infrastructure-platforms/incident-management/on-call/tier-2/)**.

For non-urgent issues, you can:

1. Label with `security` and mention the `@gitlab-com/security/appsec` team to be notified as part of the [Application Security Triage rotation](/handbook/security/product-security/security-platforms-architecture/application-security/)
1. Label with `infradev` which will be raised to the [Infra/Dev triage board](https://gitlab.com/groups/gitlab-org/-/boards/1193197?label_name[]=gitlab.com&label_name[]=infradev)
1. Raise to the respective product stage/group Slack channel
1. Ask the [#is-this-known](/handbook/communication/#asking-is-this-known) Slack channel
