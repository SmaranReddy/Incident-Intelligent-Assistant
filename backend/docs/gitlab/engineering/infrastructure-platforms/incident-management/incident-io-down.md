---
title: What if incident.io is Down?
---

## Overview

If incident.io becomes unavailable or inaccessible, follow these procedures to maintain incident management continuity.

## Login Issues (CF or Okta Outage)

If you cannot access incident.io due to Cloudflare (CF) or Okta authentication issues:

- Contact an incident.io admin (#g_networking_and_incident_management or infrastructure leadership) to grant the permission "Bypass SAML Login" temporarily to the Standard role in the [incident.io permissions settings](https://app.incident.io/gitlab/settings/permissions/account)
- This allows temporary access without SAML authentication

## Actions

1. **Centralize Communication**: Create a dedicated Slack channel for the incident (example, `#incident-io-outage` or `#incident-[date]`)

   - Use this channel as the single source of truth for all incident updates
   - Ensure all relevant stakeholders are invited

2. **Create a Google Doc**: Start a shared Google Doc using [this template](https://docs.google.com/document/d/1tFcU5tIpUin_3O50QpEMrhB_KA1vU-PgnI1fRT-xlBE/edit?usp=sharing) to track:

   - Incident timeline and key events
   - Action items and owners
   - Status updates
   - Resolution steps

3. **Share Access**: Ensure the Google Doc is accessible to all within GitLab

## Return to Normal Operations

Once incident.io is restored:

- Create a "Retrospective incident" from the [incident.io dashboard](https://app.incident.io/gitlab/dashboard)
- Migrate any critical information from the Google Doc back into incident.io
- Close the temporary Slack channel or archive it for reference
- Document lessons learned from the outage
