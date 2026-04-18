---
title: Cloud Connector - Component ownership
---

## Ownership update

The Cloud Connector team was disbanded as of 2025-06-16. Ownership of Cloud Connector components has moved to new teams as outlined below:

- [Permissions & entitlements](#permissions--entitlements)
- [Key & token management](#key--token-management)
- [Duo Healthcheck](#duo-healthcheck)
- [Network ingress & service infrastructure](#network-ingress--service-infrastructure)

### Permissions & entitlements

- **Owner:** [Fulfillment:Provision](/handbook/engineering/development/fulfillment/provision/)
- **Themes:**
  - Unit Primitive ("cloud entitlements") configuration
  - How these entitlements map to GitLab license tiers and add-ons
  - How this information propagates to and is stored by GitLab instances
  - Alerting and incident response for Cloud Connector issues as they relate to above mentioned themes

### Key & token management

- **Owner:** [SSCS:Authentication](/handbook/engineering/development/sec/software-supply-chain-security/authentication/)
- **Themes:**
  - Token and key creation, storage and distribution
  - Tooling for rotation and invalidation of keys and tokens
  - Mapping of cloud entitlements to token payloads
  - Alerting and incident response for Cloud Connector issues as they relate to above mentioned themes

### Duo Healthcheck

- **Owner:** [Custom Models](/handbook/engineering/ai/custom-models/)
- **Themes:**
  - Implementation of the Duo Health Check feature for Cloud Connector/SHM components

### Network ingress & service infrastructure

- **Owner:** [Runway](/handbook/engineering/infrastructure-platforms/gitlab-delivery/runway/)
- **Themes:**
  - Maintaining Cloudflare Terraform modules for `cloud.gitlab.com` service ingress (routing + rate limiting)
  - Service infrastructure for Cloud Connector backends running on Runway
  - Alerting and incident response for Cloud Connector issues as they relate to above mentioned themes
