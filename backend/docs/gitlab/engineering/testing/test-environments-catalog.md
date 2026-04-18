---
title: "GitLab Test Environments Catalog"
---

This page provides a catalog of test environments available at GitLab for .com, Self-Managed, and Dedicated platforms.

## Overview

This catalog documents the various test environments used across GitLab, including their purpose, ownership, deployment model, and key characteristics. The information is organized to help developers and other stakeholders understand what environments are available and how to use them.

## Environment Catalog

### Staging-Canary

| Type | Status | Platform | URL | Owner/DRI | Project | Cells | Access |
|------|--------|----------|-----|-----------|---------|-------|--------|
| Permanent | Active | GitLab.com | [staging.gitlab.com](https://staging.gitlab.com) | Infrastructure Teams | N/A | Enabled | All engineers |

**Purpose:** Canary Stage within Staging for capturing mixed deployment issues. Receives deployments before Main Stage rollout.

**Key Features:**

- Has Web, API, Git HTTPS, websockets, registry, pages, shell, and Gitaly services available
- Shares with Main (Staging): Sidekiq, PostgreSQL, Redis, and other data storage (meaning database and feature flags affect both stages)
- HTTP router is enabled
- Topology service is deployed (not serving production traffic yet)

**Access:**

You may already have an account since Production accounts are automatically brought to Staging. Create an access request in the [access-request project](https://gitlab.com/gitlab-com/team-member-epics/access-requests) and assign to your manager for approval.

Set `gitlab_canary` cookie to `true` in your browser while accessing Staging. Also, there will also be the word “next” in a green box next to the GitLab logo in the top left.

Note: Setting the `gitlab_canary` cookie to `false` can help avoid targeting Canary Stage, but this is not a 100% guarantee since a subset of traffic will be redirected to Canary based on Infrastructure ruleset.

**Automated Testing:**

- Runs two sets of blocking E2E smoke tests: one targeting Staging-Canary, one targeting Staging Main
- Both test sets must pass for deployment to succeed
- Runs full E2E environment-compatible tests

[More details about Staging-Canary](/handbook/engineering/infrastructure-platforms/environments/#staging-canary)

---

### Staging

| Type | Status | Platform | URL | Owner/DRI | Project | Cells | Access |
|------|--------|----------|-----|-----------|---------|-------|--------|
| Permanent | Active | GitLab.com | [staging.gitlab.com](https://staging.gitlab.com) | Infrastructure Team | N/A | Enabled | All engineers |

**Purpose:** Pre-production testing for GitLab.com deployments. Same topology as Production with pseudonymized production database. Deployed frequently (at least every few hours) before production deployments.

**Key Features:**

- Mirrors production topology and configuration
- Includes Staging-Canary environment for gradual rollouts
- Pseudonymized database snapshot from production
- HTTP router is enabled
- Topology service is deployed (not serving production traffic yet)

**Automated Testing:**

- Runs smoke E2E test suite at each deployment

**Access:**

You may already have an account since Production accounts are automatically brought to Staging. Create an access request in the [access-request project](https://gitlab.com/gitlab-com/team-member-epics/access-requests) and assign to your manager for approval.

[More details about Staging](/handbook/engineering/infrastructure-platforms/environments/#staging)

---

### Staging-Ref

| Type | Status | Platform | URL | Owner/DRI | Project | Cells | Access |
|------|--------|----------|-----|-----------|---------|-------|--------|
| Permanent | Active | Self-Managed | [staging-ref.gitlab.com](https://staging-ref.gitlab.com) | `group::release-and-deploy` | [Staging-Ref GET Config](https://gitlab.com/gitlab-org/quality/gitlab-environment-toolkit-configs/staging-ref) | Not Enabled | All engineers |

**Purpose:** Pre-production sandbox environment for testing latest Staging Canary code with full admin access and control over data. Covers Engineering Department testing needs in a production-like environment.

**Key Features:**

- Deployed in parallel with Staging Canary using Deployer and GET
- 3k Cloud Native Hybrid Reference Architecture
- Geo setup with US primary site and EU secondary site
- Ultimate license with Free plan by default
- Admin access available
- Auto-rebuild capability
- Pre-existing test accounts for various scenarios
- Advanced Search, Sentry, Snowplow tracking, and Audit event streaming configured
- CustomersDot for Staging Ref is available separately from the main Staging CustomersDot ([CustomersDot Ansible inventories](https://gitlab.com/gitlab-org/customersdot-ansible/-/tree/master/inventories?ref_type=heads))

**Automated Testing:**

No tests are running at deploy time.

**Access:**

Go to staging-ref.gitlab.com and use your GitLab Google account. For Admin access, Sign in with Admin credentials from 1Password Engineering vault, then promote your user in Admin Area. To access SSH/Rails console request access to `gitlab-staging-ref` project via `#staging-ref` Slack channel or through the [access-request project](https://gitlab.com/gitlab-com/team-member-epics/access-requests).

For Feature Flags, use ChatOps commands in `#staging-ref` Slack channel.

[More details about Staging Ref](/handbook/engineering/infrastructure-platforms/environments/staging-ref)

---

### Production-Canary

| Type | Status | Platform | URL | Owner/DRI | Project | Cells | Access |
|------|--------|----------|-----|-----------|---------|-------|--------|
| Permanent | Active | GitLab.com | [gitlab.com](https://gitlab.com) | Infrastructure Teams | N/A | Enabled | SREs |

**Purpose:** Gradual rollout testing before full production. Environment subset within Production for controlled rollouts. Receives deployments before main production stage to minimize impact of issues.

**Key Features:**

- Subset of community traffic automatically routed to canary
- Has Web, API, Git HTTPS, websockets, registry, pages, shell, and Gitaly services available
- Shares with Main (Production): Sidekiq, PostgreSQL, Redis, and other data storage (meaning database and feature flags affect both stages)
- HTTP router is enabled
- Topology service is deployed (not serving production traffic yet)

**Access:**

Set `gitlab_canary` cookie to `true` in your browser while accessing Production. Also, there will also be the word "next" in a green box next to the GitLab logo in the top left.

Note: Setting the `gitlab_canary` cookie to `false` can help avoid targeting Canary Stage, but this is not a 100% guarantee since a subset of traffic will be redirected to Canary based on Infrastructure ruleset.

**Automated Testing:**

- Runs smoke E2E test suite on deployment
- Runs on Feature Flag toggle

[More details about Production-Canary](/handbook/engineering/infrastructure-platforms/environments/#production-canary)

---

### Production

| Type | Status | Platform | URL | Owner/DRI | Project | Cells | Access |
|------|--------|----------|-----|-----------|---------|-------|--------|
| Permanent | Active | GitLab.com | [gitlab.com](https://gitlab.com) | Infrastructure Team | N/A | Enabled | Public |

**Purpose:** Production environment for GitLab.com. Main SaaS platform serving the GitLab community. Full scale deployment with limited access.

**Key Features:**

- Two-stage deployment: Canary stage → Main stage
- Canary stage receives deployments first, serves limited community traffic
- Main stage serves remaining traffic for wider GitLab community
- Release candidate deployments
- HTTP router is enabled
- Topology service is deployed (not serving production traffic yet)

**Access:**

Your GitLab account.

**Automated Testing:**

- Runs E2E specs on Feature Flag toggle

[More details about Production](/handbook/engineering/infrastructure-platforms/environments/#production)

---

### .org (dev.gitlab.org)

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Permanent | Active | GitLab.com | [dev.gitlab.org](https://dev.gitlab.org) | Infrastructure Team | [Infrastructure Environments](/handbook/engineering/infrastructure-platforms/environments/) | Production and build team |

**Purpose:** Tools for GitLab.com infrastructure. Critical infrastructure hosting builds and repositories needed when GitLab.com is offline.

**Key Features:**

- Not intended for testing purposes
- Runs GitLab CE (not EE)
- Mission critical instance for builds and infrastructure
- Limited access - most engineers will not have accounts or will have restricted permissions
- Access primarily for team members involved in building and releasing the product

[More details about .org](/handbook/engineering/infrastructure-platforms/environments/#org)

---

### Ops (ops.gitlab.net)

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Permanent | Active | GitLab.com | [ops.gitlab.net](https://ops.gitlab.net) | Infrastructure Team | [Infrastructure Environments](/handbook/engineering/infrastructure-platforms/environments/) | SREs |

**Purpose:** GitLab.com operations infrastructure. Holds all infrastructure critical for managing GitLab.com.

[More details about Ops](/handbook/engineering/infrastructure-platforms/environments/#ops)

**Key Features:**

- Not intended for testing purposes
- Mission critical instance for builds and infrastructure

---

### Pre

| Type | Status | Platform | URL | Owner/DRI | Project | Cells | Access |
|------|--------|----------|-----|-----------|---------|-------|--------|
| Permanent | Active | Self-Managed | [pre.gitlab.com](https://pre.gitlab.com) | Infrastructure Team | N/A | Not Enabled | SREs |

**Purpose:** Validates release candidates for final self-managed releases and production patches. Also used by SREs for infrastructure change validation.

**Key Features:**

- Does not have full production HA topology or production database copy
- Receives release candidates on a monthly cadence
- Has Gitaly Cluster(Praefect)

**Automated Testing:**

- Runs smoke E2E test suite on deployment

[More details about Pre](/handbook/engineering/infrastructure-platforms/environments/#pre)

---

### Release

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Permanent | Active | Self-Managed | [release.gitlab.net](https://release.gitlab.net) | Infrastructure Team | N/A | SREs |

**Purpose:** Self-managed release validation. Validates security releases, self-managed final monthly and patch versions.

**Key Features:**

- Single node omnibus installation
- Does not have full production HA topology or production database copy

**Automated Testing:**

- Runs smoke E2E test suite on deployment

[More details about Release](/handbook/engineering/infrastructure-platforms/environments/#release)

---

### Release Environments

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Ephemeral | Active | Self-Managed | N/A (ephemeral) | `group::release-and-deploy` | [Release Environments](https://gitlab.com/gitlab-com/gl-infra/release-environments) | Contact `#g_release_and_deploy` |

**Purpose:** Validation for backports to maintained versions (three latest minor versions following GitLab's maintenance policy). Uses GitLab Environment Toolkit (GET) for stable branch testing.

**Key Features:**

- Creation is automated via CI
- Environments deleted when versions fall out of maintenance policy support
- Two Release Environment setups:
  - Helm Chart deployment in GKE clusters, not available for manual testing
  - GitLab Environment Toolkit deployment and it is the encouraged environment for manual testing

**Automated Testing:**

- Automated E2E smoke testing after deployments (blocking step in stable branch pipeline on Helm deployments)

**Access:**

You can login to the web interface using your Google account.

For gaining admin access, access to the Rails Console or Kubernetes infrastructure follow the [instructions in the runbook](https://gitlab.com/gitlab-org/release/docs/-/blob/master/runbooks/release-environment/get/connect-to-instance.md#connect-to-a-release-environment-instance).

---

### Dedicated Test Sandbox

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Permanent | Active | Dedicated | [dedicatedtestsandbox.gitlab-private.org](https://dedicatedtestsandbox.gitlab-private.org) | Support Team | [Dedicated Test Sandbox](https://dedicatedtestsandbox.gitlab-private.org/users/sign_in) | Support team |

**Purpose:** Support team testing and reproduction

Static dedicated sandbox for Support workflows. Enables Support team to test and reproduce customer issues on GitLab Dedicated.

[Support Handbook](/handbook/support/workflows/dedicated/#test-and-reproduction-on-gitlab-dedicated-instance)

---

### Dedicated Instrumentor Review Apps

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Ephemeral | N/A | Dedicated | N/A (ephemeral) | Dedicated SRE Team | [Instrumentor](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/instrumentor) | [Setup Guide](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/team/-/blob/main/engineering/sandbox-setup.md) |

**Purpose:** Dedicated SRE sandbox for verifying changes

Ephemeral environments for Dedicated infrastructure changes. Runs E2E smoke test suite.

**Resources:**

- [Testing Documentation](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/instrumentor/-/blob/main/docs/testing.md)

---

### Dedicated UAT

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Permanent | Active | Dedicated | N/A (internal) | Dedicated Team | [Switchboard UAT](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/sandbox/switchboard_uat) | Contact Dedicated team |

**Purpose:** User acceptance testing for Dedicated. Test environment for Dedicated automation and workflows.

---

### Reference Architecture Tester (RAT)

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Ephemeral | N/A | Self-Managed | N/A (ephemeral) | `group::build` (formerly Distribution) | [Reference Architecture Tester](https://gitlab.com/gitlab-org/distribution/reference-architecture-tester) | Automatic in CI |

**Purpose:** Multi-node deployment validation in Omnibus MRs.

Provisions environment, runs full E2E suite, then destroys. Validates that Omnibus packages work correctly in multi-node configurations.

**Key Features:**

- Currently supports 1k, 2k, 3k, and 10k architectures for testing omnibus-gitlab packages in different multi-node setups.

**Automated Testing:**

- Nightly runs for EE, including FIPS testing
- Runs full E2E test suite

#### FIPS Testing

**Purpose:** Validate GitLab software meets FIPS 140-2/140-3 cryptographic standards.

Tests compliance requirements for U.S. Federal agencies and regulated industries. Infrastructure is created per pipeline run and destroyed after testing.

**Key Features:**

- Uses [RAT](https://gitlab.com/gitlab-org/distribution/reference-architecture-tester) for environment provision
- Nightly FIPS Omnibus package builds

**Automated Testing:**

- Non-blocking, runs on nightly builds or manual EE pipeline triggers
- Any scheduled FIPS E2E pipelines results available at `#e2e-run-fips`

**Owner/DRI:** No clear owner ([Issue #4022](https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks/-/work_items/4022))

---

### GitLab Sandbox Cloud

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Ephemeral | N/A | Self-managed | `*.gitlabsandbox.cloud` | [In Transition to Infrasec](https://gitlab.com/gitlab-com/gl-security/corp/issue-tracker/-/work_items/2356) | [GitLab Sandbox Cloud](/handbook/company/infrastructure-standards/realms/sandbox/) | All engineers |

**Purpose:** Enables teams to test across cloud providers. Uses `gitlabsandbox.cloud` realm. GitLab instance needs to be spun up on AWS/GCP cloud for testing.

[Support Workflows](/handbook/support/workflows/test_env/)

---

### CustomersDot Staging

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Permanent | Active | SaaS Application | customers.staging.gitlab.com | Fulfillment | [CustomersDot](https://gitlab.com/gitlab-org/customers-gitlab-com) |  |

**Purpose:** Staging environment for GitLab Customers Portal. Connected with Zuora Central Sandbox.

**Automated Testing:**

- CustomersDot E2E tests run automatically against Staging after MR deployment. Test results appear in `#e2e-run-staging` and `#s_fulfillment_status` Slack channels.

[CustomersDot Architecture](https://gitlab.com/gitlab-org/customers-gitlab-com/-/tree/main/doc/architecture#customersdot)
[CustomersDot environment mappings](https://gitlab.com/gitlab-org/customers-gitlab-com/-/blob/main/doc/environment_mapping.md)

---

### Test-on-CNG

| Type | Status | Platform | URL | Owner/DRI | Project | Cells | Access |
|------|--------|----------|-----|-----------|---------|-------|--------|
| Ephemeral | Active | Cloud Native GitLab | N/A (Ephemeral per MR) | DevEx - Test Governance | [CNG Orchestrator](https://gitlab.com/gitlab-org/gitlab/-/tree/master/qa/gems/gitlab-orchestrator) | Not Enabled | Automatic in MR pipelines |

**Purpose:** Pre-merge E2E validation for Cloud Native GitLab. Tests MR changes against a [Cloud Native GitLab (CNG)](https://gitlab.com/gitlab-org/build/CNG) installation using E2E tests. Part of pre-merge validation lifecycle - must pass for code to merge.

**Key Features:**

- Triggered automatically in merge request pipelines (`e2e:test-on-cng` child pipeline)
- Deployment managed by [`orchestrator`](https://gitlab.com/gitlab-org/gitlab/-/tree/master/qa/gems/gitlab-orchestrator) CLI tool
- The test environment can be recreated locally using orchestrator CLI

---

### Test-on-GDK

| Type | Status | Platform | URL | Owner/DRI | Project | Cells | Access |
|------|--------|----------|-----|-----------|---------|-------|--------|
| Ephemeral | Active | Development | N/A (Ephemeral per MR) | DevEx - Test Governance | [GitLab Development Kit](https://gitlab.com/gitlab-org/gitlab-development-kit) | Enabled | Automatic in MR pipelines |

**Purpose:** Fast E2E test feedback for engineers. Provides faster end-to-end test execution than Omnibus-based testing by running against [GitLab Development Kit](https://gitlab.com/gitlab-org/gitlab-development-kit) (GDK).

**Key Features:**

- Triggered in merge request pipelines (`e2e:test-on-gdk` child pipeline)
- In CI: `build-gdk-image` job builds GDK as a Docker image
- Locally: GDK is a single-node development environment for local GitLab development ([Installation Guide](https://gitlab.com/gitlab-org/gitlab-development-kit#installation))
- Faster feedback loop for engineers
- `http router` is configured

---

### Test-on-Omnibus

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Ephemeral | N/A | Self-Managed | N/A (Ephemeral per MR) | DevEx - Test Governance | [Omnibus GitLab](https://gitlab.com/gitlab-org/omnibus-gitlab) | Manual trigger in MRs |

**Purpose:** E2E validation against Omnibus installations

Tests against an [Omnibus](https://gitlab.com/gitlab-org/omnibus-gitlab) installation to validate production-like deployments. Linux package deployment managed by [`gitlab-qa`](https://gitlab.com/gitlab-org/gitlab-qa).

**Key Features:**

- Manual trigger via `e2e:test-on-omnibus-ee` job
- Not executed in MR pipelines by default
- Non-blocking the MR merge - tests are allowed to fail even when manually triggered
- Self-managed environment running in Docker

**Automated Testing:**

- Full E2E test suite, excluding tests that run exclusively on live/permanent environments such as staging

---

### Charts Review Apps

| Type | Status | Platform | URL | Owner/DRI | Project | Cells | Access |
|------|--------|----------|-----|-----------|---------|-------|--------|
| Ephemeral | N/A | Kubernetes (Cloud Native) | N/A (ephemeral) | `group::operate` | [GitLab Charts](https://gitlab.com/gitlab-org/charts/gitlab) | Not Enabled | Automatic in CI |

**Purpose:** Helm chart validation

Validates GitLab Helm charts in cloud-native Kubernetes environments. Deployed to EKS and GKE clusters. Runs E2E and RSpec tests.

[Charts Development](https://docs.gitlab.com/charts/development/#review-apps)

---

### Operator Review Apps

| Type | Status | Platform | URL | Owner/DRI | Project | Cells | Access |
|------|--------|----------|-----|-----------|---------|-------|--------|
| Ephemeral | N/A | OpenShift | N/A (ephemeral) | `group::operate` | [GitLab Operator](https://gitlab.com/gitlab-org/cloud-native/gitlab-operator) | Not Enabled | Automatic in CI |

**Purpose:** GitLab Operator validation on OpenShift and Kubernetes clusters

Only known environment for OpenShift testing. [Operator CI Documentation](https://gitlab.com/gitlab-org/cloud-native/gitlab-operator/-/blob/master/doc/developer/ci.md#openshift-ci-clusters)

---

### Upgrade Tester

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Ephemeral | Inactive | Self-Managed | N/A | DevEx | [Upgrade Tester](https://gitlab.com/gitlab-org/quality/upgrade-tester) | All engineers |

**Purpose:** Multi-level upgrade path validation. Built for testing complex upgrade scenarios across multiple GitLab versions.

[Upgrade Path Testing](https://internal.gitlab.com/handbook/engineering/infrastructure-platforms/software-delivery/upgrade/upgrade-path-testing)

---

### Backup & Restore Testing

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Ephemeral | Active | Self-Managed | N/A (ephemeral) | Geo Team | [Backup & Restore Testing](https://gitlab.com/gitlab-com/gl-infra/gitlab-restore/backup-and-restore-testing) | All engineers |

**Purpose:** Automated testing framework for validating GitLab's backup and restore functionality across different Reference Architectures. Ensures reliability and integrity of GitLab's backup and restore processes through complete lifecycle testing from environment provisioning to data verification.

**Key Features:**

- Automated environment provisioning using GitLab Environment Toolkit (GET)
- Realistic test data seeding with Test Data Generator
- Full backup and restore lifecycle automation
- Support for multiple Reference Architectures (1k, 3k, 3k-cng)
- Manual pipeline execution with configurable parameters
- GCP-based infrastructure (`gitlab-restore` project)

**Automated Testing:**

- Automated weekly runs on Sundays for 1k and 3k architectures. Slack integration for automated reporting to `#tp-backup-results`.

---

### UX Cloud Sandbox

| Type | Status | Platform | URL | Owner/DRI | Project | Cells | Access |
|------|--------|----------|-----|-----------|---------|-------|--------|
| Permanent | Active | Self-Managed | https://ux.gitlabdemo.cloud/ | UX Team | [UX Research](/handbook/product/ux/experience-research/ux-cloud-sandbox/) | Not Enabled | Contact UX team |

**Purpose:** UX research and testing. Safe environment for external participants to share screens without security/privacy concerns.

---

### Support GET Template

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Ephemeral | N/A | Self-Managed | N/A  | Support Team | [Support GET Template](https://gitlab.com/gitlab-com/infra-standards/project-templates/support-Gitlab-Environment-Toolkit-template) | [Template](https://gitlab.com/gitlab-com/infra-standards/project-templates/support-Gitlab-Environment-Toolkit-template) |

**Purpose:** Support team environment provisioning. Template project for Support team members to quickly create test environments using GitLab Environment Toolkit (GET).

---

### Performance Test RFH Environments

| Type | Status | Platform | URL | Owner/DRI | Project | Access |
|------|--------|----------|-----|-----------|---------|--------|
| Ephemeral | Active | Self-Managed | N/A (ephemeral) | DevEx - Performance Enablement | [Performance Test RFH](https://gitlab.com/gitlab-org/quality/gitlab-environment-toolkit-configs/performance-test-rfh) | Contact `#g_performance_enablement` |

**Purpose:** Ad-hoc performance testing. Framework for creating performance test environments with idempotent scripts for easy reset and reuse.

---

*Related Epic: [Understanding Test Environments](https://gitlab.com/groups/gitlab-org/quality/-/epics/262)*
