---
owning-stage: "~group::release-and-deploy"
title: "Cells ADR 022: Ring definition for Protocells"
toc_hide: true
---

## Context

The definition of rings and responsibility for ownership over deployments to them has changed since the discussion in [Application Deployment with a Cellular Architecture](../infrastructure/deployments/#ring-deployment). This ADR is an attempt to clarify the rings that will exist for [Protocells](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/1616), which rings will house the cells containing the first few cohorts of users, and the ownership boundary for auto-deploy to Protocells.[^1]

## Decision

1. Ring 0 will contain only the QA cell
1. Ring 1 will contain one (or more) cells where we will import data for the first few cohorts of users
1. Changes will be deployed to Ring 0 before the deployment to `gprd-cny`
1. Changes will be deployed to Ring 1 in parallel with `gprd`
1. Success of the QA job in Ring 0 will be a prerequisite for a package to be deployed to Ring 1
1. [Release managers](../../../../deployments-and-releases/_index.md#release-managers) will be responsible for ensuring the continuous deployment of changes in the GitLab and related codebases to Ring 0 and Ring 1 as part of the [Auto-deploy process](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/deploy/auto-deploy.md). Specifically, this includes changes to the tenant model fields `prerelease_version` and `gitlab_custom_helm_chart.version`
1. Failing deployment or QA to any ring will not block deployments to any stage in the Legacy cell
1. Ring 1 might be a few versions behind the Legacy cell, when deployment or QA fails in any of the rings. When this happens, release managers can choose to pause deployments to the Legacy cell until Ring 1 catches up

### Out of scope

1. Ability to roll-back deployments
1. Decoupling post-deployment migrations and configuration changes. While we intend to work on this before the initial rollout, there are some unknowns, so we will not introduce any dependence on this feature.
1. Ring 2 and beyond. We do not plan to create ring 2 during the initial rollout of Protocells
1. The design and implementation of QA jobs. The implementation of QA is being discussed separately in [this issue](https://gitlab.com/gitlab-com/gl-infra/delivery/-/issues/21521#note_2740966366). The outcome of that discussion may be documented in a later ADR.
1. Responsibility for configuration changes (such as Instrumentor version upgrades or changes to other fields in the tenant model) will also be discussed at a later point

## Consequences

[Protocells](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/1616) will co-exist with the legacy Cell during the initial roll-out and for the foreseeable future. So, we will continue to use the auto-deploy deployment pipeline to orchestrate deployments.

### Deployment Pipeline Schematic

The proposed pipeline below, which incorporates the decisions made in the previous section, is essentially identical to what was suggested in the [Co-existence of the legacy infrastructure and Cells](../infrastructure/deployments/#co-existence-of-the-legacy-infrastructure-and-cells) section of the [original deployments blueprint](../infrastructure/deployments/#ring-deployment), barring some minor changes.

``` mermaid
flowchart TD
    Timer[Auto Deploy Schedule] --> Build[Build package]

    Build --> Legacy
    Build --> R0

    subgraph Legacy[Legacy Deployments]
        gstg-cny --> gprd-cny
        gprd-cny --> AwaitingPromotion[Awaiting promotion]
        AwaitingPromotion --> ManualPromotion[RM promotes package]
        ManualPromotion --> gstg
        ManualPromotion --> Delay[30min delay]
        Delay --> gprd --> LegacySuccess[Complete - Legacy Cell]
    end

    subgraph Protocells[Protocell Deployments]
        subgraph R1[Ring 1 - User Cells]
            direction TB
            VersionDrift[Record version drift] --> Threshold{Has version drift<br>exceeded a threshold?} --> AlertRMs[Alert RMs about Ring 1 to gprd version drift] --> RMDecision[RM may pause deployments to legacy cell]
            R1Deploy[Deploy to Ring 1] -->|Deployment success| CellsSuccess[Complete - Cells]
            R1Deploy -->|Deployment failure| VersionDrift
        end

        subgraph R0[Ring 0 - QA Cell]
            direction TB
            R0QA -->|QA failure| RMInvestigation[RMs investigate the reason for failure]
            R0QA -->|QA success| R0QASuccessRecord[Record QA success<br>in Ring 0 for package]
            R0Deploy -->|Deploy failure| RMInvestigation
            R0Deploy[Deploy to Ring 0] -->|Deploy success| R0QA[QA in Ring 0]
        end
    end

    Delay --> CellsPromotionDecision{Has QA on this package<br>succeeded in Ring 0?} --> |Yes| R1Deploy
    CellsPromotionDecision -->|No| VersionDrift[Record version drift]

```

### Version Drift Between `gprd` and Ring 1

`gprd` and Ring 1 are both "production environments" of GitLab.com, as they will be serving user traffic. However, a failure to deploy a package to Ring 1 will not automatically block deployments to `gprd`.

This is because we are not yet confident about the reliability of deployments and QA in Protocells. So, we will accept the consequence that there may be one-sided version drift between `gprd` and Ring 1: Ring 1 may lag several versions behind `gprd`.

We will handle this version drift by monitoring it using a metric, and alerting release managers when this metric surpasses a threshold. It will be up to the release managers' discretion as to whether deployments to `gprd` should be paused in order to let Ring 1 to catch up. This decision would be influenced by various factors such as the criticality of deploying a package to `gprd` and the impact of _not_ delivering a package to users on Ring 1.

[^1]: This ADR is a result of the discussion that we had at https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/1619#note_2660377621 and https://gitlab.com/gitlab-com/gl-infra/delivery/-/issues/21384#note_2663367292.
