---
stage: enablement
group: Tenant Scale
title: 'Cells: CI/CD Catalog'
toc_hide: true
---

{{% alert %}}
This document is a work-in-progress and represents a very early state of the
Cells design. Significant aspects are not documented, though we expect to add
them in the future. This is one possible architecture for Cells, and we intend to
contrast this with alternatives before deciding which approach to implement.
This documentation will be kept even if we decide not to implement this so that
we can document the reasons for not choosing this approach.
{{% /alert %}}

## 1. Definition

The [CI/CD Catalog](https://docs.gitlab.com/ee/ci/components/) is a feature that allows users to discover and reuse pipeline configurations through components. Components are reusable pipeline snippets that can be published from projects and consumed by other projects.

**Current Architecture:** The CI/CD Catalog is **instance-wide**, meaning it lists components published from projects across the entire GitLab instance. On GitLab.com, users can browse all public components at `https://gitlab.com/explore/catalog`, and components are referenced using the instance FQDN (e.g., `component: $CI_SERVER_FQDN/components/sast/sast@1.0.0`).

## 2. Problem Statement

The instance-wide nature of the CI/CD Catalog creates two fundamental challenges with the Cells architecture:

### 2.1. Challenge 1: Using Components Across Cells

With Cells, organizations and thus groups and projects are isolated. When a project in one cell tries to include a component from a project in another cell, the Rails monolith cannot access the component data because:

- Each cell has its own database and cannot query data from other cells
- All cells share the same FQDN (`gitlab.com`), so the system cannot distinguish between local and cross-cell components based on the URL alone
- The current component resolution logic (`Ci::Components::InstancePath`) assumes all components are in the same database

**Impact:** Projects cannot use components published in different cells, severely limiting the catalog's utility on GitLab.com.

### 2.2. Challenge 2: Publishing and Listing Components

The catalog listing page (`https://gitlab.com/explore/catalog`) currently fetches all published components from the Rails monolith database (`Ci::Catalog::Listing`). With multiple cells:

- Components published in Cell 2 won't appear in the catalog when viewed from Cell 1
- There's no mechanism to aggregate and display components from all cells
- Discovery of components becomes limited to the user's own cell

**Impact:** The main value proposition of the catalog, discovering reusable components across the entire GitLab.com instance, is lost.

### 2.3. Challenge 3: Usage data and internal metrics

Partly related to the challenge above – the public Catalog also shows usage numbers and analytics. See `Ci::Catalog::Resources::Components::LastUsage` and `AggregateLast30DayUsageService` as starting points for how this tracking is done. These analytics need to continue showing data across the entire instance.

## 3. Proposed Solutions

### 3.1. Using Components Across Cells (Issue #456843)

**Approach:** Treat cross-cell component includes like remote includes.

When a component is included:

1. Check if the component project is in the same organization as the current project.
2. If it exists, fetch the component locally (same cell)
3. If it doesn't exist but the FQDN matches the current instance, assume it's in a different cell
4. Fetch the component via HTTP request to the other cell
5. The HTTP Router service (being implemented for Cells) will route the request to the correct cell

**Benefits:**

- Has potential to also enable Self-Managed/Dedicated instances to use GitLab.com components (unless they're air-gapped)
- Leverages existing remote include infrastructure
- Can use aggressive caching since published component versions are immutable
  - **Concern:** Performance impact with non-published components needs evaluation - should cross-instance access be allowed or blocked for unpublished components?

### 3.2. Publishing and Listing Components (Issue #442195)

**Approach:** Create a separate service outside the Rails monolith for GitLab.com.

This service would:

- Run only on GitLab.com (not shipped with Self-Managed/Dedicated)
- Be the target for component publishing from all cells
- Serve the `https://gitlab.com/explore/catalog` page
- Aggregate components from all cells for discovery
- Handle component analytics/usage data across the entire instance

**Alternative Approaches Considered:**

- **Elasticsearch cross-cluster search:** Could enable searching across cells, but requires investigation
- **Organization-scoped catalog:** Limit catalog to organization boundaries (conflicts with catalog's purpose)
- **Repository mirroring:** Creates global uniqueness issues and path inconsistencies

**Challenges:**

- Unprecedented for the Pipeline Authoring group to own a service with uptime requirements
- Requires SRE support and operational expertise beyond current team scope
- Metrics and usage tracking is a big challenge that also needs to be solved by the service
- Many unknowns remain about service architecture and deployment

## 4. Timeline and Scope

### Cells 1.0 (FY25-Q4)

### Protocells Migration

**Current Status:** This work is part of the Protocells Cohort B migration (Epic &1758)

**Priority:** Priority 1-2, marked as "needs attention" health status

## 5. Related Work

### AI Catalog (Issue #549767)

The AI Catalog faces similar challenges with Cells and Self-Managed/Dedicated support:

- Database architecture must store data at organization level (`gitlab_main_org`) rather than globally
- Cross-organization querying and data synchronization are challenges
- Proposed solution: API-based catalog population where organizations can download AI Catalog data from another organization's catalog
- "GitLab official" filtering would help identify trusted components

### Self-Managed and Dedicated Instances

Self-Managed and Dedicated instances will continue to operate as a single cell, so they won't face the cross-cell challenges. However:

- The cross-cell component usage solution (#456843) likely can enable SM/Dedicated to use GitLab.com components without manual mirroring
- Current workaround requires manual repository mirroring, which is cumbersome

## 6. Open Questions

1. **Service Ownership:** How will the Pipeline Authoring group handle operational responsibilities for a new service?
2. **Caching Strategy:** What caching mechanisms are needed to avoid performance degradation from cross-cell HTTP requests?
3. **Global Search:** How does this align with the broader strategy for global search in Cells (Issue #465563)?
4. **Explore Pages:** How will other `/explore` pages (projects, groups, etc.) handle cross-cell discovery?

## 7. References

- [Issue #442195: Publishing to and listing components in the GitLab.com CI/CD Catalog](https://gitlab.com/gitlab-org/gitlab/-/work_items/442195)
- [Issue #456843: How to use components from different Cells](https://gitlab.com/gitlab-org/gitlab/-/work_items/456843)
- [Issue #549767: AI Catalog support for Self Managed & Dedicated](https://gitlab.com/gitlab-org/gitlab/-/work_items/549767)
- [Epic &1758: Migrate Cohort B to Protocell](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/1758)
- [Cells 1.0 Blueprint](https://docs.gitlab.com/ee/architecture/blueprints/cells/iterations/cells-1.0.html)
- [CI/CD Components Documentation](https://docs.gitlab.com/ee/ci/components/)
