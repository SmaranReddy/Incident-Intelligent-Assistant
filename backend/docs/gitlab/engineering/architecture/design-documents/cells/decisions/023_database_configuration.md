---
title: "Database Configuration for Cells"
owning-stage: "~devops::tenant-scale"
group: cells-infrastructure
creation-date: "2025-12-08"
authors: ["@tkhandelwal3"]
approvers: ["@daveyleach", "@praba.m7n", "@krasio", "@mkozono", "@tkuah"]
coach: "@tkuah"
toc_hide: true
---

## Context

We discussed whether an individual cell should have 3 **logical** databases or a single database in [this issue](https://gitlab.com/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/team/-/issues/507).

## Decision

We decided to create a single DB instance for cells with **3 logical databases** (main, ci, sec) as part of it, maintaining consistency with our `legacy cell` architecture.

This approach reduces unknowns and provides an easier PostgreSQL rollback path from a cell back to the legacy cell.

## Consequences

- We maintain a homogeneous cell cluster architecture aligned with our existing `legacy cluster`.
- Organization PostgreSQL data migration is simplified, as the migration becomes `3→3` instead of `3→1`, and rollbacks follow the same pattern.
- Organization data migration works with cells using decomposed databases, as described in [this comment](https://gitlab.com/groups/gitlab-org/-/epics/8631#note_2926361776).
- Geo for disaster recovery continues to work for cells using decomposed databases, as described in [this comment](https://gitlab.com/groups/gitlab-org/-/epics/8631#note_2926319113).

## Alternatives

**Use a single logical database (main) for cells**

- Rejected for the following reasons:
  - Creates migration challenges for existing data with the `gitlab_shared_org` schema.
  - Increases unknowns, as some features may depend on the decomposed database architecture.
  - Complicates rollbacks from cell to legacy cell, requiring identification of which tables belong to which database (a `1→3` data conversion).
