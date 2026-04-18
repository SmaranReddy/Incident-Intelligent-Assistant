---
title: "Traversal IDs on issues"
status: ongoing
creation-date: "2025-04-22"
authors: ["@dgruzd", "@nicolasdular"]
coach: ["@ahegyi"]
approvers: []
owning-stage: "~plan::product planning"
participating-stages: []
---

## Business Objectives

To further roll out work items, we need to tackle the scaling problems that are inherent to our hierarchy-based approach to querying data.
Without addressing these performance problems, the product will not be able to scale appropriately for large organizations.

For more details about the active development, see

- [Decompose issue descriptions](https://gitlab.com/groups/gitlab-org/-/epics/18643)
- [Denormalize namespace_traversal_ids on issues](https://gitlab.com/groups/gitlab-org/-/epics/18644)
- [Remove unused issue indexes](https://gitlab.com/groups/gitlab-org/-/epics/18704)

## Problem Statement

After migrating `epics` to the `issues` table as part of the work items framework, we face a new performance problem when querying work items within the group hierarchy.
Before this migration, we only had two cases when querying within the hierarchy:

1. Issues - Fetching all issues from all projects within the group hierarchy. We're able to use the `project_id` on the `issues` table.
2. Epics - Fetching all epics from all groups within the group hierarchy. We're able to use the `group_id` on the `epics` table.

In both cases, the number of namespaces was limited to either Groups or Projects. Also, the `epics` table was considerably smaller in size (about 0.5% of the `issues` table).

With work items, everything is in one `issues` table, and we now only distinguish between different work items using the `work_item_type_id` column.
The requirements also changed, as we want to list all work items from groups and projects (both Epics, Issues, or any other work item type).

From our first experiments, we face performance problems where the queries time out for large groups like `gitlab-org` or `gitlab-com`.

### Root cause of the problem - looking up work items in many namespaces

When looking up all issues within a group, we use a query like the following (authorization checks are removed for improved readability):

```sql
WITH namespace_ids AS (
  SELECT id
  FROM namespaces
  WHERE traversal_ids @> ('{9970}')
)
SELECT * from issues WHERE issues.namespace_id IN (
  SELECT id FROM namespace_ids
)
ORDER BY created_at, id LIMIT 100;
```

For large groups, the number of namespace_ids in the `IN`-clause can be a few thousand IDs, and the query can time out.

## Proposal: Denormalize traversal_ids on issues table

Since the bottleneck is the large IN clause, we're proposing to denormalize the hierarchy on the `issues` table. With that, we can directly query all issues within a group hierarchy, instead of providing all namespace_ids to the query.
This approach has already shown great results as the `vulnerability_reads` table performed the [same optimization](https://gitlab.com/groups/gitlab-org/-/epics/12372).
Notable: The `vulnerabilities_reads` table has a similar cardinality to the `issues` table on GitLab.com, but there is a difference in requirements. For example, the sorting options are limited, which we will need to address with more indexes on the issues table.

### Setup

To validate the experiment, we copied the `issues` to the `new_issues` table and backfilled the `traversal_ids` column on a replica.
The complete setup can be found here to replicate: https://gitlab.com/gitlab-org/plan-stage/product-planning/issue-database-research-benchmarking

#### Before and after performance

As part of the project above, we're running a benchmarking setup that compares the performance of running the same set of SQL queries before and after enabling traversal_ids.
The full report with all before/after query plans, timings, and buffer usage can be found here:

https://gitlab.com/gitlab-org/plan-stage/product-planning/issue-database-research-benchmarking/-/blob/main/report.md

### Indexes

#### Top-level group index

To support querying work items on large top-level groups and ordering, we're adding a BTREE index on the root of every `traversal_ids`.
This is required to support all of our sorting options that we offer.

```sql
CREATE INDEX idx_issues_on_root_namespace_id_and_state_id_created_at ON issues USING btree ((traversal_ids[1]), state_id, created_at);
CREATE INDEX idx_issues_on_root_namespace_id_and_state_id_updated_at ON issues USING btree ((traversal_ids[1]), state_id, updated_at);
CREATE INDEX idx_issues_on_root_namespace_id_and_closed_at ON issues USING btree ((traversal_ids[1]), closed_at);
```

These are the proposed indexes, but we might need to add `work_item_type_id` as part of the index to account for filtering on work item type.

#### Index on traversal_ids

To optimize querying specific sub-groups, we add a BTREE index on `traversal_ids`. This is a trade-off to optimize querying for
all work items within that sub-group and being forced to sort in memory.

```sql
CREATE INDEX idx_issues_on_state_id_traversal_ids ON issues USING btree (state_id, traversal_ids);
```

This index works best when queried in the following way (in this case for the namespace_id `10510295`):

```sql
WHERE
    state_id = 1
    AND
        traversal_ids >= '{9970, 10510295}'::bigint[] AND
        traversal_ids < '{9970, 10510296}'::bigint[]
```

In our tests, we observed that this query still performs better for large subgroups compared to IN queries. This is the reason we decided to use traversal_ids instead of just root_namespace_id. However, if we see any performance issues for larger subgroups, we can create a more specific index for second-level subgroups. For example:

```sql
CREATE INDEX idx_issues_on_lvl2_namespace_id_and_state_id_created_at ON issues USING btree ((traversal_ids[2]), state_id, created_at);
```

#### Traversal ID Storage

**Column storage overhead:**

```sql
SELECT
    pg_size_pretty(
        (SELECT avg_width FROM pg_stats
         WHERE tablename = 'new_issues' AND attname = 'traversal_ids')::bigint
        * (SELECT reltuples FROM pg_class WHERE relname = 'new_issues')::bigint
    ) AS estimated_column_size;
```

```sql
estimated_column_size
-----------------------------
4055 MB
```

#### Index storage

For each sorting index, we can estimate around ~6GB, and for the `traversal_ids` index, we roughly account for 4GB.
Since we need three of the sorting indexes, the total storage size for the indexes will be around ~22GB.

```sql
 SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size
FROM
    pg_indexes
WHERE
    indexname IN ('idx_new_issues_on_root_namespace_id_and_state_id_created_at', 'idx_new_issues_on_state_id_traversal_ids')
ORDER BY
    pg_relation_size(indexname::regclass) DESC;
                          indexname                          | index_size
-------------------------------------------------------------+------------
 idx_new_issues_on_state_id_traversal_ids                    | 6715 MB
 idx_new_issues_on_root_namespace_id_and_state_id_created_at | 4109 MB
 ```

### Delay of syncing traversal_ids when moving namespaces

When moving issues from one namespace to another, we need to update all of the `traversal_ids`. This means that there is a timeframe where issues would not show up, as this is done
in a background job. Based on the Workers for updating `vulnerability_reads`, the P95 execution time of these jobs is within an acceptable range (<3 seconds) [[0](https://log.gprd.gitlab.net/app/lens?_g=%28filters%3A%21%28%28%27%24state%27%3A%28store%3AappState%29%2Cmeta%3A%28alias%3A%21n%2Cdisabled%3A%21f%2Cindex%3AAWNABDRwNDuQHTm2tH6l%2Ckey%3Ajson.class%2Cnegate%3A%21f%2Cparams%3A%28query%3A%27Sbom%3A%3ASyncProjectTraversalIdsWorker%27%29%2Ctype%3Aphrase%29%2Cquery%3A%28match_phrase%3A%28json.class%3A%27Sbom%3A%3ASyncProjectTraversalIdsWorker%27%29%29%29%2C%28%27%24state%27%3A%28store%3AappState%29%2Cmeta%3A%28alias%3A%21n%2Cdisabled%3A%21f%2Cindex%3AAWNABDRwNDuQHTm2tH6l%2Ckey%3Ajson.job_status.keyword%2Cnegate%3A%21f%2Cparams%3A%28query%3Adone%29%2Ctype%3Aphrase%29%2Cquery%3A%28match_phrase%3A%28json.job_status.keyword%3Adone%29%29%29%29%2Ctime%3A%28from%3Anow-1w%2Cto%3Anow%29%29#/?_g=h@97e8101)], [1](https://log.gprd.gitlab.net/app/lens?_g=%28filters%3A%21%28%28%27%24state%27%3A%28store%3AappState%29%2Cmeta%3A%28alias%3A%21n%2Cdisabled%3A%21f%2Cindex%3AAWNABDRwNDuQHTm2tH6l%2Ckey%3Ajson.class%2Cnegate%3A%21f%2Cparams%3A%28query%3A%27Vulnerabilities%3A%3AUpdateNamespaceIdsOfVulnerabilityReadsWorker%27%29%2Ctype%3Aphrase%29%2Cquery%3A%28match_phrase%3A%28json.class%3A%27Vulnerabilities%3A%3AUpdateNamespaceIdsOfVulnerabilityReadsWorker%27%29%29%29%2C%28%27%24state%27%3A%28store%3AappState%29%2Cmeta%3A%28alias%3A%21n%2Cdisabled%3A%21f%2Cindex%3AAWNABDRwNDuQHTm2tH6l%2Ckey%3Ajson.job_status.keyword%2Cnegate%3A%21f%2Cparams%3A%28query%3Adone%29%2Ctype%3Aphrase%29%2Cquery%3A%28match_phrase%3A%28json.job_status.keyword%3Adone%29%29%29%29%2Ctime%3A%28from%3Anow-1w%2Cto%3Anow%29%29#/?_g=h@97e8101)

## Table Size Challenge

Before implementing traversal IDs, we must reduce the size of the `issues` table, as it is already a large table.
The idea is similar to the existing `vulnerability_reads` table, but instead of creating an `issues_reads` table that has duplicated data, we want to trim down the `issues` table
and remove data that we do not filter or sort on.

This table decomposition is not just necessary for introducing the `traversal_ids` index, but also reduces the size of the table overall.

### Current Storage Statistics

The table has the following columns + indexes that use 60% of the table size:

- `description` (~20%)
- `description_html` (~32%)
- `index_issues_on_description_trigram_non_latin` (~10%)

The `title` and `title_html` are around 3% of the table size. However, since the access pattern of querying a list of work items with their title is common,
and the gains are minimal, we won't decompose them.
Each other column uses <=1% of the total table size.

**Combining issue_search_data**

We have an `issue_search_data` table, which contains a `search_vector tsvector` column that we can use for our full-text search indexing. However, the `issue_search_data` table only has a `project_id` and is partitioned by it.
Since we moved to epic work items, which live in groups, this is a good opportunity to move this vector index over to the new decomposed table.

**Partitioning**

Due to the size of the table, we also aim to partition the new `work_item_descriptions` table using the `root_namespace_id`. The reason for `root_namespace_id` instead of `namespace_id` is due to the access patterns
when listing work items within a group with their descriptions. If we used `namespace_id` as the partitioning key, we'd need to look up the data in multiple partitions, instead of just one.

The downside of `root_namespace_id` as a partitioning key is that we need to keep it up to date when moving issues across root namespaces.

**New work_item_descriptions table**

This leaves us with the following data for the `work_item_descriptions` table:

- `last_edited_at`, `lock_version`, `cached_markdown_version` are related to modifications of the `description` column. So it makes sense to keep them in the same table, since we also don't
    filter or sort by these columns.
- The `search_vector` also encodes the `title` from the `issues` table. This is not a perfect separation, but it provides us the ability to simplify our full-text search queries.

```sql
                                                     Table "public.work_item_descriptions"
                Column                 |            Type             | Collation | Nullable |              Default
---------------------------------------+-----------------------------+-----------+----------+------------------------------------
 work_item_id                          | bigint                      |           | not null |
 description                           | text                        |           |          |
 description_html                      | text                        |           |          |
 cached_markdown_version               | integer                     |           |          |
 last_edited_at                        | timestamp without time zone |           |          |
 last_edited_by_id                     | bigint                      |           |          |
 lock_version                          | integer                     |           |          | 0
 namespace_id                          | bigint                      |           | not null |
 root_namespace_id                     | bigint                      |           | not null |
 search_vector                         | tsvector                    |           | not null |
```

#### work_items_description table indexes

For the new `work_item_descriptions` table, we also keep using the same indexes we had before:

```sql
CREATE INDEX idx_work_item_descriptions_on_description_trigram ON work_item_descriptions USING gin (description gin_trgm_ops) WHERE description IS NOT NULL;
CREATE INDEX idx_work_item_descriptions_on_search_vector ON ONLY work_item_descriptions USING gin (search_vector);
CREATE INDEX idx_work_item_descriptions_on_last_edited_by_id ON work_item_descriptions USING btree (last_edited_by_id);
```

### Moving more columns out of the issues table

The following columns are not used for filtering or sorting, but all require an index because they have a foreign key constraint.
These columns will be moved to a separate table (name to be defined):

- `moved_to_id`
- `duplicated_to_id`
- `promoted_to_id`

```sql
                                                      Table "public.work_item_transitions"
                Column                 |            Type             | Collation | Nullable |              Default

---------------------------------------+-----------------------------+-----------+----------+------------------------------------
 work_item_id                          | bigint                      |           | not null |
 moved_to_id                           | bigint                      |           |          |
 duplicated_to_id                      | bigint                      |           |          |
 promoted_to_epic_id                   | bigint                      |           |          |
```

### New and removed indexes on the issues table

With the decomposition work, we can remove the following indexes from the `issues` table:

1. `index_issues_on_duplicated_to_id`
2. `index_issues_on_moved_to_id`
3. `index_issues_on_promoted_to_epic_id`
4. `index_issues_on_last_edited_by_id`
5. `index_issues_on_description_trigram_non_latin`

In addition to that, we found that we can remove the following indexes:

1. `idx_issues_on_health_status_not_null`
2. `index_issues_on_project_health_status_asc_work_item_type`
3. `index_issues_on_project_id_and_external_key`

We're also exploring more unused indexes that could be removed: https://gitlab.com/gitlab-org/gitlab/-/issues/557718.

For the `traversal_ids`, we need four more indexes (three for `traversal_ids[0]` sorting options, and one for `traversal_ids`). This means it is a net negative of 4 indexes on the table.

## Why not partition the issues table?

Even after decomposing the table, the table size of `issues` is still larger than 100GB, which is our limit for large tables.
So we also looked into partitioning the `issues` table, but the [downsides we found in the last research still remain](/handbook/engineering/data-engineering/database-excellence/database-frameworks/doc/issue-group-search-partitioning/). Assuming we'd partition by `root_namespace_id`:

- Access patterns do not always match the partition key (e.g., assigned work items to a user, or linked work items across hierarchy)
- Self-Managed instances have fewer root_namespace_ids compared to GitLab.com
- With cells, the table size of `issues` will become smaller
- We can't gradually roll out partitioning of the table

### Implementation plan

Both plans can happen in parallel:

#### Table Decomposition

1. Create `work_item_descriptions` and `work_item_transitions` tables
2. Build dual-write system for issues using triggers
3. Backfill existing issues descriptions to `work_item_descriptions` and `work_item_transitions` tables
4. Wait until the next release with a required stop
5. Update application code to query from the new table
6. Update application code to only write to the new table
7. Remove triggers and delete the columns

#### Traversal ID Implementation

1. Add `traversal_ids` column to `issues` table
2. Implement setting traversal IDs when creating a new issue or when an issue gets moved from one namespace to another
3. Backfill traversal IDs for all existing issues
4. Wait until the next release with a required stop
5. Create required indexes
6. Update query patterns to use traversal IDs
