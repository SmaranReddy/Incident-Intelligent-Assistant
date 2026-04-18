---
title: "Reduce the growth rate of pipeline data"
status: ongoing
creation-date: "2024-05-27"
authors: [ "@fabiopitino", "@mbobin" ]
coach: [ "@fabiopitino", "@grzesiek" ]
approvers: [ "@jreporter", "@cheryl.li" ]
owning-stage: "~devops::verify"
toc_hide: true
---

{{< engineering/design-document-header >}}

## Problem to solve

CI/CD pipeline data is one of the fastest-growing datasets in GitLab. The `p_ci_builds_metadata` table significantly contributes to this growth because each pipeline execution creates duplicate job metadata across builds, even when jobs have identical configurations.

This leads to:

- **Excessive storage consumption**: We store the same job configuration repeatedly for every build
- **Database performance degradation**: Large tables impact query performance and maintenance
- **Scalability challenges**: Managing billions of rows with duplicate information becomes difficult

The primary challenge is that job metadata (options, variables, secrets, etc.) is duplicated for every build execution, even when the job configuration hasn't changed between pipeline runs.

## Job definition deduplication

During the [Rapid Action](https://gitlab.com/groups/gitlab-org/-/epics/16520) initiative, we implemented a normalization strategy to deduplicate job metadata by introducing two new tables that separate job configuration from job execution data.

This separation distinguishes between two types of data:

- **Intrinsic data**: Information needed to display high-level pipeline and job details after archival, when jobs can no longer be retried. The pipeline in this stage remains read-only. Examples include job name, stage, status, and basic configuration needed for display purposes.
- **Processing data**: Information necessary for running and retrying jobs. This includes job execution `options`, `variables`, `secrets`, etc. This data is specific to each job execution and can be safely dropped once a job becomes un-runnable (when the pipeline is archived and can no longer be retried).

The job definitions architecture deduplicates processing data while it's still needed (during the retryable period), achieving approximately **90% deduplication rate** as most jobs run repeatedly with identical configurations. Once jobs become un-runnable, this processing data can be safely removed, significantly reducing long-term storage requirements.

### Architecture

The solution consists of two main components:

#### 1. Job definitions table (`p_ci_job_definitions`)

This table stores unique, immutable job configurations that can be shared across multiple builds.

**Key characteristics:**

- **Deduplication via checksum**: Each job configuration is hashed using SHA256 to create a unique checksum
- **Immutable records**: Once created, job definitions are read-only to ensure data integrity
- **Partitioned design**: Uses list-based partitioning by `partition_id` for scalability
- **Normalized columns**: Frequently queried attributes like `interruptible` are stored as indexed columns for performance

**Schema:**

```ruby
create_table(:p_ci_job_definitions, primary_key: [:id, :partition_id]) do |t|
  t.bigserial :id, null: false
  t.bigint :partition_id, null: false
  t.bigint :project_id, null: false
  t.timestamps_with_timezone null: false
  t.boolean :interruptible, default: false, null: false, index: true
  t.binary :checksum, null: false
  t.jsonb :config, default: {}, null: false

  t.index [:project_id, :checksum, :partition_id], unique: true
end
```

Configuration attributes stored:

- `options`: Job execution options (cache, artifacts, etc.)
- `yaml_variables`: Variables defined in .gitlab-ci.yml
- `id_tokens`: JWT tokens configuration
- `secrets`: External secrets configuration
- `interruptible`: Whether the job can be interrupted
- `tag_list`: Job tags
- `run_steps`: Execution configs for the functions

#### 2. Job definition instances table (`p_ci_job_definition_instances`)

This join table links builds to their job definitions without adding indexes to the already heavily-indexed `p_ci_builds` table.

**Why a separate join table?**

The `p_ci_builds` table has reached PostgreSQL's practical limit for indexes. Adding a foreign key to `job_definition_id` would require additional indexes, which would degrade write performance further and slow down index maintenance operations.

By using a separate join table, we can:

- Maintain the relationship without impacting `p_ci_builds` performance
- Efficiently truncate partitions to remove old associations after the pipelines are no longer retryable
- Add necessary indexes on the join table without affecting the builds table

**Schema:**

```ruby
create_table(:p_ci_job_definition_instances, primary_key: [:job_id, :partition_id]) do |t|
  t.bigint :job_id, null: false
  t.bigint :job_definition_id, null: false, index: true
  t.bigint :partition_id, null: false
  t.bigint :project_id, null: false, index: true
end
```

#### How deduplication works

1. **Job creation**: When a pipeline is created, job configurations are extracted
1. **Checksum calculation**: Configuration is serialized to JSON and hashed with SHA256
1. **Lookup or create**:
    - Query `p_ci_job_definitions` for existing definition with matching `project_id`, `checksum`, and `partition_id`
    - If found, reuse the existing definition
    - If not found, create a new job definition record
1. **Link creation**: Create a record in `p_ci_job_definition_instances` linking the build to the job definition
1. **Build execution**: Jobs reference their configuration through the join table

**Legacy data handling:**

When retrying jobs that still use the legacy data format (stored in `p_ci_builds_metadata`), job definitions are automatically created on-the-fly. This ensures backward compatibility during the migration period and allows all jobs to benefit from the deduplication architecture regardless of when they were originally created.

#### Data life cycle and cleanup

One of the key benefits of this architecture is efficient data removal:

- When a partition becomes old enough for archival, we can simply truncate the `p_ci_job_definition_instances` partition
- This immediately removes all associations between builds and job definitions for that partition
- Job definitions in `p_ci_job_definitions` can then be cleaned up if no longer referenced
- This is much faster than deleting individual rows, doesn't create database bloat and returns the disk space immediately.

**Orphaned definition cleanup:**

Job definitions that are no longer referenced by any build can be identified and removed. With [pipeline archival](https://gitlab.com/groups/gitlab-org/-/epics/19547) enabled this happens naturally as old jobs fell out of the retry window and the job definitions partitions can be truncated.

#### Migration strategy

The migration to this new architecture is being rolled out gradually:

1. New tables created in GitLab 18.3
1. New pipelines write to both old and new destinations
1. Gradually shift reads from old to new tables via feature flags with fallback on the old tables when missing
1. Existing data migrated from `p_ci_builds_metadata`, `p_ci_builds`, `p_ci_build_tags`, `p_ci_builds_execution_configs` to the other tables
1. Remove old columns and tables once migration is complete
1. [Repack `p_ci_builds` partitions](https://gitlab.com/groups/gitlab-org/-/epics/19286) to recover table bloat

The background migration ([issue #552069](https://gitlab.com/gitlab-org/gitlab/-/issues/552069)) moves data to multiple destination tables as described below:

| Source Table | Target Table | Columns | Purpose |
|--------------|--------------|---------|----------|
| `p_ci_builds_metadata` | `p_ci_job_definitions` | • `config_options`<br>• `config_variables`<br>• `id_tokens`<br>• `secrets`<br>• `interruptible` | Job configuration data |
| `p_ci_builds` | `p_ci_job_definitions` | • `options`<br>• `yaml_variables` | Legacy job configuration data |
| `p_ci_build_tags` | `p_ci_job_definitions` | • `tag_list` | Job tags |
| `p_ci_build_execution_configs` | `p_ci_job_definitions` | • `run_steps` | Execution steps |
| `p_ci_builds_metadata` | `p_ci_job_definition_instances` | *(relationship)* | Links builds to definitions |
| `p_ci_builds_metadata` | `p_ci_builds` | • `scoped_user_id`<br>• `timeout`<br>• `timeout_source`<br>• `exit_code`<br>• `debug_trace_enabled` | Build execution data |
| `p_ci_builds_metadata` | `p_ci_job_artifacts` | • `exposed_as`<br>• `exposed_paths` | Artifact configuration |
| `p_ci_builds_metadata` | `job_environments` | • `expanded_environment_name`<br>• `environment options` | Environment settings |

The migration was split into multiple jobs, one for each table partition. This takes advantage of parallel execution and we will be able to truncate the metadata partitions as soon as each migration job is complete.

## Alternative approaches considered

### Pipeline blueprint approach (not implemented)

Before implementing job-level deduplication, we considered a pipeline-level blueprint approach (detailed in [MR !11967](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/11967)). This approach would have introduced a `p_ci_pipeline_blueprints` table storing entire pipeline configurations.

**Key differences from implemented solution:**

- **Granularity**: Pipeline-level vs. job-level deduplication
- **Storage format**: Single JSON file per pipeline vs. normalized job definitions
- **Deduplication trigger**: Any job change creates new blueprint vs. only changed jobs create new definitions
- **Data access**: File download/caching required vs. direct database queries

**Why job-level deduplication was chosen:**

1. **Better deduplication ratio**: Jobs often remain stable even when other jobs in the pipeline change
2. **Simpler data access**: No need for file caching or object storage integration
3. **Incremental migration**: Easier to migrate existing data job-by-job
4. **Partition management**: Maintains top-down hierarchy for partition rebalancing
5. **Query performance**: Normalized structure supports efficient queries without file parsing

**Pipeline blueprint drawbacks:**

- Any single job change would create an entirely new blueprint for the whole pipeline
- Would require Redis caching layer to avoid downloading large JSON files repeatedly
- More complex migration path from existing metadata structure
- Job-level blueprints would have created circular dependencies in partition hierarchy

The pipeline blueprint approach is documented in detail in the [closed MR !11967](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/11967/diffs) for future reference.

### Delete pipeline processing data

Once a build gets archived, it is no longer possible to resume
pipeline processing in such pipeline. It means that all the metadata, we store
in PostgreSQL, that is needed to efficiently and reliably process builds can be
safely moved to a different data store.

Storing pipeline processing data is expensive as this kind of CI/CD
data represents a significant portion of data stored in CI/CD tables. Once we
restrict access to processing archived pipelines, we can move this metadata to
a different place - preferably object storage - and make it accessible on
demand, when it is really needed again (for example for compliance or auditing purposes).

We need to evaluate whether moving data is the most optimal solution. We might
be able to use de-duplication of metadata entries and other normalization
strategies to consume less storage while retaining ability to query this
dataset. Technical evaluation will be required to find the best solution here.

## Results and impact

After 1 week from introducing the new `p_ci_job_definitions` table, we achieved approximately **85% deduplication** of `p_ci_builds_metadata` records.

The chart below shows the growth rate of the affected CI tables from the moment we enabled writes to `p_ci_job_definitions`:

![CI partitions growth rate comparison](/images/engineering/architecture/design-documents/ci_data_decay/metadata_growth_graph_comparison.png)

This dramatic reduction in growth rate demonstrates the effectiveness of the job definition deduplication strategy. By normalizing job configurations and reusing them across multiple builds, we've successfully addressed one of the primary contributors to CI database growth while maintaining full functionality for pipeline execution and historical data access.

## When is deduplication worth the effort?

Deduplicating data is most effective when large portions of data remain constant between records--such as jobs that do not change from pipeline to pipeline. Consider the following data criteria when deciding if this pattern is suitable for your use-case:

- **Immutable across executions:** The configuration does not vary from one execution to another.
- **Shared by many records:** The same configuration is duplicated frequently.
- **Logically separable:** The data represents a coherent concept that can reasonably live in its own table without complicating the main model.

If the configuration changes often or is tightly coupled to per-run state, this pattern is unlikely to provide meaningful benefit. Teams should weigh query performance impact, refactoring effort, and migration complexity before deciding to adopt it.

### Query performance considerations

Moving data into a separate table affects how queries must be written and optimized. Consider:

- **More complex query patterns:** Filtering or sorting moved columns may require additional JOINs or alternative query strategies (e.g. `IN`/`EXISTS` clauses, subqueries, or CTEs). These add structural complexity and can increase overall query cost on large datasets.
- **Index effectiveness:** Cross-table sorts or filters may no longer benefit from single-table indexes. Some queries become less index-friendly when attributes are split across tables.
- **Index-related overhead:**
  - *Non-indexed columns:* Lower cost to migrate because queries that rely on them were not index-optimized to begin with.
  - *Indexed columns:* High cost to migrate; indexes may need to be redesigned (e.g., removing the moved column, adding a foreign key column, or restructuring existing composite indexes). For large tables, index rebuilds or significant schema churn can create long-running operations or require complex rollout strategies.

### Refactoring and migration complexity

Adopting this pattern typically requires a structured transition plan. Evaluate:

- **Migration path strategy:**
  - *Dual-read period:* Application reads from both old and new tables until migration is complete. This increases complexity and requires careful feature flag placements or fallbacks.
  - *Full backfill first:* Migrate all existing data upfront, then switch all reads to the new table at once. Simpler long-term but potentially riskier and more resource-intensive.
- **Dependencies on existing data structures:** Features, services, or reporting queries may rely heavily on the current schema. The more coupling exists, the greater the refactor risk.
- **Operational considerations:** Large data migrations and index updates must be planned to minimize impact on production workloads.

Epic: [Reduce the rate of builds metadata table growth](https://gitlab.com/groups/gitlab-org/-/epics/7434).
