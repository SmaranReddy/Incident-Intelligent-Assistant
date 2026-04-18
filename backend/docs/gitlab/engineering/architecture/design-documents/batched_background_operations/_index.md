---
title: Batched Background operations
status: proposed
creation-date: "2025-07-16"
authors: [ "@praba.m7n", "@morefice" ]
coaches: [ "@DylanGriffith", "@ahegyi" ]
dris: [ "@alexives" ]
owning-stage: "~devops::data access"
participating-stages: []
toc_hide: true
---

<!-- Design Documents often contain forward-looking statements -->
<!-- vale gitlab.FutureTense = NO -->

<!-- This renders the design document header on the detail page, so don't remove it-->
{{< engineering/design-document-header >}}

## Summary

Create a framework to do Batched Background data Operations (BBO). Data operations targeted on a large set of records should be (sub) batched
and performed in a safe manner.

There are two types of operations.

1. Cell local, it will be performed application wide and mostly triggered by the system (eg: cronjobs).
2. Organization specific, will be specific to the user and mostly triggered by user actions.

The BBOs should have enough logging and monitoring system to track the progress of the operations.

## Motivation

We tend to do large data operations from async workers and custom migrations, mostly by batching over the records.
Such operations can result in a degraded database performance, as it touches huge number
of records without proper measures in-place.

To be more safe, sidekiq workers currently can

1. Have its own batching logic to break down the records and then process them.
2. Defer on bad database health signals, using [defer_on_database_health_signal](https://docs.gitlab.com/development/sidekiq/#deferring-sidekiq-workers).
3. Sleep between batches to give some breathing space.
4. Have guard-rails to avoid running beyond the set time.

But then we can not easily track the progress of these (large) operations at a given time, and the above options can vary between workers and there is not a clear
structure as of now.

[BBM (Batched Background Migrations)](https://docs.gitlab.com/development/database/batched_background_migrations/) has in-built [throttling](https://docs.gitlab.com/development/database/batched_background_migrations/#throttling-batched-migrations), [optimal batching](https://gitlab.com/gitlab-org/gitlab/blob/9eab5b3eb225897bc6a00464f29137f8d0392d94/lib/gitlab/database/background_migration/batched_migration.rb#L277)
and [retry](https://docs.gitlab.com/development/database/batched_background_migrations/#job-retry-mechanism) mechanisms.
But they can only be queued from regular rails migrations and it needs
manual interventions to `finalize` them to ensure its run in self-managed instances as well.

Since BBM is a matured framework used widely for many large data operations, we can make use of its functionalities in BBO framework too.
So that workers doing large operations will be batched optimally and gets deferred automatically.
This would allow us to share core parts between BBM and BBO, and any future
developments (eg: adding more db health check indicator) will be applied to both.

Similar to BBM, `batched_background_operations` will have uniqueness check on `job_class` and `job_arguments`,
it makes sure duplicate operations are not created for the same purpose until the existing one is finished (or failed).

### Goals and proposal

We do not want to mix BBO with BBM framework (underlying tables/modules) for following reasons:

1. BBM related tables have details not useful for Batched Background Operations (BBOs), example: `batched_background_migrations.queued_migration_version` column and other
   migration specific tools
2. To [avoid STI](https://docs.gitlab.com/development/database/single_table_inheritance).
3. To not intervene logic in BBM framework to accommodate BBOs and then hide them from Admin UIs and other places.
4. BBMs are not triggered by user action (organization specific), so the growth of the tables tend to be in control unlike BBOs.

So we want to develop a new framework for batched background data operations by (re)using BBM modules.

#### Cell compatibility

`batched_background_operations` will have `organization_id` (sharding key) and `user_id` attributes which will be populated for organization-specific BBOs,
and those will be _NULL_ for cell-local (eg: `inactive_projects_deletion_cron_worker`).

So on moving an organization, organization-specific BBOs will be moved and the cell-local ones don't have to be moved since they will be the same across cells.
And as they are created from cronjobs, organizations moved to the cell will eventually get processed by them (in the further runs).

### Use cases

As the aim is to have right sized batches and halt BBOs on bad database health signal, `low/medium urgency` workers are better candidates than the `high` ones.
Ideally we should be able to adopt BBO for any worker that batches through the database records and perform some action which will get eventually completed.

Below are few existing scenarios which can be converted/adopted to use BBO.

#### Projects::InactiveProjectsDeletionCronWorker (source: CronJob)

`inactive_projects_deletion_cron_worker` cron runs every 10 minutes which calls `Projects::InactiveProjectsDeletionCronWorker`,
it batches over 100 projects, with a runtime limiter of 4 minutes. On lapsing the time, it caches the last ID (cursor),
which will be used for selection in the next run.

**Current state:**

```ruby
Settings.cron_jobs['inactive_projects_deletion_cron_worker'] ||= {}
Settings.cron_jobs['inactive_projects_deletion_cron_worker']['cron'] ||= '*/10 * * * *'
Settings.cron_jobs['inactive_projects_deletion_cron_worker']['job_class'] = 'Projects::InactiveProjectsDeletionCronWorker'
```

```ruby
def perform
  project_id = last_processed_project_id

  Project.where('projects.id > ?', project_id).each_batch(of: 100) do |batch|
    inactive_projects = batch.inactive.not_aimed_for_deletion

    inactive_projects.each do |project|
      if over_time?
        save_last_processed_project_id(project.id)
        raise TimeoutError
      end

      with_context(project: project, user: admin_bot) do
        deletion_warning_email_sent_on = notified_inactive_projects["project:#{project.id}"]

        if deletion_warning_email_sent_on.blank?
          send_notification(project)
          log_audit_event(project, admin_bot)
        elsif grace_period_is_over?(deletion_warning_email_sent_on)
          Gitlab::DormantProjectsDeletionWarningTracker.new(project.id).reset
          delete_project(project, admin_bot)
        end
      end
    end
  end
  reset_last_processed_project_id
rescue TimeoutError
  # no-op
end

def save_last_processed_project_id(project_id)
  with_redis do |redis|
    redis.set(LAST_PROCESSED_INACTIVE_PROJECT_REDIS_KEY, project_id)
  end
end


def reset_last_processed_project_id
  with_redis do |redis|
    redis.del(LAST_PROCESSED_INACTIVE_PROJECT_REDIS_KEY)
  end
end
```

**With BBO:**

Below options can be used to enqueue new BBO from the cronjobs.

_Option 1 (Using Worker):_

```ruby
Settings.cron_jobs['inactive_projects_deletion_cron_worker'] ||= {}
Settings.cron_jobs['inactive_projects_deletion_cron_worker']['cron'] ||= '*/10 * * * *'
Settings.cron_jobs['inactive_projects_deletion_cron_worker']['job_class'] = 'EnqueueBatchedBackgroundOperationWorker'
Settings.cron_jobs['inactive_projects_deletion_cron_worker']['job_arguments'] = { bbo_job_class_name: 'Projects::InactiveProjectsDeletionCronWorker', min_cursor: [<last_processed_project_id>] }
```

`EnqueueBatchedBackgroundOperationWorker` will create BBOs using the _job_arguments_ sent along.

_Option 2 (Using Sidekiq server middleware):_

```ruby
Settings.cron_jobs['inactive_projects_deletion_cron_worker'] ||= {}
Settings.cron_jobs['inactive_projects_deletion_cron_worker']['cron'] ||= '*/10 * * * *'
Settings.cron_jobs['inactive_projects_deletion_cron_worker']['job_class'] = 'Projects::InactiveProjectsDeletionCronWorker'
Settings.cron_jobs['inactive_projects_deletion_cron_worker']['job_arguments'] = { min_cursor: [<last_processed_project_id>] }
Settings.cron_jobs['inactive_projects_deletion_cron_worker']['batched_operation'] = true
```

As the above cronjobs are setup from the app initializer, `job_arguments.min_cursor` can't be static, can be fetched from the cache store.

A sidekiq server middleware will interpret jobs with `{ batched_operation: true }` and create new `batched_background_operations` using the _job_class_ and _job_arguments_.

`Option 1` is more intuitive but the decision will be taken during the development after considering the tradeoffs.

```ruby
module Gitlab
  module BackgroundOperation
    # BatchedOperationJob will have the sub-batch iteration logic, similar to BBM
    class Project::DeleteInactiveProjects < Gitlab::Database::BatchedOperationJob
      feature_category <feature_category>

      # 'last_processed_project_id' from current design can be set as the min_cursor
      cursor(:id)

      def perform
        each_sub_batch do |sub_batch|
          inactive_projects = sub_batch.inactive.not_aimed_for_deletion

          inactive_projects.each do |project|
            with_context(project: project, user: admin_bot) do
              deletion_warning_email_sent_on = notified_inactive_projects["project:#{project.id}"]

              if deletion_warning_email_sent_on.blank?
                send_notification(project)
                log_audit_event(project, admin_bot)
              elsif grace_period_is_over?(deletion_warning_email_sent_on)
                Gitlab::DormantProjectsDeletionWarningTracker.new(project.id).reset
                delete_project(project, admin_bot)
              end
            end
          end
        end
      end
    end
  end
end
```

#### Todos::DeleteAllDoneWorker (source: User Action)

GraphQL API is exposed to use this worker for the current user, with _delete_until_ param set by default to the current time.

**Current state:**

```ruby
module Mutations
  module Todos
    class DeleteAllDone < ::Mutations::BaseMutation
      def resolve(updated_before: nil)
        delete_until = (updated_before || Time.now).utc.to_datetime.to_s

        ::Todos::DeleteAllDoneWorker.perform_async(current_user.id, delete_until)

        {
          message: format(_('Your request has succeeded. Results will be visible in a couple of minutes.')),
          errors: []
        }
      end
    end
  end
end
```

```ruby
module Todos
  class DeleteAllDoneWorker
    include ApplicationWorker
    include EachBatch

    LOCK_TIMEOUT = 1.hour
    BATCH_DELETE_SIZE = 10_000
    SUB_BATCH_DELETE_SIZE = 100
    SLEEP_INTERVAL = 100
    MAX_RUNTIME = 2.minutes

    def perform(user_id, time)
      runtime_limiter = Gitlab::Metrics::RuntimeLimiter.new(MAX_RUNTIME)
      delete_until = time.to_datetime
      pause_ms = SLEEP_INTERVAL

      in_lock("#{self.class.name.underscore}_#{user_id}", ttl: LOCK_TIMEOUT, retries: 0) do
        Todo.where(user_id: user_id)
            .with_state(:done)
            .each_batch(of: BATCH_DELETE_SIZE) do |batch|
              batch.each_batch(of: SUB_BATCH_DELETE_SIZE) do |sub_batch|
                sql = <<~SQL
                      WITH sub_batch AS MATERIALIZED (
                        #{sub_batch.select(:id, :updated_at).limit(SUB_BATCH_DELETE_SIZE).to_sql}
                      ), filtered_relation AS MATERIALIZED (
                      SELECT id FROM sub_batch WHERE updated_at < '#{delete_until.to_fs(:db)}' LIMIT #{SUB_BATCH_DELETE_SIZE}
                    )
                    DELETE FROM todos WHERE id IN (SELECT id FROM filtered_relation)
                    SQL

                Todo.connection.exec_query(sql)

                sleep(pause_ms * 0.001) # Avoid hitting the database too hard
              end

              next unless runtime_limiter.over_time?

              self.class.perform_in(MAX_RUNTIME, user_id, time)

              break
            end
      end
    end
  end
end
```

NOTES:

1. `in_lock` is used with 1 hour TTL for deleting the DoneTodos of the same user.
2. `10_000` is used as the outer batch and `100` as the inner sub batch.
3. Sleeps in between sub-batches.
4. 2 minutes runtime limiter is used, post that it requeues the same job after 2 minutes.

**With BBO:**

```ruby
module Mutations
  module Todos
    class DeleteAllDone < ::Mutations::BaseMutation
      def resolve(updated_before: nil)
        delete_until = (updated_before || Time.now).utc.to_datetime.to_s

        queue_batched_background_operation(
          'Gitlab::Database::BackgroundOperation::DeleteAllDoneTodos',
          :todos,
          :id,
          user: current_user,
          job_arguments: {
            delete_until: delete_until
          }
        )

        {
          message: format(_('Your request has succeeded. Results will be visible in a couple of minutes.')),
          errors: []
        }
      end
    end
  end
end
```

```ruby
def queue_batched_background_operation(job_class_name, table_name, column_name, user: nil, job_arguments: {})
  user_specific_args = user.present? ? { user_id: user.id, organization_id: user.organization_id } : {}

  # Having only crucial args for simplicity, it will also assign other attrs during the development (eg: batch_size, sub_batch_size, cursors, etc.,)
  Gitlab::Database::BatchedBackgroundOperation.create!({
    job_class_name: job_class_name,
    table_name: table_name,
    column_name: column_name,
    job_arguments: job_arguments
  }.merge(user_specific_args))
end
```

```ruby
module Gitlab
  module BackgroundOperation
    # BatchedOperationJob will have the sub-batch iteration logic, similar to BBM
    class DeleteAllDoneTodos < BatchedOperationJob
      feature_category <feature_category>

      # rubocop:disable Database/AvoidScopeTo -- supporting index: index_todos_on_user_id_and_id_done ON todos USING btree (user_id, id) WHERE ((state)::text = 'done'::text);
      scope_to ->(relation) { relation.with_state(:done).where("user_id = ?", user_id) }

      cursor(:id)

      def perform
        each_sub_batch do |sub_batch|
          sub_batch.where("updated_at < ?", delete_until.to_fs(:db))
                   .delete_all
        end
      end
      # rubocop:enable Database/AvoidScopeTo
    end
  end
end
```

BBO framework will fetch scoped ToDos and create sub-batches with optimal ranges, with enough interval between them and
the entire operation will be paused/resumed based on the db health check indicator's statuses.

**Similar use cases:**

Below are few more workers which can be migrated to BBOs in a similar fashion.

- AdjournedProjectsDeletionCronWorker
- MemberInvitationReminderEmailsWorker
- Users::UnconfirmedSecondaryEmailsDeletionCronWorker
- Analytics::CycleAnalytics::ConsistencyWorker
- Analytics::ValueStreamDashboard::CountWorker
- Packages::Cleanup::DeleteOrphanedDependenciesWorker
- Vulnerabilities::DeleteExpiredExportsWorker
- ClickHouse::SyncStrategies::BaseSyncStrategy
- Gitlab::Counters::FlushStaleCounterIncrementsWorker
- LooseForeignKeys::CleanupWorker

#### Tracking large data operations

Unlike above use cases, there can be systems built to track the progress of large data operations. BBOs can be extended
for these since it already got ways to find the progress of the operations.

**Examples:**

- AI Context Abstraction Layer
- Search indexing

### Non Goals

BBO will not be a right fit for workers

- Operating on small set of records.
- Looping through an Array (ruby object) of IDs or fetching the next ID from stored cache instead of batching through database tables.
- With `high` urgency, as BBOs can get halted.

#### Selecting records not from the database

```ruby
module Ci
  class DestroyOldPipelinesWorker
    def perform_work(*)
      Project.find_by_id(fetch_next_project_id).try do |project|
        ...
        # This pushes the next ID to the redis store
        Ci::DestroyPipelineService.new(project, nil).unsafe_execute(pipelines)
      end
    end

    def fetch_next_project_id
      Gitlab::Redis::SharedState.with do |redis|
        redis.lpop(queue_key)
      end
    end
  end
```

```ruby
module Ci
  class UnlockPipelinesInQueueWorker
    def perform_work(*_)
      # `next!` fetches the next ID from cache
      pipeline_id, enqueue_timestamp = Ci::UnlockPipelineRequest.next!
      return log_extra_metadata_on_done(:remaining_pending, 0) unless pipeline_id

      Ci::Pipeline.find_by_id(pipeline_id).try do |pipeline|
        log_extra_metadata_on_done(:pipeline_id, pipeline.id)
        log_extra_metadata_on_done(:project, pipeline.project.full_path)

        result = Ci::UnlockPipelineService.new(pipeline).execute
        ...
      end
    end
  end
end
```

`Ci::DestroyOldPipelinesWorker` and `Ci::UnlockPipelinesInQueueWorker` get their next item to process from the cache,
BBO can't be adopted for them without changing these worker's nature of selection/iteration process.

#### High urgency operations

```ruby
module WorkItems
  class CopyTimelogsWorker
    include ApplicationWorker
    urgency :high

    BATCH_SIZE = 100

    def perform(from_issue_id, to_issue_id)
      ...
      ...
      from_issue = Issue.find_by_id(from_issue_id)

      reset_attributes = { project_id: to_issue.project_id, issue_id: to_issue.id }
      ApplicationRecord.transaction do
        from_issue.timelogs.each_batch(of: BATCH_SIZE) do |timelogs|
          new_timelogs_attributes = timelogs.map do |timelog|
          timelog.attributes.except('id').merge(reset_attributes)
        end

        Timelog.insert_all!(new_timelogs_attributes)
      end
    end
  end
end
```

Theoretically `WorkItems::CopyTimelogsWorker` can be done using BBO, but it is not suitable because the given issue would
not be having loads of `timelogs` records and it's an `high urgency` worker, which can (might) not endure the waiting time from BBO framework.

## Design and implementation details

Similar to `Database::BatchedBackgroundMigrationWorker` cron, `Database::BatchedBackgroundOperationWorker` will run every minute,
which will process any newly added `batched_background_operations`.

Common methods will be pulled out from modules/classes in [workers/database/batched_background_migration](https://gitlab.com/gitlab-org/gitlab/blob/676e40c4dfa0071d4931b25ddbaf1375e59baeb0/app/workers/database/batched_background_migration/)
to reuse them in BBO framework.

### Database design

#### batched_background_operations

New operations will be created in `batched_background_operations` table and it will be partitioned by `sliding_list` strategy.

Each partition will contain **7 days** of data, executable (with status active/paused) will be re-inserted before execution to ensure they always end up in the new partition and the old
partitions will eventually have only failed/finished operations, which can be detached and dropped.

Logs will capture the failed operations with error messages, would be good to have a monitoring dashboards setup for this so that the feature team can use them post deployment.

```ruby
PARTITION_DURATION = 7.days

partitioned_by :partition_id, strategy: :sliding_list,
  next_partition_if: ->(active_partition) do
    oldest_record_in_partition = BatchedBackgroundOperation
      .select(:id, :created_at)
      .for_partition(active_partition.value)
      .order(:id)
      .limit(1)
      .take

    oldest_record_in_partition.present? && oldest_record_in_partition.created_at < PARTITION_DURATION.ago
  end,
  detach_partition_if: ->(partition) do
    !BatchedBackgroundOperation
      .for_partition(partition.value)
      .executable # with status [active, paused]
      .exists?
  end
end
```

#### batched_background_operation_jobs

With BBM framework there is an ongoing performance [problem](https://gitlab.com/gitlab-org/gitlab/-/issues/434089) on deleting the _batched_background_migrations_ records, it cascades to its referenced records in
_batched_background_migration_jobs_ and _batched_background_migration_transition_logs_, for a large BBM the volume of jobs/transition logs are huge and it times out on deleting them.

To avoid this issue in BBO, as `batched_background_migration_jobs` holds the batched job information and only last job's `max_cursor` is used to create the next job, we can safely delete the operation_jobs by storing last job's max-cursor info in `batched_background_operations` table itself.

It has ['pending', 'running', 'failed', 'succeeded'] statuses, and to avoid a bulk deletes we can create a new partition `batched_background_migration_jobs_executed` to hold failed/succeeded jobs with a retention of 7 days.
So that `batched_background_operations` deletes don't have to cascade to the jobs table.

#### batched_background_operation_transition_logs

There is an ongoing [discussion](https://gitlab.com/gitlab-org/gitlab/-/issues/434089#note_2653929336) around not having this table (in BBM framework) and handle exceptions elsewhere.

We will try to avoid this additional table, if not this will be partitioned based on the created timestamp with a retention period similar to BBM framework.

##### Decision: Not storing logs in the database

In BBM, `batched_background_migration_job_transition_logs is used to store the state transition logs, but we decided not to store them in the database for background operations monitoring.

Instead, we will use other types of metrics:

- **Prometheus** for time-series metrics
- **Kibana** for log aggregation and analysis

This approach reduces maintenance overhead associated with managing large log tables in the database, including:

- Partitioning and retention policies
- Index optimization
- Data cleanup procedures
- Database growth and storage requirements

These tools are purpose-built for handling high-volume metrics and logs, providing better scalability and operational simplicity compared to database tables.

See [gitlab-org/gitlab!218794](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/218794) for the decision discussion.

### Considerations

- Preference should be given to organization specific operations than the cell local ones, to reduce the waiting time on user facing operations.
- Sidekiq worker has [data_consistency](https://docs.gitlab.com/development/sidekiq/worker_attributes/#job-data-consistency-strategies) strategies to use database load balancing.
  We might have to add this to BBO as well.
- Having FF to control pause/resume certain BBO will be helpful to avoid running buggy/destructive BBOs.
