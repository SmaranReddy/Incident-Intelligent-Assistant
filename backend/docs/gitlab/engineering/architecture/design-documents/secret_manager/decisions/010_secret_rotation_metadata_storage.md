---
owning-stage: "~sec::software supply chain security"
title: 'GitLab Secrets Manager ADR 010: Using Rails ActiveRecord for Secret Rotation Metadata'
toc_hide: true
---

### Context

Background jobs need to efficiently identify secrets requiring rotation reminders across all projects.

Our design goals are:

1. Support efficient queries for daily/hourly background job processing.
2. Enable both standard intervals (30/60/90 days) and custom cron schedules.
3. Track compliance status for overdue rotations.
4. Keep OpenBao as the single source of truth for secrets.

### Decision

We will use PostgreSQL to store rotation scheduling metadata while OpenBao remains the single source of truth for secret values. This enables background jobs to efficiently query across all projects to identify secrets needing rotation reminders, supporting both standard intervals and custom cron-based schedules. The trade-off of potential data synchronization issues is acceptable because we're only storing rotation metadata, not duplicating secret values or access controls.
We intentionally deviate from ADR 008's principle of avoiding database storage.

### Implementation Details

#### Database Schema

```sql
CREATE TABLE rotation_infos (
  id SERIAL PRIMARY KEY,
  project_id INTEGER NOT NULL REFERENCES projects(id),
  secret_name VARCHAR NOT NULL,

  -- Rotation scheduling
  rotation_interval_days INTEGER,  -- Computed days for standard intervals (30, 60, 90)
  rotation_interval_raw_value VARCHAR NOT NULL,  -- Original input: "30", "60", "90", or cron syntax
  next_reminder_at TIMESTAMP NOT NULL,

  -- Notification tracking
  last_reminder_sent_at TIMESTAMP,

  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

-- Indexes for performance
CREATE UNIQUE INDEX idx_rotation_infos_project_secret ON rotation_infos(project_id, secret_name);
CREATE INDEX idx_rotation_infos_reminder_at ON rotation_infos(next_reminder_at);
```

#### Data Consistency Strategy

We use a **"rotation-first, verify-later"** approach to avoid critical failures:

**Creation Flow:**

1. Create rotation record when user configures rotation (before secret exists)
2. Create secret in OpenBao using the same deterministic path

**Failure Scenarios:**

- **Prevented:** Secret exists without rotation tracking → Cannot happen since rotation records are created first
- **Handled:** Rotation record exists without secret → Background job detects and cleans up orphaned records
- **Handled:** Secret deleted outside Rails → Background job detects missing secret and removes rotation record

This strategy ensures we never miss rotation reminders for secrets that need them, while gracefully handling orphaned records.

### Background Job Processing

The following sample Ruby code illustrates how a background job would efficiently process rotation reminders (actual production implementation may differ):

```ruby
class SecretRotationReminderWorker
  def perform
    # Single database query to find all secrets due for rotation
    RotationInfo
      .where('next_reminder_at <= ?', Time.current)
      .find_in_batches(batch_size: 1000) do |batch|

      batch.each do |rotation_info|
        # Verify secret exists in OpenBao before sending reminder
        if secret_exists_in_vault?(rotation_info)
          send_rotation_reminder(rotation_info)
          rotation_info.update!(
            last_reminder_sent_at: Time.current,
            next_reminder_at: calculate_next_reminder(rotation_info)
          )
        else
          # Handle orphaned rotation records
          rotation_info.destroy
        end
      end
    end
  end
end
```

This approach enables:

- Efficient batch processing across all projects in minutes instead of hours
- Lazy validation of secret existence only when needed
- Automatic cleanup of orphaned rotation records

#### Future scaling consideration

If we encounter performance issues with hundreds of thousands of secrets across thousands of projects, we can partition the background job:

- Split into one worker per project (or group of projects)
- Workers run in parallel, each querying only their assigned projects
- Enables horizontal scaling without changing the core design

This partitioning strategy would be straightforward to implement since our schema already indexes by `project_id`.

### Alternative approach: OpenBao metadata storage

We evaluated storing all rotation attributes in OpenBao secret metadata ([Option 1 from #547863](https://gitlab.com/gitlab-org/gitlab/-/issues/547863)):

```ruby
# Rejected approach - requires constant OpenBao queries even when no rotations are due
class RotationCheckerWithOpenBao
  def perform
    Project.find_each do |project|
      # LIST operation for every project, every hour
      secrets = vault_client.logical.list("#{project.path}/secret/metadata")
      next if secrets.blank?

      secrets.each do |secret|
        # READ operation for each secret to check rotation date
        metadata = vault_client.logical.read("#{project.path}/secret/metadata/#{secret}")
        rotation_date = metadata.data[:custom_metadata][:next_rotation_at]

        if Time.parse(rotation_date) <= Time.current
          send_rotation_reminder(project, secret)
        end
      end
    end
  end
end
```

This approach was rejected because:

- **Constant API load**: Background jobs running hourly must query OpenBao even when no secrets need rotation.
- **Performance**: For 10,000 projects with 100 secrets each, this requires 1,000,000 API calls every hour.
- **No early termination**: Cannot quickly determine if any reminders need to be sent.
- **No query filtering**: Does not support time-based queries to efficiently identify secrets that need to be rotated. This would require us to check every secret.

### References

- [Issue #547863](https://gitlab.com/gitlab-org/gitlab/-/issues/547863): Technical Evaluation for Storage Architecture
- [Issue #499945](https://gitlab.com/gitlab-org/gitlab/-/issues/499945): Secret Rotation API Implementation
- [ADR 008](/handbook/engineering/architecture/design-documents/secret_manager/decisions/008_no_database/): No Database Storage for Secrets
