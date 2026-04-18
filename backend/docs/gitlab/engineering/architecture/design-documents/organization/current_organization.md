---
title: "Current Organization"
owning-stage: "~devops::tenant scale"
group: Organizations
toc_hide: true
---

We will ensure that the current organization id is defined on every entry point. These entry points include web requests, background jobs, and scheduled tasks.

## Web Requests

Including rails controllers, Grape API, GraphQL, ActionCable.

For web requests the current organization will be determined in the following order of precedence with details listed in sub-sections below:

1. Path params. E.g. `/o/my-org/my-group`, `/my-group/my-project`
1. Header field (`X-GitLab-Organization-ID`)
1. Users Home Organization (`user.organization`)
1. The default organization (ID = 1).

### Path Params

The current Organization will be the parent Organization for the resource requested.
The Organization is prefixed to paths using the Organization scope `o/` such as
`https://gitlab.com/o/my-org/my-group/my-project/-/issues/1234`. Details on this
decisions are in the [Organization path scope ADR](decisions/004_path_scope.md).

For example:

- `/o/my-organization/my-group`: The Organization will be `my-organization`
- `/o/my-organization/engineering/backend`: The organization will be `my-organization` since the project `engineering/backend` is scoped.
- `/top-level-group/my-project`: The organization will be `my-organization` if the `top-level-group` was moved into the `my-organization`.

### Header Field

- Organization context included in `X-GitLab-Organization-ID` header
- Frontend JavaScript automatically includes the context in AJAX/API calls
- Maintains consistent context during interactive sessions

### Users Home Organization

Every user belongs to one Organization. This organization is known as the Home Organization.

Used as a fallback for any ambiguous authenticated request.

### The Default Organization

This is a fallback for unauthenticated requests not handled by the higher precedent mappings.

Use of the Default Organization will break Cells compatibility because the Default Organization is only available on one Cell.

## API Requests

REST and GraphQL requests will remain at `/api/v4` and `/api/graphql`.

You must specify the Organization context for each request with either:

- An `X-GitLab-Organization-ID` header
- The organization associated with the used secret (personal access token, etc.)

With no parameter a fallback to the Default Organization will occur.

## Background Jobs

Jobs will be enqueued with the current organization context and automatically applied during execution.

For example:

```ruby
# Sidekiq client middleware - preserves context when enqueuing
class OrganizationContextClientMiddleware
  def call(worker_class, job, queue, redis_pool)
    # Capture current organization context
    job['organization_id'] = Current.organization_id
    yield
  end
end

# Sidekiq server middleware - restores context during processing
class OrganizationContextServerMiddleware
  def call(worker, job, queue)
    organization_id = job['organization_id']
    raise "Missing organization context" unless organization_id

    # Set thread-local context that query executor will use
    Current.organization_id = organization_id
    yield
  ensure
    Current.organization_id = nil
  end
end
```

## Scheduled Tasks

Cron jobs will be required to define a current Organization. There will be provision to skip this requirement, but it will be considered an exceptional circumstance rather than typical.

## Exceptions

The Current Organization will be defined as a kind of global execution context. There is an expectation that Current Organization is always defined, however there are recognized circumstances where this is not possible or desirable.

Admin Operations
: System-wide admin tasks will use explicit context exemption blocks

Scheduled Tasks
: Cron jobs will likely initalize the job without a Current Organization and then define Current Organization per Organization batch.

An example exception may look like:

```ruby
class DropPendingBuildsWorker
  def perform
    # Process each organization's builds with proper context
    organizations_with_pending_builds.each do |org_id|
      # Set thread-local context for automatic query wrapping
      Current.organization_id = org_id
      process_builds_for_organization(org_id)
    end
  ensure
    Current.organization_id = nil
  end

  def organizations_with_pending_builds
    Organizations::Organization.without_current_organization do
      # make some query
    end
  end
end
```
