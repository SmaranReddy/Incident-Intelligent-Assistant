---
title: Activation Engine
status: proposed
creation-date: "2025-11-18"
authors: [ "@jmontal", "@dstull" ]
coaches: [ ]
dris: [ "@jtucker_gl", "@ghosh-abhinaba", "@jmontal", "@dstull" ]
owning-stage: "~devops::growth"
participating-stages: []
toc_hide: true
---

{{< engineering/design-document-header >}}

## Summary

The Activation Engine is a foundational system designed to drive user activation
and engagement by tracking user progress through key milestones (Setup, Aha,
Habit) and orchestrating personalized experiences based on that state. This
design document outlines the initial implementation approach using a user metric
completions tracking system that provides a foundation for activation
experiments while establishing patterns for future extensibility.

The initial implementation focuses on tracking specific user actions and
milestones to enable targeted activation experiments for new users, with a
database-driven approach that allows the Growth team to iterate quickly without
dependencies on external data sources. The engine orchestrates multiple delivery
mechanisms (banners, notifications, page routing, scheduled notifications) to present
personalized experiences based on user activation state.

## Glossary

- **CQRS (Command Query Responsibility Segregation)**: An architectural pattern that separates read and write operations
  into different models, allowing independent optimization and scaling of each
- **DIP (Data Insights Platform)**: GitLab's standardized data access layer designed to provide a unified querying
  interface across multiple data sources
- **MVC (Minimum Viable Change)**: The smallest change that delivers value and can be iterated upon
- **Snowplow**: GitLab's event tracking pipeline that collects and routes analytics events
- **ClickHouse**: A columnar database management system optimized for analytical queries and high-volume data ingestion
- **Internal Events**: GitLab's unified event tracking system that routes events to multiple destinations (Snowplow,
  Redis counters, etc.)
- **Root Namespace**: The top-level namespace in GitLab's hierarchy (group or personal namespace)

## Motivation

GitLab users face several challenges when navigating and utilizing the platform effectively:

- **Information Overload**: Users are presented with too much information and too many options, making it difficult to
  focus on what matters most to them.
- **Generic Experience**: All users receive the same interface and recommendations regardless of their role, experience
  level, or usage patterns
- **Poor Discoverability**: Valuable features and capabilities remain hidden or hard to find for users who would benefit
  from them.
- **Inefficient Workflows**: Users spend time navigating to frequently used features or recreating workflows that could
  be streamlined.
- **Onboarding Friction**: New users struggle to understand how to get started and achieve their goals quickly.
- **Feature Adoption Gaps**: Users don't discover relevant features that could improve their productivity and success.

### Goals

1. **Enable Rapid Experimentation**: Provide the Growth team with the ability to
   run activation experiments quickly
   without waiting for complex data infrastructure.
2. **Track User Activation Milestones**: Capture key user actions and
   completions that indicate progress through Setup, Aha, and Habit stages.
3. **Orchestrate Personalized Experiences**: Enable targeted UI elements,
   notifications, page routing, and scheduled notifications based on user
   activation state.
4. **Establish Extensible Patterns**: Create a foundation that can evolve to
   support more sophisticated activation and engagement strategies in the
   future.
5. **Maintain Performance**: Ensure the tracking and delivery system has minimal
   impact on application performance and user experience.

### Non-Goals

1. **Dynamic Audit Event Filtering**: The initial implementation will not dynamically pull data from existing audit
   events or other data sources.
2. **Snowflake Integration**: This implementation does not integrate with Snowflake or other analytics data warehouses
   for real-time personalization.
3. **Self-Managed Support (Initial)**: The first version is scoped to GitLab.com (SaaS) and will not support
   self-managed instances.
4. **Comprehensive User Profiling**: This is not a complete user profiling system; it focuses on specific, actionable
   metrics.
5. **Machine Learning Recommendations**: The initial version does not include ML-based recommendation engines.

## Proposal

### Path Forward: User Metric Completions

We propose implementing a database table design for tracking user metric completion in the Activation Engine.
This provides a static solution for v1 with possible extensibility for future dynamic audit event filtering.

#### Overview

The Activation Engine will track specific user actions and milestones through a dedicated database table that
records completion status for predefined metrics.
This approach allows the Growth team to:

- Start working on activation experiments faster
- Avoid dependencies on Snowflake or other external data sources
- Maintain control over what metrics are tracked
- Iterate quickly based on experiment results

#### User Journey Flow

![User Flow Diagram](/images/engineering/architecture/design-documents/personalization_engine/user_flow_diagram.png)

### Metrics-Driven Personalization Flow

The Activation Engine operates through a continuous feedback loop that drives user activation and engagement:

#### 1. User Activation State Tracking

User activation state is stored as metrics in the database, capturing progress
through key milestones:

- **Setup Metric**: User has completed initial account and project setup
- **Aha Metric**: User has experienced the core value of GitLab (e.g., first code push, successful pipeline, merged MR)
- **Habit Metric**: User demonstrates regular engagement and adoption of platform features

These metrics represent distinct activation stages that inform personalization decisions.

#### 2. Activation Decision Logic

The engine evaluates user metrics to determine readiness for personalization:

- **Metric Evaluation**: Check if user has completed prerequisite metrics (e.g., has user reached Aha moment?)
- **Readiness Assessment**: Determine if user is in the right activation stage for a specific experience
- **Eligibility Filtering**: Apply experiment-specific rules to decide whether to present personalization

This decision logic ensures personalized experiences are relevant and timely for
each user.

#### 3. Personalized UX Presentation

Based on activation state, the engine orchestrates personalized experiences
through multiple delivery mechanisms:

- **Page Routing**: Direct users to relevant features or onboarding flows based on their activation stage
- **Notifications**: Send targeted notifications encouraging next steps in the activation journey
- **Banners/Alerts**: Display contextual UI elements highlighting features or actions relevant to user's current stage
- **Scheduled Notifications**: Deliver targeted notifications to users at specific activation stages

#### 4. Delivery Timing

Personalized experiences are delivered through two complementary approaches:

- **On-Demand**: Experiences triggered when users interact with the product (
  e.g., banner shown when user visits a feature)
- **Pre-Determined Schedule**: Experiences delivered on a schedule (e.g.,
  re-activation email sent 7 days after signup if user hasn't reached Aha)

#### 5. Engagement and Completion Tracking

The engine tracks how users interact with personalized experiences:

- **Completion Metrics**: Record when users complete actions encouraged by
  personalization (e.g., "user merged first MR after seeing banner")
- **Engagement Metrics**: Measure user reactions to nudges (e.g., banner
  dismissed, email opened, notification clicked)
- **State Updates**: Update user activation metrics based on completed actions,
  feeding back into the decision loop

#### 6. Continuous Improvement Loop

Tracked metrics inform ongoing optimization:

- **Engagement Analysis**: Understand which delivery mechanisms and messages resonate with users
- **Iteration**: Refine personalization strategies based on measured results and engagement patterns

#### Advantages

- **Faster Time to Market**: Allows Growth to start working on personalization experiments faster
- **Proven Pattern**: Similar to implementation patterns that were proposed in user-centric onboarding experiences and
  current namespace-based onboarding progress
- **Performance Control**: Direct database access provides predictable performance characteristics
- **Flexibility**: Easy to add new metrics as experiments evolve

#### Trade-offs

- **Not Dynamically Driven**: Data is not pulled from existing data sources (such as audit events and Snowflake)
- **Scope Limitation**: Most likely scoped to new users initially
- **Maintenance Overhead**: Requires maintaining separate tracking logic rather than leveraging existing event streams
- **Data Duplication**: Some tracked data may duplicate information available elsewhere in the system

## Ownership and Scope

**v1 Ownership:** The Activation Engine is owned and operated by the Growth team for activation experiments on
GitLab.com.

**Decision-making:** The Growth team determines which metrics to track and which experiences to present based on the Activation Engine goals (Setup → Aha → Habit moments).

**Implementation:** Growth engineers implement both the metric tracking and the personalized UI elements.
This is not a self-service platform for other teams in v1.

**Future evolution:** If other product teams express interest in using the Activation Engine for their own
activation and engagement needs, we'll evaluate expanding it into a shared platform with documented contribution
processes.
For now, it's scoped to Growth's activation experiments.

**Concrete examples of activation experiences (v1):**

- Contextual prompts encouraging users to complete their first merge request (Aha moment)
- Onboarding tooltips that appear/disappear based on user progress milestones
- Personalized guidance in empty states based on what the user has already accomplished
- Targeted feature discovery based on activation stage (Setup vs. Aha vs. Habit)
- Re-activation notifications for users who haven't reached Aha moment
- Notifications encouraging users to take next steps in their activation journey

## Design and implementation details

![Data Model](/images/engineering/architecture/design-documents/personalization_engine/data_model_diagram.png)

### Database Schema

#### Personalization metrics table

The design uses a single table with normalization on read, keeping user-level tracking as the primary model while adding optional namespace context for future aggregation capabilities.

```ruby
create_table :personalization_metrics do |t|
    t.bigint :user_id, null: false
    t.bigint :namespace_id, null: true
    t.integer :metric, null: false, limit: 2 # ActiveRecord enum
    t.timestamps_with_timezone null: false

    t.index [:user_id, :namespace_id, :metric], unique: true, name: 'unique_personalization_metric_user_id_namespace_id_and_metric'
    t.index [:namespace_id, :metric], name: 'index_personalization_metrics_on_namespace_id_and_metric'
end
```

Metric column: ActiveRecord enum defined in the PersonalizationMetric model. Example values:

- code_push_activity (0)
- pipeline_success (1)
- merge_request_completion (2)

The smallint type (limit: 2) supports up to 32,767 metric types.

#### Key Design Decisions

1. **User-level tracking**: Maintains user-level metrics as the primary tracking mechanism (as currently designed)
2. **Optional namespace context**: Adds `namespace_id` to capture the context where the action occurred, enabling future namespace-level personalization
3. **Normalization on read**: Data is stored at the user level and normalized/aggregated when querying for namespace-level personalization
4. **Metric identifier**: Integer-based identifier for flexibility in adding new metrics
5. **Uniqueness constraint**: Ensures data integrity at the user-metric level
6. **Composite indexing**: Supports efficient queries for both user-level and namespace-level lookups
7. **Safe concurrent writes**: Upsert operations will follow established GitLab patterns
   (similar to [OrganizationUser](https://gitlab.com/gitlab-org/gitlab/blob/afe043ec9c9abc62feb949c7f10bd4f3dd079f9d/app/models/organizations/organization_user.rb#L46-46))
   to handle race conditions when metrics are recorded concurrently

#### Constraint Considerations with Namespace Deletion

The `namespace_id` foreign key will use `ON DELETE SET NULL` to preserve user metric data when namespaces are deleted.
When a namespace is deleted, the `namespace_id` is nullified while keeping the user's metric record intact,
since the user is the primary entity being tracked.

**PostgreSQL NULL Handling**: PostgreSQL treats NULL values as DISTINCT in unique constraints ([reference](https://www.linkedin.com/pulse/postgresql-unique-constraint-null-values-mykhaylo-symulyk-fpkwf)),
which means the unique constraint `[:user_id, :namespace_id, :metric]` allows multiple rows with the same `user_id` and `metric` when `namespace_id` is NULL.
This behavior supports our design goals:

- Users can have the same metric recorded multiple times across different namespaces
- When namespaces are deleted, the nullified `namespace_id` values don't create uniqueness violations
- User-level metrics (recorded without namespace context) coexist with namespace-specific metrics

**Implementation**: No special trigger or application-level conflict handling is required for the deletion scenario,
as PostgreSQL's NULL handling naturally supports our data model.

#### Trade-offs

**Query complexity vs. write simplicity**:

- **Write simplicity**: Maintaining user-level tracking keeps the write path straightforward and performant. Recording metrics remains a simple insert/upsert operation without complex aggregation logic.
- **Query complexity**: Namespace-level personalization requires aggregating user-level data at query time. This adds computational overhead to read operations but provides flexibility in how data is aggregated and interpreted.
- **Scalability consideration**: As the number of tracked metrics and users grows, namespace-level queries, may require optimization through caching or materialized views.

#### Maintaining Consistency Between Event Definitions and Metric Mappings

The `METRIC_MAPPINGS` hash creates a link between event YAML files (e.g., `config/events/push_code.yml`) and the `PersonalizationMetric` enum values. To ensure consistency:

**v1 validation:**

- Custom linting validates `METRIC_MAPPINGS` keys match event YAML files
- CI check fails if mappings are out of sync
- Integration tests exercise the full event → metric recording flow

**Sources to keep in sync:**

1. Event YAML files (`config/events/*.yml`) - Define which events trigger tracking
2. Model enum (`PersonalizationMetric`) - Define metric types in database
3. `METRIC_MAPPINGS` hash - Links events to metrics

**Future considerations:**

- Code generation from YAML to enum (or vice versa) if the number of metrics grows significantly
- Shared constants file if other systems need to reference these metrics

### Service Layer Architecture

The Personalization Engine will use a service-oriented architecture with clear separation of concerns and an abstraction
layer to enable future storage backend changes.

## Internal Events Integration

### Leveraging Analytics Single instrumentation layer

The Personalization Engine will integrate with GitLab's existing Internal Events system using the proven [Single Instrumentation Layer](https://docs.gitlab.com/development/internal_analytics/instrumentation_layer/), similar to how AI usage tracking is
[implemented](https://docs.gitlab.com/development/internal_analytics/instrumentation_layer/#additional-tracking-systems).

#### Architecture Overview

Instead of modifying core tracking logic, we'll extend the event tracking system through event definition configuration:

1. **PersonalizationTracking Module:** A dedicated tracking module that receives events from the Internal Events router
2. **Event Configuration:** YAML-based event definitions that declare personalization tracking
3. **Storage Adapter:** An abstraction layer supporting multiple backends (PostgreSQL, and ClickHouse)

#### Implementation Pattern

**Tracking Module Example**
_lib/gitlab/tracking/personalization_tracking.rb_

```ruby
module Gitlab
    module Tracking
        module PersonalizationTracking
            class << self
                def track_event(event_name, **context)
                    storage_adapter.record_metric(event_name, context)
                end

                private

                def storage_adapter
                    if Feature.enabled?(:personalization_clickhouse_storage)
                        ClickHouseAdapter.new
                    else
                        PostgresAdapter.new
                    end
                end
            end

            class PostgresAdapter
                def record_metric(event_name, context)
                    user = context[:user]
                    namespace = context[:namespace]&.root_namespace || context[:project]&.root_namespace
                    metric_type = METRIC_MAPPINGS[event_name]

                    return unless user && metric_type

                    # Check if metric already exists to avoid unnecessary upsert
                    return if PersonalizationMetric.exists?(
                        user_id: user.id,
                        namespace_id: namespace&.id,
                        metric: metric_type
                    )

                    PersonalizationMetric.upsert(
                        {
                            user_id: user.id,
                            metric: metric_type,
                            namespace_id: namespace&.id
                        },
                        unique_by: [:user_id, :namespace_id, :metric]
                    )
                end

                METRIC_MAPPINGS = {
                    'push_code' => :code_push_activity,
                    'pipeline_succeeded' => :pipeline_success,
                    'merge_request_merged' => :merge_request_completion
                }.freeze
            end

            class ClickHouseAdapter
                def record_metric(event_name, context)
                    # Future implementation for ClickHouse/DIP
                end
            end
        end
    end
end
```

**Event Configuration**
_config/events/push_code.yml_

```yml
action: push_code
category: source_code
internal_events: true
extra_trackers:
    - tracking_class: Gitlab::Tracking::PersonalizationTracking
      protected_properties: {}
```

#### Event Flow

**Application code triggers event**
`Gitlab::InternalEvents.track_event('push_code', user: current_user, project: project)`

Automatic routing:

1. Snowplow tracking
2. Redis counter updates if metric definitions exist
3. PersonalizationTracking.track_event (via extra_trackers)
4. Record to personalization_metrics table

#### Advantages

- Non-Invasive: No modifications are required to Internal events core logic
- Proven Pattern: Follows the same architecture as [ee/lib/gitlab/tracking/ai_tracking.rb](https://gitlab.com/gitlab-org/gitlab/blob/dabea7ad3f68f0ae1d0bac09191f4b82b30dc3a6/ee/lib/gitlab/tracking/ai_tracking.rb#L7-7)
- Declarative Configuration: Events configured via YAML, similar to [ee/config/events/code_suggestion_rejected_in_ide.yml](https://gitlab.com/gitlab-org/gitlab/blob/dabea7ad3f68f0ae1d0bac09191f4b82b30dc3a6/ee/config/events/code_suggestion_rejected_in_ide.yml#L1-1)
- Independent Testing: Personalization tracking can be tested in isolation
- Feature Flag Control: Easy to enable/disable without code changes
- Future-Proof: Storage adapter pattern enables seamless migration to ClickHouse/DIP

#### Reference Implementations

- Core Internal Events: [lib/gitlab/internal_events.rb](https://gitlab.com/gitlab-org/gitlab/blob/dabea7ad3f68f0ae1d0bac09191f4b82b30dc3a6/lib/gitlab/internal_events.rb#L4-4) - Event routing and extra_trackers invocation
- Contribution Analytics: [lib/gitlab/tracking/contribution_analytics_tracking.rb](https://gitlab.com/gitlab-org/gitlab/blob/d39e4b11f287cc4e862f760c7f692e7429b025e4/lib/gitlab/tracking/contribution_analytics_tracking.rb#L5-5) - Similar tracking pattern
- AI Tracking Example: [ee/lib/gitlab/tracking/ai_tracking.rb](https://gitlab.com/gitlab-org/gitlab/blob/dabea7ad3f68f0ae1d0bac09191f4b82b30dc3a6/ee/lib/gitlab/tracking/ai_tracking.rb#L7-7) - Similar tracking pattern
- Event Schema: [db/docs/ai_code_suggestion_events.yml](https://gitlab.com/gitlab-org/gitlab/blob/dabea7ad3f68f0ae1d0bac09191f4b82b30dc3a6/db/docs/ai_code_suggestion_events.yml#L5-5) - Documentation pattern for tracked events
- Event Configuration: [ee/config/events/code_suggestion_rejected_in_ide.yml](https://gitlab.com/gitlab-org/gitlab/blob/dabea7ad3f68f0ae1d0bac09191f4b82b30dc3a6/ee/config/events/code_suggestion_rejected_in_ide.yml#L1-1) - YAML configuration example

#### Migration Path to CQRS

The storage adapter pattern naturally supports future CQRS evolution:

- Write Path: PostgresAdapter → ClickHouseAdapter for high-volume event ingestion (via batched writes or Snowplow
  pipeline ingestion to avoid ClickHouse antipatterns)
- Read Path: Direct PostgreSQL queries → DIP querying API for analytics
- Transition: Feature flags control gradual migration without API changes
- Compatibility: Maintains same interface as DIP support evolves

**Note on ClickHouse Write Patterns:** Small, frequent writes to ClickHouse are an antipattern that can degrade query
performance due to inefficient data merges. The migration path accounts for this by either:

- Batching writes at the application layer before sending to ClickHouse
- Leveraging the Snowplow pipeline for event ingestion (which handles batching)
- Using DIP's ingestion API once available, which will handle optimal write patterns

This approach aligns with the design document's goals of rapid experimentation while establishing extensible patterns
for future sophistication.

#### Storage Abstraction

To support future migration to alternative storage backends (such as
ClickHouse, [Data Insights Platform (DIP)](../data_insights_platform/), or other data stores), the
implementation should use an adapter pattern or similar abstraction:

- **Storage Interface**: Define a clear interface for metric recording and querying operations
- **Initial Implementation**: ActiveRecord-backed adapter using the application database
- **Future Flexibility**: Enable swapping storage backends without rewriting application logic
- **Insulation Layer**: Isolate storage implementation details from business logic and API consumers

This abstraction allows the MVC to start simple with ActiveRecord and the local database while preserving the ability to
migrate to more specialized storage solutions as requirements evolve. For example,
the [Data Insights Platform (DIP)](../data_insights_platform/) is GitLab's standardized data access layer designed to
provide a unified querying interface across multiple data sources. Once DIP supports the necessary backends (like
ClickHouse), we can migrate to use it as our read interface while maintaining the same API surface for consumers.

#### Metric Recording

A metric recording service will be responsible for:

- Validating and recording user metric completions
- Supporting user-level metrics with optional namespace context
- Capturing the namespace where the action occurred when available
- Delegating storage operations to the storage adapter
- **Optimized upsert logic:** Check for existing records before attempting upsert to minimize unnecessary database operations
  (following the [OrganizationUser pattern](https://gitlab.com/gitlab-org/gitlab/blob/afe043ec9c9abc62feb949c7f10bd4f3dd079f9d/app/models/organizations/organization_user.rb#L46-46))

**Recording Location Guidelines:**

To maintain database performance and proper replica utilization, metrics should only be recorded in contexts that already write to the database:

- **State-changing requests:** PUT, POST, DELETE operations that modify data
- **Background jobs:** Sidekiq workers processing async events

Metrics should **never** be recorded during:

- **GET requests:** To avoid session stickiness to the primary database and maintain read replica utilization
- **Read-only operations:** Any operation that doesn't already require a database write

This ensures metric recording doesn't introduce unnecessary primary database load.

**v1 Implementation:**

- Synchronous recording within the request cycle (similar to `onboarding_progresses` table, stable since GitLab 13.8)
- Metrics recorded at root namespace level (no hierarchy aggregation needed)
- Expected load similar to existing onboarding progress tracking

**Future Evolution:**

- Async processing via Redis/queue for high-volume metrics
- Migration to analytical database (ClickHouse/DIP) when write volume exceeds PostgreSQL comfort zone
- Event-driven architecture for real-time personalization

#### Metric Querying

A metric query service will provide:

- Checking if a user has completed specific metrics
- Retrieving user progress across all tracked metrics
- Normalizing and aggregating user-level data for namespace-level activation queries
- Supporting queries for activation experiments and UI customization
- Delegating read operations to the storage adapter
- **Activation Decision Logic**: Evaluating user metrics to determine readiness
  for specific personalized experiences

The service will support queries like:

- "Has this user reached their Aha moment?" (used to decide whether to show advanced feature discovery)
- "What is the user's current activation stage?" (used to route to appropriate onboarding flow)
- "How long has it been since user signup?" (used to trigger re-activation campaigns)

The storage abstraction enables caching strategies and query optimizations to be implemented independently of the
business logic.
For namespace-level queries, the service will aggregate user-level metrics at read time, allowing
flexible interpretation of namespace progress and engagement patterns.

### Integration Points

The metric recording service will need to be triggered when specific user actions occur. The specific integration points
and implementation approach will be determined during development based on existing GitLab patterns and performance
requirements.

### Data Access

The Activation Engine will need to expose metric data to support:

- **UI Personalization**: Frontend components need to query user metric completion status to customize the interface
- **Activation Decision Logic**: Backend services need to evaluate user metrics to determine which experiences to present

The specific access patterns and technologies will be determined during implementation based on performance requirements
and existing GitLab patterns.

### Personalization Delivery Mechanisms

The Activation Engine orchestrates personalized experiences through multiple delivery channels:

#### Page Routing

Direct users to relevant features or onboarding flows based on their activation
stage:

- Route new users to Setup-focused onboarding
- Route users who've completed Setup to Aha-focused feature discovery
- Route activated users to advanced features and workflows

#### Notifications

Send targeted notifications encouraging progression through activation stages:

- Notify users about features relevant to their current stage
- Encourage completion of key activation milestones
- Celebrate achievements (e.g., "Congratulations on your first merge request!")

#### Banners and Alerts

Display contextual UI elements highlighting features or actions relevant to user's current stage:

- Show Setup guidance in empty states
- Highlight Aha-moment features when user is ready
- Promote advanced features to activated users

#### Scheduled Notifications

Deliver targeted notifications to users at specific activation stages:

- Welcome series for new users
- Re-activation notifications for users who haven't reached Aha moment
- Engagement notifications for users at Habit stage

#### Delivery Timing

Experiences are delivered through two complementary approaches:

- **On-Demand**: Triggered when users interact with the product (e.g., banner
  shown when user visits a feature, notification sent when user takes an action)
- **Pre-Determined Schedule**: Delivered on a schedule (e.g., re-activation
  email sent 7 days after signup if user hasn't reached Aha, weekly engagement
  digest for active users)

### Engagement Tracking and Metrics

The Activation Engine tracks how users interact with personalized experiences to measure effectiveness:

#### Completion Metrics

Record when users complete actions encouraged by personalization:

- User merged first MR after seeing banner
- User created first pipeline after receiving notification
- User completed Setup milestone after receiving scheduled notification

#### Engagement Metrics

Measure user reactions to nudges and personalized experiences:

- Banner impressions and dismissals
- Notification open rates and click-through rates
- Scheduled notification open rates, click rates, and conversion rates
- Feature adoption rates among users who received personalization

#### State Updates

Update user activation metrics based on completed actions, feeding back into the decision loop:

- When user completes an action, update their activation state
- Trigger new personalization opportunities based on updated state
- Enable continuous progression through activation stages

### Performance Considerations

The Activation Engine must maintain GitLab's performance standards:

- **Metric Recording**: Should not block or slow down user-facing operations
- **Metric Querying**: Must support real-time UI personalization without introducing latency
- **Database Impact**: Minimize load on the primary database
- **User Scalability**: Design should accommodate growth in number of users
- **Metric Extensibility**: As an MVC, adding new metrics will require code changes; future iterations may explore more
  dynamic approaches

Specific optimization strategies (caching, indexing, partitioning, etc.) will be determined during implementation based
on measured performance characteristics and actual usage patterns.

### Monitoring and Observability

The Personalization Engine will establish comprehensive monitoring to ensure reliable operation and inform scaling
decisions.

**Operational Metrics:**

- **System Health:**
  - Recording service errors and success rates
  - Query service errors and response times
  - Database connection pool utilization
  - Background job processing metrics

- **Performance Metrics:**
  - Query latency (p50, p95, p99) for metric lookups
  - Write throughput: Metric records created per day/hour
  - Read throughput: Query frequency and patterns
  - Database query performance and slow query tracking

- **Storage Metrics:**
  - Table size and growth rate
  - Index efficiency and usage

- **Usage Patterns:**
  - Which metrics are being tracked and their frequency
  - User and namespace coverage
  - Feature flag adoption rates

- **Business Impact:**
  - Activation experiment effectiveness (Setup, Aha, Habit completion rates)
  - User engagement with personalized experiences (banner
    impressions/dismissals, notification engagement, scheduled notification metrics)
  - A/B test results and conversion metrics
  - Uplift in activation metrics for treatment vs. control groups
  - Engagement patterns by delivery mechanism (on-demand vs. scheduled)

Specific logging formats, alerting thresholds, and monitoring dashboards will be defined during implementation.

### Capacity Planning

**Initial Load Estimation:**

To ensure the database schema is appropriately designed and set expectations for backend transitions, we'll use existing
metrics as proxies:

- **Write volume proxy:** Use `onboarding_progresses` table as a reference - it tracks similar user milestone
  completions with comparable write patterns and scale
- **User base:** Scope v1 to new users (estimated subset of total GitLab.com users) to bound initial scale
- **Metric cardinality:** Start with 3-5 tracked metrics per user to establish baseline storage requirements
- **Read patterns:** Personalization UI queries will be user-scoped (single-user lookups), not analytical aggregations

**Expected v1 Characteristics:**

- Write path: Asynchronous metric recording via background jobs to avoid blocking user requests
- Read path: Simple indexed lookups by user_id and metric type
- Storage: Estimated growth based on (new users per month × metrics tracked × record size)

**Baseline Establishment:**

- **Timeline:** First 3 months of production use
- **Approach:** Monitor actual usage patterns against initial estimates to validate assumptions and establish
  performance baselines

**Scaling Decision Points:**

Based on monitored metrics, we'll trigger scaling actions when:

- **Write volume** consistently exceeds PostgreSQL comfort zone → Evaluate async batching or ClickHouse migration
- **Read latency** degrades beyond acceptable thresholds → Implement caching layer or read replicas
- **Storage growth** trajectory indicates capacity issues → Plan migration to DIP or implement partitioning
- **Query complexity** impacts performance → Consider materialized views or CQRS pattern

The storage adapter pattern enables backend migration based on observed data, allowing us to scale incrementally as
actual needs emerge while initial estimates ensure the schema is appropriately designed.

### Implementation Approach

The Activation Engine will be implemented in phases to enable rapid iteration and learning:

#### Phase 1: Foundation and Initial Experiments

1. Create database table and indexes with system for recording and querying metrics
2. Extend internal events to include activation tracking outlined in [internal events integration section](#internal-events-integration)
3. Implement activation metrics (Setup, Aha, and Habit)
4. Build metric querying service with activation decision logic
5. Implement delivery mechanisms (banners, page routing, scheduled notifications)
6. Add engagement tracking for personalized experiences
7. Run initial activation experiments

#### Phase 2: Expansion and Modularity

1. Build reusable, modular components for common activation patterns
2. Establish component library for future activation concepts
3. Begin running activation experiments

## Alternative Solutions

### Alternative 1: POC Connector - Direct Snowflake Integration

**Description**: Query Snowflake directly for user behavior data.

**Pros**:

- Access to comprehensive historical data
- Leverage existing analytics infrastructure
- No need to maintain separate tracking

**Cons**:

- Violates architectural principle that Snowflake is for analytics, not operational use (
  see [Data Insights Platform design](../data_insights_platform/) for the established data access patterns)
- High latency for real-time activation decisions
- Requires complex data access patterns
- Dependency on external system for core functionality
- Does not use existing infrastructure (DIP)
- **Rejected**: Does not align with architectural principles

### Alternative 2: DIP Extension - Access to Snowflake Data

**Description**: Extend DIP to support Snowflake connectivity and use it as the data access layer.

**Pros**:

- Aligns with architectural mandate to use DIP (see [Data Insights Platform design](../data_insights_platform/) which
  establishes DIP as the standardized data access layer)
- Centralized data access layer
- Future-proof solution

**Cons**:

- DIP does not currently support Snowflake
- Not on DIP roadmap for FY26
- Timeline exceeds Growth team's needs
- Blocks activation experiments
- **Rejected**: Timeline incompatible with business needs

### Alternative 3: DIP Extension - Snowplow Events Direct to ClickHouse

**Description**: Add ClickHouse as an additional destination in the Snowplow pipeline so DIP can access it.

**Pros**:

- Aligns with DIP architecture
- Leverages existing event streams
- Scalable for high-volume data

**Cons**:

- Requires infrastructure work with unclear ownership
- Timeline uncertainty (likely FY27Q1)
- Dependency on Analytics team roadmap
- Blocks immediate activation experimentation needs
- **Deferred**: May be considered for future iterations

**Blockers**

- This is currently blocked by [DIP ClickHouse support work](https://gitlab.com/groups/gitlab-org/architecture/gitlab-data-analytics/-/work_items/10)

## Future Considerations

### Extensibility Path

The current design provides a foundation for future enhancements:

1. **Dynamic Metric Definitions**: Allow defining new metrics without code changes
2. **Metric Aggregation**: Support complex aggregations across multiple base metrics
3. **Predictive Metrics**: Use ML to predict user behavior and needs
4. **Real-time Streaming**: Integrate with event streaming for real-time updates

### CQRS Pattern Evolution

The storage abstraction layer enables evolution toward a possible Command Query Responsibility Segregation ([CQRS](https://martinfowler.com/bliki/CQRS.html)) pattern:

- **Separate Write and Read Models**: The adapter pattern allows different storage backends for writes (commands) and
  reads (queries)
- **Write Optimization**: Metric recording could write directly to ClickHouse, optimized for high-volume event ingestion
- **Read Optimization**: Metric querying could use DIP (Data Insights Platform) as the read interface, providing a
  standardized querying layer
- **Async Synchronization**: ClickHouse data would be accessible through DIP's querying API once DIP supports ClickHouse
  as a backend
- **Independent Scaling**: Write and read workloads could scale independently based on their specific performance
  characteristics

This approach would allow the Activation Engine to handle high-volume metric recording while providing fast,
complex queries for activation decisions through GitLab's standard data access patterns.

### Migration to DIP

When DIP supports the required data sources, we can:

1. Maintain the same API surface
2. Gradually migrate backend to query DIP instead of local database
3. Use feature flags to control rollout
4. Deprecate local tracking once DIP is feature-ready for this use case

### Self-Managed Support

Future iterations may include:

1. Opt-in metric tracking for self-managed instances
2. Activation engine capabilities for self-managed deployments

## References

### Related Epics and Issues

- [Parent Epic: Activation Engine](https://gitlab.com/groups/gitlab-org/-/epics/18239)
- [Implementation Epic](https://gitlab.com/groups/gitlab-org/-/epics/19831)
- [Drive Activation: Setup metric](https://gitlab.com/groups/gitlab-org/-/epics/19917)
- [Drive Activation: Aha metric](https://gitlab.com/groups/gitlab-org/-/epics/19918)
- [Drive Activation: Habit metric](https://gitlab.com/groups/gitlab-org/-/epics/19919)
- [Prompting users to merge their first MR](https://gitlab.com/groups/gitlab-org/-/epics/19912)
- [Create a user-centric onboarding experience](https://gitlab.com/groups/gitlab-org/-/epics/18068)
- [Spike Analysis: Activation Data](https://gitlab.com/gitlab-org/gitlab/-/issues/568940)
- [DIP ClickHouse Support](https://gitlab.com/groups/gitlab-org/architecture/gitlab-data-analytics/-/work_items/10)

### Implementation References

- [Database Schema MR](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/213365)
- [OrganizationUser Upsert Pattern](https://gitlab.com/gitlab-org/gitlab/blob/afe043ec9c9abc62feb949c7f10bd4f3dd079f9d/app/models/organizations/organization_user.rb#L46-46)
- [Internal Events Core](https://gitlab.com/gitlab-org/gitlab/blob/dabea7ad3f68f0ae1d0bac09191f4b82b30dc3a6/lib/gitlab/internal_events.rb#L4-4)
- [AI Tracking Example](https://gitlab.com/gitlab-org/gitlab/blob/dabea7ad3f68f0ae1d0bac09191f4b82b30dc3a6/ee/lib/gitlab/tracking/ai_tracking.rb#L7-7)
- [Contribution Analytics Tracking](https://gitlab.com/gitlab-org/gitlab/blob/d39e4b11f287cc4e862f760c7f692e7429b025e4/lib/gitlab/tracking/contribution_analytics_tracking.rb#L5-5)
- [Event Configuration Example](https://gitlab.com/gitlab-org/gitlab/blob/dabea7ad3f68f0ae1d0bac09191f4b82b30dc3a6/ee/config/events/code_suggestion_rejected_in_ide.yml#L1-1)
- [Event Schema Documentation](https://gitlab.com/gitlab-org/gitlab/blob/dabea7ad3f68f0ae1d0bac09191f4b82b30dc3a6/db/docs/ai_code_suggestion_events.yml#L5-5)

### Design Documents

- [Data Insights Platform Design](../data_insights_platform/)
- [DIP Querying API](../data_insights_platform_querying_api/)

### Documentation

- [Internal Analytics Documentation](https://docs.gitlab.com/ee/development/internal_analytics/)
- [Single Instrumentation Layer](https://docs.gitlab.com/development/internal_analytics/instrumentation_layer/)
- [Personalization Data Document](https://docs.google.com/document/d/136wSMfHG7yK-5iIJEKu9B6N-5weDaWy30dh1NnHPdUE/edit?tab=t.0#heading=h.qg4rq88euedq)
