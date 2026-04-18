---
title: "Pipeline Mini Graph"
status: ongoing
creation-date: "2024-05-27"
authors: [ "@bsandlin" ]
coach: "@ntepluhina"
approvers: [ "@dhershkovitch", "@nmezzopera" ]
owning-stage: "~devops::verify"
participating-stages: []
toc_hide: true
---

{{< engineering/design-document-header >}}

This blueprint serves as living documentation for the Pipeline Mini Graph. The Pipeline Mini Graph is used in various places throughout the platform to communicate to users the status of the relevant pipeline. Users are able to re-run jobs directly from the component or drilldown into said jobs and linked pipelines for further investigation.

![Pipeline Mini Graph](/images/engineering/architecture/design-documents/pipeline_mini_graph/pipeline_mini_graph.png)

## Motivation

The Pipeline Mini Graph originally relied on REST for its functionality. We have updated the component to support GraphQL and are in the process of migrating all instances to this API. This documentation will act as the single source of truth (SSOT) for the refactor. Developers have found it challenging to contribute to the component due to the coexistence of two APIs, making it essential to simplify the code while ensuring compatibility with both REST and GraphQL.

### Goals

- Improved maintainability
- Backwards compatibility
- Improved query performance
- Deprecation of REST support
- Real-time pipeline status updates

### Non-Goals

- Redesign of the Pipeline Mini Graph UI

## Proposal

To break down implementation, we are taking the following steps:

1. [COMPLETE] Separate the REST version and the GraphQL version of the component into 2 directories called `pipeline_mini_graph` and `legacy_pipeline_mini_graph`. This way, developers can contribute with more ease and we can easily remove the REST version once all apps are using GraphQL.
1. [COMPLETE] Optimize GraphQL query structure to be more performant.
1. [COMPLETE] Finish updating the newer component to fully support GraphQL
1. [COMPLETE] Roll out `ci_graphql_pipeline_mini_graph` to globally enable GraphQL instances of the component.

## Implementation Details

| Issue | Milestone  | Status |
| ----- | ---------  | ------ |
| [Move legacy files to new directory](https://gitlab.com/gitlab-org/gitlab/-/work_items/464375) | [17.1](https://gitlab.com/groups/gitlab-org/-/milestones/99#tab-issues) | ✅ |
| [Move remaining legacy code](https://gitlab.com/gitlab-org/gitlab/-/work_items/464379) | [17.1](https://gitlab.com/groups/gitlab-org/-/milestones/99#tab-issues) | ✅ |
| [Create README for PMG](https://gitlab.com/gitlab-org/gitlab/-/work_items/464632) | [17.1](https://gitlab.com/groups/gitlab-org/-/milestones/99#tab-issues) | ✅ |
| [GraphQL Query Optimization](https://gitlab.com/gitlab-org/gitlab/-/issues/465309) | [17.1](https://gitlab.com/groups/gitlab-org/-/milestones/99#tab-issues) | ✅ |
| [Dedicated component for downstream pipelines](https://gitlab.com/gitlab-org/gitlab/-/issues/466238) | [17.1](https://gitlab.com/groups/gitlab-org/-/milestones/99#tab-issues) | ✅ |
| [Fetch Stage by ID](https://gitlab.com/gitlab-org/gitlab/-/issues/464100) | [17.2](https://gitlab.com/groups/gitlab-org/-/milestones/100#tab-issues) | ✅ |
| [Job Item](https://gitlab.com/gitlab-org/gitlab/-/issues/467278) | [17.2](https://gitlab.com/groups/gitlab-org/-/milestones/100#tab-issues) | ✅ |
| [Job Actions](https://gitlab.com/gitlab-org/gitlab/-/issues/467279) | [17.3](https://gitlab.com/groups/gitlab-org/-/milestones/101#tab-issues) | ✅ |
| [Rollout `ci_graphql_pipeline_mini_graph`](https://gitlab.com/gitlab-org/gitlab/-/issues/407818) | [17.8](https://gitlab.com/groups/gitlab-org/-/milestones/107#tab-issues) | ✅ |
| [Migrate pipeline editor PMG to GraphQL instance](https://gitlab.com/gitlab-org/gitlab/-/issues/466275) | [17.8](https://gitlab.com/groups/gitlab-org/-/milestones/107#tab-issues) | ✅ |
| [Migrate commit page PMG to GraphQL instance](https://gitlab.com/gitlab-org/gitlab/-/issues/466274) | [17.8](https://gitlab.com/groups/gitlab-org/-/milestones/107#tab-issues) | ✅ |
| [Remove dead logic from PMG codebase](https://gitlab.com/gitlab-org/gitlab/-/issues/466277) | TBD | To Do |
| [Migrate MR PMG to GraphQL instance](https://gitlab.com/gitlab-org/gitlab/-/issues/419725) | TBD | To Do |

## Design Details

### Structure

All data for the pipeline mini graph is passed into the component. This data comes from various API calls throughout different apps which use the component. The data passed in is formatted to GraphQL structure within the entry file. All API calls which then occur in the component itself are GraphQL queries/mutations.

#### Properties

| Name | Type | Required | Description |
| ---- | ---- | -------- | ----------- |
|`downstreamPipelines` | Array | false | pipelines triggered by current pipeline |
|`isMergeTrain` | Boolean | false | whether the pipeline is part of a merge train |
|`pipelinePath` | String | false | pipeline URL |
|`pipelineStages` | Array | true | stages of current pipeline |
|`upstreamPipeline` | Object | false | upstream pipeline which triggered current pipeline |

#### File Structure

```plaintext
├── pipeline_mini_graph/
│   ├── downstream_pipelines.vue
│   ├── pipeline_mini_graph.vue << entry point
│   ├── pipeline_stage_dropdown.vue
│   └── pipeline_stages.vue
├────── graphql/
│       └── fragments/
│           └── job.fragment.graphql
│       └── mutations/
│           ├── job_cancel.mutation.graphql
│           ├── job_play.mutation.graphql
│           ├── job_retry.mutation.graphql
│           └── job_unschedule.mutation.graphql
│       └──queries/
│           └── get_pipeline_stage_jobs.query.graphql
```

#### Considerations

##### Properties

- `isMergeTrain`: This property is specific to the MR page and is used to display a message in the job dropdown to warn users that merge train jobs cannot be retried. This is an odd flow. Perhaps we should consider either having this data come from somewhere else within the pipeline mini graph, or living in the merge train widget itself. It is worth noting here that this boolean is not used for any other logic outside of displaying this message.

## Future Improvements

- [GraphQL Subscriptions](https://gitlab.com/gitlab-org/gitlab/-/issues/406652)
- [Show downstream pipeline jobs](https://gitlab.com/gitlab-org/gitlab/-/issues/345571)
- Possible redesign
