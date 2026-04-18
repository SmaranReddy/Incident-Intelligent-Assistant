---
title: "Work Item REST API"
status: proposed
creation-date: "2026-02-05"
authors: ["@nicolasdular"]
coaches: ["ntepluhina", "@engwan"]
dris: ["@nicolasdular"]
owning-stage: "~devops::plan"
toc_hide: true
---

{{< engineering/design-document-header >}}

## Summary

The Work Item REST API extends the Work Items architecture with a first-class, resource-oriented interface that aligns with the broader GitLab REST conventions. It enables third-party integrations, automations, and CLI tooling to access Work Items without adopting GraphQL while maintaining a single Work Item domain model that powers issues, incidents, tasks, epics, and future types. This document proposes the foundational design for the REST surface, its evolution model, and the supporting backend components required to reach feature parity with the GraphQL API iteratively.

## Motivation

Work Items have become our preferred framework for representing planning entities across GitLab. The existing GraphQL endpoint offers breadth, but many users rely on or prefer a REST API. Establishing a documented and stable REST interface will unlock Work Item adoption, simplify migrations from legacy issue APIs, and reduce duplication of business logic between the existing REST APIs and the GraphQL API.

### Goals

- Provide a versioned REST surface for Work Items that aligns with existing GitLab REST patterns and authentication flows.
- Design for feature parity between the REST and GraphQL APIs.
- Provide flexibility when shaping responses to maximize client efficiency.
- Aim for parity in parameters and responses so switching between the GraphQL and REST APIs is straightforward.
- Deliver an excellent developer experience for our users and for us internally when we want to switch endpoints from GraphQL to REST.

### Non-Goals

- Replacing the Work Items GraphQL API.

## Proposal

We expose the Work Item REST API under the following endpoints:

1. `/namespaces/:full_path/-/work_items`.
    Notably, we only support `full_path` because the ID would need to be a `Namespace` ID, and we do not expose Namespace IDs of projects transparently in our APIs. This is a trade-off between developer experience and capabilities. I expect that this endpoint will be used mostly internally.
2. `/projects/:id_or_full_path/-/work_items` and `/groups/:id_or_full_path/-/work_items`
    For an easy migration from the existing issues endpoints, we also introduce endpoints for `/groups` and `/projects`. In this case, we allow the IDs as well, as we share the ID of a Group or Project in our APIs.

### Listing and returning single Work Items

#### Filtering and pagination

For filtering, we can either:

1. Generate the filter params from the GraphQL definitions, so new filters stay in sync automatically (see [POC](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/221749/diffs#diff-content-99fa4134f164348eb12207e36f4f325203311e1c))
    The advantage is to have it generated automatically. The disadvantage is that GraphQL is the source of truth of the REST API, and we have a different deprecation policy for GraphQL APIs.
2. Add tests to ensure that the filters are the same on both the GraphQL and the REST API
    The advantag is to ensure this in tests, but we still need to add them manually.

Regardless of what we use, we want parity by design and never ship updates to either the REST or GraphQL exclusively.

For pagination, we require keyset pagination for these endpoints, asking clients to replace `page` identifiers with cursors derived from the sort order.

#### Flexible response

We retain some of the GraphQL flexibility by adapting some of the JSON:API concept of [sparse fieldsets](https://jsonapi.org/format/#fetching-sparse-fieldsets).

1. By default, we only return the Work Item `id`, `global_id`, `iid`, `title`, and `title_html`.
2. By default, no feature or widget is added as part of the response.
3. Other top-level fields must be requested specifically via a `fields` param.
4. For the features/widgets, we add a separate `features` param.
5. We don't allow to select nested fields within `features`.

`features` is the flattened representation we recently added to GraphQL; once all consumers switch over, we plan to drop the old `widgets` array for parity.

The reasons for supporting sparse fields are:

1. It avoids serializing unnecessary fields.
2. It reduces payload for clients, which can be especially important for agents that need to minimize context size.
3. It gives us insights into how fields are used within our API.

Note: We will not allow requesting all types of `features` in the listing endpoint. For example, we will allow the `hierarchy` feature only as part of a single Work Item request. When requesting `hierarchy` as part of the listing endpoint, we would return an error.

#### Example requests

- **List Work Items within a namespace**

  ```shell
  curl --request GET \
    --header "PRIVATE-TOKEN: <your_access_token>" \
    "https://gitlab.example.com/api/v4/namespaces/gitlab-org%2Fplan/-/work_items?fields=title,state,confidential&features=labels,assignees&work_item_type_id=task"
  ```

  This call lists Work Items of type `task`, requesting the `title`, `state`, and `confidential` fields, plus the `labels` and `assignees` features.

- **Retrieve a single Work Item**

  ```shell
  curl --request GET \
    --header "PRIVATE-TOKEN: <your_access_token>" \
    "https://gitlab.example.com/api/v4/projects/gitlab-org%2Fplan/-/work_items/42?features=labels,hierarchy"
  ```

  This call returns Work Item `42` with only `id`, `iid`, `global_id`, `title`, and `title_html` and includes the `labels` and `hierarchy` features in the response.

### Creating Work Items

Creating Work Items should remain straightforward while still translating into the existing feature service layer we use for GraphQL. The REST contract flattens feature inputs into a single `features` object whose nested keys mirror the GraphQL widget inputs.

#### Example create request

```shell
curl --request POST \
  --header "PRIVATE-TOKEN: <your_access_token>" \
  --header "Content-Type: application/json" \
  --data '{
    "title": "Draft Work Item REST API ADR",
    "work_item_type_id": 1,
    "features": {
      "description": { "description": "Capture the architectural decisions about the REST API." },
      "labels":   { "label_ids": [23, 47] },
      "assignees": { "assignee_ids": [42] }
    }
  }' \
  "https://gitlab.example.com/api/v4/namespaces/gitlab-org%2Fplan/-/work_items"
```

This example creates a Work Item using a flattened `features` object whose nested structures align with the existing GraphQL widget inputs. The REST API accepts integer IDs and translates them to the same service-layer payloads currently produced by the GraphQL prepare hooks.

### Updating Work Items

Updating Work Items should use a different payload structure to follow a more typical REST API schema. We plan to offer separate endpoints for each feature while also supporting a flattened `features` object on the main update endpoint. The nested keys continue to align with the GraphQL widget inputs to minimize translation.

#### Example update requests

- **Update core fields**

  ```shell
  curl --request PUT \
    --header "PRIVATE-TOKEN: <your_access_token>" \
    --header "Content-Type: application/json" \
    --data '{
      "title": "Work Item REST API rollout",
      "state": "closed",
      "features": {
        "description": { "description": "Track the rollout milestones and metrics." }
      }
    }' \
    "https://gitlab.example.com/api/v4/namespaces/gitlab-org%2Fplan/-/work_items/42"
  ```

  This call updates the title and state of Work Item `42`, while refreshing the description through the flattened `features` object.

- **Update labels via a dedicated feature endpoint**

  ```shell
  curl --request PATCH \
    --header "PRIVATE-TOKEN: <your_access_token>" \
    --header "Content-Type: application/json" \
    --data '{ "add_label_ids": [81], "remove_label_ids": [23, 47] }' \
    "https://gitlab.example.com/api/v4/groups/gitlab-org/-/work_items/42/labels"
  ```

  This endpoint focuses solely on label updates, demonstrating how feature-specific routes can avoid large, multi-purpose payloads while reusing the same field names as the GraphQL inputs.

### Deleting Work Items

Deleting a Work Item follows the standard REST pattern by issuing a `DELETE` request to the Work Item resource. The endpoint returns `204 No Content` on success when the Work Item is removed.

#### Example delete request

```shell
curl --request DELETE \
  --header "PRIVATE-TOKEN: <your_access_token>" \
  "https://gitlab.example.com/api/v4/projects/gitlab-org%2Fplan/-/work_items/42"
```

## Concerns and open questions

1. **Response casing**: We have to decide whether REST responses use snake_case like the rest of the GitLab REST API or camelCase to align with GraphQL field names. Mixing both would hurt consistency, so we need a clear guideline before GA.
2. We could expose a single `PUT /work_items/:id` endpoint that shares the same payload structure as the GraphQL API. It is not very REST-like, but it would save us a lot of work.

## Rollout plan

The API will be marked as `experimental` and controlled via the `:work_items_rest_api` flag (default off with the user as actor). Since it is crucial to get the REST API right and we cannot introduce breaking changes, we will remove the `experimental` tag only after we are certain that the API meets our expectations.
