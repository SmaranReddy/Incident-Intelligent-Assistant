---
title: "Create:Code Review AI prompts"
description: Common prompts we can use with Duo Chat to be more efficient
---

## Responding to requests for help (RFH)

### Use Case / Impact

When responding to a [request for help (RFH)](/handbook/engineering/devops/create/code-review/#requests-for-help-rfh) issue,
we need to generate ideas based on knowledge of the product (features and implementation) to determine a root cause.

To triage and respond to RFH issues faster, Duo Chat can help identify potential root causes,
questions to ask the customer/Support to narrow down the problem, or find relevant past issues.

### Prompt

Open the RFH issue and use these prompts:

```markdown
Investigate this issue, read the feature docs, suggest possible root causes and questions to ask the customer.
```

```markdown
Identify if any other issues in the `request-for-help` project could be related.
```

## Triaging ToDos

### Use Case / Impact

GitLab ToDos can quickly accumulate, especially when returning from PTO.
When facing 30+ ToDos, it's hard to know at a glance what's the most important one to address first.

Duo Chat can help categorize and prioritize open ToDos.

### Prompt

1. Run a graphlQL query in https://gitlab.com/-/graphql-explorer to get your todos in JSON format (see below, replace `francoisrose` with your GitLab username).
1. Copy the output.
1. Go to Duo Chat and use this prompt:

```markdown
Here are my open todos in gitlab. Help me prioritize them. Categorize them by theme and highlight the more urgent ones.

<paste raw JSON from step 2>
```

```graphql
query GetUserTodos {
  user(username: "francoisrose") {
    id
    username
    name
    todos {
      nodes {
        id
        action
        state
        createdAt
        targetType
        target {
          ... on Issue {
            id
            iid
            title
            webUrl
            state
            projectId
          }
          ... on MergeRequest {
            id
            iid
            title
            webUrl
            state
            project {
              name
              fullPath
            }
          }
          ... on Epic {
            id
            iid
            title
            webUrl
            state
            group {
              name
              fullPath
            }
          }
        }
        author {
          id
          username
          name
        }
        note {
          id
          body
          createdAt
        }
      }
      pageInfo {
        hasNextPage
        hasPreviousPage
        startCursor
        endCursor
      }
    }
  }
}
```
