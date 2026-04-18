---
title: "Pipeline Authoring Duo prompts "
description: Common prompts we can use with Duo
---

## Responding to Requests for Help (RFH) issues

### Use Case / Impact

When triaging a new [Requests for Help (RFH)](https://gitlab.com/gitlab-com/request-for-help) issue,
Duo Chat can help identify potential root causes, search for related issues and determine follow-up
questions to ask Support or the customer to help narrow down the problem.

### Prompt

In the RFH issue, use this prompt with agentic chat:

```markdown
This Request for Help issue has been opened by a GitLab team member to get
help from Engineering with a problem one or more customers are facing.

Read this issue. Identify and read relevant documentation to understand the
subject matter. Review older Requests for Help issues in this project that
are also labelled for `Help group::pipeline authoring` to understand what
information is needed for troubleshooting issues in this product group.

Determine what information might be missing for a root cause analysis and suggest
follow-up questions or troubleshooting steps that will help gather that information.

Suggest possible root causes if the available information is sufficient.

Furthermore, search the `gitlab-org/gitlab` namespace for possibly related issues
and provide a list of relevant issues if any exist.
```

## Issue triage

### Use Case / Impact

In agentic mode, Duo can be very efficient at performing issue search at scale.
That makes it suited well to identify duplicate issues, or thematic clusters of
related issues / epics.

### Prompt

When suspecting an issue is likely a duplicate, use this prompt with agentic chat:

```markdown
Read through <ISSUE_URL> and search the `gitlab-org/gitlab` namespace for duplicates
of it. Search both open and closed issues. If there is no clear duplicates, present
a list of the most closely related issues. If applicable, group them into thematic
clusters that would make sense to collect in epics.
```

## Issue Refinement

### Use Case / Impact

We often discover that certain use cases have not been properly defined before we
move an issue either into "planning breakdown" or even "ready for development".
This slows down development as we either uncover these details only at review
time or, worse, after rolling things out and having actual usage reveal flaws.

### Prompt

For a feature proposal issue, or a bug issue with a proposed fix/solution,
use this prompt with agentic chat:

```markdown
Analyze the proposed solution in this issue and determine if all necessary
details and edge cases have been considered before implementation can begin.

Read relevant documentation to get an understanding of what to consider from
a product perspective. Search for comparable proposals and their related
merge requests to get an idea of typical concerns that are uncovered at
review time.

Based on that, suggest use cases for which the expected behavior is not
clearly defined, and highlight where existing definitions given in the
proposal are not comprehensive enough. Suggest questions for the product
and UX counterpart that will help clarify these details.

Analyze the implementation plan on the issue and suggest whether or not it
is sufficiently detailed so that any engineer with some domain knowledge
would be able to begin implementation without significant further planning.
If the implementation plan is missing or lacking, suggest one based on what
has been discussed on the issue. Point out parts of the plan where no clear
technical path towards the solution has been proposed.
```
