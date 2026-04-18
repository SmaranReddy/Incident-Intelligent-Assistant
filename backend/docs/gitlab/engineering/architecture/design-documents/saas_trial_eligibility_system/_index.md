---
# This is the title of your design document. Keep it short, simple, and descriptive. A
# good title can help communicate what the design document is and should be considered
# as part of any review.
title: SaaS trial eligibility system
status: implemented
creation-date: "2025-01-24"
authors: [ "@dstull" ]
dris: [ "@p_cordero", "@courtmeddaugh", "@dstull", "@qzhaogitlab" ]
owning-stage: "~devops::growth"
participating-stages: ["~devops::growth", "~devops::fulfillment" ]
# Hides this page in the left sidebar. Recommended so we don't pollute it.
toc_hide: true
---

<!-- vale gitlab.FutureTense = NO -->

<!-- This renders the design document header on the detail page, so don't remove it-->
{{< engineering/design-document-header >}}

## Summary

This design document outlines the planned implementation of a new trial eligibility system for GitLab SaaS.
The system aims to improve trial management by ensuring users can only start a trial on a namespace according to various internal business conditions,
preventing abuse of the trial system, and implementing efficient caching mechanisms for trial eligibility checks from the GitLab side.
GitLab will utilize the trial eligibility system in many scenarios:

- CTAs throughout the UI.
- Listing namespaces that a user owns that are eligible for a trial in the UI.
- Applying for a trial on a namespace from the UI.

## Motivation

Currently, GitLab checks trial eligibility using GitLab database level queries.
A recent business added eligibility qualification that triggered this was solved in the
short term with https://gitlab.com/gitlab-org/gitlab/-/issues/500359.
However, these queries are becoming more complex and are merely a duplication of logic that is already defined
in CustomersDot.
That issue highlighted the growing need to rely on CustomersDot for the SSOT for namespace trial eligibility.
CustomersDot already defines the eligibility conditions, and we'd like to move towards using that as the SSOT.
This also allows us to implement a solution that builds this SSOT mechanism and convert GitLab to use it in order to allow longer
term expansion of business needs in the trial eligibility space and rely on CustomersDot to be that SSOT.

reference epic: https://gitlab.com/groups/gitlab-org/-/epics/16169

### Goals

- Implement a robust and accurate system for checking trial eligibility from GitLab.
- Enable easy extension of the trial eligibility rules.

### Out of Scope

- Self-managed solution.
- Duo Pro/Duo Enterprise add on only trials. See https://gitlab.com/gitlab-org/gitlab/-/issues/507859#note_2364566118.

## Proposal

The new trial eligibility system will consist of the following components:

1. CustomersDot API Endpoint: A new endpoint in the Customer Dot system to check namespace trial eligibility.
2. Caching Mechanism: Implement caching in GitLab to store trial eligibility information for a namespace.
3. Trial Application Process: Update the existing processes to use the new eligibility criteria.
4. Cache Invalidation: Implement a mechanism to invalidate the cache when necessary.

## Design and implementation details

<!--
This section should contain enough information that the specifics of your
change are understandable. This may include API specs (though not always
required) or even code snippets. If there's any ambiguity about HOW your
proposal will be implemented, this is the place to discuss them.

If you are not sure how many implementation details you should include in the
document, the rule of thumb here is to provide enough context for people to
understand the proposal. As you move forward with the implementation, you may
need to add more implementation details to the document, as those may become
valuable context for important technical decisions made along the way. A
document is also a register of such technical decisions. If a technical
decision requires additional context before it can be made, you probably should
document this context in a document. If it is a small technical decision that
can be made in a merge request by an author and a maintainer, you probably do
not need to document it here. The impact a technical decision will have is
another helpful information - if a technical decision is very impactful,
documenting it, along with associated implementation details, is advisable.

If it's helpful to include workflow diagrams or any other related images.
Diagrams authored in GitLab flavored markdown are preferred. In cases where
that is not feasible, images should be placed under `images/` in the same
directory as the `index.md` for the proposal.
-->

### 1. CustomersDot API Endpoint

Create a new API endpoint in CDOT to check namespace trial history. https://gitlab.com/gitlab-org/customers-gitlab-com/-/issues/11481

1. The endpoint should accept a list of namespace IDs as input.
2. For each request, the endpoint should return:
    - `trial_type` with the namespace id's that are eligible for that type.
3. The response should be in a standardized format (e.g., JSON) for easy parsing by GitLab.

```json
{
  "namespaces": {
    "1": ["ultimate_with_gitlab_duo_enterprise","ultimate_on_premium_with_gitlab_duo_enterprise"],
    "2": ["ultimate_with_gitlab_duo_enterprise","ultimate_on_premium_with_gitlab_duo_enterprise"],
    "3": ["ultimate_with_gitlab_duo_enterprise"]
  },
  "success": true
}
```

### 2. Caching Mechanism

- Implement a caching system in GitLab to store trial eligibility information for a namespace after a call to CustomersDot for that information.
- Use Redis as the caching backend for consistency with existing GitLab architecture.
- Cache trial eligibility information with a TTL to balance performance and data freshness.
  - Default GitLab setting of 8 hours should be fine here.

### 3. Trial Application Process

- Update the trial application process to check the new eligibility criteria.
- Integrate the CustomersDot API call and caching mechanism into the process.

### 4. Cache Invalidation

- Implement a cache invalidation mechanism to ensure data consistency.
- Invalidate the cache when relevant namespace data changes, such as after a plan update or after applying for a new trial.

## Performance Considerations

- The caching mechanism should significantly reduce the load on the CustomersDot API.
- With caching, the GUI response time should not noticeably increase compared to the current solution that queries data from the GitLab database.
- Optimize cache TTL and invalidation strategies to balance data freshness and system performance.

## Scalability Considerations

- The caching system should be designed to handle GitLab.com scale for this added use.

## Drawbacks

- Increased complexity in the trial eligibility checking process.
- Potential for inconsistencies if cache invalidation is not handled properly.
- May slightly increase the time taken to troubleshoot trial eligibility issues from the GitLab side due to the CustomersDot API calls and checks.
- Testing the trial flow in development may require more setup to test areas where we would call the CustomersDot API.
  Today we merely create the data in the database. Perhaps we can do the same by populating Redis(our cache).

## Alternative Solutions

- Do nothing and use the current GitLab database queries.
  - Rejected due to increasing complexity in the trial eligibility rules and
    duplication in these rules between CustomersDot and GitLab.
- Storing trial history directly in GitLab instead of CustomersDot
  - Rejected due to the need for consistency with existing systems and potential data duplication.
- Not implementing caching.
  - Rejected due to potential performance issues with frequent API calls to CustomersDot.
