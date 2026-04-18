---
# This is the title of your design document. Keep it short, simple, and descriptive. A
# good title can help communicate what the design document is and should be considered
# as part of any review.
title: Shift Left and Right Performance
status: proposed
creation-date: "2024-12-13"
authors: [ "@AndyWH" ]
coaches: [ ]
dris: [ "@ksvoboda" ]
owning-stage: ""
participating-stages: []
# Hides this page in the left sidebar. Recommended so we don't pollute it.
toc_hide: true
---

<!--
Before you start:

- Copy this file to a sub-directory and call it `_index.md` for it to appear in
  the design documents list.
- Remove comment blocks for sections you've filled in.
  When your document ready for review, all of these comment blocks should be
  removed.

To get started with a document you can use this template to inform you about
what you may want to document in it at the beginning. This content will change
/ evolve as you move forward with the proposal.  You are not constrained by the
content in this template. If you have a good idea about what should be in your
document, you can ignore the template, but if you don't know yet what should
be in it, this template might be handy.

- **Fill out this file as best you can.** At minimum, you should fill in the
  "Summary", and "Motivation" sections.  These can be brief and may be a copy
  of issue or epic descriptions if the initiative is already on Product's
  roadmap.
- **Create a MR for this document.** Assign it to an Architecture Evolution
  Coach (i.e. a Principal+ engineer).
- **Merge early and iterate.** Avoid getting hung up on specific details and
  instead aim to get the goals of the document clarified and merged quickly.
  The best way to do this is to just start with the high-level sections and fill
  out details incrementally in subsequent MRs.

Just because a document is merged does not mean it is complete or approved.
Any document is a working document and subject to change at any time.

When editing documents, aim for tightly-scoped, single-topic MRs to keep
discussions focused. If you disagree with what is already in a document, open a
new MR with suggested changes.

If there are new details that belong in the document, edit the document. Once
a feature has become "implemented", major changes should get new blueprints.

The canonical place for the latest set of instructions (and the likely source
of this file) is
[content/handbook/engineering/architecture/design-documents/_template.md](https://gitlab.com/gitlab-com/content-sites/handbook/-/blob/main/content/handbook/engineering/architecture/design-documents/_template.md).

Document statuses you can use:

- "proposed"
- "accepted"
- "ongoing"
- "implemented"
- "postponed"
- "rejected"

-->

<!-- Design Documents often contain forward-looking statements -->
<!-- vale gitlab.FutureTense = NO -->

<!-- This renders the design document header on the detail page, so don't remove it-->
{{< engineering/design-document-header >}}

<!--
Don't add a h1 headline. It'll be added automatically from the title front matter attribute.

For long pages, consider creating a table of contents.
-->

## Summary

Performance is a critical aspect of software quality that directly impacts user satisfaction and business success. We recognize that performance testing is not just a final step in development but an integral part of our entire software lifecycle. This blueprint outlines our comprehensive approach to performance testing, which goes beyond traditional load testing to encompass a wide range of techniques and practices.

We believe in shifting performance testing both left (early in development) and right (into production monitoring) to create a culture of continuous performance awareness. By integrating performance considerations at every stage - from unit testing to production monitoring - we aim to identify and address performance issues early, optimize our systems proactively, and deliver a consistently high-performing product to our users.

This guide serves as a resource, showcasing our strategies, tools, and best practices for performance engineering. It demonstrates our commitment to excellence and our innovative approach to ensuring that GitLab remains fast, responsive, and scalable as we continue to grow and evolve.

## Motivation

We have been successfully using tools like [GPT](https://gitlab.com/gitlab-org/quality/performance) for load testing against [Reference Architectures](https://docs.gitlab.com/ee/administration/reference_architectures/). We have had experiences, such as [database performance testing with large datasets](https://gitlab.com/gitlab-org/gitlab/-/issues/434465), that have demonstrated the benefits of shifting performance testing left in the development process.

By adopting a broader range of performance engineering practices, we can take a more proactive stance on solving performance concerns before they surface as issues in live environments. This approach aligns with our commitment to delivering excellent user experiences and maintaining GitLab's scalability as we continue to grow.

### Goals

* More complete documentation on Performance Engineering practices in [the handbook page](/handbook/engineering/testing/performance-tools/)
* Develop a plan for adopting these practices across all Engineering teams.
* Create a culture of performance awareness throughout the entire software development lifecycle.
* Improve early detection and resolution of performance issues.
* Enhance GitLab's overall performance and scalability.
* Standardize prioritization of performance issues at GitLab, so it no longer varies team to team with the lowest possible standard being “in response to production incident”.
* Ingrain performance earlier into the development process

### Non-Goals

* Performance becomes something that another team is responsible for owning and driving (the individual teams should have ownership of their own performance goals)
* Performance is seen as a checklist / gatekeeper item that has to be done to progress

## Proposal

Our proposed Shift Left and Right Performance Testing Strategy includes:

1. Shift Left approaches:
   * Unit Testing: Implement performance-focused unit tests using tools like benchmark-ips for Ruby.
   * Early Profiling: Integrate profiling tools like ruby-prof into the development pipeline.
   * Continuous Performance Awareness: Use instrumented tests and observability tools throughout development.

2. Shift Right approaches:
   * Load Testing: Continue and enhance our use of GitLab Performance Tool (GPT) for load testing.
   * Production Monitoring: Implement real-time observability and user-centric performance metrics in production.
   * Chaos Engineering: Introduce controlled failures to test system resilience.

3. Tool and technique adoption:
   * Drive adoption of modules for unit-level performance testing
   * Integrate profiling tools into the development pipeline
   * Implement observability tools for ongoing performance monitoring

4. Training and culture:
   * Develop training programs for engineers on performance testing techniques
   * Establish performance benchmarks and goals for each development stage
   * Create a system for sharing performance insights across teams

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

## Alternative Solutions

1. Maintain current GPT-focused approach:
   * Pros: Familiar process, less initial effort
   * Cons: Misses opportunities for early performance optimization, may lead to costly late-stage fixes

2. Focus solely on production monitoring:
   * Pros: Real-world data, less impact on development process
   * Cons: Reactive rather than proactive, potential for user-impacting issues

3. Implement full shift-left without shift-right:
   * Pros: Early issue detection, potentially faster development
   * Cons: May miss real-world performance issues, less comprehensive approach

The proposed Shift Left and Right strategy provides a balanced and comprehensive approach to performance testing, addressing the limitations of these alternatives.
