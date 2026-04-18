---
title: "The New Auth Stack in GitLab"
status: ongoing
creation-date: "2025-02-17"
authors: [ "@maw", "@kamil", "@grzesiek" ]
coach: [ "@maw" ]
approvers: [ "@maw" ]
toc_hide: true
---

{{< engineering/design-document-header >}}

## Summary

GitLab authentication and authorization code has been evolving organically for
more than decade so far. Most of the authentication and authorization
mechanisms live inside GitLab Rails monolith, with hundreds of software
engineers contributing to the project each day. This results in code complexity
growing:

1. We have 10+ different token types used for authentication.
1. We have 5+ different ways to authenticate a user session.
1. We have 3+ ways to use GitLab as an identity provider.
1. We use a tightly-coupled and complex Declarative Policies language built by us.
1. Our authorization tooling is causing frequent performance-related incidents.
1. Teams are building more of their own authentication methods in silo.

Today GitLab is a mission-critical component for many customers, who rely on
our authentication and authorization mechanisms to also grant or reject access
to their infrastructure.

Based on the challenges listed above we decided to focus on building the
**GitLab Adaptive Trust Environment (GATE)**, following three core principles
aligned under a single overarching technical strategy aimed to **consolidate
and centralize our core authentication and authorization mechanisms**:

1. **Zero Trust**: Never trust, always verify. Access is granted based on
  verified identity, context, and explicit policy. Verify trust continuously,
  on every level of application stack.
1. **Least Privilege**: Grant the minimum necessary permissions for the
  shortest possible time required to complete a task for all tokens and
  authenticated principals.
1. **Ambient Security**: Authentication, authorization and continuous
  verification are always present. Most engineers and customers don’t have to
  deal with this complexity: it is largely abstracted away and enabled by default.

To provide more clarity around how to deliver **GATE** we will focus on
building three key features anchoring this redesign in us delivering customer
value innovation:

1. **GitLab Workload Identity Federation**: built to reduce reliance on Personal Access Tokens, and long-lived credentials.
1. **Ambient Credentials in GitLab CI**: significantly reducing the need of using tokens / credentials / keys in CI jobs.
1. **Fine-grained access controls for all authenticated principals**: to improve security posture across the whole feature set.

## Details

The architectural decisions register and more details can be found in the private
project in **[The New Auth Stack Design Doc](https://gitlab.com/gitlab-org/architecture/auth-architecture/design-doc)**.

The main GitLab epic with issues and workstreams is [Deliver The New Auth Stack in GitLab](https://gitlab.com/groups/gitlab-org/-/epics/17711).
