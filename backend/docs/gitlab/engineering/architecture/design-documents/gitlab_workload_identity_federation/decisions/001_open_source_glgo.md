---
title: 'GitLab WLIF: STS-001 Open-Source GLGO'
toc_hide: true
---

## Context

We developed GLGO as an identity translation layer between GitLab and cloud providers like
Google Cloud Platform and AWS. We now plan to extend it into a general-purpose
GitLab Secure Token Service (STS). This service will be responsible for minting tokens that
users can use to authenticate with GitLab APIs.

## Decision

We have decided to make GLGO an open-source project under a permissive license.

## Consequences

Making GLGO open-source will allow the community to audit the code and contribute to its
development.

## Alternatives

Keeping GLGO as a closed-source project would limit its availability in GitLab distribution
packages. This could hinder our plans to deliver GitLab Workload Identity Federation and other
engineering projects that depend on GLGO's availability.

While we could build this functionality directly into GitLab Rails, it would not provide
sufficient isolation for authentication data and signing materials from the rest of the
GitLab monolith.
