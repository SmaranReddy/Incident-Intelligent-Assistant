---
# This is the title of your design document. Keep it short, simple, and descriptive. A
# good title can help communicate what the design document is and should be considered
# as part of any review.
title: Field Standardisation in Observability
status: ongoing
creation-date: "2025-10-15"
coaches: []
authors: [ "@e_forbes" ]
dris: [ "@swiskow", "@amyphillips" ]
owning-stage: "~devops::developer experience"
participating-stages: []
# Hides this page in the left sidebar. Recommended so we don't pollute it.
toc_hide: true
---

<!-- This renders the design document header on the detail page, so don't remove it-->
{{< engineering/design-document-header >}}

## Summary

This document outlines our approach to introducing standardisation in our observability
setup when it comes to field names used across services.

## Terminology/Glossary

- MTTR - Mean Time To Recovery - the mean time it takes to recover to a healthy state once an issue has been identified.
- SSOT - Single Source of Truth

## Motivation

One of the challenges that has cropped up regularly in discussions with
staff from different sections has been one around field names, specifically the lack of
consistency in how our field names are defined.

One service defines a field name X to represent information pertinent
to a request being handled, whilst another service defines the field name of X_ID or Y.

This leads to developers spending time building separate queries that are bespoke to different
parts of the system which then have to be married up to each other.

This then manifests itself in an increase in the MTTR for incidents, which, in turn, results in less happy customers.

This also represents one of the biggest problem areas identified by the DX survey in August, and whilst this doesn't directly tackle
all of the problems identified by engineers in that survey, it should tackle one of the biggest.

### Goals

- Define a simple pattern that helps to improve consistency between all runtimes and services being developed internally
- Provide automated enforcement of usage for previously defined fields
- Provide guidance around migration strategies and ensuring we maintain backwards compatability

#### Secondary Goals

- To reduce the field mappings in the observability stack and thus make working in the Kibana UI a nicer developer experience
- To help reduce or entirely remove dynamic field names which in turn should further reduce the number of incidents that
the dedicated team are pulled into
- To provide a pattern that the observability team can lean on when building out an explicit field schema

#### What Does This Unlock?

The Observability team have plans to provide tracing. The proposed design should help to bring consistency across both
logging setups and any tracing implementations that are delivered.

### Non-Goals

This work explicitly does not define the standard schema, as that ownership will lie in the hands of the
observability team.

## Proposal

The solution here is standardisation. The organisation has previously not put enough value on enforcing a consistent setup when it comes to common tasks such as logging, and this is one such example of this.

The proposal is to introduce consistency and standardisation across all of the "big 3" language runtimes: Go, Ruby, and Python.

This will be implemented through LabKit. There will be new packages defined called Fields - This will be an SSOT for all the standard field names.

As adoption of these fields grows in time, so too will the consistency in logging systems, and there will be improvement in the way that log lines emitted from distinct services are married up.

## Design and implementation details

The ask here is that developers will start to develop the habit of using these new Fields packages when instrumenting their code.

If the field name doesn't exist and they're emitting a value that multiple domains care about,
then they should endeavour to add that field name to the 3 LabKit libraries so that others can take advantage of it in the future.

This does incur additional overhead; however, the cost of longer MTTR in incidents,
as well as additional toil and overhead when performing business-as-usual tasks, should outweigh this overhead.

The overhead should also reduce over time as the schema becomes more established.

### Example

What we'd like to avoid, is the amount of free-text field names that are used to represent well-defined objects within our system.

Let's take for example this line:

```ruby
Gitlab::AppLogger.info(message: "Pipeline #{log_message}", user_id: current_user.id)
```

It features 3 plaintext field names that are translated into:

```json
{"message": "Pipeline #{log_message}", "user_id": 2345}
```

This is error-prone and can lead to discrepancies between those that instrument different parts of the system.

An improvement on this to ensure we're emitting the user ID in a consistent way would be:

```ruby
Gitlab::AppLogger.info(message: "Pipeline #{log_message}", Labkit::Fields::GL_USER_ID => current_user.id)
```

> Note: When making these changes, you should validate if these fields are already present in the logs.
Removing duplication will help reduce the demands on our observability infrastructure.

### Labkit

The Ruby and Go LabKit changes have already been published:

- https://gitlab.com/gitlab-org/ruby/gems/labkit-ruby/-/blob/master/lib/labkit/fields.rb?ref_type=heads
- https://gitlab.com/gitlab-org/labkit/-/tree/master/fields?ref_type=heads

These are explicitly separated out from any existing log packages as they're intended to be used across logging, metrics, and tracing.

## Maintaining Consistency

One of the raised challenges with this approach is the question of "how do we maintain consistency across all of the language runtimes?".

The idea of implementing a generator is one that has been raised to tackle this, however, it's possible that this would be an
over-engineered solution.

The alternative is delegating the Go Labkit variant to be the SSOT for these fields and the other variants would be derived from
this one.

Engineers are asked to raise a single MR into Go variant and to iterate on the MR there. Once accepted, subsequent MRs will be raised
by either the requesting team, or maintainers of the Labkit packages in order to ensure the other variants are aligned.

## Safe Migrations

It's recommended to validate that the fields being emitted aren't being utilised in downstream systems. Feature teams will have the
strongest understanding of whether or not this is the case, so please ensure that there has been some consultation with the owning teams
prior to rolling out changes.

For higher-risk migrations, it will be worthwhile utilising feature flags in order to retain the ability to switch back to
older field-names quickly and minimise the potential for impact.

Please see - https://gitlab.com/gitlab-org/gitlab/-/merge_requests/207458/diffs - as a reference of how one can do this.

In some situations, you may need to emit both field names simultaneously in order to facilitate a straightforward migration.

## Alternative Solutions

Few alternative solutions to this have been identified, however, if one should exist then folks are encouraged to highlight it.

Do nothing - continue to allow fields to be a chaotic free-for-all with entirely subjective and subtly distinct field names.

This approach will likely increase field heterogeneity and subsequently makes any future standardisation efforts harder to accomplish.

## Decision log

- [ADR 001: Standardized application field naming strategy](decisions/001_logging_field_name_standardization.md)
