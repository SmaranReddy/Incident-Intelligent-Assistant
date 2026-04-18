---
title: "Data Lifecycle"
---

**Important note:** The governance and enforcement of of this best practice is
still a work in progress.

## Introduction

**Data doesn't disappear; it accumulates—and it will bury your application.**

Data an application generates has a built-in cost that compounds over
time. Without deliberate lifecycle management, unbounded data growth leads to
performance challenges, scalability limitations, and increased cloud spend.
Further, customers develop dependencies on historical data that make retroactive
policies very difficult to implement.

Managing data bloat after the fact is expensive and difficult. Thinking through
an application's data lifecycle at the design phase is critical to developing
performant and scalable applications.

## Why?

Without proper data lifecycle planning, applications accumulate data
indefinitely, leading to several serious consequences:

- **Performance Degradation**: Database performance deteriorates as tables grow,
   leading to issues such as plan flipping, and increased query latency due to
   TOAST storage.
- **Escalating Cloud Costs**: Unmanaged data growth leads to increased storage
  costs and excessive compute resources needed to process bloated datasets.
- **Operational Complexity**: Altering the database like adding indices and
  columns become increasingly difficult.
- **Compliance Risks**: Failure to properly manage data retention violates
  regulations like GDPR that mandate data minimization and limited retention
  periods.
- **Customer Dependency Trap**: Without clear lifecycle policies established
  from the beginning, customers may develop dependencies on historical data,
  making it nearly impossible to implement retention policies later without
  disrupting user experiences. This is a version of [Hyrum's Law](https://www.hyrumslaw.com/)
  in the context of data.

## Core Data Lifecycle Phases

All application designs must account for the following data lifecycle phases:

### 1. Data Creation

When designing data models, engineers must categorize each data type according
to its purpose and anticipated lifecycle:

- **Purpose Classification**:
  - **Operational**: Data required for core application functions
  - **User-Generated**: Data created and owned by users
  - **Derivative**: Data generated for performance optimization
  - **Metadata**: Data about system operations and events

- **Critical Questions**:
  - How is this data generated? (Automated, user input, derived)
  - What business function does this data serve?
  - Who owns this data? (System, user, third-party)
  - What is the expected frequency and volume of data creation?
  - What are the limits that will be set on how much data is allowed to be
    generated?

### 2. Data Aging

All data models must include explicit consideration of how data value changes
over time.

The following diagram shows different phases of data as it ages.

```mermaid
flowchart LR
    A[Fresh] -->B[Recent]
    B --> C[Aged]
```

- **Value Degradation Analysis**:
  - Define how rapidly each data type loses operational value
  - Establish criteria for determining when data transitions to "aged" status
  - Example: Pipeline results from merge requests closed more than a year ago
    has less value than those for an open merge request.

- **Critical Questions**:
  - How quickly does this data type become stale or less relevant?
  - What's the threshold age at which we should turn longer retention into a
    paid feature?
  - Are there regulatory reasons to maintain data beyond its operational usefulness?
  - Can aged data be stored more cheaply?

### 3. Data Offloading

By default, all data should have a [data retention policy](data_retention). Any data that needs to
 be kept around must have a strategy to offload to cheaper storage.

- **Tiered Storage Strategy**:
  - Define specific thresholds and triggers for data migration
  - Implement tablespace partitioning or object storage to facilitate seamless
    offloading
  - Design query patterns that work efficiently with offloaded data

- **Critical Questions**:
  - What storage tiers are appropriate for this data type?
  - What access patterns will still be needed for offloaded data?
  - How will offloaded data be indexed and retrieved when needed?

### 4. Data Deletion

By default, all data must have a retention policy for old data to be removed.

- **Deletion Framework**:
  - Implement automated processes for identifying deletion-eligible data
  - Create audit trails for deletion actions to demonstrate compliance
  - Consider selective archiving capabilities as paid features

- **Critical Questions**:
  - What is the maximum useful retention period for this data type?
  - Are there regulatory requirements mandating deletion (e.g., GDPR right to be forgotten)?
  - What retention extensions should be offered as premium features?
