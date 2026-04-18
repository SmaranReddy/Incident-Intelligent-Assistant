---
title: Technical Blueprint
description: "The Technical Blueprint for the Analytics Instrumentation team"
---

## Overview

The Technical Blueprint documents the following:

1. The current state of the platform.
2. The current understanding of the various user personas that interact with the systems and processes.
3. Currently known gaps, challenges, and opportunities.
4. Our desired architecture, knowing what we know today.
5. Our current stance on how we realize our desired architecture, i.e., our path from the current to the desired state.

This technical blueprint is intended to be living documentation of the current collective understanding of the Analytics Instrumentation team.
The commit history for the blueprint is intended to be used to understand how the team's collective understanding of our scope, processes, and systems has evolved over time.

## Why does the Analytics Instrumentation team exist?

The Analytics Instrumentation Group is part of the Analytics section.
We build product usage data collection and instrumentation tools within the GitLab product in a privacy-focused manner.

Our group focuses on:

### 1. Empowering internal teams within GitLab with data-driven product insights to build a better GitLab

Insights generated from Analytics Instrumentation enable teams to identify the best places to invest people and resources, what product categories mature faster, where our user experience can be improved, and how product changes impact the business.

### 2. Evolving GitLab's offerings to include usage-based services that align costs with customer value

The billing events collected by the Analytics Instrumentation tooling enable GitLab to provide robust usage metering capabilities across GitLab's product portfolio.

## What are our goals?

These are our high-level product goals:

## 1. Ease of Instrumentation

Our goal is to make instrumentation easy: we do this by building self-service tooling and infrastructure that allows product teams to add instrumentation independently with minimal friction or support requests. This means creating clear documentation, automated workflows, and intuitive APIs that reduce the time from "we need to track this" to "data is flowing!" We want instrumentation to feel effortless rather than burdensome.

## 2. Broaden Instrumentation Support

Our goal is to enable every GitLab product team to instrument their product usage, billing events, and metrics using our tools and processes, regardless of how they are developed or deployed. We aim to provide consistent instrumentation coverage across GitLab's entire product architecture: from our main Ruby-based monolith to all satellite services, spanning .com, self-managed, and dedicated deployments. We currently have strong instrumentation support for the main GitLab monolith, but we're working to bridge the gap and achieve parity across all our satellite services and deployment types. No codebase should be left behind when it comes to usage tracking and data collection capabilities.

## 3. Deepen Instrumentation Adoption

We're working to achieve thorough, comprehensive instrumentation within our existing codebases rather than settling for superficial coverage. Some parts of our platform are well-instrumented while others have significant gaps - this creates an uneven understanding of user behavior across our product. We're particularly focused on the GitLab monolith, which contains critical features that users rely on daily but where we often lack sufficient visibility into actual usage patterns and user journeys.

## 4. Data Quality

We want the Analytics Instrumentation team to be a trusted source of usage data by ensuring accuracy, reliability, and consistency in everything we collect. We're implementing data validation, monitoring pipelines for anomalies, and maintaining strict adherence to privacy policies and legal compliance.

The data quality for the Analytics Instrumentation team is evaluated using the [following principles](https://internal.gitlab.com/handbook/product-usage-data-architecture/governance-framework):

1. Accuracy
2. Completeness
3. Consistency
4. Uniqueness and Integrity
5. Timeliness
6. Validity
7. Relevance
8. Accessibility
9. Traceability
10. Interpretability

## 5. Data to Insights

Our goal is to ensure that our collected data integrates well with internal analytics and visualization tools, supports both internal decision-making as well as in-product analytics and insights for customers, and enables teams to quickly answer questions about user behavior and product performance.
