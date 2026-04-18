---
title: "Custom Tooling and Dogfooding"
description: "Developer Experience custom tooling and dogfooding practices"
---

GitLab [dogfoods](/handbook/values/#dogfooding) extensively. The Developer Experience stage builds custom tooling to address GitLab functionality gaps, creating a valuable feedback loop between internal needs and product development.

## The Custom Data ↔ Custom Tooling Loop

Over the years, we've built custom tooling for missing GitLab features (triage-ops, test health management, CI observability, in-job metrics). This custom tooling generates **custom data** that's not part of GitLab's production database.

**By integrating custom tooling as product features, we achieve:**

- Features that benefit customers (if it helps us as customer zero, it likely helps others)
- Custom data becomes product data, available by default in visualizations
- Reduced technical debt from maintaining custom solutions
- Every improvement benefits both customers and internal teams

## Current Custom Tooling

### CI/CD Pipeline Observability

| Tool | Purpose | Repository | Status | Product Potential |
|------|---------|------------|--------|-------------------|
| **CI Alerts** | Real-time alerting for pipeline failures and performance issues | [ci-alerts](https://gitlab.com/gitlab-org/quality/analytics/ci-alerts) | Active | High - Could become native pipeline alerting |
| **CI/CD Pipelines Observability** | Pipeline visualizer, failure categorization, CI log scraping for custom metrics, and comprehensive analytics requiring tests as first-class citizens and runner improvements | [Epic #22](https://gitlab.com/groups/gitlab-org/quality/analytics/-/epics/22), [Pipeline Visualizer](https://pipeline-visualizer-gitlab-org-quality-engineeri-bcf92e4999c4df.gitlab.io/) and [the issue to add it to the product](https://gitlab.com/gitlab-org/gitlab/-/work_items/508903), [Failure Categories](https://gitlab.com/gitlab-org/quality/triage-ops/-/blob/master/doc/failure_categories.md) | Active | High - Requires runner epic and native test concept in product |
| **Snowflake Observability** | Custom dashboards and analytics for GitLab.com operations and performance | [snowflake-dashboard-sql](https://gitlab.com/gitlab-org/quality/analytics/snowflake-dashboard-sql) | Active | High - Could be integrated directly in product |

### Triage and Issue Management Automation

| Tool | Purpose | Repository | Status | Product Potential |
|------|---------|------------|--------|-------------------|
| **Triage Ops (Reactive)** | Real-time automated issue and MR triage using custom policies and reactive engine | [triage-ops](https://gitlab.com/gitlab-org/quality/triage-ops) | Active | High - Advanced real-time triage automation |
| **Triage Ops (Scheduled)** | Scheduled triage operations including weekly team reports and batch processing | [triage-ops](https://gitlab.com/gitlab-org/quality/triage-ops) | Active | High - Could benefit many customers (for example, weekly team reports) |

### Review and Code Quality Tools

| Tool | Purpose | Repository | Status | Product Potential |
|------|---------|------------|--------|-------------------|
| **GitLab Roulette** | Intelligent reviewer assignment system that considers domain expertise, availability, and workload | [gitlab-roulette](https://gitlab-org.gitlab.io/gitlab-roulette/) | Active | High - Core developer workflow improvement |
| **GitLab Danger Files** | Standardized CI-based code review automation and policy enforcement | [gitlab-dangerfiles](https://gitlab.com/gitlab-org/ruby/gems/gitlab-dangerfiles) | Active | High - Demonstrates advanced merge request automation |
| **Renovate Bot** | Automated dependency management across GitLab projects | [renovate-gitlab-bot](https://gitlab.com/gitlab-org/frontend/renovate-gitlab-bot) | Active | High - Multiple teams built similar solutions, clear product need |

### Test Health and Quality Management

| Tool | Purpose | Repository | Status | Product Potential |
|------|---------|------------|--------|-------------------|
| **Flaky Tests Management** | RSpec report parsing from job artifacts, automated test behavior analysis, GitLab issue creation for test health tracking, and data pipeline to GCS/Snowflake | [gitlab_quality-test_tooling](https://gitlab.com/gitlab-org/ruby/gems/gitlab_quality-test_tooling) | Active | High - Native test results concept missing from product |
| **Slow Tests Management** | RSpec profiling and analysis with custom frontend/backend for test performance insights | [RSpec profiling stats](https://gitlab-org.gitlab.io/rspec_profiling_stats/), [gitlab_quality-test_tooling](https://gitlab.com/gitlab-org/ruby/gems/gitlab_quality-test_tooling) | Active | Medium - Built but limited team adoption/reaction |

### Incident and Process Management

| Tool | Purpose | Repository | Status | Product Potential |
|------|---------|------------|--------|-------------------|
| **Main Branch Broken Process** | Automated identification, incident management, response, and revert MR handling | [broken main branch workflow](/handbook/engineering/workflow/#broken-master) | Active | High - Critical workflow for maintaining stable main branches |

### Data and Analytics Infrastructure

| Tool | Purpose | Repository | Status | Product Potential |
|------|---------|------------|--------|-------------------|
| **Data Pipelines** | Custom data processing pipelines feeding Snowflake dashboards and analytics | [test_tooling project](https://gitlab.com/gitlab-org/ruby/gems/gitlab_quality-test_tooling), various GCS buckets, [internal events](https://docs.gitlab.com/development/internal_analytics/internal_event_instrumentation/), [Snowflake SQL queries](https://gitlab.com/gitlab-org/quality/analytics/snowflake-dashboard-sql), [alerting](https://gitlab.com/gitlab-org/quality/analytics/ci-alerts) | Active | Medium - Could be integrated directly in product |
