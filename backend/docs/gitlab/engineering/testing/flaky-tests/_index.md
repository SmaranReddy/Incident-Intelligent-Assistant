---
title: "Flaky tests"
---

## Introduction

This page describes GitLab's organizational process for detecting, reporting, and managing flaky tests. For technical guidance on debugging and fixing flaky tests, see [Unhealthy Tests (Developer Docs)](https://docs.gitlab.com/development/testing_guide/unhealthy_tests/). For quarantine procedures and syntax, see [Quarantine Process (Handbook)](../quarantine-process/) and [Quarantining Tests (Developer Docs)](https://docs.gitlab.com/development/testing_guide/quarantining_tests/).

A flaky test is an unreliable test that occasionally fails but passes eventually if you retry it enough times. Flaky tests can be a result of brittle tests, unstable test infrastructure, or an unstable application. We should try to identify the cause and remove the instability to improve quality and build trust in test results.

## Why is flaky tests management important?

- Flaky tests undermine test results, leading to engineers disregarding test failures as flaky.
- Manual retries to try to get flaky tests to pass, and the effort needed to investigate flaky tests as failures are a significant waste of time.
- Managing flaky tests by quickly fixing the cause or removing the test from the test suite allows test time and costs to be used where they add value.

## Reporting of Top Flaky Test Files

GitLab uses custom tooling to automatically identify and report the most impactful flaky test files that block CI/CD pipelines. The [ci-alerts automation](https://gitlab.com/gitlab-org/quality/analytics/ci-alerts) creates issues for test files causing repeated pipeline failures, which are then triaged and assigned to Engineering Managers for resolution.

**View all top flaky test file issues:** [automation:top-flaky-test-file label](https://gitlab.com/gitlab-org/quality/test-failure-issues/-/issues/?label_name%5B%5D=automation%3Atop-flaky-test-file)

### How It Works

The ci-alerts system analyzes test failure data from ClickHouse to identify test files with the highest impact on pipeline stability. It classifies test files into three categories:

1. **Flaky**: Failures spread over 3+ days, still actively failing (≤3 days since last failure)
2. **Master-broken**: High-volume incidents (≥30 in 12h with 40%+ concentration OR 60+ absolute)
3. **Unclear**: Don't meet classification criteria

For detailed information about the classification algorithm and configuration, see the [ci-alerts flaky tests reporting documentation](https://gitlab.com/gitlab-org/quality/analytics/ci-alerts/-/blob/main/doc/flaky_tests_reporting.md).

### Frequency

We create top flaky tests issues weekly (Sundays at 10:00 UTC)

### Triage Process

Issues created by the automation are triaged by the Development Analytics team and dispatched to the responsible Engineering Managers. The complete triage workflow is documented in the [ci-alerts TRIAGE.md](https://gitlab.com/gitlab-org/quality/analytics/ci-alerts/-/blob/main/TRIAGE.md).

**Key steps:**

1. Initial triage to verify genuine flakiness
2. Dispatch to responsible product group with EM mention

### For Engineering Managers

If you've been assigned a top flaky test file issue:

1. **Review the issue description** - Contains impact metrics, Grafana dashboard link, and recommended actions
2. **Assess the situation** - Use the Grafana dashboard to understand failure patterns
3. **Take action** - See [Urgency Tiers and Response Timelines](#urgency-tiers-and-response-timelines) for timeline guidance

For guidance on quarantining tests, see the [Quarantine Process (Handbook)](../quarantine-process/) and [Quarantining Tests (Developer Docs)](https://docs.gitlab.com/development/testing_guide/quarantining_tests/).

### Urgency Tiers and Response Timelines

Flaky tests are categorized by urgency based on their impact on pipeline stability:

- 🔴 **Critical**: 48 hours - Tests blocking critical workflows, deployment pipelines or affecting multiple teams
- 🟠 **High**: 1 week - Tests with significant pipeline impact
- 🟡 **Medium**: 2 weeks - Tests with moderate impact

These timelines guide when a test should be quarantined if it cannot be fixed. For quarantine procedures and technical implementation, see [Quarantine Process (Handbook)](../quarantine-process/) and [Quarantining Tests (Developer Docs)](https://docs.gitlab.com/development/testing_guide/quarantining_tests/).

### What About Other Flaky Test Reporting Systems?

You may have noticed other flaky test issues with different labels. GitLab runs multiple flaky test detection systems in parallel:

You may have noticed older flaky test issues with `flakiness::*` labels (e.g., `flakiness::1`, `flakiness::2`). These are from a previous reporting system that runs in parallel with the top flaky test file automation.

**Key differences:**

| Old System (`flakiness::*` labels) | New System (`automation:top-flaky-test-file`) |
|---|---|
| Only shows failures on master branch | Includes both MRs and master |
| Only shows tests that failed then succeeded in the same job (via retry) | Shows tests that actually blocked pipelines |
| Doesn't impact pipeline stability | Directly measures pipeline blocking impact |

**The old system has significant shortcomings** - it misses the tests that actually block engineers from shipping features and fixes.

**Recommendation:** Prioritize issues with the `automation:top-flaky-test-file` label, as these represent tests actively blocking engineers from shipping features and fixes.

The two systems will eventually be merged into a unified reporting mechanism.

## Additional resources

- [Quarantine Process (Handbook)](../quarantine-process/) - Overall process for quarantined tests at GitLab
- [Unhealthy Tests (Developer Docs)](https://docs.gitlab.com/development/testing_guide/unhealthy_tests/) - Technical reference for debugging and reproducing flaky tests
- [Quarantining Tests (Developer Docs)](https://docs.gitlab.com/development/testing_guide/quarantining_tests/) - Technical reference for quarantine syntax and implementation
- [Flaky tests dashboard](https://dashboards.devex.gitlab.net/d/ddjwrqc/flaky-tests-overview)
