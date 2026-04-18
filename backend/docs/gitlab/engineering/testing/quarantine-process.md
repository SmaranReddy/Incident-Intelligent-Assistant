---
title: Test Quarantine Process
description: Complete guide to GitLab's test quarantine process
---

This page describes GitLab's quarantine process for managing flaky and broken tests. For technical implementation syntax (RSpec and Jest), see [Quarantining Tests (Developer Docs)](https://docs.gitlab.com/development/testing_guide/quarantining_tests/). For debugging flaky tests, see [Unhealthy Tests (Developer Docs)](https://docs.gitlab.com/development/testing_guide/unhealthy_tests/).

The quarantine process helps maintain pipeline stability by temporarily removing flaky or broken tests from CI execution while preserving them to be fixed in the future. It helps reduce test failure noise and ensures engineers are not blocked by unrelated failures. Quarantining a test should be temporary: tests must be fixed, removed, or moved to a lower test level.

## 🗺️ Quick navigation

| Scenario                                  | What to do                                                                |
|-------------------------------------------|---------------------------------------------------------------------------|
| 📬 **I've been assigned a quarantine MR** | [Review and take action](#youve-been-assigned-a-quarantine-merge-request) |
| 🚨 **I need to quarantine a test**        | [Quarantine Lifecycle](#quarantine-lifecycle)                             |
| ✅ **I want to dequarantine a test**      | [Follow the dequarantine process](#dequarantine-a-test)                   |
| 🔍 **How do tests get quarantined?**      | [Automated detection & manual process](#how-tests-become-quarantined)     |
| 📊 **Where's the data?**                  | [Flaky test detection and tracking](#flaky-test-detection-and-tracking)   |
| 🆘 **I need help**                        | [Slack channels & support](#get-help)                                     |

## You've been assigned a quarantine merge request

If you have been assigned a quarantine merge request, you have been identified as an appropriate DRI to review the quarantine for the test. These merge requests are created automatically for [consistently failing flaky tests](./flaky-tests/_index.md), but can also be created manually by team members.

**What to do:**

1. **Review and decide the best approach**:
    - **Fix immediately**: If the root cause is clear and a fix is known.
    - **Delete the test**: If it's low-value or redundant.
    - **Convert to lower level**: If it can be tested more reliably at unit or integration level.
    - **Quarantine**: If investigation or fix will take longer than the required response time (see [Flaky tests - Urgency Tiers and Response Timelines](./flaky-tests/_index.md#urgency-tiers-and-response-timelines) for specific timelines).
1. **Take action according to the response timelines** (see [Flaky tests - Urgency Tiers and Response Timelines](./flaky-tests/_index.md#urgency-tiers-and-response-timelines) for specific timelines):
    - Merge the merge request (test enters quarantine), OR
    - Fix the test and close the merge request, OR
    - Provide feedback on why it shouldn't be quarantined.

**If no action is taken:**

- The merge request is approved by the Pipeline DRI after the urgency timeline expires.
- The test enters long-term quarantine.
- The 3-month deletion countdown begins.

**Your responsibilities after merging:**

- Investigate the root cause.
- Update the issue with progress weekly.
- Resolve or remove within 3 months.

## How tests become quarantined

Tests are identified for quarantine through automated detection and manual identification:

### Automated processes

1. **Automated Flaky Test Reporting** (recommended) - Identifies the most impactful flaky test files and creates quarantine merge requests automatically, assigned to Engineering Managers based on `feature_category` metadata. See [Flaky Tests: Reporting of Top Flaky Test Files](./flaky-tests/#reporting-of-top-flaky-test-files) for details.

2. **Test Failure Issues** (deprecated) - Automatically creates and updates GitLab issues when tests fail in CI pipelines. This system has significant shortcomings compared to the Automated Flaky Test Reporting system. See [comparison of the two systems](./flaky-tests/#what-about-other-flaky-test-reporting-systems) for details.

For details on how these systems work, see [Flaky Tests: Detection and Tracking](./flaky-tests/).

### Manual identification

Pipeline DRIs or other team members identify repeated failures and use the associated issue from the [Test Failure Issues](https://gitlab.com/gitlab-org/quality/test-failure-issues/-/issues) project to manually quarantine the test.

## Prerequisites

Before you quarantine a test, you must have a relevant issue. Test failure and flaky issues are created automatically when a test fails on a scheduled pipeline. Issues such as bug defects may have been manually created.

The pipeline job retries failed tests. When the test fails again, a test failure issue is either updated (if it is found to exist) or created. If environment issues are found on an end-to-end test pipeline, a test environment issue is raised instead.

You can also create a test failure issue manually in the [Test Failure Issues](https://gitlab.com/gitlab-org/quality/test-failure-issues/-/issues/new) project.

The issue must include:

- The correct `group::*` label (for example, `group::code_review`). For failure issues created due to a failing pipeline test, these labels are applied automatically by the labeler.
- A link to the failing pipeline or job.
- The stack trace and failure pattern.
- The `~"failure::flaky-test"` label.

## Flaky test detection and tracking

Before quarantining a test, it must be identified as flaky or failing. GitLab uses automated systems to detect and track test failures.

For complete information about how flaky tests are detected, tracked, and reported, see the [Flaky Tests handbook page](./flaky-tests/).

**Key systems:**

- **Automated Flaky Test Reporting** (recommended): Identifies the most impactful flaky test files and creates quarantine merge requests
- **Test Failure Issues** (deprecated): Automatically creates and updates issues when tests fail in CI pipelines

These systems provide the foundation for the quarantine process by identifying which tests need attention.

## Quarantine lifecycle

Choose the right quarantine type based on urgency and your ability to resolve the test failure.
Fast quarantine uses a separate file in a dedicated repository for rapid merging, while long-term quarantine modifies test metadata directly in the GitLab codebase.

### Quarantine phase durations

| Phase                | Duration         | Action required                      |
|----------------------|------------------|--------------------------------------|
| Fast quarantine      | 3 days maximum   | Fix, remove, or convert to long-term |
| Long-term quarantine | 3 months maximum | Investigation and resolution         |
| Deletion warning     | 1 week           | Final opportunity to resolve         |
| Automatic deletion   | After 3 months   | Test permanently removed             |

### Fast quarantine

Use fast quarantine when:

- A test failure is blocking critical development work and needs immediate quarantine.
- The root cause is known or easily identifiable.
- You can commit to providing an update within 3 days.

**Process:**

1. **Immediate action**:
    - Follow [the fast quarantining process](https://gitlab.com/gitlab-org/quality/engineering-productivity/fast-quarantine/-/blob/main/.gitlab/merge_request_templates/Default.md#fast-quarantine-process)
    - Update related test failure issue in the [Test Failure Issues](https://gitlab.com/gitlab-org/quality/test-failure-issues/) project with correct stage and group labels

1. **Re-running failed jobs with the latest fast quarantine file**:
    - **RSpec tests (unit/integration/system)**: Re-trigger the `retrieve-tests-metadata` job, then retry the failed RSpec job. Simply restarting the job will NOT pick up new fast quarantine updates.
    - **E2E tests**: Simply retry the failed E2E job - E2E tests automatically download the latest fast quarantine file.
    - **Alternative**: Running a new pipeline will pick up the latest fast quarantine for all test types.

1. **Within 3 days** (fast quarantine expectation):
    - **Option A**: Implement a fix for the flaky test
    - **Option B**: Remove the test entirely (if it's duplicating other tests or the flakiness cannot be fixed)
    - **Option C**: If investigation reveals the issue is more complex, convert to long-term quarantine with updated timeline.

1. **After merging fast quarantine**:
    - You will receive an automated reminder comment with instructions for creating a permanent quarantine

1. **After long-term quarantine MR is merged**:
    - If the test was failing in the default branch only, revert the fast-quarantine MR you created earlier
    - If the test was failing in a test environment, wait for the long-term quarantine to reach the desired environment before removing the fast-quarantine
    - Update the test failure issue with long term quarantine MR

**Important notes:**

- Fast quarantine commits you to providing an update within 3 days
- If you cannot meet the 3-day timeline, always convert to long-term quarantine
- Fast-quarantined tests still run locally by default
- Fast-quarantined tests apply to all branches including stable branches (e.g., `17-6-stable-ee`)
- Fast-quarantined tests can be ignored in merge requests by applying the ~"pipeline:run-flaky-tests" label

> ⚠️ Warning: The fast quarantine file is automatically cleared every Sunday at 10:00 AM UTC via scheduled pipeline. Please follow up with a permanent quarantine or fix, otherwise the tests may start failing again.

### Long-term quarantine

Use long-term quarantine when:

- The root cause is unknown and requires extensive investigation.
- You don't have immediate bandwidth available for this specific test.
- Investigation is expected to take longer than 3 days.

To use long-term quarantine, create a quarantine merge request with the appropriate metadata. For technical implementation details (RSpec syntax, Jest syntax, metadata types), see [Quarantining Tests (Developer Docs)](https://docs.gitlab.com/development/testing_guide/quarantining_tests/).

The maximum duration for long-term quarantine is 3 months (3 milestones or releases). The owner or Engineering Manager is alerted by the Quarantine Notification System that the test is to be deleted. The Quarantine Cleanup System then creates the deletion merge request to be actioned by the Pipeline DRI within a week.

## Dequarantine a test

### Prerequisites

Before you dequarantine a test, ensure that one of the following is true:

- The test passes consistently (more than 100 local runs), and the root cause is identified and fixed.
- The test is removed or replaced with better coverage.

You must also have a plan to monitor the test for 1 week after dequarantine, and commit to re-quarantine immediately if failures occur.

### Steps

To dequarantine a test:

1. Verify the fix locally by running the test multiple times without failure.
1. Create a merge request:
    1. Remove the `quarantine: 'issue-url'` metadata from the test.
    1. Link to the original quarantine issue.
    1. Include evidence of your verification.
1. After the merge request is merged, monitor the test for 1 week:
    1. Watch `#master-broken` for failures.
    1. Check test failure issues for new reports.
    1. Be prepared to re-quarantine immediately if needed.
1. After 1 week of stable runs, close the quarantine issue:
    1. Document the root cause and fix.
    1. Add the `~"quarantine::resolved"` label.

### If the test fails again

If a dequarantined test fails again:

1. Re-quarantine immediately using fast quarantine.
1. Update the original issue with new failure information.
1. Reassess whether the test should be:
    - Moved to a lower test level.
    - Removed entirely.
    - Subject to deeper investigation.

If a test is quarantined 3 or more times:

- The test is a candidate for permanent removal.
- Discuss with the Test Governance team in `#g_test_governance`.
- Consider alternative testing approaches.

## Ownership and accountability

### Who owns a quarantined test

Ownership is determined by `feature_category`. Each feature category maps to a specific engineering group, which is responsible for all tests with their feature category.

### Ownership responsibilities

When your test is quarantined:

1. Acknowledge within 48 hours.
1. Investigate the root cause.
1. Provide a timeline for resolution.
1. Update the issue with progress weekly.
1. Resolve or remove within 3 months.

If you don't maintain ownership:

- The test is automatically deleted after 3 months.
- Your team's test health metrics might be impacted.
- Repeated issues might require Test Governance intervention.

## Automated cleanup process

### Three-month deletion timeline

- **Months 1-2**: Active quarantine period. Teams are expected to investigate and resolve. Weekly progress updates are encouraged.
- **Month 3**: Final warning period. One week advance notification is sent to test owners. A deletion merge request is created and assigned to the team. This is the last opportunity to resolve or justify an extension.
- **After 3 months**: Automatic deletion. The test is permanently removed from the codebase. The quarantine issue is closed with a deletion note. The team is notified of removal.

### Notification process

Throughout the quarantine lifecycle, owners receive automated notifications:

- **Weekly reminders**: While test is in quarantine, to encourage progress updates.
- **One week before deletion**: Final warning with link to deletion merge request.
- **At deletion**: Notification that test has been removed.

The deletion merge request:

- Is assigned to the relevant Engineering Manager or code owners.
- Is tagged with mentions.
- **Requires manual approval by the Pipeline DRI** (semi-automatic process, not fully automatic).
- Provides one final opportunity to resolve or justify an extension.

## Quarantine reporting

To maintain visibility and accountability:

- **Weekly summaries**: Sent to engineering teams with current quarantine status and encouragement to resolve tests.
- **Monthly rollups**: Provided to senior management on test health metrics.
- **Team dashboards**: Available to track your team's quarantined tests by feature category.

These reports help ensure quarantine remains a temporary state, not a backlog.

For more information, see the [quarantine improvement initiative epic](https://gitlab.com/groups/gitlab-org/quality/-/epics/259).

## Special cases

### Environment-specific issues

For live environment test failures:

- Only applies to E2E tests
- Infrastructure team involvement is required.
- Different SLA expectations apply.

### Backport support (stable branches)

Fast quarantine applies to all stable branches also but should only be used to unblock pipelines until a long term quarantine is merged into the stable branch with the failure.

## Tools and resources

### Fast quarantine tool

The [fast-quarantine](https://gitlab.com/gitlab-org/quality/engineering-productivity/fast-quarantine) repository automates merge request creation for immediate quarantine.

### Flaky test tracking and dashboards

For information about flaky test tracking systems, metrics, and dashboards, see the [Flaky Tests handbook page](./flaky-tests/).

## Glossary

- **Test Failure Issue**: Failure issues raised when a test fails in a pipeline. They can also be manually created. See the [Test Failure Issue project](https://gitlab.com/gitlab-org/quality/test-failure-issues/).
- **Quarantine merge request**: A merge request that quarantines a test such that it is ignored on test pipeline executions. The metadata includes the type of failure and the Test Failure Issue.
- **Flaky test**: An unreliable test that occasionally fails but passes eventually if you retry it enough times. See [Flaky Tests handbook page](./flaky-tests/).

## Related topics

- [Top Flaky Tests: Reporting](./flaky-tests/_index.md#reporting-of-top-flaky-test-files) - How the top flaky tests are automatically detected and reported
- [Quarantine improvement initiative epic](https://gitlab.com/groups/gitlab-org/quality/-/epics/259) - Full context on the quarantine improvement project

## Get help

- **Process questions**: `#g_test_governance` - [Test Governance Group](../infrastructure-platforms/developer-experience/test-governance)
- **General questions**: `#s_developer_experience` - [Developer Experience Stage](../infrastructure-platforms/developer-experience)
- **Technical help**: `#development` - General development help
- **Urgent issues**: `#master-broken` - Immediate pipeline issues
- **Test Ownership questions**: Check the feature category mapping, ask in relevant team channels, or escalate to `#g_test_governance`
