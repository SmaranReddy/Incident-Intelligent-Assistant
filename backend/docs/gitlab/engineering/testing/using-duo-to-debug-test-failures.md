---
title: Debug Test Failures and Live Issues with Duo
description: Concise guide to using Duo to diagnose and fix test failures in MRs and live environment E2E test pipelines.
---

GitLab Duo can help you quickly diagnose and resolve test failures in two key scenarios:

- **[Debug MR Test Failures](#-using-duo-to-debug-and-fix-test-failures-in-your-merge-request)** - Determine if failures are related to your changes and get suggested fixes
- **[Debug Live Environment Failures](#-using-duo-to-debug-live-environment-test-failures)** - Diagnose issues in staging, canary, and production monitoring pipelines

---

## 🪄 Using Duo to Debug and Fix Test Failures in Your Merge Request

When your merge request has a failing test, use Duo to quickly determine if it's related to your changes and get a suggested fix.

> **💡 Tip:** Consider retrying the job immediately in case it's a transient failure. If it fails again, proceed with the debugging steps below.

1. **Navigate to your MR pipeline** and locate the failing job
1. **Open the failing job** that displays the test failure
1. **Invoke GitLab Duo** in the job log view (press `d` or click the Duo button in the top right corner)

   > **📝 Note:** Duo will automatically truncate lengthy job logs by removing the middle section. For greater accuracy, you can copy and paste the specific stack trace and error message into your prompt.
   >
   > **🌐 For browser-based tests:** Duo cannot download artifacts automatically. If needed, you can manually download the DOM from the `/tmp/capybara` directory (feature specs) or `failure_screenshots` directory (E2E tests) in the job artifacts and paste it into your prompt to help Duo debug browser-based failures. This also applies to other artifacts such as application logs.

1. **Clear previous context** to avoid confusion with other MRs:

   ```text
      /clear
   ```

1. **Prompt: Check if the test failure is widespread:**

   ```text
      I have a test failure in this job. Can you:
      1. Check recent issues at https://gitlab.com/gitlab-org/quality/engineering-productivity/master-broken-incidents/-/issues and https://gitlab.com/gitlab-org/quality/engineering-productivity/approved-mr-pipeline-incidents/-/issues/ to see if this failure is occurring in other merge requests or in master
      2. Let me know if this appears to be related to my MR changes or a widespread issue
   ```

1. **Prompt: If the failure appears to be related to the changes in the MR, prompt for a fix:**

   ```text
      Can you analyze what's causing this failure and provide a fix? Please:
      1. Infer the test type from the test file path
      2. Explain what's causing the failure
      3. If this is a stale test (test needs updating), provide a diff to fix the test
      4. If this is a bug in the application, explain the bug and provide a diff to fix the application code
   ```

**✅ What to expect:**

- Duo will provide a diff for specific code changes required
- Always review suggestions carefully before applying them
- Test the fix locally if possible before committing

### ⚠️ When Duo Can't Help

If Duo's analysis does not resolve your issue, follow these steps in order:

   1. **🔍 Verify it's a known failure** (< 2 minutes):
      - Check the Danger bot comment on your MR under the **"Pipelines failing on master"** section for links to known master-broken issues
      - Scan [**#master-broken**](https://gitlab.enterprise.slack.com/archives/CR6QH3D7C) (master branch failures) and [**#mr-blocked-by-master-broken**](https://gitlab.enterprise.slack.com/archives/C06KFK5EE73) (approved MR failures) Slack channels
   2. **💻 Try reproducing locally** (~10 minutes):
      - Execute the test against your GDK to confirm if it's environment-specific or a genuine issue
   3. **🚧 Request quarantine if needed**:
      - If the failure is blocking `master` and is unrelated to your changes, consider the [Test Quarantine Process](quarantine-process.md)

---

## 🔥 Using Duo to Debug Live Environment Test Failures

When automated E2E tests fail in staging or production pipelines, use Duo to quickly diagnose whether it's an environmental issue, bug or a test problem.

   > **✨ Note:** GitLab Duo is available on the ops instance (ops.gitlab.net) and can be used directly in job logs there.
   >
   > **⚠️ Critical: Staging-Canary Impact**
   > Smoke test failures (`qa-smoke` jobs) in the [staging-canary pipeline](https://ops.gitlab.net/gitlab-org/quality/staging-canary/-/pipelines) **block deployments to production**. When debugging these failures, prioritize determining whether the issue is a genuine application problem or a test issue that can be safely quarantined.

1. **Navigate to the failing pipeline** on ops.gitlab.net:
    - [Staging-Canary Pipeline](https://ops.gitlab.net/gitlab-org/quality/staging-canary/-/pipelines)
    - [Staging Pipeline](https://ops.gitlab.net/gitlab-org/quality/staging/-/pipelines)
    - [Canary Pipeline](https://ops.gitlab.net/gitlab-org/quality/canary/-/pipelines)
    - [Production Pipeline](https://ops.gitlab.net/gitlab-org/quality/production/-/pipelines)

2. **Open the failing job** that displays the test failure

3. **Invoke GitLab Duo** in the job log view (press `d` or click the Duo button in the top right corner)

   > **📝 Note:** Duo will automatically truncate lengthy job logs by removing the middle section. For greater accuracy, you can copy and paste the specific stack trace and error message into your prompt.
   >
   > **🌐 For browser-based tests:** Duo cannot download artifacts automatically. If needed, you can manually download the DOM from the `failure_screenshots` directory or relevant artifacts in the job and paste it into your prompt to help Duo debug browser-based failures.

4. **Clear previous context** to avoid confusion with other investigations:

   ```text
      /clear
   ```

5. **Prompt: Analyze the failure:**

   ```text
      Analyze this test failure in [staging-canary/staging/canary/production]:

      **Context:** Tests are automatically retried within a job. If a test passed on that automatic retry, ignore it completely.

      1. **Check automatic retry status first** - Look for retry sections in the log. **Do not mention any tests that passed on automatic retry** - we only care about tests that failed both attempts within the job.
      2. For persistent failures (failed both the initial attempt AND the automatic retry):
         - What failed and why? (error type, correlation IDs, specific error messages like 404s)
         - Likely cause: environment issue, flaky test, or genuine application problem?
         - Search https://gitlab.com/gitlab-org/quality/engineering-productivity for similar issues
      3. **Urgency:**
         - ⚠️ Persistent `qa-smoke` failure in staging-canary = DEPLOYMENT BLOCKER
         - Other persistent failures = Assess user impact

      **Recommended actions (in order):**
      1. **Retry the entire job first** (even persistent-within-job failures often pass on full job retry)
      2. **If still failing AND blocking deployment:**
         - If clearly a flaky/environment issue (not a real application bug): **Use fast-quarantine immediately** to unblock deployment
           - Link: https://gitlab.com/gitlab-org/quality/engineering-productivity/fast-quarantine
         - If genuine application issue that should not be released to customers: **create an incident** - DO NOT quarantine
      3. **If not blocking deployment but is causing too much failure noise:** Follow standard quarantine process

      **Do not mention:**
      - Tests that passed on automatic retry
      - Test case reporting output (test_case iid, Labels updated, etc.)

      Note: Distinguish application issues from test problems - don't quarantine real bugs.

      Provide issue links at the end.
   ```

**✅ What to expect:**

- Duo will help distinguish between real environment issues and flaky/broken tests
- Provides suggested fixes for test issues
- Identifies potential service/application issues for escalation
- Helps assess impact and urgency

### ⚠️ When Duo Can't Help

If Duo's analysis does not resolve your issue:

1. **💻 Try reproducing locally** (~10 minutes):
    - Try logging onto the environment manually and reproducing the test case
    - Execute the test against the target environment by using credentials from 1Password
2. **🔍 Cross-check with related systems**:
    - Check [**#incident-management**](https://gitlab.enterprise.slack.com/archives/CB7P5CJS1) Slack channel for recent incidents
    - Check [GitLab.com status page](https://status.gitlab.com) for known incidents
    - Review recent deployments that might correlate with the failure
    - Look for patterns across multiple environment pipelines
3. **🚧 Quarantine if needed:**
    - **For urgent deployment-blocking smoke tests:** Use [fast-quarantine](https://gitlab.com/gitlab-org/quality/engineering-productivity/fast-quarantine) to immediately unblock deployments
    - **For non-urgent test issues:** Follow the [Test Quarantine Process](quarantine-process.md)
    - **Important:** Only quarantine test issues, not genuine application bugs - escalate those instead

---

## 📚 Related Resources

- [Testing Guide](_index.md) - Complete testing overview
- [GitLab Testing Guide](https://docs.gitlab.com/development/testing_guide) - Technical implementation details
- [Test Quarantine Process](quarantine-process.md) - How to quarantine tests
- [Guide to E2E Test Failure Issues](guide-to-e2e-test-failure-issues.md) - Product engineer debugging guide

**Need help?** Reach out in [**#s_developer_experience**](https://gitlab.enterprise.slack.com/archives/C07TWBRER7H) or create a [Request for Help issue](https://gitlab.com/gitlab-org/quality/test-governance/request-for-help/-/issues/new)
