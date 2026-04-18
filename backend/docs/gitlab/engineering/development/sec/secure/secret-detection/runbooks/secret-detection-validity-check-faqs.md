---
title: "Secret Detection Validity Checks: General FAQs"
---

### When to use this runbook?

Use this runbook to understand the validity checks feature, answer common questions, and
troubleshoot basic configuration issues. For detailed troubleshooting, see the
[troubleshooting guide](secret-detection-validity-check-troubleshooting).

### What is validity checks?

Validity checks verifies token status by calling partner APIs (AWS, GCP, Postman) and checking
GitLab token database. Status: **Active**, **Inactive**, or **Unknown**.

## Feature availability

| Offering | Tier | Status |
|----------|------|--------|
| GitLab.com | Ultimate | GA |
| GitLab Dedicated | Ultimate | GA |
| GitLab Self-Managed | Ultimate | GA |

## How it works

GitLab tokens are checked immediately through database query during report ingestion.

Partner tokens are verified asynchronously through Sidekiq worker with rate limiting.
Tokens show "Checking..." until verification completes (typically less than one minute).

## Enabling the feature

1. Go to **Secure > Security configuration > Secret detection**.
2. Turn on **Secret validity checks**.

## Status meanings

| Status | Meaning |
|--------|---------|
| Active | Token is valid |
| Inactive | Token revoked or expired |
| Unknown | Verification failed |

## Manual refresh

Select **Verify token status** on the vulnerability details page to refresh status on-demand.

## Token privacy

Token strings are sent to partner APIs for validation:

- AWS, GCP, Postman: Tokens sent to partner endpoints
- GitLab: Tokens never leave GitLab (database only)

## Disable the feature

Project-level: Users with at least the Maintainer role can turn on the setting in **Security configuration**.

## Where to find status

- Vulnerability details page: Under the **Validity** badge
- Security dashboard: Filter by token status
- Vulnerability API: Included in response

## Performance impact

Validity checks run asynchronously and have minimal performance impact. GitLab tokens take less than 100 ms to verify.
Partner tokens take 500 ms to 5 seconds to verify.

## Token check frequency

Tokens are checked once per scan automatically. You can also check tokens manually on-demand through the UI.

## Unknown token status

A token might show as `Unknown` for several reasons:

- Partner API is down
- Rate limited
- Invalid token format
- Unexpected response

For more information, see [troubleshooting secret detection validity checks](secret-detection-validity-check-troubleshooting).

## Report issues

File issues in the `#g_ast-secret-detection` Slack channel or
[GitLab tracker](https://gitlab.com/gitlab-org/gitlab/-/issues?label_name=group%3A%3Asecret%20detection).

## Related documentation

- [Monitoring guide](secret-detection-validity-check-monitoring.md)
- [Troubleshooting](secret-detection-validity-check-troubleshooting.md)
- [Architecture design](/handbook/engineering/architecture/design-documents/secret_detection_validity_checks/)
