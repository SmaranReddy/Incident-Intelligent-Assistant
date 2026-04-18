---
title: "Secret Detection Validity Checks Monitoring"
---

### When to use this runbook?

Use this runbook to monitor the validity checks feature health, performance, and usage.
For more information, see the [validity checks dashboard](https://dashboards.gitlab.net/d/secret-detection-sd-partner-token-verify/93b6ec2).

### What to monitor?

Validity checks spans these components:

- Sidekiq workers (background job processing)
- Partner APIs (AWS STS, GCP OAuth2, Postman)
- Rails (finding ingestion and status updates)
- Redis (rate limiter, queue depth)
- PostgreSQL (`secret_detection_token_statuses` and `security_finding_token_statuses` tables)

## Key metrics

### Request rate

| Metric | Source | Meaning |
|--------|--------|---------|
| Requests per sec by partner | `validity_check_partner_api_requests_total` | API call volume per partner |
| Success rate by partner     | `validity_check_partner_api_requests_total{status="success"}` | Percentage of successful verifications |
| Failure rate by partner     | `validity_check_partner_api_requests_total{status="failure"}` | Percentage of failed verifications |

Dashboard: **Requests per Second by Partner**, **Success Rate by Partner**, and **Failure rate by partner**

### Latency

| Metric | Target | Alert |
|--------|--------|-------|
| API Response P95 | < 5s | > 5s for 5+ min |

Measured using `validity_check_partner_api_duration_seconds_bucket` (95th percentile).

Dashboard: **API Response Time (P95)**

### Errors

| Type | Metric | Meaning |
|------|--------|---------|
| Overall error rate | `validity_check_partner_api_requests_total{status="failure"}` | % of all requests failing |
| Network errors | `validity_check_network_errors_total` by error class | Connectivity issues |
| Error breakdown | `validity_check_partner_api_requests_total` by error type | Failure categorization |

Possible error classes: `Timeout`, `ConnectionRefused`, `HTTPError`

Dashboard: **Errors by Type**, **Network Errors by Partner**, and **Error breakdown**

### Rate limiting

| Metric | Alert |
|--------|-------|
| Rate limit hits/sec | > 0.1 req/sec sustained |

Measured with `validity_check_rate_limit_hits_total` (requests rejected due to limits).

Location: **Rate Limit Hits**, **Rate Limits by Partner**

## Common alerts

### `SecretDetectionPartnerAPIHighErrorRate`

- Severity: S3
- Threshold: > 10% for 5+ min
- Check: **Current Error Rate** gauge, **Success Rate by Partner** chart
- Action: Identify affected partner(s). Check partner status pages. For more information, see the
[troubleshooting](secret-detection-validity-check-troubleshooting.md#partner-api-high-error-rate) section.

### `SecretDetectionPartnerAPIHighLatency`

- Severity: S3
- Threshold: P95 > 5s for 5+ min
- Check: **API Response Time (P95)** chart
- Action: Check if systemic (all partners) or partner-specific. For more information, see the [troubleshooting](secret-detection-validity-check-troubleshooting.md#partner-api-high-latency) section.

### `SecretDetectionPartnerAPIRateLimitHit`

- Severity: S4
- Threshold: > 0.1 req/sec sustained
- Check: **Rate Limit Hits** stat, **Rate Limits by Partner** chart
- Action: For more information, see the [troubleshooting](secret-detection-validity-check-troubleshooting.md#rate-limits-hit-frequently) section.

### `SecretDetectionPartnerAPINetworkErrors`

- Severity: S3
- Threshold: > 0.5 errors/sec, 5+ min
- Check: **Network Errors by Partner** chart
- Action: Identify error type (`Timeout`, `ConnectionRefused`, `HTTPError`). For more information, see the [troubleshooting](secret-detection-validity-check-troubleshooting.md#network-errors-to-partner-apis) section.

## Dashboards

| Dashboard | Purpose |
|-----------|---------|
| [Validity checks](https://dashboards.gitlab.net/d/secret-detection-sd-partner-token-verify/93b6ec2) | Request rate, latency, errors, rate limits |
| [Sidekiq workers](https://dashboards.gitlab.net/dashboards?query=sidekiq) | Job processing, retries |
| [PostgreSQL tables](https://dashboards.gitlab.net/dashboards/f/postgresql/postgresql) | `finding_token_status` growth |

## Panel descriptions

### Current Error Rate

Real-time error rate (%) across all partners. Combines success and failure counts. Alert fires if the rate exceeds 10%.

### API Response Time (P95)

95th percentile latency for each partner API call. Spikes indicate partner slowdowns. Alert fires if the latency exceeds 5 seconds for 5 or more minutes.

### Rate Limit Hits

Current rate at which partner limits are being hit (req/sec). Zero is healthy. Alert fires if the rate exceeds 0.1 req/sec.

### Requests per Second by Partner

API call volume to each partner. Use to see usage patterns.

### Success Rate by Partner

Percentage of successful verifications per partner. Should stay above 95%.

### Errors by Type

Failure reasons (network, rate limit, response parsing). Stack chart shows composition.

### Network Errors by Partner

Detailed breakdown: `Timeout`, `ConnectionRefused`, `HTTPError`. Diagnose connectivity issues.

### Rate Limits by Partner

Shows which partner limits you're hitting. Shows `limit_type` (per-second threshold).

## Performance baselines

No published SLOs exist. Use dashboard P95 latency as baseline. Expect fluctuation based on partner API health.

## Escalation

- Team: Secret Detection (@gitlab-org/secure/secret-detection)
- Slack: `#g_ast-secret-detection`
- On-call: Check `#production` for SRE availability
