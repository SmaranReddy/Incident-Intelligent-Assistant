---
title: "Secret Detection Validity Checks Troubleshooting"
---

### When to use this runbook?

Use this runbook for troubleshooting production issues related to the validity checks feature.

## Monitoring

[Validity checks monitoring](secret-detection-validity-check-monitoring) is the preferred
dashboard for identifying reliability issues.
[Sidekiq dashboard](https://dashboards.gitlab.net/d/sidekiq-workers) can supplement, follow [runbook](https://gitlab.com/gitlab-com/runbooks/-/tree/master/docs/sidekiq).

## Partner API high error rate

### Symptom

Error rate exceeds 10% on [dashboard](https://dashboards.gitlab.net/d/secret-detection-sd-partner-token-verify/93b6ec2)
(alert: `SecretDetectionPartnerAPIHighErrorRate`).

### Investigation

1. Identify affected partner(s) in dashboard breakdown.
1. Check error type:
   - `network_error`: Connectivity issues
   - `rate_limit`: Rate limit exceeded
   - `response_error`: Invalid/unparseable response
1. Review partner status pages:
   - [AWS](https://status.aws.amazon.com)
   - [GCP](https://status.cloud.google.com)
   - [Postman](https://status.postman.com)
1. Check recent deployments:

   ```shell
   git log --oneline -10 ee/lib/security/secret_detection/partner_tokens/

### Resolution

- If partner has known incident (< 1 hour): Wait for recovery
- If network error (multiple partners): Coordinate with SRE in #production
- If response error: Likely API contract change - file issue in `#g_ast-secret-detection`
- If persistent (> 6 hours): [Disable partner](#disable-partners)

## Partner API high latency

### Symptom

P95 latency exceeds 5 seconds on the [dashboard](https://dashboards.gitlab.net/d/secret-detection-sd-partner-token-verify/93b6ec2) (alert: `SecretDetectionPartnerAPIHighLatency`).

### Investigation

1. Check if systemic (all partners) or partner-specific.
1. Review partner status pages for degraded performance.
1. Test partner API from teleport Rails console:

   ```ruby
   uri = URI('https://sts.amazonaws.com/')
   start = Time.now
   Net::HTTP.start(uri.host, uri.port, use_ssl: true) do |http|
     http.request(Net::HTTP::Get.new(uri.path))
   end
   puts "Latency: #{Time.now - start}s"
   ```

### Resolution

- P95 less than 10 seconds and improving: Monitor—issue resolves itself
- P95 exceeds 10 seconds sustained (more than 24 hours): [Disable partner](#disable-partners)

## Rate limits hit frequently

### Symptom

Exceeds 0.1 req/sec sustained for 5 or more minutes on the [dashboard](https://dashboards.gitlab.net/d/secret-detection-sd-partner-token-verify/93b6ec2) (alert: `SecretDetectionPartnerAPIRateLimitHit`).

### Investigation

1. Identify partner(s) hitting limits.
1. Check current limits in `ee/lib/gitlab/application_rate_limiter.rb`:

   ```ruby
   partner_aws_api: { threshold: -> { 400 }, interval: 1.second }
   partner_gcp_api: { threshold: -> { 500 }, interval: 1.second }
   partner_postman_api: { threshold: -> { 4 }, interval: 1.second }
   ```

1. Check Sidekiq queue depth using teleport:

   ```ruby
   Sidekiq::Queue.new('security_secret_detection_partner_token_verification').size
   ```

1. Check for burst patterns: Large pipeline or multiple projects scanning.

### Resolution

1. Queue less than 1k jobs: Normal, will clear within 1 hour
1. Queue 1k–10k jobs: Monitor queue size—rate limiter auto-throttles processing
1. Queue exceeds 50k jobs: Temporarily [disable partner](#disable-partners)
1. Persistent throttling (more than 24h): Update rate limits or disable partner

## Network errors to partner APIs

### Symptom

Exceeds 0.5 errors/sec for 5 or more minutes on the [dashboard](https://dashboards.gitlab.net/d/secret-detection-sd-partner-token-verify/93b6ec2) (alert: `SecretDetectionPartnerAPINetworkErrors`).

### Investigation

1. Check which partner(s) affected.
1. Verify connectivity from Rails console using teleport:

   ```ruby
   # Test AWS
   uri = URI('https://sts.amazonaws.com/')
   begin
     Net::HTTP.get_response(uri)
     puts "✓ AWS reachable"
   rescue => e
     puts "✗ AWS unreachable: #{e.message}"
   end

   # Test GCP
   uri = URI('https://www.googleapis.com/oauth2/v1/tokeninfo')
   begin
     Net::HTTP.get_response(uri)
     puts "✓ GCP reachable"
   rescue => e
     puts "✗ GCP unreachable: #{e.message}"
   end
   ```

1. Check DNS:

   ```shell
   dig sts.amazonaws.com
   ```

1. Check for firewall changes in `#infrastructure`.
1. Check logs for SSL errors using data view `pubsub-sidekiq-inf-gprd*`:

   ```plaintext
   json.class:PartnerTokenVerificationWorker AND "SSL"
   ```

### Resolution

- Single partner affected: Likely partner-side disabled partner, monitor status page.
- Multiple partners affected: Likely GitLab network issue, coordinate with SRE.
- SSL/TLS errors: Check certificate validity, might need CA bundle update.

## Disable partners

### Emergency disable (GitLab.com)

Edit `ee/lib/security/secret_detection/partner_tokens/registry.rb`:

```ruby
'AWS' => {
  client_class: ::Security::SecretDetection::PartnerTokens::AwsClient,
  rate_limit_key: :partner_aws_api,
  enabled: false  # ← Set to false
}
```

### Re-enable after incident

1. Verify issue is resolved: Partner status is operational, network OK, error rate down.
1. Make code change to re-enable partner.
1. Monitor the [dashboard](https://dashboards.gitlab.net/d/secret-detection-sd-partner-token-verify/93b6ec2)
for 10 minutes: Error rate less than 2%, queue depth less than 1k, P95 less than 5 seconds.

## Manual token verification

To verify a token without waiting for the background job, use the Rails console (teleport):

```ruby
finding = Vulnerabilities::Finding.find(FINDING_ID)
token_type = finding.identifiers.find { |i|
  i['external_type'] == 'gitleaks_rule_id'
}&.dig('external_id')

partner_config = Security::SecretDetection::PartnerTokens::Registry.partner_for(token_type)
client = partner_config[:client_class].new
result = client.verify_token(finding.metadata['raw_source_code_extract'])

puts "Valid: #{result.valid}"
puts "Metadata: #{result.metadata}"
```

## Escalation

- Team: Secret Detection (@gitlab-org/secure/secret-detection)
- Slack : `#g_ast-secret-detection`
- On-call: Check `#production` for SRE availability
- Quick checklist:
  - [ ] Issue confirmed on the [dashboard](https://dashboards.gitlab.net/d/secret-detection-sd-partner-token-verify/93b6ec2)
  - [ ] Checked partner status pages
  - [ ] Reviewed recent code deployments
  - [ ] Checked logs for errors
  - [ ] Tried turning off partner (for emergency situations)
