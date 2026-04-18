#!/usr/bin/env python3
"""
Synthetic Incident Intelligence Data Generator
===============================================
Generates causally linked synthetic data for an Incident Intelligence System:
  PRs → Incidents → Slack Threads → Runbooks

Causal chain:
  A PR introduces a risky change → an incident is triggered later →
  engineers debug it on Slack → runbooks capture the remediation steps.

Author: SmaranReddy
"""

import json
import os
import random
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SERVICES = [
    "payment-service",
    "auth-service",
    "order-service",
    "notification-service",
    "checkout-service",
    "inventory-service",
    "user-service",
    "gateway-api",
]

ENGINEERS = [
    "alice_chen", "bob_martinez", "carol_kim", "dave_patel",
    "eve_thompson", "frank_nguyen", "grace_okonkwo", "henry_silva",
    "iris_nakamura", "jake_wilson", "kate_brennan", "liam_foster",
]

ONCALL = ["alice_chen", "bob_martinez", "carol_kim", "dave_patel", "eve_thompson"]

# Anchor date — all timestamps branch from here
BASE_DATE = datetime(2024, 9, 1, 0, 0, 0)


def ts(dt: datetime) -> str:
    """Return ISO-8601 timestamp string."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def rand_engineer(exclude=None):
    pool = [e for e in ENGINEERS if e != exclude]
    return random.choice(pool)


def rand_oncall():
    return random.choice(ONCALL)


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO DEFINITIONS
# Each scenario defines a causal chain: PR change → incident symptoms → resolution
# ─────────────────────────────────────────────────────────────────────────────

SCENARIOS = [
    # ── 1 ── Rate limiter misconfiguration ───────────────────────────────────
    {
        "id": "rate_limiter_misconfiguration",
        "service": "checkout-service",
        "pr": {
            "title": "feat: tighten rate limits on {service} to prevent API abuse",
            "summary": (
                "Reduces per-user rate limit from 1 000 req/min to 100 req/min following "
                "an observed scraping campaign. Also migrates rate limiter from Redis-based "
                "sliding window algorithm to a simpler fixed-window counter for operational "
                "simplicity. Burst allowance of 2× removed."
            ),
            "files_changed": [
                "src/middleware/rate_limiter.py",
                "config/rate_limits.yaml",
                "tests/test_rate_limiter.py",
            ],
            "risk_level": "high",
        },
        "incident": {
            "title": "{service} rate limiter throttling legitimate users — 40% checkout failure rate",
            "severity": "SEV1",
            "symptoms": [
                "HTTP 429 rate spiked from near-zero to 38% of all responses",
                "Checkout funnel abandonment increased 4× in real-time analytics",
                "p99 latency on POST /checkout jumped from 320 ms to 5 200 ms",
                "Support queue: 200+ tickets in 90 minutes about failed payments",
                "order-service receiving 60% fewer order-created events than baseline",
            ],
            "root_cause": (
                "The fixed-window counter was applied at the global service level rather than "
                "per user. All traffic shares a single counter that resets every 60 seconds. "
                "At 08:45 UTC normal morning traffic naturally exceeded 100 requests within "
                "the window, causing the counter to block ALL subsequent requests for the "
                "remainder of the minute, including legitimate checkouts."
            ),
            "resolution": (
                "Rolled back rate_limiter.py to the previous sliding-window implementation. "
                "Temporarily set global limit to 10 000 req/min as blast radius control. "
                "Created follow-up ticket to implement true per-user Redis sorted-set rate "
                "limiting with a load test gate before merge."
            ),
        },
        "slack_scenario": "rate_limiter",
    },

    # ── 2 ── Connection pool exhaustion ──────────────────────────────────────
    {
        "id": "connection_pool_exhaustion",
        "service": "order-service",
        "pr": {
            "title": "feat: add synchronous inventory availability checks during order placement",
            "summary": (
                "Calls inventory-service for each line item synchronously before confirming "
                "an order to prevent overselling. Previously the check was async and "
                "best-effort. Each cart item now triggers one DB lookup in inventory-service. "
                "Connection pool size unchanged at 20."
            ),
            "files_changed": [
                "src/services/order_validator.py",
                "src/clients/inventory_client.py",
                "config/database.yaml",
                "tests/test_order_validator.py",
            ],
            "risk_level": "high",
        },
        "incident": {
            "title": "{service} DB connection pool exhausted — orders failing at 88% error rate",
            "severity": "SEV1",
            "symptoms": [
                "Error log: 'QueuePool limit overflow, max_overflow 0 current overflow 20'",
                "Order creation success rate fell from 99.9% to 12% within 8 minutes of deploy",
                "DB connection wait p99 jumped from 2 ms to 48 000 ms",
                "inventory-service receiving 11× its normal request volume",
                "Kubernetes pod restarts on order-service every ~4 minutes (liveness probe)",
            ],
            "root_cause": (
                "Each cart with N items now opens N concurrent DB connections in inventory-service "
                "(one per SELECT). An average cart has 8 items; peak traffic has 300 concurrent "
                "checkouts, yielding 2 400 simultaneous DB connections against a pool capped at 20. "
                "A missing connection.close() in the error path leaks connections that are never "
                "returned to the pool, accelerating exhaustion."
            ),
            "resolution": (
                "Hotfix deployed: added finally: conn.close() in inventory_client.py. "
                "Batched all per-item lookups into a single SELECT ... WHERE id IN (...) query. "
                "Increased pool size to 50 as temporary buffer. Added Prometheus alert on "
                "pool utilization > 75%."
            ),
        },
        "slack_scenario": "connection_pool",
    },

    # ── 3 ── Memory leak in JWT cache ─────────────────────────────────────────
    {
        "id": "memory_leak_jwt_cache",
        "service": "auth-service",
        "pr": {
            "title": "perf: cache decoded JWT payloads to reduce redundant CPU overhead",
            "summary": (
                "JWT decode was identified as 15% of auth-service CPU in profiling. "
                "Adding an in-process LRU cache keyed by token signature. Max entries: 50 000. "
                "No TTL set — tokens carry their own exp claim. Memory impact estimated at "
                "~50 MB at full cache capacity per calculation in PR description."
            ),
            "files_changed": [
                "src/auth/jwt_validator.py",
                "src/auth/token_cache.py",
                "config/auth_config.yaml",
                "tests/test_jwt_validator.py",
            ],
            "risk_level": "medium",
        },
        "incident": {
            "title": "{service} OOM-killed every 4–6 hours — memory growing unbounded",
            "severity": "SEV2",
            "symptoms": [
                "Kubernetes events: 'OOMKilled' on all 3 auth-service replicas in rotation",
                "Memory usage grows linearly from 280 MB baseline to ~2.1 GB before kill",
                "Intermittent 5xx errors during pod restart window (45–90 s)",
                "Token validation p99 latency spikes from 2 ms to 400 ms ahead of each OOM",
                "Heap dump shows LRUCache holding 50 000 entries at ~42 KB each",
            ],
            "root_cause": (
                "The PR description mis-estimated cache memory at ~50 MB. Actual decoded "
                "JWT objects include full user claims, roles, and metadata averaging 42 KB "
                "each, not 1 KB. At 50 000 entries the cache consumes 2.1 GB. Additionally, "
                "the cache was bounded by count not by memory, and no TTL was set, so entries "
                "only evict after 50 001 distinct tokens are seen — meaning active sessions "
                "are never evicted."
            ),
            "resolution": (
                "Added 5-minute TTL to all cache entries. Reduced max-entries to 10 000. "
                "Added memory-based eviction threshold at 150 MB via cachetools MemoryCache. "
                "Added Grafana panel for cache memory and eviction rate. "
                "Auth-service memory now stable at 310–340 MB."
            ),
        },
        "slack_scenario": "memory_leak",
    },

    # ── 4 ── Circuit breaker removal → cascading retry storm ─────────────────
    {
        "id": "circuit_breaker_removed",
        "service": "notification-service",
        "pr": {
            "title": "refactor: remove deprecated Hystrix circuit breaker from notification pipeline",
            "summary": (
                "Hystrix is EOL and flagged by our dependency scanner. Removing it from the "
                "notification dispatcher as part of the Hystrix deprecation project. "
                "Notification delivery is best-effort so circuit breaking was considered "
                "unnecessary overhead. Resilience4j migration tracked in NOTIF-412."
            ),
            "files_changed": [
                "src/pipeline/notification_dispatcher.py",
                "src/resilience/circuit_breaker.py",
                "config/notification_config.yaml",
                "requirements.txt",
            ],
            "risk_level": "high",
        },
        "incident": {
            "title": "{service} retry storm overloading SendGrid — cascading to payment receipts",
            "severity": "SEV1",
            "symptoms": [
                "SendGrid API returning 503 with message 'too many requests from single source'",
                "notification-service thread pool at 100% — 2 000 threads all in retry backoff",
                "Retry queue depth growing at 45 000 messages/minute",
                "payment-service: order confirmation emails not delivered for 55 minutes",
                "auth-service: password reset emails silently dropped",
                "CPU on notification-service sustained at 98% for 60 minutes",
            ],
            "root_cause": (
                "SendGrid had a 20-minute partial outage. Without circuit breaking, all 2 000 "
                "worker threads continued attempting delivery. Exponential backoff state is "
                "in-memory per-thread and does not coordinate across threads, so the aggregate "
                "retry rate actually increased over time. By the time SendGrid recovered, our "
                "retry storm was generating 80 000 req/min against their API, triggering "
                "their global rate limiter and extending the incident by 35 minutes."
            ),
            "resolution": (
                "Deployed emergency env-var-configured circuit breaker (open after 10 failures "
                "in 30 s). Drained retry queue via dead-letter queue replay with rate limiting. "
                "Coordinated with SendGrid support to whitelist our IPs temporarily. "
                "Added circuit breaker as required field in PR template for any external HTTP client."
            ),
        },
        "slack_scenario": "cascade_failure",
    },

    # ── 5 ── Stale price cache ─────────────────────────────────────────────────
    {
        "id": "stale_price_cache",
        "service": "checkout-service",
        "pr": {
            "title": "perf: extend product price cache TTL from 5 min to 24 h",
            "summary": (
                "Price cache hit rate is 24% due to aggressive 5-minute TTL. Prices change "
                "at most once per day per marketing ops. Extending TTL to 24 h will raise "
                "hit rate to ~96% and reduce load on pricing-service by 12×. Cache busting "
                "on price updates handled by existing webhook in price_update_handler.py."
            ),
            "files_changed": [
                "src/cache/price_cache.py",
                "config/cache_config.yaml",
                "src/webhooks/price_update_handler.py",
            ],
            "risk_level": "medium",
        },
        "incident": {
            "title": "{service} serving stale prices — 40% flash sale discount not applied",
            "severity": "SEV2",
            "symptoms": [
                "47 customer support tickets in 2 hours: 'discount not showing at checkout'",
                "Revenue analytics: 14% below forecast for past 4 hours",
                "Flash sale price of $29.99 showing as $49.99 at checkout",
                "Prices on product listing page differ from cart total",
                "New A/B test pricing cohort not receiving test prices",
            ],
            "root_cause": (
                "The cache invalidation webhook began silently failing 6 days ago after a "
                "service account token was rotated as part of a security audit. The webhook "
                "endpoint returned 401 on every price update, but there was no alerting on "
                "4xx responses from this webhook. With a 24 h TTL and broken invalidation, "
                "all price changes were invisible to checkout-service for up to 24 hours. "
                "The 09:00 flash sale prices never propagated."
            ),
            "resolution": (
                "Rotated webhook service account token. Manually flushed all price cache entries "
                "via cache admin API. Reverted TTL to 5 minutes as interim measure. Added "
                "Datadog monitor on webhook endpoint HTTP response codes (alert on any 4xx/5xx). "
                "Added cache invalidation lag metric — alert if any SKU price not refreshed in >10 min."
            ),
        },
        "slack_scenario": "stale_cache",
    },

    # ── 6 ── N+1 query regression ─────────────────────────────────────────────
    {
        "id": "n_plus_one_query",
        "service": "order-service",
        "pr": {
            "title": "feat: enrich order history API with product metadata server-side",
            "summary": (
                "Currently /orders/history returns only product IDs; the mobile client "
                "makes N individual /products/{id} calls to render the page. Moving this "
                "server-side to reduce mobile chattiness and improve load time on 3G. "
                "Uses existing ProductService.get_by_id() in a loop for simplicity."
            ),
            "files_changed": [
                "src/api/order_history.py",
                "src/services/product_enrichment.py",
                "tests/test_order_history.py",
            ],
            "risk_level": "medium",
        },
        "incident": {
            "title": "{service} DB CPU at 100% — N+1 query regression in order history",
            "severity": "SEV2",
            "symptoms": [
                "Primary DB CPU jumped from 18% to 99% within 12 minutes of deploy",
                "Slow query log: 80 000 individual SELECT on products table per minute",
                "GET /orders/history p99 latency: 140 ms → 9 200 ms",
                "Read replica replication lag: 52 seconds behind primary",
                "DB connection pool at 97% — downstream services starting to queue",
            ],
            "root_cause": (
                "ProductService.get_by_id() is called once per order item in a Python for-loop, "
                "issuing one SELECT per call. A user with 20 orders × 6 items = 120 individual "
                "SELECTs per page load. At peak of 700 concurrent users loading order history: "
                "84 000 queries/minute against a DB cluster tuned for 6 000 queries/minute. "
                "The test suite used a fixture with 1 order × 1 item, masking the N+1 completely."
            ),
            "resolution": (
                "Hotfix: replaced loop with single SELECT ... WHERE id IN (all_item_ids) batch query. "
                "Deployed behind feature flag, validated 99.8% query reduction in staging. "
                "DB CPU back to 16% within 3 minutes of hotfix. Added query count assertion "
                "to integration test: assert mock_db.call_count == 1."
            ),
        },
        "slack_scenario": "n_plus_one",
    },

    # ── 7 ── TLS certificate expiry ────────────────────────────────────────────
    {
        "id": "tls_cert_expiry",
        "service": "auth-service",
        "pr": {
            "title": "chore: update cert-manager renewal window from 30 days to 7 days",
            "summary": (
                "Aligns with new security policy requiring shorter certificate lifetime "
                "awareness. cert-manager CertificateRequest resource updated. "
                "Tested in staging — cert-manager picked up the new renewBefore and "
                "issued a new cert within 5 minutes."
            ),
            "files_changed": [
                "k8s/cert-manager/certificate.yaml",
                "scripts/cert_health_check.sh",
                "docs/tls_management.md",
            ],
            "risk_level": "high",
        },
        "incident": {
            "title": "{service} mTLS certificate expired — 100% login failure",
            "severity": "SEV1",
            "symptoms": [
                "SSL handshake errors: 'certificate has expired or is not yet valid'",
                "auth-service → user-service gRPC calls failing with UNAVAILABLE",
                "All mobile/web login attempts returning 503",
                "cert-manager logs: 'Not renewing, renewBefore window not reached'",
                "Oncall paged: auth-service availability dropped to 0% for 8 minutes",
            ],
            "root_cause": (
                "When the PR changed renewBefore from 30 days to 7 days, cert-manager "
                "re-evaluated all managed certificates. The existing cert had 23 days "
                "remaining — within the old 30-day window but outside the new 7-day window. "
                "cert-manager determined 'no renewal needed' since 23 > 7. The cert was "
                "never renewed. 23 days later it expired with no automated action triggered "
                "and no monitoring alert because cert-expiry alerting only fires at <7 days."
            ),
            "resolution": (
                "Manually triggered cert rotation: kubectl annotate cert auth-service-tls "
                "cert-manager.io/issueAsCertificateRequest=true. Services recovered in 4 minutes "
                "after cert propagation. Added cert expiry alert at 14 days AND 7 days. "
                "Added CI check: cert expiry must be >30 days after any cert-manager config change."
            ),
        },
        "slack_scenario": "cert_expiry",
    },

    # ── 8 ── Payment SDK major version upgrade ────────────────────────────────
    {
        "id": "sdk_major_version_upgrade",
        "service": "payment-service",
        "pr": {
            "title": "chore: upgrade Stripe SDK from v2.1.3 to v3.0.0",
            "summary": (
                "v3.x required for BNPL and crypto payment methods on Q4 roadmap. "
                "Breaking changes in CHANGELOG: error class hierarchy changed, webhook "
                "signature verification API updated. All 47 tests passing. Staged deployment "
                "to payment-service-canary confirmed no errors in 30-minute soak."
            ),
            "files_changed": [
                "requirements.txt",
                "src/payments/stripe_client.py",
                "src/payments/webhook_handler.py",
                "tests/test_payments.py",
            ],
            "risk_level": "high",
        },
        "incident": {
            "title": "{service} Stripe webhook validation broken — orders stuck in payment_pending",
            "severity": "SEV1",
            "symptoms": [
                "Stripe dashboard: webhook delivery failure rate 100%, all returning HTTP 400",
                "Order status stuck in 'payment_pending' for all orders since 14:32 UTC",
                "Customers charged on Stripe but receiving no order confirmation",
                "payment-service logs: 'SignatureVerificationError: No signatures found'",
                "Support: 300+ tickets 'I was charged but my order disappeared'",
            ],
            "root_cause": (
                "Stripe SDK v3.0.0 changed Webhook.construct_event() to require raw request "
                "bytes for HMAC verification. webhook_handler.py called request.json() (which "
                "parses then re-serializes the body) before passing to construct_event(). "
                "Re-serialization changes key ordering and whitespace, invalidating the HMAC. "
                "The 30-minute canary soak did not catch this because no Stripe webhooks were "
                "delivered to the canary instance during that window."
            ),
            "resolution": (
                "Fix: changed request.json() to request.get_data(as_text=False) in webhook_handler.py. "
                "Replayed all failed webhook events from Stripe dashboard (Events API, last 6 hours). "
                "Manually reconciled 847 orders stuck in payment_pending. Added integration test "
                "using real raw bytes from Stripe test fixture, not reconstructed JSON."
            ),
        },
        "slack_scenario": "sdk_upgrade",
    },

    # ── 9 ── NGINX timeout config drift ────────────────────────────────────────
    {
        "id": "nginx_timeout_config_drift",
        "service": "gateway-api",
        "pr": {
            "title": "infra: increase read timeout for ML recommendations endpoint to 30 s",
            "summary": (
                "New ML inference endpoint /api/v2/recommendations can take 6–8 s for cold "
                "requests. Current 3 s gateway timeout causes premature 504s. Increasing "
                "proxy_read_timeout to 30 s. Change scoped to /api/v2/recommendations only."
            ),
            "files_changed": [
                "k8s/gateway/nginx.conf",
                "k8s/gateway/configmap.yaml",
                "terraform/gateway_lb.tf",
            ],
            "risk_level": "medium",
        },
        "incident": {
            "title": "{service} global 30-second timeout regression — thread pool saturating",
            "severity": "SEV2",
            "symptoms": [
                "All endpoint p99 latencies jumped to exactly 30 000 ms",
                "Requests that previously failed fast at 3 s now hold threads for 30 s",
                "DB connection pool exhausted as long-running queries held connections open",
                "Kubernetes: gateway-api HPA scaling to max replicas (50) but not recovering",
                "Error budget: 9% of monthly SLO budget consumed in 50 minutes",
            ],
            "root_cause": (
                "The NGINX config change placed proxy_read_timeout 30s in the http {} block "
                "instead of inside the location /api/v2/recommendations {} block. This set "
                "the global read timeout for ALL upstream connections to 30 s. Slow upstream "
                "responses that would have been killed at 3 s now hold goroutine/thread resources "
                "for 30 s, cascading into resource exhaustion under normal traffic."
            ),
            "resolution": (
                "Rolled back nginx.conf to previous configmap version via kubectl rollout undo. "
                "Traffic recovered in 90 seconds. Applied corrected config with timeout scoped "
                "inside the specific location block. Added required nginx-config-review step "
                "to gateway PR template. Mandatory staging soak of 15 min for any gateway change."
            ),
        },
        "slack_scenario": "nginx_timeout",
    },

    # ── 10 ── DB deadlock from idempotency locking ────────────────────────────
    {
        "id": "idempotency_deadlock",
        "service": "payment-service",
        "pr": {
            "title": "feat: add idempotency key support to payment API to prevent duplicate charges",
            "summary": (
                "Mobile clients retry payment requests on network failure, causing duplicate "
                "charges. Implementing idempotency keys via DB row lock on idempotency_keys table. "
                "Lock is held for the duration of the Stripe API call to ensure exactly-once "
                "semantics. Includes DB migration adding the idempotency_keys table."
            ),
            "files_changed": [
                "src/payments/idempotency.py",
                "db/migrations/0042_add_idempotency_keys_table.sql",
                "src/api/payment_endpoints.py",
                "tests/test_idempotency.py",
            ],
            "risk_level": "high",
        },
        "incident": {
            "title": "{service} DB deadlocks spiking — 23% payment failure rate",
            "severity": "SEV1",
            "symptoms": [
                "MySQL: 'Deadlock found when trying to get lock; try restarting transaction'",
                "POST /payments error rate: 0.1% → 23% within 6 minutes of deploy",
                "p99 payment latency: 45 ms → 14 000 ms",
                "Retry storms from iOS/Android clients amplifying deadlock contention",
                "Slow query log: SELECT FOR UPDATE on idempotency_keys holding locks >8 s",
            ],
            "root_cause": (
                "The idempotency lock (SELECT FOR UPDATE on idempotency_keys) is held while "
                "waiting for Stripe's response, which takes 2–8 s. A mobile client that retries "
                "after a network timeout creates a second request with the same idempotency key. "
                "Both requests acquire overlapping row locks — one on idempotency_keys, one on "
                "payment_transactions — in opposite order, creating a classic AB/BA deadlock. "
                "Mobile retry logic amplifies this: each deadlock triggers 3 retries, worsening "
                "contention exponentially under load."
            ),
            "resolution": (
                "Replaced SELECT FOR UPDATE with INSERT ... ON DUPLICATE KEY UPDATE (optimistic lock). "
                "Moved Stripe API call outside the transaction boundary. Added innodb_lock_wait_timeout=2. "
                "Added client-side retry jitter (100–500 ms random backoff) in mobile SDK. "
                "Payment failure rate returned to 0.08% within 5 minutes of hotfix."
            ),
        },
        "slack_scenario": "deadlock",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# SLACK CONVERSATION TEMPLATES
# Each maps scenario_id → list of message templates.
# {responder}, {oncall}, {author} are filled in at generation time.
# ─────────────────────────────────────────────────────────────────────────────

SLACK_CONVOS = {

    "rate_limiter": [
        ("ALERT",     "pagerduty-bot",   "🔴 [SEV1] checkout-service: p99 latency exceeded 5000 ms SLO threshold. Runbook: https://wiki/checkout-latency"),
        ("T+1m",      "{oncall}",        "Taking this. Checking dashboards now."),
        ("T+3m",      "{oncall}",        "Seeing massive 429 spike on the ingress. This looks like the WAF or a DDoS. @{eng1} can you check CloudFlare?"),
        ("T+5m",      "{eng1}",          "CloudFlare is clean. No traffic anomaly externally. Traffic profile looks normal."),
        ("T+7m",      "{oncall}",        "Interesting. So the 429s are being generated internally. Checking rate limiter metrics."),
        ("T+9m",      "{eng2}",          "Wait, there was a PR merged this morning — rate limiter config change. PR #112. Changing from sliding window to fixed window."),
        ("T+10m",     "{oncall}",        "Yeah I see it. Fixed window counter is global not per-user. As soon as we hit 100 req total in a minute window, EVERYONE gets blocked. Classic."),
        ("T+11m",     "{eng1}",          "So we're throttling ourselves. The limit was meant to be per-user but it's actually shared. Yikes."),
        ("T+12m",     "{oncall}",        "Rolling back rate_limiter.py to previous commit. Deploying now."),
        ("T+15m",     "{eng2}",          "429s dropping. Latency recovering. Looks like the rollback is working."),
        ("T+17m",     "{oncall}",        "Confirmed. Error rate back to baseline. Incident resolved. Will write postmortem."),
        ("T+20m",     "{eng1}",          "One thing I noticed — the test only covered a single-user scenario. We need a load test that validates multi-user behavior for rate limiter changes."),
        ("T+21m",     "{oncall}",        "Good catch. Adding that to the postmortem action items."),
    ],

    "connection_pool": [
        ("ALERT",     "pagerduty-bot",   "🔴 [SEV1] order-service: error rate exceeded 50% for 5 minutes. Current: 88%."),
        ("T+2m",      "{oncall}",        "On it. order-service is throwing 500s. DB connection errors in logs: 'QueuePool limit overflow'. Checking DB health."),
        ("T+4m",      "{eng1}",          "RDS metrics look fine — CPU at 22%, no long queries. This isn't a DB performance issue."),
        ("T+5m",      "{oncall}",        "Connection pool is exhausted. Max 20 connections, all in use. Why are we using 20 connections simultaneously?"),
        ("T+7m",      "{eng2}",          "There was a deploy 15 minutes ago. The inventory availability check PR. Let me look at it... oh no."),
        ("T+8m",      "{eng2}",          "Found it. For every item in a cart, we're opening a new DB connection to inventory-service. Cart with 10 items = 10 concurrent connections. At 300 checkouts/min that's 3000 simultaneous connections."),
        ("T+9m",      "{oncall}",        "And we have a connection pool of 20. We blow past that in like 2 orders. Plus connections aren't being closed on error paths."),
        ("T+10m",     "{eng1}",          "I'm looking at inventory_client.py. Yep. No finally block. Connection leaked on every exception."),
        ("T+11m",     "{oncall}",        "Two fixes needed: 1) add finally: conn.close(), 2) batch the inventory lookups into a single query. Deploying fix 1 first as a hotfix."),
        ("T+14m",     "{eng2}",          "Fix 1 deployed. Error rate dropping. Down to 35%."),
        ("T+16m",     "{oncall}",        "Fix 2 deployed (batch query). Error rate at 1.2%. Back to normal. Phew."),
        ("T+18m",     "{eng1}",          "We should add a connection pool utilization alert. This would have fired 5 minutes before the incident escalated."),
        ("T+19m",     "{oncall}",        "Agreed. Adding to postmortem. Also need load tests that simulate multi-item carts."),
    ],

    "memory_leak": [
        ("ALERT",     "pagerduty-bot",   "⚠️ [SEV2] auth-service: pod auth-service-7d9f8b-xkp2q OOMKilled. Third OOM in 18 hours."),
        ("T+2m",      "{oncall}",        "Another OOM. Memory grows linearly then crashes. Not a traffic spike — traffic is flat. Something is accumulating."),
        ("T+4m",      "{eng1}",          "Could be a connection leak? Auth-service makes a lot of outbound connections."),
        ("T+6m",      "{oncall}",        "Checked connection count. Stable. It's heap memory growing. I'm pulling a heap dump from the next pod before it OOMs."),
        ("T+12m",     "{eng2}",          "Got the heap dump. Analyzing with MAT... LRUCache is the biggest object. 50,000 entries. Each ~42 KB. That's 2.1 GB right there."),
        ("T+13m",     "{oncall}",        "The JWT cache PR! Was merged 3 days ago. Been gradually filling up ever since. The PR said 50 MB estimate but it's clearly wrong."),
        ("T+14m",     "{eng1}",          "Yeah the estimate assumed 1 KB per entry but a full JWT payload with roles is way larger. And there's no TTL so entries never leave."),
        ("T+15m",     "{eng2}",          "With 50,000 daily active users each getting a new token, the cache fills up and then just sits there forever."),
        ("T+16m",     "{oncall}",        "Immediate fix: add 5-minute TTL and cap memory at 150 MB. Deploying now."),
        ("T+19m",     "{oncall}",        "Deployed. Memory stabilized at 315 MB. OOM-kill loop stopped."),
        ("T+22m",     "{eng1}",          "We need better tooling for cache sizing estimates. The PR calculation was way off."),
        ("T+23m",     "{eng2}",          "And a memory growth alert. If a pod grows by more than X MB/hour, we should know before it OOMs."),
    ],

    "cascade_failure": [
        ("ALERT",     "pagerduty-bot",   "🔴 [SEV1] notification-service: thread pool at 100%. CPU sustained at 98% for 10 minutes."),
        ("T+2m",      "{oncall}",        "notification-service is thrashing. Checking what's filling the thread pool."),
        ("T+3m",      "{eng1}",          "All threads are in retry backoff loops. SendGrid is returning 503."),
        ("T+4m",      "{oncall}",        "Is SendGrid having an outage? Checking their status page."),
        ("T+5m",      "{eng2}",          "SendGrid status shows 'Investigating elevated error rates' as of 35 minutes ago. But why are we still hammering them?"),
        ("T+6m",      "{oncall}",        "We should have backed off by now. Wait — was the circuit breaker removed last week?"),
        ("T+7m",      "{eng1}",          "...yes it was. PR #287. 'remove Hystrix circuit breaker'. Without it, we just retry infinitely."),
        ("T+8m",      "{eng2}",          "And the retry backoff is per-thread in memory. So 2000 threads all in different phases of backoff = constant hammering. We're actually making SendGrid worse."),
        ("T+9m",      "{oncall}",        "We need to stop the retries immediately. I'm deploying an emergency circuit breaker via env vars — CIRCUIT_BREAKER_ENABLED=true CIRCUIT_BREAKER_THRESHOLD=10."),
        ("T+12m",     "{eng1}",          "Thread pool clearing. CPU dropping. SendGrid calls stopped."),
        ("T+14m",     "{eng2}",          "SendGrid recovered. We can start replaying the dead-letter queue, but slowly — 100 msg/min max."),
        ("T+16m",     "{oncall}",        "Replay started. Email backlog clearing. Incident contained."),
        ("T+25m",     "{eng1}",          "Lessons learned: removing resilience patterns needs more scrutiny. The Hystrix removal PR was merged without SRE review."),
        ("T+26m",     "{oncall}",        "Adding SRE mandatory review to any PR touching resilience patterns. And circuit breaker as a required checklist item for external clients."),
    ],

    "stale_cache": [
        ("ALERT",     "pagerduty-bot",   "⚠️ [SEV2] checkout-service: revenue anomaly detected — 14% below forecast for past 4 hours."),
        ("T+3m",      "{oncall}",        "Revenue alert. Checking checkout error rates... looks normal. No 5xx spike. Customers are completing checkouts but... at wrong prices?"),
        ("T+5m",      "{eng1}",          "Support just escalated — 47 tickets about flash sale discount not showing. The sale started at 09:00. It's 13:00 now."),
        ("T+6m",      "{oncall}",        "Product page shows the discounted price. Cart shows full price. Smells like a caching issue."),
        ("T+8m",      "{eng2}",          "Checked the price cache — it's serving prices from last night. Cache TTL was recently changed to 24h. But the webhook should bust the cache on price updates."),
        ("T+9m",      "{oncall}",        "Let me check webhook delivery logs... oh. The webhook has been returning 401 for 6 days."),
        ("T+10m",     "{eng1}",          "The service account token was rotated in the security audit last week. Webhook was never updated. Silent failure — no alerting on the 401s."),
        ("T+11m",     "{eng2}",          "So for 6 days all price updates have been going nowhere. The 24h TTL means the cache is serving week-old prices."),
        ("T+12m",     "{oncall}",        "Two actions: 1) rotate webhook token right now, 2) flush the price cache entirely."),
        ("T+13m",     "{oncall}",        "Token updated. Running cache flush... done. Prices are now current."),
        ("T+15m",     "{eng1}",          "Revenue recovering. Checkout prices now match product pages. Flash sale discount visible."),
        ("T+17m",     "{eng2}",          "We need monitoring on that webhook. If it starts returning 4xx we need to know immediately, not 6 days later."),
        ("T+18m",     "{oncall}",        "Also, the TTL change from 5min to 24h made the blast radius of a broken webhook enormous. We should revert TTL to 5min until we have proper invalidation health monitoring."),
    ],

    "n_plus_one": [
        ("ALERT",     "pagerduty-bot",   "⚠️ [SEV2] order-service: DB CPU at 99%. Slow query log flooded."),
        ("T+1m",      "{oncall}",        "DB is on fire. Checking slow query log... thousands of 'SELECT * FROM products WHERE id = ?' queries. All from order-service."),
        ("T+3m",      "{eng1}",          "Was there a recent deploy? Checking deployment history... order-service deployed 14 minutes ago. 'order history enrichment' feature."),
        ("T+4m",      "{oncall}",        "Let me look at order_history.py... ProductService.get_by_id() called in a loop for each item in each order. Classic N+1."),
        ("T+5m",      "{eng2}",          "A user with 20 orders × 6 items per order = 120 SELECTs just to load their order history page. And we have 700 concurrent users."),
        ("T+6m",      "{oncall}",        "700 × 120 = 84,000 queries/minute. Our DB handles 6,000. We're 14× over capacity."),
        ("T+7m",      "{eng1}",          "Can we feature-flag this off while we fix it?"),
        ("T+8m",      "{oncall}",        "No feature flag in place. I'm reverting the deploy. ETA 3 minutes."),
        ("T+11m",     "{eng2}",          "Revert deployed. DB CPU dropping. 45%... 28%... back to 17%. Read replica catching up."),
        ("T+13m",     "{oncall}",        "Good. Now let's fix the query properly. We need a single batch query: SELECT * FROM products WHERE id IN (all_item_ids_in_all_orders)."),
        ("T+30m",     "{eng1}",          "Fix implemented and tested. Reducing 120 queries to 1 per page load. Deploying with query count assertion in the test."),
        ("T+32m",     "{oncall}",        "Fix deployed. Order history p99 back to 130ms. DB CPU at 16%. All good."),
        ("T+35m",     "{eng2}",          "We need to add query count assertions to any endpoint touching ORM loops. This is the third N+1 this quarter."),
    ],

    "cert_expiry": [
        ("ALERT",     "pagerduty-bot",   "🔴 [SEV1] auth-service: availability 0% for 8 minutes. All health checks failing."),
        ("T+1m",      "{oncall}",        "auth-service completely down. Checking pods... pods are running but not serving traffic. gRPC health check failing."),
        ("T+2m",      "{eng1}",          "Logs: 'tls: failed to verify certificate: x509: certificate has expired'. TLS certificate expired??"),
        ("T+3m",      "{oncall}",        "Certificate expired? cert-manager should handle this. Let me check cert-manager logs."),
        ("T+4m",      "{eng2}",          "cert-manager logs: 'Not renewing certificate auth-service-tls, not within renewBefore window'. The window is 7 days. Cert expires today."),
        ("T+5m",      "{oncall}",        "But renewBefore is 7 days... the cert should have been renewed 7 days ago. Unless the cert had more than 7 days left when the config changed?"),
        ("T+6m",      "{eng1}",          "Found it. PR #341 changed renewBefore from 30d to 7d. At the time of the PR the cert had 23 days left. Outside the new 7-day window. cert-manager never triggered renewal. Then the cert just expired naturally."),
        ("T+7m",      "{eng2}",          "So the config change created a dead zone between 7 and 30 days. Any cert in that range gets stranded."),
        ("T+8m",      "{oncall}",        "I'm manually forcing renewal now: kubectl annotate cert... done. cert-manager is issuing new cert."),
        ("T+12m",     "{oncall}",        "New cert propagated. auth-service responding. Login success rate 100%."),
        ("T+14m",     "{eng1}",          "We need a cert expiry alert at 14 days, not just 7. And a CI check that validates the renewBefore policy won't strand existing certs."),
        ("T+15m",     "{eng2}",          "Also, any cert-manager config change should trigger a full audit of all managed certs to check for stranding."),
    ],

    "sdk_upgrade": [
        ("ALERT",     "pagerduty-bot",   "🔴 [SEV1] payment-service: webhook endpoint returning HTTP 400 for 100% of Stripe events."),
        ("T+2m",      "{oncall}",        "Stripe webhooks all failing with 400. Checking payment-service logs."),
        ("T+3m",      "{eng1}",          "Logs: 'SignatureVerificationError: No signatures found matching the expected signature for payload'. Webhook signature validation is broken."),
        ("T+4m",      "{oncall}",        "Did Stripe change their signing format? Checking Stripe changelog... no changes on their side."),
        ("T+5m",      "{eng2}",          "payment-service had a deploy this afternoon. Stripe SDK v3 upgrade. Let me check the v3 migration guide."),
        ("T+7m",      "{eng2}",          "Found it. v3 changelog says: 'construct_event() now requires raw bytes, not parsed string'. We're passing request.json() which re-serializes the body."),
        ("T+8m",      "{oncall}",        "Re-serialization changes whitespace and key ordering. HMAC over different bytes = signature mismatch. Every time."),
        ("T+9m",      "{eng1}",          "And the canary didn't catch this because no webhooks were delivered to the canary instance during the 30-min soak window."),
        ("T+10m",     "{oncall}",        "Fix: change request.json() to request.get_data(as_text=False). Deploying hotfix."),
        ("T+13m",     "{eng2}",          "Stripe webhooks now returning 200. Webhook delivery recovering."),
        ("T+14m",     "{oncall}",        "Now we need to replay missed events. Pulling event IDs from Stripe API for the last 6 hours."),
        ("T+20m",     "{eng1}",          "847 events replayed. Order statuses reconciled. Revenue impact under assessment."),
        ("T+22m",     "{oncall}",        "Lesson: mock-based webhook tests don't catch raw-vs-parsed body issues. We need integration tests with actual raw bytes from Stripe fixtures."),
    ],

    "nginx_timeout": [
        ("ALERT",     "pagerduty-bot",   "⚠️ [SEV2] gateway-api: p99 latency 30,000 ms across all endpoints. SLO breach."),
        ("T+2m",      "{oncall}",        "Everything is timing out at exactly 30 seconds. Not random — exactly 30s. That's a configured timeout value."),
        ("T+3m",      "{eng1}",          "Was there a gateway deploy today? Checking... yes, NGINX config change for the ML recommendations endpoint. 30-second timeout."),
        ("T+4m",      "{oncall}",        "Checking the nginx.conf diff... the proxy_read_timeout 30s is in the http block, not in the location block. It's global."),
        ("T+5m",      "{eng2}",          "So all endpoints now have a 30-second timeout. Requests that should fail at 3s are now hanging for 30s. Thread pool fills up."),
        ("T+6m",      "{oncall}",        "Rolling back the gateway configmap now. kubectl rollout undo deployment/gateway-api."),
        ("T+8m",      "{oncall}",        "Rollback complete. Latency recovering. 8s... 4s... back to normal."),
        ("T+9m",      "{eng1}",          "I'll reapply the config correctly — timeout inside the location block only."),
        ("T+11m",     "{eng2}",          "That's live. /api/v2/recommendations gets 30s, everything else stays at 3s."),
        ("T+12m",     "{oncall}",        "All clear. Gateway healthy."),
        ("T+14m",     "{eng1}",          "We need mandatory nginx-config-review for gateway changes. And a staging soak that actually exercises all endpoint types, not just the changed one."),
        ("T+15m",     "{oncall}",        "Adding those to the gateway PR template as required steps."),
    ],

    "deadlock": [
        ("ALERT",     "pagerduty-bot",   "🔴 [SEV1] payment-service: error rate 23%. DB deadlock errors spiking."),
        ("T+2m",      "{oncall}",        "Payment failures at 23%. MySQL deadlock errors. Multiple transactions are locking each other."),
        ("T+3m",      "{eng1}",          "Deadlock graph from information_schema shows: T1 holds idempotency_keys row lock, waiting on payment_transactions. T2 holds payment_transactions lock, waiting on idempotency_keys. Classic AB/BA."),
        ("T+4m",      "{oncall}",        "The idempotency PR was merged yesterday. That introduced the idempotency_keys table with SELECT FOR UPDATE."),
        ("T+5m",      "{eng2}",          "The lock is held for the duration of the Stripe API call. Stripe takes 2-8 seconds. So you have a lock held for up to 8 seconds. If a retry comes in during that window — deadlock."),
        ("T+6m",      "{eng1}",          "And mobile clients retry aggressively. Each deadlock triggers 3 retries, which create more deadlocks. It's a feedback loop."),
        ("T+7m",      "{oncall}",        "I'm adding innodb_lock_wait_timeout=2 immediately. That'll at least fail fast instead of holding for 8 seconds."),
        ("T+9m",      "{eng2}",          "Error rate still 18%. Failing faster but still failing."),
        ("T+10m",     "{oncall}",        "We need to fix the root cause. The Stripe call cannot happen inside a transaction holding a row lock. Replace SELECT FOR UPDATE with INSERT ... ON DUPLICATE KEY UPDATE."),
        ("T+25m",     "{eng1}",          "Fix deployed. Optimistic lock replaces pessimistic. Stripe call is now outside the transaction."),
        ("T+27m",     "{oncall}",        "Error rate dropping. 8%... 3%... 0.1%. Back to baseline."),
        ("T+29m",     "{eng2}",          "Also pushed mobile SDK update with retry jitter (100-500ms random backoff). That'll reduce the thundering herd if we ever deadlock again."),
        ("T+31m",     "{oncall}",        "Confirmed stable. Writing incident report now."),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# ADDITIONAL LOWER-FIDELITY SCENARIO SEEDS
# Used to generate the remaining incidents beyond the 10 detailed ones
# ─────────────────────────────────────────────────────────────────────────────

FILLER_INCIDENT_SEEDS = [
    {
        "service": "user-service",
        "title": "user-service pagination returning duplicate records",
        "severity": "SEV3",
        "symptoms": [
            "GET /users?page=N returning overlapping records with page N-1",
            "Frontend infinite scroll showing duplicate user entries",
            "Data inconsistency reported by QA in staging canary",
        ],
        "root_cause": "ORDER BY clause missing on paginated query; cursor-based pagination requires stable sort order.",
        "resolution": "Added ORDER BY created_at, id to paginated query. Deployed hotfix. No data loss.",
        "related_pr_summary": "feat: add cursor-based pagination to user list API",
    },
    {
        "service": "inventory-service",
        "title": "inventory-service bulk import timing out for large catalogs",
        "severity": "SEV3",
        "symptoms": [
            "POST /inventory/bulk-import timing out after 30 seconds for catalogs >5000 SKUs",
            "Partial imports leaving inventory in inconsistent state",
            "Merchants reporting missing SKUs after import",
        ],
        "root_cause": "Bulk import wrapped in single DB transaction. 5000 individual INSERTs in one transaction held table locks for 30+ seconds, triggering gateway timeout.",
        "resolution": "Batched INSERTs into chunks of 500 within separate transactions. Import now runs in ~4 seconds for 5000 SKUs.",
        "related_pr_summary": "feat: implement bulk SKU import endpoint for merchant catalog sync",
    },
    {
        "service": "notification-service",
        "title": "notification-service sending duplicate push notifications",
        "severity": "SEV2",
        "symptoms": [
            "Users receiving 3–5 identical push notifications per order event",
            "Push notification provider billing anomaly: 4× expected volume",
            "App store reviews mentioning notification spam",
        ],
        "root_cause": "Kafka consumer group rebalance during deploy caused messages to be re-processed without idempotency check. Each partition reassignment replayed last 1000 messages.",
        "resolution": "Added deduplication check on event_id before sending push notification. Kafka consumer configured with explicit partition assignment during deploys.",
        "related_pr_summary": "feat: migrate push notification consumer from polling to Kafka",
    },
    {
        "service": "auth-service",
        "title": "auth-service returning 500 for users with unicode characters in display name",
        "severity": "SEV3",
        "symptoms": [
            "Users with emoji or CJK characters in display name getting 500 on profile update",
            "auth-service logs: UnicodeEncodeError: 'latin-1' codec can't encode character",
            "Affects approximately 3% of user base",
        ],
        "root_cause": "New audit logging module used Python's logging with latin-1 encoding for file handler. Any user with non-latin character in their name caused logging to crash mid-request.",
        "resolution": "Updated FileHandler to use UTF-8 encoding. Audit logs now handle full Unicode range.",
        "related_pr_summary": "feat: add user activity audit logging for compliance",
    },
    {
        "service": "payment-service",
        "title": "payment-service refund endpoint returning 200 but not processing refund",
        "severity": "SEV2",
        "symptoms": [
            "POST /payments/{id}/refund returns HTTP 200 but refund_status stays as 'pending'",
            "Stripe dashboard shows no refund initiated",
            "Customer service: 80 tickets about refunds not appearing",
        ],
        "root_cause": "Exception in Stripe API call was silently caught and swallowed by a bare except: pass block introduced in error handling refactor. Endpoint returned 200 even on failure.",
        "resolution": "Removed bare except: pass. Added proper exception handling with logging and 500 response. Manually initiated 80 pending refunds via Stripe dashboard.",
        "related_pr_summary": "refactor: standardize error handling across payment endpoints",
    },
    {
        "service": "checkout-service",
        "title": "checkout-service applying wrong discount tier for bulk orders",
        "severity": "SEV2",
        "symptoms": [
            "B2B customers reporting incorrect discount tiers applied at checkout",
            "Orders with qty > 100 items receiving 5% discount instead of contracted 20%",
            "Finance flagging margin anomaly on B2B segment",
        ],
        "root_cause": "Off-by-one error in discount tier boundary check. Condition was cart.quantity > 100 but should be cart.quantity >= 100. Exactly 100-item orders fell into wrong tier.",
        "resolution": "Fixed boundary condition to >=. Issued credit notes for affected orders in past 30 days.",
        "related_pr_summary": "feat: implement tiered B2B discount pricing engine",
    },
    {
        "service": "order-service",
        "title": "order-service search returning results from wrong tenant in multi-tenant environment",
        "severity": "SEV1",
        "symptoms": [
            "Merchant A seeing order data from Merchant B in search results",
            "tenant_id filter missing from Elasticsearch query",
            "Potential data privacy compliance issue — legal notified",
        ],
        "root_cause": "New Elasticsearch query builder did not include tenant_id filter clause. Single-tenant Elasticsearch index had tenant isolation enforced at query level, not index level.",
        "resolution": "Hotfix added mandatory tenant_id term filter to all search queries. Full audit of all ES queries conducted. No evidence of bulk data exfiltration.",
        "related_pr_summary": "feat: migrate order search from PostgreSQL full-text to Elasticsearch",
    },
    {
        "service": "gateway-api",
        "title": "gateway-api CORS preflight failing for new mobile API version",
        "severity": "SEV3",
        "symptoms": [
            "Mobile app v4.2 users unable to reach any API endpoints on iOS",
            "Browser console: 'CORS policy: No Access-Control-Allow-Origin header'",
            "Affects only requests from app version sending X-App-Version: 4.2",
        ],
        "root_cause": "New gateway CORS policy whitelisted specific Origin headers. Mobile app v4.2 sends a new custom header X-App-Version not in the allowed headers list, causing preflight to fail.",
        "resolution": "Added X-App-Version to Access-Control-Allow-Headers in gateway CORS config. Mobile traffic restored immediately.",
        "related_pr_summary": "security: restrict CORS origins and headers to approved list",
    },
    {
        "service": "user-service",
        "title": "user-service password reset tokens expiring prematurely",
        "severity": "SEV3",
        "symptoms": [
            "Users clicking password reset links within 5 minutes receiving 'token expired'",
            "Token TTL shows as 0 seconds in Redis",
            "Support volume up 3× for password reset issues",
        ],
        "root_cause": "Token expiry calculation used datetime.utcnow() but Redis TTL was set using local server time (UTC+5:30). Net result: tokens expired 5.5 hours before they should, sometimes immediately.",
        "resolution": "Standardized all timestamp generation to UTC. Used int(time.time()) for Redis TTL calculation. All servers confirmed to run in UTC timezone.",
        "related_pr_summary": "feat: implement secure password reset flow with time-limited tokens",
    },
    {
        "service": "inventory-service",
        "title": "inventory-service stock count going negative due to race condition",
        "severity": "SEV2",
        "symptoms": [
            "inventory.stock_count reaching -47 for high-demand SKU",
            "Overselling: 47 orders for an item with 0 remaining stock",
            "Warehouse team unable to fulfill 47 orders",
        ],
        "root_cause": "Stock decrement used read-then-write pattern without atomic operation: read stock, check > 0, subtract 1, write back. Under concurrent checkouts, multiple threads read stock > 0 simultaneously and all decrement, yielding negative values.",
        "resolution": "Replaced read-then-write with UPDATE inventory SET stock = stock - 1 WHERE id = ? AND stock > 0. Affected orders manually cancelled with customer notification.",
        "related_pr_summary": "feat: add real-time stock reservation during checkout",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# RUNBOOK TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

RUNBOOK_TEMPLATES = [
    {
        "service": "checkout-service",
        "issue_type": "high_error_rate",
        "title": "checkout-service High Error Rate",
        "steps": [
            "1. Open Grafana checkout-service dashboard. Check error rate by endpoint.",
            "2. Check rate limiter metrics: `kubectl exec -it <pod> -- curl localhost:9090/metrics | grep rate_limit`",
            "3. If 429 spike: verify rate limiter scope (per-user vs global). Check `config/rate_limits.yaml`.",
            "4. If 500 spike: check application logs for stack traces: `kubectl logs -l app=checkout-service --since=10m | grep ERROR`",
            "5. Check downstream dependency health: payment-service /health, inventory-service /health.",
            "6. If downstream degraded: enable circuit breaker via `CIRCUIT_BREAKER_ENABLED=true` env var.",
            "7. If error rate >30% sustained for 5 min: initiate rollback to previous deployment.",
            "8. Rollback: `kubectl rollout undo deployment/checkout-service`",
            "9. Verify recovery: confirm error rate below 1% for 5 consecutive minutes.",
            "10. Post-incident: collect slow query log, heap dump, and nginx access log for postmortem.",
        ],
        "escalation": "Page checkout-service oncall. If SEV1, notify #incidents and loop in payment-service oncall.",
    },
    {
        "service": "order-service",
        "issue_type": "database_connection_exhaustion",
        "title": "order-service Database Connection Pool Exhaustion",
        "steps": [
            "1. Confirm connection pool exhaustion: check logs for 'QueuePool limit overflow' or 'connection pool exhausted'.",
            "2. Get current pool utilization: `kubectl exec -it <pod> -- curl localhost:9090/metrics | grep db_pool`",
            "3. Identify which query is holding connections: query `information_schema.processlist` on RDS.",
            "4. Check for connection leaks: look for connections in 'Sleep' state held >30s.",
            "5. If leak found: restart affected pods to flush leaked connections: `kubectl rollout restart deployment/order-service`",
            "6. As emergency measure: increase pool size via env var `DB_POOL_SIZE=50` and redeploy.",
            "7. Identify root cause: check recent deploys for new DB-touching code paths. Look for missing connection.close().",
            "8. If N+1 query suspected: check slow query log for repeated single-row SELECTs on same table.",
            "9. Deploy fix or rollback if root cause identified.",
            "10. Add monitoring alert: `db_pool_utilization > 0.75` should page before full exhaustion.",
        ],
        "escalation": "Page order-service oncall. Alert DBA team if RDS CPU >80%. Notify #incidents for SEV1.",
    },
    {
        "service": "auth-service",
        "issue_type": "high_memory_oom",
        "title": "auth-service OOM / High Memory Usage",
        "steps": [
            "1. Confirm OOM: `kubectl describe pod <pod-name> | grep -A5 'Last State'` — look for OOMKilled.",
            "2. Check memory growth rate: Grafana → auth-service → Container Memory. Is it linear (leak) or spike (traffic)?",
            "3. If linear growth: pull heap dump before next OOM: `kubectl exec <pod> -- curl localhost:6060/debug/pprof/heap > heap.prof`",
            "4. Analyze heap dump: look for unbounded in-memory caches (LRU, dict, set) without TTL or size caps.",
            "5. Common culprits: JWT cache (token_cache.py), session store, request/response logging buffers.",
            "6. Immediate relief: increase memory limit temporarily via `MEMORY_LIMIT=2Gi` env var to buy time.",
            "7. Deploy fix: add TTL and memory-based eviction to identified cache. Reduce max entries.",
            "8. Verify fix: monitor memory over 2 hours. Should plateau, not grow.",
            "9. Add alert: `container_memory_usage_bytes > 1.5 * memory_request` → page oncall.",
        ],
        "escalation": "Page auth-service oncall. If all replicas OOMKilling simultaneously, page incident commander — auth down = full site impact.",
    },
    {
        "service": "auth-service",
        "issue_type": "tls_certificate_expiry",
        "title": "auth-service TLS Certificate Expired or Expiring",
        "steps": [
            "1. Confirm cert issue: `kubectl describe cert auth-service-tls | grep -A10 'Status'`",
            "2. Check expiry: `openssl s_client -connect auth-service:443 2>/dev/null | openssl x509 -noout -dates`",
            "3. Check cert-manager logs: `kubectl logs -n cert-manager deploy/cert-manager | grep auth-service-tls`",
            "4. If cert expired and cert-manager not renewing: check renewBefore window vs remaining validity.",
            "5. Force manual renewal: `kubectl annotate cert auth-service-tls cert-manager.io/issueAsCertificateRequest=$(date +%s)`",
            "6. Wait for cert issuance (typically 60–120 seconds). Monitor: `kubectl get cert auth-service-tls -w`",
            "7. Verify new cert propagated to all pods: check READY column shows True.",
            "8. Test mTLS: `grpcurl -insecure auth-service:443 grpc.health.v1.Health/Check`",
            "9. If cert-manager ACME challenge failing: check DNS records and verify challenge solver config.",
            "10. After resolution: audit ALL managed certs for any in the 7–30 day renewal window gap.",
        ],
        "escalation": "Cert expiry = SEV1 (complete auth failure). Page incident commander immediately. Notify #security.",
    },
    {
        "service": "payment-service",
        "issue_type": "webhook_validation_failure",
        "title": "payment-service Stripe Webhook Signature Validation Failures",
        "steps": [
            "1. Confirm via Stripe dashboard: Developers → Webhooks → view recent events. Look for 4xx response codes.",
            "2. Check payment-service logs: `kubectl logs -l app=payment-service | grep -i 'SignatureVerification\\|webhook\\|400'`",
            "3. Verify webhook signing secret is correct: compare env var STRIPE_WEBHOOK_SECRET with Stripe dashboard value.",
            "4. If secret mismatch: update Kubernetes secret: `kubectl create secret generic stripe-webhook-secret --from-literal=key=whsec_... --dry-run=client -o yaml | kubectl apply -f -`",
            "5. If secret correct but still failing: check that raw request body is being passed to construct_event(), not parsed JSON.",
            "6. Test with Stripe CLI: `stripe trigger payment_intent.succeeded` and watch logs.",
            "7. Once fixed, replay missed events: use Stripe API to list events since incident start, replay each via POST /internal/webhook-replay.",
            "8. Reconcile affected orders: query orders with status='payment_pending' and payment_intent_id matching replayed events.",
            "9. Notify finance team of impacted transaction window for reconciliation.",
        ],
        "escalation": "Page payment-service oncall. Notify finance for any reconciliation gaps. SEV1 if duration >15 min.",
    },
    {
        "service": "payment-service",
        "issue_type": "database_deadlocks",
        "title": "payment-service Database Deadlock Surge",
        "steps": [
            "1. Confirm deadlocks: `kubectl logs -l app=payment-service | grep -i deadlock`",
            "2. Get deadlock graph: `SHOW ENGINE INNODB STATUS\\G` on primary DB — look for LATEST DETECTED DEADLOCK.",
            "3. Identify contended tables from deadlock graph. Common: idempotency_keys, payment_transactions.",
            "4. Immediate mitigation: set `innodb_lock_wait_timeout=2` to fail fast and reduce cascading waits.",
            "5. Check if any transaction holds locks during external API calls (Stripe, banks) — this is always a bug.",
            "6. If confirmed: move external API call outside transaction scope. Deploy hotfix.",
            "7. Add retry jitter to clients: `time.sleep(random.uniform(0.1, 0.5))` before retry.",
            "8. Monitor: deadlock rate should drop to near zero within 2 minutes of fix.",
            "9. Review optimistic vs pessimistic locking strategy for affected table.",
            "10. Add deadlock rate alert: >5 deadlocks/minute should page oncall.",
        ],
        "escalation": "Page payment-service oncall. If payment error rate >5%, escalate to SEV1 and notify #incidents.",
    },
    {
        "service": "notification-service",
        "issue_type": "retry_storm_external_provider",
        "title": "notification-service Retry Storm Against External Email/Push Provider",
        "steps": [
            "1. Confirm: check thread pool saturation and external provider error rate in Grafana.",
            "2. Check if external provider has an active incident: status.sendgrid.com or equivalent.",
            "3. Immediate action: enable circuit breaker via env var: `CIRCUIT_BREAKER_ENABLED=true CIRCUIT_BREAKER_THRESHOLD=10`",
            "4. Redeploy notification-service to pick up env var change.",
            "5. Monitor thread pool utilization. Should drop as circuit opens and retries stop.",
            "6. Set retry queue to drain slowly: `RETRY_RATE_LIMIT=100` (messages/minute).",
            "7. If provider recovers: close circuit breaker: `CIRCUIT_BREAKER_ENABLED=false`, replay queue.",
            "8. Monitor replay rate to avoid overwhelming provider on restart.",
            "9. Check dead-letter queue for permanently failed messages: replay after provider recovery.",
            "10. Post-incident: ensure circuit breaker is NOT optional for any external HTTP client.",
        ],
        "escalation": "Page notification-service oncall. Alert downstream teams (payment, auth) if their notification flows are impacted.",
    },
    {
        "service": "checkout-service",
        "issue_type": "stale_price_cache",
        "title": "checkout-service Serving Stale or Incorrect Prices",
        "steps": [
            "1. Confirm stale data: compare price in checkout-service response vs pricing-service API for same SKU.",
            "2. Check cache invalidation webhook health: `kubectl logs -l app=checkout-service | grep price_update_webhook`",
            "3. If webhook returning 4xx: check service account token expiry. Token likely rotated without updating webhook config.",
            "4. Fix token: `kubectl create secret generic pricing-webhook-token --from-literal=token=<new_token> --dry-run=client -o yaml | kubectl apply -f -`",
            "5. Restart checkout-service to pick up new token.",
            "6. Flush price cache for all affected SKUs: `curl -X POST http://checkout-service/admin/cache/flush/prices`",
            "7. Verify prices now correct: spot-check 10 SKUs against pricing-service.",
            "8. If TTL is >5 minutes: revert to 5-minute TTL as a safety measure until invalidation is verified.",
            "9. Add monitoring: webhook endpoint health check with alert on any 4xx/5xx responses.",
        ],
        "escalation": "Page checkout-service oncall. Notify finance and marketing of impacted pricing window. Check if active promotions are affected.",
    },
    {
        "service": "gateway-api",
        "issue_type": "nginx_timeout_misconfiguration",
        "title": "gateway-api NGINX Timeout Misconfiguration",
        "steps": [
            "1. Confirm: check if ALL endpoints are timing out at exactly the same value (e.g., 30s). Uniform timeout = config issue.",
            "2. Check recent gateway deployments: `kubectl rollout history deployment/gateway-api`",
            "3. Diff nginx.conf: `kubectl get configmap gateway-nginx-config -o yaml` and compare with git HEAD.",
            "4. Look for timeout directives in wrong scope (http block vs server block vs location block).",
            "5. Immediate rollback: `kubectl rollout undo deployment/gateway-api`",
            "6. Verify recovery: p99 latency should return to baseline within 60 seconds.",
            "7. Apply corrected config with timeout scoped to specific location block only.",
            "8. Test: verify ML endpoint has 30s timeout, all other endpoints have 3s timeout.",
            "9. Deploy corrected config with canary (10% traffic) and monitor for 15 minutes before full rollout.",
        ],
        "escalation": "Page gateway-api oncall. Global timeout misconfiguration is effectively a site-wide incident — notify #incidents.",
    },
    {
        "service": "order-service",
        "issue_type": "n_plus_one_query",
        "title": "order-service N+1 Query Performance Regression",
        "steps": [
            "1. Confirm via slow query log: `kubectl exec <db-pod> -- mysql -e 'SELECT * FROM slow_log ORDER BY start_time DESC LIMIT 50'`",
            "2. Look for pattern: same table queried N times in rapid succession from same host.",
            "3. Correlate with recent deploy: check if N+1 pattern started at specific deployment time.",
            "4. Identify endpoint: match slow query timestamps with application request logs.",
            "5. Immediate relief: rollback the offending deployment: `kubectl rollout undo deployment/order-service`",
            "6. Verify DB CPU returns to baseline within 3 minutes of rollback.",
            "7. Root cause fix: replace loop-based queries with batch query using WHERE id IN (...).",
            "8. Add query count assertion to integration test before re-deploying.",
            "9. Add DB query rate alert: >20k queries/minute should page oncall.",
        ],
        "escalation": "Page order-service oncall. Alert DBA if DB CPU >80%. If read replica lag >60s, pause async workloads.",
    },
    {
        "service": "inventory-service",
        "issue_type": "stock_race_condition",
        "title": "inventory-service Stock Count Race Condition / Overselling",
        "steps": [
            "1. Confirm negative stock: `SELECT sku, stock_count FROM inventory WHERE stock_count < 0`",
            "2. Identify affected orders: `SELECT * FROM orders WHERE sku IN (<negative_skus>) AND created_at > <incident_start>`",
            "3. Immediate: set stock to 0 for negative-count SKUs to prevent further overselling.",
            "4. Verify the stock update query uses atomic operation: `UPDATE inventory SET stock = stock - 1 WHERE id = ? AND stock > 0`",
            "5. If read-then-write pattern found: deploy hotfix with atomic UPDATE.",
            "6. Contact warehouse team with list of oversold orders.",
            "7. Determine customer impact: cancel orders that cannot be fulfilled or source from alternate warehouse.",
            "8. Add SELECT FOR UPDATE or atomic compare-and-swap to all stock mutation code paths.",
            "9. Add alert: `SELECT COUNT(*) FROM inventory WHERE stock_count < 0` running every 5 minutes.",
        ],
        "escalation": "Page inventory-service oncall. Alert warehouse ops immediately. Notify customer service with affected order IDs.",
    },
    {
        "service": "user-service",
        "issue_type": "password_reset_token_failure",
        "title": "user-service Password Reset Token Failures",
        "steps": [
            "1. Reproduce: request password reset and click link within 2 minutes. If 'token expired' → timestamp bug.",
            "2. Check server timezone: `kubectl exec <pod> -- date` — should output UTC.",
            "3. Check token TTL in Redis: `redis-cli TTL password_reset_token:<token_id>` — should be close to 3600.",
            "4. If TTL is 0 or very small: timezone mismatch in token expiry calculation.",
            "5. Check token generation code for datetime.now() vs datetime.utcnow() vs time.time().",
            "6. Fix: standardize all timestamp generation to UTC. Use `int(time.time())` for Redis TTL.",
            "7. Ensure all Kubernetes nodes and pods are configured with TZ=UTC.",
            "8. Test fix: generate token, check Redis TTL, click link — should succeed within TTL window.",
            "9. Issue new tokens for any users currently locked out.",
        ],
        "escalation": "Page user-service oncall. Notify customer support with workaround (manual account unlock) for affected users.",
    },
    {
        "service": "notification-service",
        "issue_type": "duplicate_notifications",
        "title": "notification-service Duplicate Push Notification Delivery",
        "steps": [
            "1. Confirm: check push notification provider dashboard for delivery volume vs expected.",
            "2. Check Kafka consumer group lag and rebalance events: `kafka-consumer-groups.sh --describe --group notification-consumer`",
            "3. Look for recent consumer group rebalance (common during deploys).",
            "4. Check if deduplication is implemented: grep for event_id or idempotency check in dispatcher.",
            "5. If no deduplication: add event_id check before sending: `if already_sent(event_id): return`",
            "6. Use Redis SET with NX flag and TTL as idempotency store: `SET notif:{event_id} 1 NX EX 86400`",
            "7. For Kafka: reset consumer group offset to after the duplicate window: `kafka-consumer-groups.sh --reset-offsets --to-datetime <time>`",
            "8. Deploy deduplication fix and monitor notification volume returns to expected rate.",
            "9. Notify push provider if over-billing from spike to request credit.",
        ],
        "escalation": "Page notification-service oncall. Alert mobile team if user complaints surge. Check app store reviews.",
    },
    {
        "service": "payment-service",
        "issue_type": "silent_exception_swallowing",
        "title": "payment-service Silent Failure in Payment Processing",
        "steps": [
            "1. Confirm: POST /payments returns 200 but payment_status stays pending in DB.",
            "2. Check Stripe dashboard for corresponding PaymentIntent — was it created?",
            "3. Search code for bare except: pass or except Exception: return None patterns.",
            "4. Add logging to all exception handlers: `except Exception as e: logger.error('Payment failed', exc_info=True)`",
            "5. Replace silent failures with explicit error responses: return HTTP 500 with error details.",
            "6. Deploy fix and verify: payment failures now return non-200 status codes.",
            "7. Replay affected payments: identify orders with status='pending' and created_at during incident window.",
            "8. For each affected order: manually trigger payment retry or notify customer.",
            "9. Run integration test suite to catch any other silent failure patterns.",
        ],
        "escalation": "Page payment-service oncall. Notify finance for reconciliation. Customer service needs list of affected orders.",
    },
    {
        "service": "gateway-api",
        "issue_type": "cors_misconfiguration",
        "title": "gateway-api CORS Misconfiguration Blocking Clients",
        "steps": [
            "1. Reproduce: check browser network tab for preflight (OPTIONS) request. Look for missing CORS headers.",
            "2. Identify which Origin or header is being blocked: check Access-Control-Request-Headers in failed preflight.",
            "3. Check nginx CORS config: `kubectl get configmap gateway-nginx-config -o yaml | grep -A10 cors`",
            "4. Compare allowed headers list with headers being sent by the blocked client.",
            "5. Add missing header to Access-Control-Allow-Headers in nginx config.",
            "6. Also ensure Access-Control-Allow-Methods includes all required HTTP methods.",
            "7. Apply config change and reload nginx: `kubectl rollout restart deployment/gateway-api`",
            "8. Test: trigger preflight request from affected client and confirm 204 response with correct headers.",
            "9. Add CORS smoke test to CI that verifies all known client headers are allowed.",
        ],
        "escalation": "Page gateway-api oncall. Identify which app versions are affected and coordinate with mobile team on timeline.",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# PR GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_prs(scenarios: list, total: int = 50) -> list:
    """
    Generate PR records. Primary PRs come from scenario templates (causally linked).
    Additional 'background' PRs fill the remaining count.
    """
    prs = []
    pr_day = 0  # Offset in days from BASE_DATE

    # ── Primary PRs from scenarios ──────────────────────────────────────────
    for scenario in scenarios:
        pr_id = f"PR-{len(prs) + 100}"
        merged_at = BASE_DATE + timedelta(days=pr_day, hours=random.randint(9, 17))
        service = scenario["service"]
        tmpl = scenario["pr"]

        pr = {
            "pr_id": pr_id,
            "title": tmpl["title"].replace("{service}", service),
            "service": service,
            "summary": tmpl["summary"],
            "files_changed": tmpl["files_changed"],
            "author": rand_engineer(),
            "reviewers": random.sample(ENGINEERS, 2),
            "merged_at": ts(merged_at),
            "risk_level": tmpl["risk_level"],
            "scenario_id": scenario["id"],  # causal link marker
        }
        prs.append(pr)
        pr_day += random.randint(3, 8)

    # ── Background PRs (low-risk, no incident) ──────────────────────────────
    bg_templates = [
        ("fix: correct typo in {service} API documentation", "docs", "low",
         ["docs/api_reference.md"]),
        ("chore: update {service} dependencies to latest patch versions", "dependency-update", "low",
         ["requirements.txt", "package.json"]),
        ("test: add unit tests for {service} input validation layer", "test", "low",
         ["tests/test_validation.py"]),
        ("refactor: extract {service} config loader into separate module", "refactor", "medium",
         ["src/config/loader.py", "src/config/__init__.py"]),
        ("feat: add Prometheus metrics endpoint to {service}", "observability", "low",
         ["src/metrics/prometheus.py", "src/server.py"]),
        ("fix: handle empty response body in {service} API client", "bugfix", "low",
         ["src/clients/api_client.py", "tests/test_api_client.py"]),
        ("chore: upgrade Python base image from 3.11 to 3.12 in {service}", "infra", "medium",
         ["Dockerfile", ".github/workflows/ci.yml"]),
        ("feat: add structured JSON logging to {service}", "observability", "low",
         ["src/logging/structured_logger.py", "src/server.py"]),
        ("fix: handle timezone-aware datetimes in {service} serialization", "bugfix", "low",
         ["src/serializers/datetime_serializer.py"]),
        ("perf: add database index on {service} most-queried columns", "performance", "medium",
         ["db/migrations/add_indexes.sql"]),
    ]

    while len(prs) < total:
        tmpl_title, category, risk, files = random.choice(bg_templates)
        service = random.choice(SERVICES)
        pr_id = f"PR-{len(prs) + 100}"
        merged_at = BASE_DATE + timedelta(days=pr_day, hours=random.randint(9, 17))

        pr = {
            "pr_id": pr_id,
            "title": tmpl_title.replace("{service}", service),
            "service": service,
            "summary": f"Routine {category} change for {service}. No functional behavior changes.",
            "files_changed": [f.replace("{service}", service) for f in files],
            "author": rand_engineer(),
            "reviewers": random.sample(ENGINEERS, 2),
            "merged_at": ts(merged_at),
            "risk_level": risk,
            "scenario_id": None,
        }
        prs.append(pr)
        pr_day += random.randint(1, 4)

    return prs


# ─────────────────────────────────────────────────────────────────────────────
# INCIDENT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_incidents(scenarios: list, prs: list, filler_seeds: list, total: int = 75) -> list:
    """
    Generate incident records. Primary incidents map 1:1 with scenarios
    (and reference their causal PR). Filler incidents add volume.
    """
    incidents = []

    # Build a PR lookup by scenario_id for linking
    pr_by_scenario = {pr["scenario_id"]: pr for pr in prs if pr["scenario_id"]}

    # ── Primary incidents (causally linked to PRs) ───────────────────────────
    for scenario in scenarios:
        causal_pr = pr_by_scenario.get(scenario["id"])
        pr_merged_at = datetime.strptime(causal_pr["merged_at"], "%Y-%m-%dT%H:%M:%SZ")

        # Incident occurs hours to a few days after the PR merges
        incident_delay = timedelta(
            hours=random.randint(2, 72),
            minutes=random.randint(0, 59)
        )
        incident_start = pr_merged_at + incident_delay
        incident_end = incident_start + timedelta(
            minutes=random.randint(20, 120)
        )

        inc_tmpl = scenario["incident"]
        service = scenario["service"]
        incident_id = f"INC-{1000 + len(incidents)}"

        incident = {
            "incident_id": incident_id,
            "title": inc_tmpl["title"].replace("{service}", service),
            "service": service,
            "severity": inc_tmpl["severity"],
            "timestamp": ts(incident_start),
            "resolved_at": ts(incident_end),
            "duration_minutes": int((incident_end - incident_start).total_seconds() / 60),
            "symptoms": inc_tmpl["symptoms"],
            "root_cause": inc_tmpl["root_cause"],
            "resolution": inc_tmpl["resolution"],
            "services_affected": _get_affected_services(service),
            "related_prs": [causal_pr["pr_id"]],
            "oncall_engineer": rand_oncall(),
            "postmortem_url": f"https://wiki.internal/postmortems/{incident_id.lower()}",
            "scenario_id": scenario["id"],  # causal link marker
        }
        incidents.append(incident)

    # ── Filler incidents ─────────────────────────────────────────────────────
    filler_day = 90  # Start filler incidents after primary batch
    for seed in filler_seeds:
        if len(incidents) >= total:
            break

        incident_start = BASE_DATE + timedelta(
            days=filler_day + random.randint(0, 5),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )
        incident_end = incident_start + timedelta(minutes=random.randint(15, 90))
        incident_id = f"INC-{1000 + len(incidents)}"

        # Find a background PR from the same service to link
        same_service_prs = [p for p in prs if p["service"] == seed["service"] and p["scenario_id"] is None]
        related_pr = random.choice(same_service_prs)["pr_id"] if same_service_prs else None

        incident = {
            "incident_id": incident_id,
            "title": seed["title"],
            "service": seed["service"],
            "severity": seed["severity"],
            "timestamp": ts(incident_start),
            "resolved_at": ts(incident_end),
            "duration_minutes": int((incident_end - incident_start).total_seconds() / 60),
            "symptoms": seed["symptoms"],
            "root_cause": seed["root_cause"],
            "resolution": seed["resolution"],
            "services_affected": _get_affected_services(seed["service"]),
            "related_prs": [related_pr] if related_pr else [],
            "oncall_engineer": rand_oncall(),
            "postmortem_url": f"https://wiki.internal/postmortems/{incident_id.lower()}",
            "scenario_id": None,
        }
        incidents.append(incident)
        filler_day += random.randint(2, 6)

    # Pad to target count with variations if needed
    while len(incidents) < total:
        base = random.choice(incidents[:len(scenarios)])
        padded = deepcopy(base)
        padded["incident_id"] = f"INC-{1000 + len(incidents)}"
        padded["timestamp"] = ts(
            BASE_DATE + timedelta(days=random.randint(120, 180), hours=random.randint(0, 23))
        )
        padded["oncall_engineer"] = rand_oncall()
        padded["scenario_id"] = None
        incidents.append(padded)

    return incidents[:total]


def _get_affected_services(primary_service: str) -> list:
    """Return a realistic set of services affected by an incident in primary_service."""
    dependency_map = {
        "checkout-service":     ["payment-service", "inventory-service", "order-service"],
        "order-service":        ["inventory-service", "notification-service", "user-service"],
        "payment-service":      ["checkout-service", "order-service", "notification-service"],
        "auth-service":         ["gateway-api", "user-service"],
        "notification-service": ["order-service", "payment-service"],
        "inventory-service":    ["checkout-service", "order-service"],
        "user-service":         ["auth-service", "checkout-service"],
        "gateway-api":          [s for s in SERVICES if s != "gateway-api"],
    }
    downstream = dependency_map.get(primary_service, [])
    count = random.randint(1, min(3, len(downstream)))
    return [primary_service] + random.sample(downstream, count)


# ─────────────────────────────────────────────────────────────────────────────
# SLACK THREAD GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_slack_threads(incidents: list, prs: list, total: int = 50) -> list:
    """
    Generate Slack threads. Primary threads are tied to scenario-linked incidents
    with rich, realistic conversation. Additional threads cover filler incidents.
    """
    threads = []

    pr_by_scenario = {pr["scenario_id"]: pr for pr in prs if pr["scenario_id"]}

    # ── Primary threads (scenario incidents with full conversation) ──────────
    primary_incidents = [inc for inc in incidents if inc.get("scenario_id")]

    for incident in primary_incidents:
        scenario_id = incident["scenario_id"]
        convo_template = SLACK_CONVOS.get(
            next((s["slack_scenario"] for s in SCENARIOS if s["id"] == scenario_id), None)
        )
        if not convo_template:
            continue

        causal_pr = pr_by_scenario.get(scenario_id)
        incident_time = datetime.strptime(incident["timestamp"], "%Y-%m-%dT%H:%M:%SZ")

        # Pick engineers for this thread
        oncall = incident["oncall_engineer"]
        eng1 = rand_engineer(exclude=oncall)
        eng2 = rand_engineer(exclude=oncall)
        while eng2 == eng1:
            eng2 = rand_engineer(exclude=oncall)

        substitutions = {
            "{oncall}": oncall,
            "{eng1}": eng1,
            "{eng2}": eng2,
            "{pr_id}": causal_pr["pr_id"] if causal_pr else "PR-???",
        }

        messages = []
        offset_minutes = 0

        for (time_label, author_tmpl, text_tmpl) in convo_template:
            # Parse time offset
            if time_label == "ALERT":
                msg_time = incident_time - timedelta(minutes=2)
            else:
                # e.g. "T+7m" → 7 minutes
                mins = int(time_label.replace("T+", "").replace("m", ""))
                msg_time = incident_time + timedelta(minutes=mins)

            # Substitute author and text
            author = author_tmpl
            for k, v in substitutions.items():
                author = author.replace(k, v)
                text_tmpl = text_tmpl.replace(k, v)

            messages.append({
                "timestamp": ts(msg_time),
                "author": author,
                "text": text_tmpl,
            })

        thread = {
            "thread_id": f"SLACK-{len(threads) + 1:04d}",
            "incident_id": incident["incident_id"],
            "service": incident["service"],
            "channel": f"#incidents-{incident['service']}",
            "thread_start": ts(incident_time - timedelta(minutes=2)),
            "thread_end": ts(incident_time + timedelta(minutes=40)),
            "participants": list({oncall, eng1, eng2, "pagerduty-bot"}),
            "messages": messages,
        }
        threads.append(thread)

    # ── Fill remaining threads with shorter filler conversations ────────────
    filler_incidents = [inc for inc in incidents if not inc.get("scenario_id")]

    generic_convos = [
        [
            ("ALERT", "pagerduty-bot", "⚠️ [{severity}] {service}: alert threshold breached."),
            ("T+2m", "{oncall}", "Checking. Looks like {symptom}. Investigating."),
            ("T+5m", "{eng1}", "I see it too. Could be related to recent deploy."),
            ("T+8m", "{oncall}", "Found the issue: {root_cause_short}. Deploying fix."),
            ("T+12m", "{eng1}", "Fix deployed. Metrics recovering."),
            ("T+15m", "{oncall}", "Confirmed resolved. Writing up incident report."),
        ],
        [
            ("ALERT", "pagerduty-bot", "🔴 [{severity}] {service}: error rate exceeding SLO."),
            ("T+3m", "{oncall}", "On it. {symptom}. Checking recent changes."),
            ("T+6m", "{eng1}", "There was a deploy 20 minutes ago. Checking diff."),
            ("T+9m", "{oncall}", "Root cause: {root_cause_short}. Rolling back."),
            ("T+13m", "{eng1}", "Rollback successful. Error rate normalizing."),
            ("T+16m", "{oncall}", "Incident closed. Postmortem scheduled for tomorrow."),
        ],
    ]

    for incident in filler_incidents:
        if len(threads) >= total:
            break

        incident_time = datetime.strptime(incident["timestamp"], "%Y-%m-%dT%H:%M:%SZ")
        oncall = incident["oncall_engineer"]
        eng1 = rand_engineer(exclude=oncall)

        convo = random.choice(generic_convos)
        messages = []

        for (time_label, author_tmpl, text_tmpl) in convo:
            if time_label == "ALERT":
                msg_time = incident_time - timedelta(minutes=2)
            else:
                mins = int(time_label.replace("T+", "").replace("m", ""))
                msg_time = incident_time + timedelta(minutes=mins)

            author = author_tmpl.replace("{oncall}", oncall).replace("{eng1}", eng1)
            text = (text_tmpl
                    .replace("{severity}", incident["severity"])
                    .replace("{service}", incident["service"])
                    .replace("{symptom}", incident["symptoms"][0] if incident["symptoms"] else "anomaly detected")
                    .replace("{root_cause_short}", incident["root_cause"][:80] + "...")
                    .replace("{oncall}", oncall)
                    .replace("{eng1}", eng1))

            messages.append({
                "timestamp": ts(msg_time),
                "author": author,
                "text": text,
            })

        thread = {
            "thread_id": f"SLACK-{len(threads) + 1:04d}",
            "incident_id": incident["incident_id"],
            "service": incident["service"],
            "channel": f"#incidents-{incident['service']}",
            "thread_start": ts(incident_time - timedelta(minutes=2)),
            "thread_end": ts(incident_time + timedelta(minutes=20)),
            "participants": [oncall, eng1, "pagerduty-bot"],
            "messages": messages,
        }
        threads.append(thread)

    return threads[:total]


# ─────────────────────────────────────────────────────────────────────────────
# RUNBOOK GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_runbooks(runbook_templates: list) -> list:
    """
    Generate runbook records from templates. Returns structured runbook objects
    and also writes individual Markdown files.
    """
    runbooks = []

    for tmpl in runbook_templates:
        runbook_id = f"RB-{len(runbooks) + 1:03d}"

        runbook = {
            "runbook_id": runbook_id,
            "title": tmpl["title"],
            "service": tmpl["service"],
            "issue_type": tmpl["issue_type"],
            "last_updated": ts(BASE_DATE + timedelta(days=random.randint(0, 30))),
            "owner_team": f"{tmpl['service'].split('-')[0]}-eng",
            "steps": tmpl["steps"],
            "escalation": tmpl.get("escalation", "Page service oncall. Escalate to SEV1 if not resolved in 30 minutes."),
            "related_runbooks": [],
        }
        runbooks.append(runbook)

    # Add cross-references between related runbooks
    service_groups = {}
    for rb in runbooks:
        service_groups.setdefault(rb["service"], []).append(rb["runbook_id"])

    for rb in runbooks:
        related = [r for r in service_groups.get(rb["service"], []) if r != rb["runbook_id"]]
        rb["related_runbooks"] = related

    return runbooks


def runbook_to_markdown(runbook: dict) -> str:
    """Render a runbook dict to Markdown format."""
    lines = [
        f"# {runbook['title']}",
        f"",
        f"**Runbook ID:** {runbook['runbook_id']}  ",
        f"**Service:** `{runbook['service']}`  ",
        f"**Issue Type:** `{runbook['issue_type']}`  ",
        f"**Owner Team:** {runbook['owner_team']}  ",
        f"**Last Updated:** {runbook['last_updated']}  ",
        f"",
        f"---",
        f"",
        f"## Remediation Steps",
        f"",
    ]
    for step in runbook["steps"]:
        lines.append(step)

    lines += [
        f"",
        f"---",
        f"",
        f"## Escalation",
        f"",
        runbook["escalation"],
    ]

    if runbook.get("related_runbooks"):
        lines += [
            f"",
            f"---",
            f"",
            f"## Related Runbooks",
            f"",
        ]
        for rb_id in runbook["related_runbooks"]:
            lines.append(f"- [{rb_id}](../{rb_id}.md)")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def save_output(prs: list, incidents: list, slack_threads: list, runbooks: list, base_dir: str = "data"):
    """Write all generated data to structured folders."""
    dirs = {
        "prs":       Path(base_dir) / "prs",
        "incidents": Path(base_dir) / "incidents",
        "slack":     Path(base_dir) / "slack",
        "runbooks":  Path(base_dir) / "runbooks",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # PRs — one file per PR + combined index
    for pr in prs:
        path = dirs["prs"] / f"{pr['pr_id']}.json"
        path.write_text(json.dumps(pr, indent=2), encoding="utf-8")

    (dirs["prs"] / "_index.json").write_text(
        json.dumps(prs, indent=2), encoding="utf-8"
    )

    # Incidents
    for inc in incidents:
        path = dirs["incidents"] / f"{inc['incident_id']}.json"
        path.write_text(json.dumps(inc, indent=2), encoding="utf-8")

    (dirs["incidents"] / "_index.json").write_text(
        json.dumps(incidents, indent=2), encoding="utf-8"
    )

    # Slack threads
    for thread in slack_threads:
        path = dirs["slack"] / f"{thread['thread_id']}.json"
        path.write_text(json.dumps(thread, indent=2), encoding="utf-8")

    (dirs["slack"] / "_index.json").write_text(
        json.dumps(slack_threads, indent=2), encoding="utf-8"
    )

    # Runbooks — JSON + Markdown
    for rb in runbooks:
        json_path = dirs["runbooks"] / f"{rb['runbook_id']}.json"
        json_path.write_text(json.dumps(rb, indent=2), encoding="utf-8")

        md_path = dirs["runbooks"] / f"{rb['runbook_id']}.md"
        md_path.write_text(runbook_to_markdown(rb), encoding="utf-8")

    (dirs["runbooks"] / "_index.json").write_text(
        json.dumps(runbooks, indent=2), encoding="utf-8"
    )

    return dirs


def print_summary(prs, incidents, slack_threads, runbooks, dirs):
    print("\n" + "=" * 60)
    print("  SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"  PRs generated:           {len(prs)}")
    print(f"  Incidents generated:     {len(incidents)}")
    print(f"  Slack threads generated: {len(slack_threads)}")
    print(f"  Runbooks generated:      {len(runbooks)}")
    print()
    print("  Causal chain coverage:")
    linked = sum(1 for inc in incidents if inc.get("scenario_id"))
    print(f"    Scenario-linked incidents:  {linked}/{len(incidents)}")
    print(f"    Rich Slack threads:         {sum(1 for t in slack_threads if len(t['messages']) > 8)}")
    print()
    print("  Output directories:")
    for name, path in dirs.items():
        count = len(list(path.glob("*.json"))) - 1  # exclude _index.json
        print(f"    {path}  ({count} files)")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Generating PRs...")
    prs = generate_prs(SCENARIOS, total=55)

    print("Generating incidents...")
    incidents = generate_incidents(SCENARIOS, prs, FILLER_INCIDENT_SEEDS, total=75)

    print("Generating Slack threads...")
    slack_threads = generate_slack_threads(incidents, prs, total=50)

    print("Generating runbooks...")
    runbooks = generate_runbooks(RUNBOOK_TEMPLATES)

    print("Saving output...")
    dirs = save_output(prs, incidents, slack_threads, runbooks, base_dir="data")

    print_summary(prs, incidents, slack_threads, runbooks, dirs)


if __name__ == "__main__":
    main()
