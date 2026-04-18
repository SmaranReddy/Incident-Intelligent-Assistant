---
title: "Sidekiq - Background Processing"
---

## Status

**Candidate** - Under consideration for approval

## Owner

[To be assigned]

## Description

Sidekiq is a background job processing framework for Ruby applications that uses Redis for job storage and queuing.

## Why This Should Be a Key Abstraction

Sidekiq is being considered as the standard background processing system for GitLab because:

1. **Proven at Scale**: Successfully handles millions of jobs across GitLab.com
2. **Ruby Native**: Seamlessly integrates with GitLab's Rails application
3. **Self-Managed Compatible**: Works reliably in all GitLab deployment environments

## Use Cases

Use Sidekiq for:

- Standard background job processing
- Asynchronous task execution
- Scheduled jobs and cron-like tasks
- Any feature that needs to work in self-managed instances (SMF/SMA)
- Core GitLab functionality requiring background processing

## Do Not Use For

Consider alternatives for:

- **Very high throughput scenarios**: Use NATS (candidate key abstraction) for extreme throughput requirements
- **Real-time event streaming**: NATS may be more appropriate for under 1ms latency requirements

## Integration Guidance

### For New Features

When implementing background processing for new features:

1. **Default to Sidekiq**: Unless you have specific performance requirements, use Sidekiq
2. **Measure First**: If you think you need NATS, gather performance data first
3. **Consider Self-Managed**: If the feature must work for self-managed customers, use Sidekiq
4. **Follow Patterns**: Use established Sidekiq patterns and worker conventions in the GitLab codebase

## Related Key Abstractions

- [**NATS**](nats/) - Alternative for very high throughput scenarios
- **Redis** - Required dependency for Sidekiq
- **PostgreSQL** - Often used alongside Sidekiq for persistent data storage

## Support and Resources

- GitLab [Sidekiq development documentation](https://docs.gitlab.com/development/sidekiq/)
- Sidekiq [development configuration](https://docs.gitlab.com/development/sidekiq/worker_attributes/) and [operator configuration](https://docs.gitlab.com/administration/sidekiq/)
- Internal [monitoring dashboards](https://dashboards.gitlab.net/dashboards/f/sidekiq/sidekiq-service) and [runbooks](https://gitlab.com/gitlab-com/runbooks/-/tree/master/docs/sidekiq)
- Infrastructure team office hours and Slack channels
