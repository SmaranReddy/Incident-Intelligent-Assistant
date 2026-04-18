---
title: Testing
---

Welcome to the Testing Guide. Pages in this section provides information about
testing practices, methodologies, and tools used in our development workflow.
Effective testing is crucial for maintaining code quality, preventing
regressions, and ensuring that our software meets requirements.

## Introduction to Testing at GitLab

This introduction provides new engineers with an
overview of our testing philosophy, practices, and the support available to help
you contribute effectively to our quality engineering efforts.

## How We Test

### Our Testing Philosophy

At GitLab, we believe that quality is everyone's responsibility, and testing is
integrated into every stage of our development process rather than being a
separate phase. Our approach is built on industry best practices and GitLab's
core values.

**Test Pyramid Approach**: [We champion the concept of the test pyramid](https://docs.gitlab.com/development/testing_guide/testing_levels/),
prioritizing fast, reliable tests at the base (unit tests) while using fewer,
more focused tests at higher levels (integration and end-to-end). This approach
gives us:

- Rapid feedback during development
- Reliable detection of regressions
- Efficient use of CI/CD resources
- Maintainable test suites

**Strategic Testing Focus**: Our testing strives to consider risk analysis and
input on test strategy, helping us focus testing efforts on critical user
journeys and high-impact areas. We make strategic decisions about where to
invest our testing efforts based on user impact and business needs.

**Quality Gates**: Testing is embedded throughout our [product development workflow](/handbook/product-development/how-we-work/product-development-flow/):

- Pre-commit and pre-receive hooks for immediate feedback
- [Merge request pipelines with mandatory code reviews](../../engineering/workflow/code-review/) that must pass before code integration
- Deployment pipelines with comprehensive test suites
- Post-deployment monitoring and validation

### Testing Ownership Model

**Everyone Tests**: While we have dedicated the Test Governance team and greater
Developer Experience department, every product engineering team is responsible for creating and
maintaining tests for their features. Developer Experience are there for supporting
engineers in creating comprehensive test coverage that includes:

- Writing unit tests for new functionality
- Adding integration tests for API endpoints and service interactions
- Contributing to end-to-end test coverage for critical user flows
- Maintaining and fixing flaky or outdated tests

**Test Governance Support**: The Test Governance team provides:

- Testing infrastructure and tooling through [Developer Experience teams](../../engineering/infrastructure-platforms/developer-experience/)
- Collaboration with [Developer Experience for development workflow support](../../engineering/infrastructure-platforms/developer-experience/)
- Guidance on testing strategies and best practices
- Support for complex testing scenarios
- Test automation frameworks and libraries

## When We Test

### Development Workflow Integration

**Testing Early and Often**: We encourage writing tests alongside feature
development to ensure clear requirements understanding, better code design,
and comprehensive coverage from the start.

**Continuous Integration**: Every merge request triggers automated testing:

- Unit and integration tests run on every push
- Feature tests execute for UI changes
- Performance tests validate critical paths
- Security scans check for vulnerabilities
- End-to-end tests for critical user journeys

### Release and Deployment Testing

**Pre-Deployment Validation**: Before code reaches production:

- [Smoke tests verify basic functionality](https://docs.gitlab.com/development/testing_guide/end_to_end/debugging_end_to_end_test_failures/#staging-canary) with staging-canary blocking deployments
- Performance tests ensure acceptable response times
- End-to-end tests validate critical user journeys
- Canary deployments allow gradual rollout with monitoring

**Post-Deployment Monitoring**: Testing doesn't stop at deployment. Post-deployment monitoring is also done including:

- Synthetic monitoring simulating user interactions
- Performance monitoring tracking application health
- Error tracking identifying issues in real-time
- [Feature flag testing enabling safe experimentation](https://docs.gitlab.com/development/testing_guide/end_to_end/feature_flag_testing/)

## Support Available

### Getting Help with Testing

**Request for Help Process**: When you need testing support or guidance, use
our established request process:

- Create an issue using the [testing support template](https://gitlab.com/gitlab-org/quality/test-governance/request-for-help)
- Include context about your testing challenges or requirements
- Expect response within our defined SLA timeframes

### Developer Experience Department

**Developer Experience Team**: [Provides testing infrastructure, tools, and frameworks](../../engineering/infrastructure-platforms/developer-experience/) including:

- Testing pipeline optimization
- Test automation libraries and utilities
- CI/CD testing infrastructure
- Performance testing capabilities

**Test Governance Team**: Support specific product areas with:

- Testing strategy guidance
- Complex test scenario design
- Flaky test investigation and resolution
- Test coverage analysis and recommendations

**On-Call Support**: [Engineering teams participate in incident management rotations](../../engineering/on-call/) to ensure rapid response to production issues

### Self-Service Resources

#### Documentation and Guides

- [GitLab Testing Guide](https://docs.gitlab.com/development/testing_guide) - Guidelines for automated testing in the GitLab project
- [Testing Levels, Tooling, and Strategy](https://docs.gitlab.com/development/testing_guide/testing_levels/) - Detailed technical implementation guide
- [Testing Best Practices](https://docs.gitlab.com/development/testing_guide/best_practices/) - Everything you should know about how to write good tests in the GitLab project
- [Code Review Guidelines](../../engineering/workflow/code-review.md) - Mandatory review process for all merge requests

#### Test Health and Pipeline Stability

- [Flaky Tests](flaky-tests/_index.md) - Automated detection and reporting
  - [Reporting of Top Flaky Test Files](flaky-tests/_index.md#reporting-of-top-flaky-test-files) - Weekly assignments for high-impact flaky tests
- [Product Engineer guide to E2E test failure issues](guide-to-e2e-test-failure-issues.md)
- [Unhealthy Tests (Developer Docs)](https://docs.gitlab.com/development/testing_guide/unhealthy_tests/) - Technical debugging reference for GitLab contributors
- [🪄 Debug MR Test Failures with Duo](using-duo-to-debug-test-failures.md#-using-duo-to-debug-and-fix-test-failures-in-your-merge-request) - Use Duo to quickly diagnose and fix test failures in your MR
- [🔥 Debug Live Environment Test Failures with Duo](using-duo-to-debug-test-failures.md#-using-duo-to-debug-live-environment-test-failures) - Use Duo to quickly diagnose and fix test failures in your MR

#### 📹 GitLab End-to-End Testing Overview (Video)

<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/KbQzrVJMvNQ" frameborder="0" allowfullscreen="true"> </iframe>
</figure>

**Duration:** ~30 minutes
**Level:** Beginner to Intermediate

This video covers:

- 📁 [Directory structure and test organization](https://www.youtube.com/watch?v=KbQzrVJMvNQ&t=150)
- 🕵️ [Finding and understanding existing tests](https://www.youtube.com/watch?v=KbQzrVJMvNQ&t=322)
- 🤿 [Deep dive into E2E test architecture](https://www.youtube.com/watch?v=KbQzrVJMvNQ&t=399)
- 📍 [Where E2E tests run in our infrastructure](https://www.youtube.com/watch?v=KbQzrVJMvNQ&t=978)
- 🐛 [Debugging test failures from merge requests](https://www.youtube.com/watch?v=KbQzrVJMvNQ&t=1084)
- 🚩 [Working with feature flags in tests](https://www.youtube.com/watch?v=KbQzrVJMvNQ&t=1406)
- 🔧 [Troubleshooting common failure issues](https://www.youtube.com/watch?v=KbQzrVJMvNQ&t=1552)
- 💻 [Running tests locally in your GDK](https://www.youtube.com/watch?v=KbQzrVJMvNQ&t=1721)

- [Presentation Slides](https://docs.google.com/presentation/d/1eYLuTdSpI-H0ZalzoqH7Ee8cprjmL1FNoV0XwRPY4-4/edit?usp=sharing)

#### Further Reading

- [The Practical Test Pyramid](https://martinfowler.com/articles/practical-test-pyramid.html) - A deep dive into the "Test Pyramid"

#### Community and Communication

- Testing-focused Slack channels for questions and discussions are namely #g_test_governance and the broader #s_developer_experience

#### Tooling and Automation

- Test generators and templates for common scenarios
- [Automated workflow tooling](../../engineering/infrastructure-platforms/developer-experience/workflow-automation/) for issue and MR triage
- CI/CD pipeline templates with testing best practices
- Performance and coverage monitoring dashboards

---

*For detailed technical implementation guidance, refer to our comprehensive
[Development Testing Guide](https://docs.gitlab.com/development/testing_guide/).
For immediate testing support, use our [request for help process](https://gitlab.com/gitlab-org/quality/test-governance/request-for-help).*
