---
title: Observability Based Performance Testing at GitLab
---

## Description

Observability Based Performance Testing is a proactive approach to understanding system performance through comprehensive instrumentation and real-time data collection. Unlike traditional performance testing, which relies on specific test scenarios, observability testing provides deep visibility into application behavior under real-world conditions.

## Approach

Observability testing is actively making use of our [Observability tools](#tools-and-technologies) to detect trends that would develop into performance issues. A couple common approaches:

- Have the development teams monitor the dashboards on their components and proactively pickup performance concerns
- Build dashboards/tooling that support doing exploratory testing on the Observability data, looking for linkages that may not be obvious (system A causes system B to slow down)
  - Tools like the [Performance Bar](https://docs.gitlab.com/ee/administration/monitoring/performance/performance_bar.html) can enable someone to notice a performance oddity and start the investigation into the root cause
- Extend existing Observability tooling to development/test environments
  - This enables teams to get performance metrics earlier in the development process
  - This increases team familiarity with the tooling which facilitates easier adoption/use

## Key Components

- **Instrumentation**: Embedding traces, metrics, and structured logging across the application stack.
- **Real-time Data Collection**: Gathering performance data continuously during actual usage.
- **Holistic Analysis**: Examining the entire system to identify bottlenecks and patterns.

## Goals

- **Enhanced Performance Visibility**: Increase team awareness of their work's performance impact.
- **Proactive Issue Detection**: Identify potential problems before they affect customers.
- **Team Empowerment**: Provide accessible performance analysis tools that don't require specialized expertise.

## Non-Goals

- **Replace Load Testing**: Observability-based testing complements, but does not replace, well-designed load tests like those using [GPT](https://gitlab.com/gitlab-org/quality/performance).

## Implementation Approach

### Phase 1: Foundation Setting

1. Education & Documentation
   - Add [documentation](https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks/-/work_items/3322) on what this process is so teams understand it
   - Create [training](https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks/-/work_items/3325) to help teams understand and adopt
   - Create [guides](https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks/-/work_items/3323) to enable teams to build / use performance dashboards
2. Baseline Assessment & Infrastructure
   - There are a number of existing dashboards, identify existing ones that already cover what we need
   - Identify gaps and develop plans to fill

### Phase 2: Initial Implementation

1. Team Selection
   - Select a single team to pair with
   - Work with to identify / build dashboards for their use
   - Integrate with their workflow
2. Feedback Loop & Documentation
   - Bring lessons learned back into documentation for other teams

### Phase 3: Practice and Culture Building

1. Expand to Early Adopters
2. Process Integration
   - Document how to use in GitLab Dev workflows

### Phase 4: Expansion and Maturity

1. Full Scale Rollout
2. Continuous Improvement
3. Advanced Capabilities
   - **Machine Learning Integration**: Implement ML algorithms to predict performance issues based on observability data patterns.
   - **Enhanced Developer Tools**: Create IDE plugins that provide real-time performance insights during code writing.

## Case Studies

### Case Study 1: Scrum team using Observability Based Performance Testing

#### Background

The team was building an ETL (Extract, Transform, Load) project with critical performance requirements. A few hours of system delay would create an insurmountable backlog in the ETL process, making performance monitoring essential to business operations. The team was small (9 developers, 2 QA, 1 Ops) and deployed to production every other week, adding a heavy weight process would of introduced a significant burden. So, we implemented using our existing Observability stack.

#### Observability implementation

We implemented a three-tiered monitoring approach:

1. User Experience Monitoring
   - Used existing Apdex-based dashboards for both production and test environments
   - Tracked end-user performance metrics like page load times and API response times
   - Set clear thresholds for acceptable performance levels
2. Service-Level Monitoring
   - Created detailed dashboards for key services showing:
     - Resource utilization (CPU, memory, I/O)
     - Service dependencies and their performance
     - Queue lengths and processing rates
   - Established KPI tracking for critical ETL operations
3. Error Tracking
   - Implemented comprehensive error logging
   - Created error pattern recognition dashboards
   - Set up alerting for error rate thresholds

#### Daily Workflow

1. Dashboard Review (5 minutes in daily standup)
   - Team reviewed all three dashboard tiers
   - Focused on identifying patterns rather than immediate problem-solving
   - Used a structured approach:
     - Review user experience metrics
     - Check service health indicators
     - Examine error patterns
2. Issue Management
   - Created issues for concerning patterns
   - Prioritized performance issues alongside feature work
   - Allocated dedicated investigation time in sprint planning

#### Implementation Challenges

- Small team size and fast release cycle was a constraint on what we could deliver
- Balancing prioritizing performance investigation with new feature development was difficult
  - Performance improvements often competed with feature deadlines
  - Sometimes had to make trade-offs between immediate fixes and long-term solutions
- It took some trial and error to identify the right metrics/dashboard configuration for our team
  - What we thought was the right metric to monitor proved not to be correct
  - The infrastructure evolved incrementally so we had to evolve the dashboards to keep in sync

#### Outcomes

- Met performance KPIs consistently
- Maintained 99.999% uptime during 10x growth period
  - Zero performance-related incidents
  - No emergency load testing needed
  - Proactively scaled based on observability data
- Lessons Learned:
  - Early warning from observability prevented major incidents
  - Daily review habit kept performance top-of-mind
  - Small, regular improvements are better than big reactive changes

## Tools and Technologies

- Prometheus
- Grafana
- [Add other relevant tools used at GitLab]

## Future Directions

[TBD]

## Resources

- GitLab Handbook pages
  - [GitLab Performance Monitoring](https://gitlab.com/help/administration/monitoring/performance/index)
  - [Observability for stage groups](https://gitlab.com/help/development/stage_group_observability/index)
- [Link to related blog posts or external resources]
- [Information on available training or workshops]
