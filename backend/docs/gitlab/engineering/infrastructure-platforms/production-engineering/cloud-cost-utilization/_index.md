---
title: "Cloud Cost Utilization Team"
---

The [Cloud Cost Utilization function](../../../../../job-description-library/engineering/infrastructure/site-reliability-engineer#finops) brings experience in both Site Reliability Engineering (SRE) and Software Engineering (SWE), leveraging these skills to optimize the financial operations of our cloud services and data resources. This technical expertise enables us to not only ensure alignment with financial objectives but also drive operational efficiency at scale. Our team’s deep understanding of cloud cost structures, infrastructure, data management, and automation empowers us to manage the full lifecycle of cloud consumption, from cost allocation to detailed analysis, while maintaining the high standards of reliability and performance expected in modern cloud environments.

|  |  |
|--|--|
| Team | EM: @lmcandrew, SRE: @tonyganga, @irotman |
| Issue Tracker | [Cloud Cost Utilization Issue Tracker](https://gitlab.com/gitlab-com/gl-infra/finops/team/-/boards/5046766) |
| Framework Board | [Cloud Cost Utilization Framework Activities](https://gitlab.com/gitlab-com/gl-infra/finops/team/-/boards/5046766?group_by=epic) |
| Slack | [#g_cloud-cost-utilization](https://gitlab.enterprise.slack.com/archives/C09MUMXRECC) |

## Our Core Responsibilities

We focus on several key activities:

- Analysis of costs related to cloud infrastructure.
- Providing insights into cloud resource utilization, identifying inefficiencies, and helping stakeholders optimize their spend.
- Provide analysis on spend trends, forecasting future expenses, and helping teams plan their budgets accordingly.
- Helping stakeholders access and interpret the cloud cost data they need for decision-making.

We achieve these responsbilies by:

- **Forecasting**: To help predict future cloud spending based on historical trends, planned initiatives, and business growth patterns. Accurate forecasting enables proactive budget planning, prevents unexpected cost overruns, and supports strategic decision-making around infrastructure investments and capacity needs.

- **Budgeting**: To ensure consistent and expected cloud spend across teams and projects by establishing clear financial guardrails. This process involves setting spending limits, allocating costs to appropriate cost centers, and creating accountability mechanisms that align cloud consumption with business objectives and financial constraints.

- **Wastage Monitoring (Resource Optimization)**: To proactively ensure efficient resource utilization by identifying and eliminating unnecessary spending. This includes detecting idle resources, rightsizing over-provisioned instances, removing orphaned assets, and optimizing reserved capacity to maximize the value derived from every dollar spent on cloud infrastructure.

- **Label Compliance (Tagging Strategy)**: To ensure the fundamentals of our infrastructure are properly attributed, which forms the bedrock of proper reporting. Consistent tagging enables accurate cost allocation and granular visibility into spending patterns across teams, projects, environments, and business units. See our [Labeling Strategy](labeling-strategy) page for details on the core labels we're implementing.

It's important to also call out there is a lifecycle to these items and it all starts with **Label Compliance**.

Effective cloud resource and cost management begins with Label Compliance to establish proper attribution and visibility, which then enables three parallel disciplines:

- **Budgeting** to set spending guardrails and accountability,
- **Forecasting** to predict and plan for future spend
- **Wastage Monitoring** to continuously identify optimization opportunities

All three work simultaneously and interdependently once the foundational tagging strategy is in place.

## Our Goals

- **Enable comprehensive cost visibility and accountability**
  - Build and maintain the foundational infrastructure for accurate cost tracking (tagging, account structure, governance)
  - Establish clear cost allocation and real-time spending visibility across teams and business units
  - Create dashboards and reporting that tie infrastructure cloud costs directly to business outcomes and ownership (DRI-level attribution)

- **Drive proactive cost optimization and efficiency**
  - Identify and eliminate infrastructure cloud waste through continuous monitoring of spending patterns
  - Implement cost-saving strategies like right-sizing, reserved instances, and spot pricing
  - Work with engineering teams to optimize resource usage without compromising performance
  - Track and measure optimization impact

- **Establish governance and enable cost-conscious culture**
  - Implement automated controls, approval workflows, and policies to prevent cost overruns while enabling velocity
  - Support budget planning, spending alerts, and proactive cost management
  - Create infrastructure-as-code standards that enforce cost best practices
  - Foster cost awareness and accountability throughout the engineering organization

## How We Work

If something requires immediate attention, please tag the relevant person/team in the GitLab issue and include a clear description of the urgency.
DMs and ad-hoc requests can lead to duplicated work or missed context and should be avoided in favor of formal issue submission. Slack should be used primarily for quick clarifications or urgent escalations but not for initiating work requests.

Whenever possible, we encourage stakeholders to refer to the data available in our data warehouses (e.g., Snowflake, Google BigQuery, etc) to find the data they need. When in doubt, [check dbt](https://dbt.gitlabdata.com/#!/overview). If you still can't find what you need, reach out to us in an issue.

- **Work Requests:** All tasks and requests should be tracked via [issues in GitLab](https://gitlab.com/gitlab-com/gl-infra/finops/team/-/issues/new?description_template=issue). This allows for clear documentation, prioritization, and tracking of requests.
- **Roadmap**: Our [roadmap](https://gitlab.com/gitlab-com/gl-infra/finops/team/-/issues/198) is updated weekly.

## Educating Stakeholders

- We encourage stakeholders to access our data repositories and take ownership of their use cases and views. We do our best to ensure data we find useful finds its way into our [data warehouse](../../../../enterprise-data/platform/_index.md).

> **Important:** The Cloud Cost Utilization team does not create or maintain dashboards (including Tableau) for other teams, as this requires deep domain knowledge that resides with the business stakeholders. Creating dashboards for the entire business is not scalable for our small team and would detract from our core responsibilities of cost analysis and optimization.

- We provide the necessary data foundation, but visualization and dashboard creation are the responsibility of the teams who best understand their specific needs and data context.

📊 **Need Tableau help?** The Data team provides documentation on [getting started with Tableau](../../../../enterprise-data/organization/programs/data-for-product-managers/index.md).

### Engaging with Stakeholders

We manage a variety of stakeholder relationships across the organization. Each stakeholder group has specific needs, and we aim to support them in a way that’s both efficient and scalable.

- **FP&A:** Provide cost insights and analysis for financial forecasting.
- **Data Teams (PDI & AI):** Collaborate to ensure the data required from our stakeholders is present in the data warehouse. This ensures structure and aligns with both operational and financial needs.
- **Engineering and Infrastructure:** Work closely with engineering teams to analyze resource utilization and identify cost-saving opportunities in cloud infrastructure.
