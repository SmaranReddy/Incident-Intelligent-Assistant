---
title: "P&L Allocation"
description: "P&L allocation documentation"
---

## Overview of the Data Pipeline

The Google Cloud Platform (GCP) billing Profit & Loss (P&L) allocation pipeline takes raw GCP billing data and allocates each line item to P&L categories for analysis and reporting. It provides a daily overview of GCP costs by P&L category (free. internal, paid) and GitLab plan (Free, Ultimate, Premium, etc.).

The pipeline consists of the following key steps:

1. Combine GCP billing line items with credits to caclulate the final charges (`fct_gcp_billing_line_item`)
2. Calculate usage percentages by P&L categories and GitLab plan and apply these to known free/paid/internal costs per input from FP&A and FinOps (`prep_gcp_billing_attribute_ratio_daily`, `prep_repo_storage_ratio_daily`, `prep_container_registry_ratio_daily`, `prep_ci_build_artifact_ratio_daily`, `prep_ci_runner_ratio_daily`, `prep_haproxy_ratio_daily`, `prep_pubsub_ratio_daily`)
3. Apply the combined P&L mappings to the raw GCP data, finding the best match for each line item (`mart_gcp_billing_line_item`)
4. Apply the lookback mappings to the raw GCP data (`rpt_gcp_billing_lookback`)
5. Apply the lookback mapping and output final allocated costs by P&L category and applies a hierarchy for exploration on Tableau (`rpt_gcp_billing_line_item_with_lookback`)

## Lineage

### Step 1: GCP Billing Line Items

First, combine the GCP Billing line items (`summary_gcp_billing_source`) with the credits (`prep_gcp_billing_credit`) to calculate the true `net_cost`.

### Step 2: P&L Metric Definitions

The next step in the process is calculating metrics for known GCP use cases so we can apply these ratios to the actual costs.

These metric definitions are owned by FinOps and FP&A and come from various sources:

1. GitLab.com: Calculate the daily ratio of actual usage of various GitLab features (i.e. CI Runners, repository storage) by `plan` and `pl_category` by mapping the usage to the namespace's GitLab plan at the time of usage
2. GCP Billing attributes: Based on a sheetload, map GCP folders, projects, labels, etc. to `pl_category` and apply the supplied ratio
3. Prometheus (`periodic_queries`): Calculate the raily ratio of HAProxy egress bytes usage and map it to `pl_category` based on a sheetload of backend mappings

| Model Name    | Source(s) |   Metric(s) Definition    |
|-----------|--------|--------------|
|`prep_ci_build_artifact_ratio_daily`    |GitLab Project Statistics|    % Build artifacts usage per plan/P&L Category per day |
|`prep_repo_storage_ratio_daily` |  GitLab Project Statistics|   % Repo storage usage per plan/P&L Category per day |
|`prep_container_registry_ratio_daily`    | GitLab Project Statistics|    % Container registry usage per plan/P&L Category per day |
|`prep_ci_runner_ratio_daily`   | GitLab CI Runner Activity |    % CI consumption in ci.minutes per runner type per plan/P&L Category per day |
|`prep_gcp_billing_attribute_ratio_daily`|  Sheetload  | GCP projects by parent folder, Specific GCP project costs, GCP resources by infrastructure label mapped to P&L categories |
|`prep_haproxy_ratio_daily` |    Prometheus, Sheetload |    Network usage per backend in gigabyte per day |
|`prep_pubsub_ratio_daily`|   Manual  |    PubSub usage per day |

### Step 3: First Attribution of GCP Line Items

Next, in `mart_gcp_billing_line_item`, we attribute each GCP line item to the metric which matches best, based on the following identifiers (in order of importance):

- GCP full path
- GCP service description
- GCP SKU description
- Infrastructure label
- Environment label
- Runner label
- GCP Project ID

Once the best match is selected, the line is split across the P&L categories and GitLab plans. The usage amount and net cost fields are adjusted based on the percent of metric usage calculated in `Step 2`.

### Step 4: Calculate Lookback

The lookback mappings are used to retroactively apply updated profit and loss (P&L) allocations to historical data. This ensures consistency in P&L reporting over time.
In Google's Cloud Billing data, our commitment costs are still incurred on the projects using the eligible compute resources. Once we apply the P&L split to a certain area of costs, the Committed Use Discount (CUD) lines are not mapped and must be mapped by looking back on the same parameters.

There are two types of Commited Use Discounts to apply:

- **Flex CUD**: A type of Committed Use Discount in Google Cloud Platform that provides flexibility in usage.
- **T2D**: Refers to a specific type of Google Cloud Platform resource or service, likely in the context of a Committed Use Discount.

These two lookbacks are generated in `rpt_gcp_billing_lookback` where we calculate the daily ratio of billed net cost for each type of CUD based on rules that define the Commited Use Discount line items.

### Step 5: Apply Lookback and Generate Reporting Fields

The final step is to reattribute the lines previously attributed in `mart_gcp_billing_line_item` based on the CUD lookbacks calculted in `Step 3`. We find the best match between the GCP line items and the lookback mappings, peferring matches based on the same attributes.

Lastly, we derive fields for reporting such as `product_category`, `finance_sku_type`, and hierarchical columns.
