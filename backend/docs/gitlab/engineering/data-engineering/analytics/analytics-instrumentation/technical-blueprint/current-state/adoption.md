---
title: Adoption Across Teams and Categories
description: "How good is the instrumentation coverage across teams and categories"
---

## Overview

This [Current Metrics](https://10az.online.tableau.com/#/site/gitlab/views/AnalyticsInstrumentationCoverage/CurrentMetrics?:iid=1) tab within the [Analytics Instrumentation Coverage](https://10az.online.tableau.com/#/site/gitlab/workbooks/2437638/views) dashboard provides an overview of the coverage of metrics across the GitLab product. This page presents an aggregated view of the data displayed in the [Metrics Dictionary](https://metrics.gitlab.com/).

The categories listed in the dashboard are described in the [product categories](/handbook/product/categories/) handbook page, and sourced from the [categories.json](https://about.gitlab.com/categories.json) file. Teams are responsible for adding accurate category information at the time of instrumentation. The [categories lookup](/handbook/product/categories/lookup/) page provides a lookup table with the details around each product category.

Selecting the `Product Categories` dimension and the `Total Metric Count` metric gives a breakdown of the different metrics by each category.

The information regarding Sections, Stages and Groups is sourced from [stages.yaml](https://gitlab.com/gitlab-com/www-gitlab-com/blob/master/data/stages.yml) and [sections.yaml](https://gitlab.com/gitlab-com/www-gitlab-com/blob/master/data/sections.yml).

## Observations

### Product category is missing in 53.2% of the metrics

- As of October 2025, there are a total of 4522 metrics
- Out of these, 4010 metrics are active.
- Out of the 4010 active metrics, 111 metrics do not have any section+stage+group information.
- Out of 4010 active metrics, 2134 metrics do not have any product category information.
- Out of 2134 metrics without any product category information, only 5 metrics are missing the team information as well.
- **This means that 2129 active metrics can be labelled with product category information with the help of the teams that own them.**

### Not all teams have adopted Instrumentation

- The [stages.yaml](https://gitlab.com/gitlab-com/www-gitlab-com/blob/master/data/stages.yml) and [sections.yaml](https://gitlab.com/gitlab-com/www-gitlab-com/blob/master/data/sections.yml) list 77 groups, 20 stages and 15 sections.
- The metrics dashboard shows that the metrics are owned by 39 distinct groups, across 16 stages and 11 sections.
- This means that 37 groups, 4 stages, and 4 sections do not own any metrics.
