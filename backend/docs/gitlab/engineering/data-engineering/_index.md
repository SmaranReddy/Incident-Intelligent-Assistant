---
title: "Data Engineering and Monetization"
description: "Building the unified data foundation, both operational and analytical, that scales GitLab across every deployment model and unlocks intelligent monetization."
---

## Mission

We build the unified data foundation, both operational and analytical, that scales GitLab across every deployment model and unlocks intelligent monetization. By connecting fragmented systems into a seamless, low-touch ecosystem and ensuring zero data issues during migrations and upgrades, we enable customers to adopt new features faster while transforming raw data into leading indicators across customer journeys that accelerate growth and competitive advantage.

## Vision

We envision GitLab to define Developer-Led Economy: A global shift where software developers, empowered by agents and data-driven platforms, are the core drivers of innovation, growth, and competitive advantage—similar to how oil defined industrial power in the 20th century.

## Organization Structure

```mermaid
flowchart LR
    DEAM[Data Engineering and Monetization]
    click DEAM "/handbook/engineering/data-engineering/"

    DEAM --> AN[Analytics]
    click AN "/handbook/engineering/data-engineering/analytics"
    DEAM --> MON[Monetization]
    DEAM --> DE[Database Excellence]
    click DE "/handbook/engineering/data-engineering/database-excellence/"

    AN --> AI[Analytics Instrumentation]
    click AI "/handbook/engineering/data-engineering/analytics/analytics-instrumentation"
    AN --> Optimize
    click Optimize "/handbook/engineering/data-engineering/analytics/optimize"
    AN --> PI[Platform Insights]
    click PI "/handbook/engineering/data-engineering/analytics/platform-insights"

    MON --> Growth
    click Growth "/handbook/engineering/development/growth"
    MON --> Fulfillment
    click Fulfillment "/handbook/engineering/development/fulfillment"

    Fulfillment --> FP[Fulfillment Platform]
    click FP "/handbook/engineering/development/fulfillment/fulfillment-platform"
    Fulfillment --> Provision
    click Provision "/handbook/engineering/development/fulfillment/provision"
    Fulfillment --> SEATM[Seat Management]
    click SEATM "/handbook/engineering/development/fulfillment/seat-management"
    Fulfillment --> SUBM[Subscription Management]
    click SUBM "/handbook/engineering/development/fulfillment/subscription-management"

    Growth --> Acquisition
    click Acquisition "/handbook/engineering/development/growth"
    Growth --> Activation
    click Activation "/handbook/engineering/development/growth"
    Growth --> Engagement
    click Engagement "/handbook/engineering/development/growth"

    DE --> DBF[Database Frameworks]
    click DBF "/handbook/engineering/data-engineering/database-excellence/database-frameworks"
    DE --> DBO[Database Operations]
    click DBO "/handbook/engineering/data-engineering/database-excellence/database-operations"
    DE --> DBH[Database Health]
```
