---
title: OAK Testing and Integration Ownership
owning-stage: "~devops::gitlab delivery"
toc_hide: true
---

## Summary

This ADR defines the testing responsibilities and integration ownership model for OAK (Omnibus-Adjacent Kubernetes) as it scales to support multiple advanced components. It establishes clear boundaries between Omnibus OAK core functionality and advanced component integrations to prevent breaking changes and ensure maintainability.

## Problem Statement

As OAK grows to support multiple advanced components (OpenBao, future components), each managed by different Product teams, there is a risk of:

1. **Integration drift**: When an advanced component's Helm chart or configuration changes, Omnibus automation may become outdated without anyone noticing
2. **Unclear ownership**: It's unclear who is responsible for updating Omnibus when a component changes
3. **Testing gaps**: No clear process for ensuring component integrations remain functional across OAK updates
4. **Maintenance burden**: Omnibus team cannot realistically test and maintain all component integrations

## Decision

We establish a **distributed responsibility model** where:

### Operate Team Responsibilities (Initial Implementation)

1. The Operate team is responsible for the **first iteration** of OAK:
    - **Define OAK patterns**: Establish patterns for NGINX configuration generation, Helm values generation, PostgreSQL/Redis network exposure
    - **Implement OpenBao integration**: Demonstrate the pattern with OpenBao as the first advanced component
    - **Create integration framework**: Build the mechanism for Omnibus to generate component-specific Helm values and NGINX configs
    - **Document patterns**: Document the required patterns and expectations for future components
    - **OAK core CI/CD pipeline**: Automated tests for OAK core functionality
    - **Breaking Changes**: When initiated by Omnibus core framework changes, the Operate team owns the whole deprecation announcement and breaking change process on both projects, the component project, and the Omnibus. The Product team may
    the process, but the Operate team coordinates.
2. The Operate team is responsible for defining the patterns of what Omnibus configures/automates.
3. What Operate does NOT own (after Beta):
    - Maintaining individual component integrations
    - Component-specific Helm chart validation
    - Component-specific configuration details

### Advanced Component Product Team Responsibilities (Future Components)

1. Each Product team managing an advanced component owns:
    - **Helm chart maintenance**: Maintaining their component's Helm chart and ensuring compatibility with OAK patterns
    - **Omnibus integration updates**: Updating Omnibus automation when their Helm chart structure changes
    - **Integration testing**: Tests that verify their component works with OAK-generated values
    - **Configuration documentation**: Documenting expected values and any breaking changes
    - **Breaking Changes**: When initiated by chart changes, the Product teams own the whole deprecation announcement and breaking change process on both projects, the component project, and the Omnibus. The Operate team may support the process,
    but the Product team coordinates.
    - **Implmenting Omnibus changes**: When ther Helm chart structure changes
2. What Product teams do NOT own:
    - OAK core patterns and framework
    - Omnibus automation for other components
