---
# This is the title of your design document. Keep it short, simple, and descriptive. A
# good title can help communicate what the design document is and should be considered
# as part of any review.
title: Dashboard Customization Framework
status: "ongoing"
creation-date: "2025-06-16"
authors: [ "@jiaan" ]
coaches: [ "@ahegyi" ]
dris: [ "@lfarina8", "@nicholasklick" ]
owning-stage: "~devops::analytics"
participating-stages: [ "~devops::analytics", "~devops::security risk management" ]
# Hides this page in the left sidebar. Recommended so we don't pollute it.
toc_hide: true
---

<!-- Design Documents often contain forward-looking statements -->
<!-- vale gitlab.FutureTense = NO -->

<!-- This renders the design document header on the detail page, so don't remove it-->
{{< engineering/design-document-header >}}

<!--
Don't add a h1 headline. It'll be added automatically from the title front matter attribute.

For long pages, consider creating a table of contents.
-->

## Table of Contents

- [Summary](#summary)
- [Motivation](#motivation)
  - [Goals](#goals)
  - [Non-Goals](#non-goals)
- [Proposal](#proposal)
- [Design and implementation details](#design-and-implementation-details)
  - [Dashboard layout](#dashboard-layout)
  - [Panels](#panels)
  - [Filters](#filters)
  - [Customization persistence](#customization-persistence)
- [Getting started](#getting-started)
- [Upgrading from static dashboards](#upgrading-from-static-dashboards)

## Summary

This document builds on the dashboard foundation established by the [Dashboard Layout Framework](../dashboard_layout_framework/_index.md),
it outlines the architecture for a dashboard customization framework that enables user-driven dashboard creation and personalization across GitLab.

The dashboard customization framework provides Vue components and UI patterns that allow for user-driven customization of dashboard layouts.
Users can use it to add/remove panels, position & resize elements, edit text, and to save their personalized dashboard configurations. The framework
is agnostic about where dashboard configurations are stored, instead focusing purely on the customization user experience.

The customizable dashboard component is a drop-in replacement for existing use cases of the dashboard layout framework. Internally it wraps the
dashboard layout component and extends its capabilities to allow for user-driven customization that align with GitLab's Pajamas design system.

## Motivation

Currently dashboards provide valuable insights but are static in nature. Users cannot personalize their dashboards to focus on the metrics most
relevant to their workflows, responsibilities, or user roles. These severely reduce the effectiveness of dashboards as an analytics tool
for our customers.

Without user-customization, product teams must anticipate, design, and continuously maintain multiple dashboards to satisfy ever changing user personas,
use cases, and organizational requirements. This creates a significant maintenance burden and resource allocation challenge, as teams must spend considerable
time building and updating dashboard layouts and visualizations instead of focusing on core feature development and data quality improvements.

### Goals

- Enable users to personalize dashboard layouts, panel selection, and configurations to match their specific needs
- Establish the UI pattern for user-driven dashboard customization to deliver a consistent UX across our applications
- Adopt the customization framework across GitLab where customization would increase user satisfaction and experience
- The customization framework must be a drop-in replacement for use cases of the dashboard layout framework
- Outline a clear upgrade path for static dashboards and how teams can integrate customization into their existing dashboards

### Non-Goals

The dashboard customization framework does not:

- Include [Data Exploration](../gitlab_data_exploration/_index.md) outside defined panel visualizations
- Define where [dashboards should be listed or placed in the navigation](https://gitlab.com/gitlab-org/gitlab/-/issues/536612)
- Define how or where dashboard configurations are stored

## Proposal

Create a new customizable dashboard component to help teams extend their dashboard layouts with user customization in accordance with our modular
dashboard foundations [epic](https://gitlab.com/groups/gitlab-org/-/epics/18072). The component must be a drop-in replacement for existing
use cases of the dashboard layout component. The component should wrap the dashboard layout component and extend its capabilities to deliver
a consistent dashboard customization UX while maintaining the core dashboard layout component's functionality and API.

## Design and implementation details

The framework extends the dashboard layout with a new editing state that users can enter to customize their dashboards. When a user is done
editing the dashboard layout, the dashboard should return back to the static dashboard state. Any changes made by the user should be preserved
in the static state. The user should be provided the option to return to the edit state through a button or secondary option. Any saved changes
should be visible to all authorized users the next time they load the dashboard.

![dashboard_customization_framework_outline](/images/engineering/architecture/design-documents/dashboard_customization_framework/dashboard_customization_framework_outline.png)

### Dashboard layout

When not editing:

- The dashboard behaves the same as a static dashboard
- The edit action is available to authorized users who can make dashboard changes

When editing:

- Users can edit the title and description
- Users can discard unsaved changes and revert the dashboard back to its original state
- The dashboard produces a modified configuration when the user opts to save changes made
- The dashboard shows a warning message when unsaved changes will be lost
- Allow users to add new or GitLab-provided visualizations via a custom interface (optional)

### Panels

When not editing, panels look and behave the same as on a static dashboard.

When editing:

- Panels can be dragged and snapped into a new grid position
- Panels can be resized with a new height and width
- Panels can be removed (optional)

### Filters

When not editing, dashboard-level filters should look and behave the same as on a static dashboard.

When editing, the framework should hide dashboard-level filters by default to reduce visual clutter.

### Customization persistence

The framework does not define how dashboard configurations should persist and consumers are responsible for implementing this feature.
Persistence is planned as an upcoming framework feature as part of the dashboard foundations roadmap [epic](https://gitlab.com/groups/gitlab-org/-/epics/18072).
Any customization persistence should adhere to the following:

- Customizations made to dashboards should be traceable with change events persisted
- Customizations must be stored using the dashboard configuration object, see an example in the dashboard layout
[documentation](https://docs.gitlab.com/development/fe_guide/dashboard_layout_framework/#basic-implementation)

## Getting started

_TODO: Add link to the relevant developer documentation section: https://gitlab.com/gitlab-org/gitlab/-/issues/549982_

## Upgrading from static dashboards

_TODO: Add link to the relevant developer documentation section: https://gitlab.com/gitlab-org/gitlab/-/issues/549982_
