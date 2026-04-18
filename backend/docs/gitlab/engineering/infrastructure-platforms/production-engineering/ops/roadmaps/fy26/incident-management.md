---
title: "FY26 - Incident Management"
description: "Goals and plans for incident management in FY26"
---

## Prior art and links

1. [Incident Management Overview](https://docs.google.com/document/d/1NRo3AmttIw9aHN5KlRfeHIkWLIqKChumx3dMRSFrI44/edit?tab=t.0#heading=h.qhr2mkyyxfpj) from 2024-02
1. [Overhauling Incident Management](https://docs.google.com/document/d/13Iukuc3rYoF8TO_GSUxiA8MFm1QsBoloFXSkP8n7BHI/edit?tab=t.0) from 2024-02

## Background

Incident Management includes:

1. Incident Management Processes
1. Staffing for Incident Response  
1. Tooling

In this document, we are setting out the FY26 goals for these areas.

## North star

Teams can support the services they build, including taking ownership of on-call for them. The Ops team offers a platform that enables teams to easily find help when they need it and ensures they adhere to standard processes.

## Goals for FY26

### Incident management processes

1. Standard, well-defined incident workflows across all incident types
   1. Clear role definitions and responsibilities
   1. Uniform processes for SIRT and regular incidents
   1. Effective handling of concurrent incidents
1. High-quality, maintained runbooks owned by service teams
1. Comprehensive, measurable IMOC program
   1. Clear performance expectations and feedback
   1. Standardized training and documentation
1. Data-driven incident management improvements through robust reporting

### Tooling

1. Implement incident.io for GitLab.com and Dedicated
1. Modify post-incident process to incorporate incident.io
1. Ensure accurate incident data collection occurs on all incidents

### Staffing for incident response

1. Focused EOC role with clear scope and responsibilities
1. Reliable incident coordinator role with consistent performance
1. Well-defined infra leader role with clear regional coverage
1. Engineering-wide on-call program

Each of these goals represents a desired end state. The specific implementation plans, including how we'll achieve these goals, what tools we'll use, and what processes we'll follow, will be determined separately based on the prioritization of these goals.

We need to have team members to staff incident response and they need to be equipped to execute on the workflows.

**The EOC role** currently includes many non-incident related tasks. This role needs to be refined and focused. The EOC should not be used for activities that should be fed into specific teams.

**The IM role** needs support to become a reliable incident coordinator. With a large membership pool and a long time between participation on incidents, it is easy for best practices to be forgotten. For the most critical incidents, we often default to relying on specific team members to carry us though. If we want this to be the standard, we should be specific about this.

**The Infra Leader role** thankfully has not needed to be used much in the past few months. However, we needed clearer guidance about what this role entails, who is part of this rotation, and how handoffs between regions are handled.

### EOC in Engineering

There is a project to establish wider on-call responsibilities in Engineering. Please see [FY26 Directional Theme - On-Call Enhancements across Engineering](https://docs.google.com/document/d/1KGFMyHR2pTw_mkeOGxhG1YBY-MZDb3mwEnUd4Zdwl8M/edit?tab=t.0) for details.
