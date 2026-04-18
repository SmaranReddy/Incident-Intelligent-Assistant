---
title: "Dependency Scanning Analyzer"
status: ongoing
creation-date: "2024-08-14"
authors: [ "@hacks4oats", "@gonzoyumo" ]
coaches: [ ]
dris: [ "@johncrowley", "@thiagocsf", "@nilieskou" ]
owning-stage: "~devops::application security testing"
participating-stages: []
# Hides this page in the left sidebar. Recommended so we don't pollute it.
toc_hide: true
---

<!-- This renders the design document header on the detail page, so don't remove it-->
{{< engineering/design-document-header >}}

<!--
Don't add a h1 headline. It'll be added automatically from the title front matter attribute.
For long pages, consider creating a table of contents.
-->

## Summary

The dependency scanning feature has been historically powered by a set of analyzers - `gemnasium`, `gemnasium-maven`, and `gemnasium-python`. Associated with CI templates, these analyzers have the responsibility of detecting supported projects, building the dependency graph or list when needed, parsing the detected dependencies, and finally, producing a security report with detected vulnerabilities alongside a CycloneDX SBOM that contains the dependencies. This approach has worked well, but over time it's become evident that the actions required to build a project's dependency graph exports come with a lot of complexity. This complexity negatively impacts the maintenance and creation of features, and the user experience of setting up and maintaining the dependency scanning analyzer.

To address these challenges, we are redesigning the dependency scanning analyzer to follow a multi-tiered approach that balances accuracy with ease of use. This document outlines the overall vision and architecture of the new analyzer, while specific implementation decisions are documented in the [Architectural Decision Records (ADRs)](#decisions) section.

## Motivation

The high cost associated with building the dependency graphs/list exports motivates us to rethink how we can structure the dependency scanning feature. Instead of building the project dependency graphs or lists on behalf of customers and within the analyzer, we can delegate this responsibility to a job that runs before the analyzer does. A build stage is a very common part of the development cycle, and generating the dependency artifacts during this stage is a lot simpler than mapping existing build system configuration values to the ones used by the gemnasium set of analyzers.

The high maintenance cost associated with building the dependency graphs/list exports has pushed us to rethink how we can structure the dependency scanning feature. Instead of building the project dependency graphs or lists on behalf of customers and within the analyzer, we can delegate this responsibility to a job that runs before the analyzer does. A build stage is a very common part of the development cycle, and generating the dependency artifacts during this stage is a lot simpler than mapping existing build system configuration values to the ones used by the gemnasium set of analyzers. So we initially considered deferring this entirely to users (see [ADR 001: Graph Export Only](./decisions/001_graph_export_only.md)) but eventually faced customer feedback and other challenges that forced us to revisit this design.

## Goals

- Provide a simplified, maintainable analyzer that reduces the attack surface and maintenance burden
- Support multiple dependency detection strategies to accommodate different project configurations
- Enable out-of-the-box dependency scanning for projects with committed lockfiles or graphfiles
- Support automatic dependency resolution for projects that require build steps
- Provide a fallback mechanism for projects without pre-generated dependency artifacts
- Reduce security maintenance costs by eliminating bundled runtimes and package managers from the analyzer image
- Removal of historical limitations like single project analysis for Java and Python monorepos

## Non-Goals

- Supporting 3rd party SBOM generators. We can still support this in a future iteration.

## Proposal

### Design Principles

- **Separation of Concerns**: Dependency detection (what components exist) is separated from vulnerability analysis (which components have vulnerabilities)
- **Minimal Image Footprint**: The analyzer image contains only the scanning logic, not build tools or runtimes
- **Flexibility**: Different projects can use different dependency detection strategies based on their needs

### Dependency detection

The new dependency scanning analyzer follows a multi-tiered approach to dependency detection, providing flexibility while maintaining accuracy.

For more details on the dependency detection approach, including the service-based resolution pattern and manifest parsing implementation, see [ADR 003: Dependency Resolution and Manifest Scanning](./decisions/003_dependency_resolution_and_manifest_scanning.md).

#### Tier 1: Lockfile/Graphfile Present (Highest Accuracy)

When projects have committed or pre-generated lockfiles or graphfiles, the analyzer consumes them directly. This provides the most accurate dependency information with minimal processing overhead.

#### Tier 2: Automatic Dependency Resolution

For projects that require build steps to generate dependency artifacts, the analyzer supports automatic dependency resolution through preceding CI jobs that run in the `.pre` stage. These jobs:

- Use ecosystem-native tools (Maven, Gradle, Python's `uv`) in vanilla public images
- Run the Dependency Scanning analyzer as a CI service to provide the necessary detection logic and generate the instructions for dependency resolution
- Execute these instructions to produce lockfiles or graphfiles and export them as artifacts for the DS analyzer CI job to consume

This approach avoids bundling multiple runtimes and package managers into the analyzer image, reducing maintenance burden and security surface area.

#### Tier 3: Manifest Parsing Fallback (Lowest Accuracy)

When neither lockfiles nor graphfiles are available, the analyzer can parse dependency manifests directly to extract minimal dependency information. This provides basic coverage for projects without pre-generated artifacts, though with lower accuracy and completeness than lockfiles since it cannot capture transitive dependencies and the actual version used.

### Vulnerability Scanning

The analyzer integrates vulnerability scanning directly into the CI pipeline, providing immediate security feedback to developers. After generating CycloneDX SBOMs from detected dependencies, the analyzer:

1. **Uploads SBOMs to the GitLab SBOM Scan API**: The generated SBOM files are sent to GitLab's backend vulnerability scanning service
2. **Polls for scan results**: The analyzer waits for the backend to complete vulnerability analysis using the unified GitLab SBOM Vulnerability Scanner
3. **Aggregates findings**: Results from multiple SBOMs are combined into a single security report
4. **Generates security report**: A standardized GitLab dependency scanning report is produced with detected vulnerabilities

This approach maintains separation of concerns by delegating the actual vulnerability detection logic to the unified Dependency Scanning engine using the [GitLab SBOM Vulnerability Scanner](../dependency_scanning_engine/decisions/001_gitlab_sbom_vulnerability_scanner.md), while the analyzer handles orchestration and result aggregation.

For more details on the vulnerability scanning implementation, including error handling strategies, retry logic, and the concurrent processing model, see [ADR 002: Vulnerability Scanning using SBOM Scan API](./decisions/002_vulnerability_scanning.md).

## Decisions

- [ADR 001: Graph Export Only](./decisions/001_graph_export_only.md) - Documents the initial vision of supporting only lockfiles and graphfiles
- [ADR 002: Vulnerability Scanning using SBOM Scan API](./decisions/002_vulnerability_scanning.md) - Documents the decision to reintroduce vulnerability scanning capabilities within the DS analyzer
- [ADR 003: Dependency Resolution and Manifest Scanning](./decisions/003_dependency_resolution_and_manifest_scanning.md) - Documents the approach with automatic dependency resolution and manifest parsing fallback

## Appendix

- [dependency graph export](https://docs.gitlab.com/ee/user/application_security/terminology/#dependency-graph-export)
- [package manager](https://docs.gitlab.com/ee/user/application_security/terminology/#package-managers)
- [lock file](https://docs.gitlab.com/ee/user/application_security/terminology/#lock-file)

## References

- [Bring security scan results back into the Dependency Scanning CI job Epic](https://gitlab.com/groups/gitlab-org/-/work_items/17150)
- [Dependency Resolution Epic](https://gitlab.com/groups/gitlab-org/-/work_items/20461)
- [Manifest scanning Epic](https://gitlab.com/groups/gitlab-org/-/work_items/20457)
- [Dependency Scanning Engine](../dependency_scanning_engine/_index.md)
- [Dependency Scanning Engine ADR003: SBOM-based CI Pipeline Scanning](../dependency_scanning_engine/decisions/003_sbom_based_scans_for_ci_pipelines.md)
