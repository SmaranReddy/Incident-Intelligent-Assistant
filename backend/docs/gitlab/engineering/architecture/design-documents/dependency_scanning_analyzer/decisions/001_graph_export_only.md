---
title: "Dependency Scanning Analyzer ADR 001: Graph Export Only"
---

## Context

This ADR documents the initial vision for the new Dependency Scanning analyzer, which focused on supporting only [dependency graph exports](https://docs.gitlab.com/ee/user/application_security/terminology/#dependency-graph-export) and lockfiles. This approach was motivated by the desire to simplify the analyzer by removing the need to build projects and manage multiple language runtimes. The initial proposal was to delegate the responsibility of generating dependency artifacts to users through preceding CI jobs, allowing the analyzer to focus solely on parsing and analyzing these artifacts.

## Decision

> :warning: Some aspects of this decision have been [revised](#subsequent-revisions).

Create a new analyzer that focuses on supporting only [dependency graph exports](https://docs.gitlab.com/ee/user/application_security/terminology/#dependency-graph-export) and lockfiles. Document how to generate the exports with example projects, and provide a dependency scanning CI/CD component that scans the generated artifacts.

Because of the change to SBOM-based scanning in [epic 8026](https://gitlab.com/groups/gitlab-org/-/epics/8026), do not port over the vulnerability matching done by the Gemnasium analyzers, as this functionality is already [planned for deprecation](https://gitlab.com/groups/gitlab-org/-/epics/14146).
The new analyzer should be based on a scratch image to reduce the attack surface introduced by container dependencies.

### Pros

- Simplified integration tests. No need to test against various permutations of package managers, runtime, and compiler versions.
- We should always have zero container-scanning vulnerabilities. This translates to a reduced workload on the engineers going through reaction rotation.
- Smaller image sizes. Fast CI job start-up, reduced network traffic.
- Simplified FIPS-compliance as the library does not use crypto libraries.
- Improved community contribution experience due to simplified permissions for development pipeline execution.

### Cons

- Additional documentation required with examples and guides on getting started with a dependency scanning for certain package managers.
- Requires the establishment of new graph export naming standards.
- Users need to configure their build jobs as instructions. It doesn't work out of the box.

## Design and Implementation Details

At a high level, the graph export only approach would operate as follows.

```mermaid
sequenceDiagram
    autonumber
    actor user
    participant build job
    participant analyzer
    participant sbom ingestion service
    participant database

    user->>build job: triggers pipeline on default branch
    build job->>analyzer: passes all dependency graph exports generated as artifacts
    analyzer->>sbom ingestion service: converts dependency graph exports to CycloneDX SBOMs
    sbom ingestion service->>database: stores occurrences of sbom components
```

### Build job(s)

It's important to note that we cannot expect for a dependency graph export to be checked into a project's repository. This is likely to happen in cases where the dependency graph export does not also function as a lock file like in the cases of `pipdeptree` and `pipenv graph` dependency graph exports. In such cases, we will expect the build job to generate the dependency graph exports, and for the job to store these as [job artifacts](https://docs.gitlab.com/ee/ci/jobs/job_artifacts.html).

We'll use the following naming conventions to establish a contract with users on what file's we'll detect in cases where the dependency graph export does not function as a lock file, and thus does not have a canonical name.

| Pattern | Description
| ------- | -----------
| `**/go.graph` | Generated via `go mod graph > go.graph`
| `**/pipenv.graph.json` | Generated via `pipenv graph --tree-json > pipenv`

It's required for the build job to run in a stage that precedes the one in which the dependency scanning analyzer runs. This is true by default, since the analyzer runs in the `test` stage which runs after the `build` stage.

### Analyzer

Once the build jobs complete, and the artifacts are stored, they will be passed on to [proceeding jobs](https://docs.gitlab.com/ee/ci/jobs/job_artifacts.html#prevent-a-job-from-fetching-artifacts) unless specifically asked not to do so. The analyzer takes advantage of this and expects that users have configured the build jobs to pass on the artifacts using the documented naming patterns. It will then search the entire target directory, by default this is the project's repository, detect all supported files, parse them, and convert them into a CycloneDX SBOM that can be utilized by the services running in the GitLab monolith.

### Pros

- No preinstalled compilers, runtimes or system dependencies required.
- Small attack surface.
- Runs offline by default.

### Cons

- Graph export documentation varies in quality. Some package managers like `npm` document each version of the lock file, while others like `pnpm` do not.
- Java and Python projects require additional configuration since they do not capture graph information in their lock files by default.

## Alternative Solutions

### Require lock file, add graph information to it

One alternative solution to dependency graph exports is to make every supported lock file a dependency graph export by default. In this scenario, we would work directly with package manager maintainers to enhance lock files with transitive dependency relationships, and dependency group metadata. For example, we could work with the Gradle maintainers to add a new version of their `gradle.lockfile` that includes parent dependencies. Our contributions would have the added benefit of improving the experience for our users by including the necesssary tooling out of the box, overall improving the workflow for getting started with GitLab's dependency scanning feature.

#### Pros

- Does not require establishing new file requirements.
- Works out of the box in majority of cases. Package managers usually generate a lock file if one doesn't exist.

#### Cons

- Package managers tend to have large code bases that increase the onboarding time required.
- Lock files require domain expertise. For example, in [pnpm's issue 7685](https://github.com/pnpm/pnpm/issues/7685) you can see the discussion of a very specific corner case that must be handled.
- Project maintainers have their own sets of concerns that may not align with our own. For example, they may prioritize stability and maintenance over new features.
- Older versions of package managers, or build tools, would not be compatible with new additions.

### Rely on 3rd party CycloneDX generators

This approach moves the direction of composition analysis so that we interface only with user provided `cyclonedx` CI reports from 3rd party CycloneDX generators.

#### Pros

- No CI/CD component integration testing.
- No analyzer maintenance required.

#### Cons

- Tied to the GitLab release schedule, so we can't deploy new features, enhancements, and bug fixes mid milestone.
- There are a lot of third party analyzers that can generate a CycloneDX report. Supporting all of their custom [metadata properties](https://cyclonedx.org/docs/1.5/json/#metadata_properties) and [component properties](https://cyclonedx.org/docs/1.5/json/#components_items_properties) can be challenging.
- Requires time to get up to speed with third party SBOM generator code bases.
- Proposals to the generators may be rejected. If required, we could fork the project, but that comes with its own set of challenges.
- Dependency graphs may be incomplete like `cyclondex_py` in some circumstances.

### Generate custom dependency graph exports with package manager plugins

In the cases where package managers expose a public API, we are able to write a plugin to generate the dependency graph in a format of our choosing. This has been used for `gemansium-maven` dependency analysis.

#### Pros

- Choice of output format.
- Can re-use the bundled `gemnasium-maven` plugins.

#### Cons

- Not all package managers have support for third party plugins. For example, `pnpm` does not have documented plugin support.
- Plugins that do not use Ruby or Go require new language expertise, which lead to a smaller pool of plugin maintainers, and higher review load per maintainer.
- Additional overhead required for the maintenance, improvements, and deployments of plugin projects.

## Subsequent Revisions

This initial approach was later revised through the following ADRs:

- [ADR 002: Vulnerability Scanning using SBOM Scan API](./002_vulnerability_scanning.md) - Reintroduces vulnerability scanning capabilities within the analyzer to provide immediate security feedback in CI pipelines
- [ADR 003: Dependency Resolution and Manifest Scanning](./003_dependency_resolution_and_manifest_scanning.md) - Addresses user experience gaps and enables dependency scanning at scale through automatic dependency resolution and manifest parsing fallback
