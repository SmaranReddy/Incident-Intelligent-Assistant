---
title: "GitLab Secret Detection ADR 008: Unified Scanner"
---

## Context

Currently, the Secret Detection scans are running for different scan target types like Source code (via Pipeline SD), Git commits (Push Protection), Issue description/comments (Client-side SD), and more target types to follow. All the mentioned scan target types have different scan engines (Gitleaks, [one managed by](https://gitlab.com/gitlab-org/security-products/secret-detection/secret-detection-service) the team, another one by Web frontend), and to add complexity to the mix, there are certain teams at GitLab maintaining their version of secret detection. This leads to multiple issues (outlined below) and an inconsistent experience for customers:

* It causes a significant burden of maintaining feature parity across multiple engines.
* It limits us from optimizing the core scanning logic for performance and efficiency.
* It becomes difficult to ensure our ruleset quality(precision and recall) against all the scan engines.
* It also becomes difficult to optimize rule patterns which are generally done against a specific regex engine.
* Inconsistent integration of intelligent scanning capabilities with the engines based on their compatibility.

## Proposal

The proposal is to build a unified Secret Detection Scanner that will be used for all the scan target types. Its core responsibility is to run Secret Detection scan on a given set of payload(s) against the provided scan configuration. The Scanner may contain one or more underlying `Scan Engine` and possibly run the scan using multiple scan engines simultaneously to obtain accurate findings.

The Scanner internally uses Processors that run before and after the scan operation to either facilitate the scanning process or enhance the findings (more details [below](#processors)). This approach ensures that the core scanning logic can be reused across all scan target implementations.

Domain-specific preprocessing (such as gathering and structuring data into a scannable format) and postprocessing steps (including validity checks) fall outside the scope of the Scanner module and they are instead handled by the calling application.

### Characteristics

* The Scanner is portable. This is crucial considering some scan target types(source code/artifacts) running in an isolated environment like GitLab CI.
* The Scanner can run transactional or streaming-based scan.
* The Scanner is cross-platform compatible.
* The Scanner is stateless by nature.
* The Scanner is highly resource efficient to be able to run in a resource-constrained environment (like in client IDE or small CI environment).
* The Scanner allows customization the scan behavior by accepting user inputs (with defaults):
  * exclusions (rule/path/value)
  * custom ruleset
  * ruleset version
  * timeout constraints
  * payload size limit
  * payloads acceptable via file-path/dir-path/stdin (binary executable)
  * configurable resource(memory/cpu cores) limit (binary executable)
* The Scanner is able to run as a binary executable in air-gapped environments.

Some of the above characteristics are achieved through Scanner Clients (more details below).

### Pre-requisite

The idea of unified Scanner could hold true only if we have an efficient yet portable Scan Engine. As per the [decision](007_switch_to_go_scan_engine.md), we need to have Vectorscan-based Go scan engine implemented.

### Mode of using Scanner

The minimalistic scope and the stateless nature of the proposed Scanner will open up the _portability_ advantage, which is a necessity for certain target scan types (source code or job artifacts running in CI env). Therefore, the scan engine could be adopted in one of the following three modes depending on the nature of the scan target type (traffic,size,etc.):

* **As a Distributed Service**: The Scanner will be wrapped with a REST/gRPC layer having scan API endpoints. The caller makes the scan request over the network. This mode is suitable for SD features having high traffic with lightweight payloads (\<1MB). Example: Scanning Work Items via [Secret Detection Service](https://gitlab.com/gitlab-org/security-products/secret-detection/secret-detection-service)

* **As an Embedded module**: The Scanner in this mode is _embedded_ within the same host as the caller application. The caller invokes the scan for a scannable payload. This mode is transactional by nature. We are already using this mode for the Push Protection feature where the engine is embedded as a Ruby Gem and installed in the Rails monolith. The Rails monolith makes the scan request (including `git diff` data as a scannable payload) to the gem.

![Embedded Mode](/images/engineering/architecture/design-documents/secret_detection/008_scan_mode_embedded.png "Embedded Scan Mode")

* **As a Batch processor**: This is a special case to support [in-storage processing](https://en.wikipedia.org/wiki/In-situ_processing) where the Secret Detection program (+engine) runs where the data resides. This reverse approach is suitable for scan target types having larger data sizes, like source code or job artifacts, to avoid data-transfer costs incurred b/w data storage and scan servers. The primary difference when compared to Embedded mode is that the caller includes the scannable payload within the scan request whereas in Batch mode, the caller points at the scannable payload(s) available at the target host (where the program and data reside), e.g. passing a file path along with the request.

![Batch Mode](/images/engineering/architecture/design-documents/secret_detection/008_scan_mode_batch.png "Batch Scan Mode")

These modes are implemented as Scanner Clients. A `Scanner Client` is a wrapper interface that allows callers to invoke the `Scanner` module. Scanner Clients can be either gRPC service (for distributed processing) or CLI application (for embedded/batch processing).

### Components

The unified Scanner architecture introduces the following components:

* **Scanner**: The core component that orchestrates secret detection scans. It receives scannable payloads from Scanner Clients along with Scan configuration, applies preprocessing, invokes the underlying Scan Engine(s), and performs postprocessing on the findings. It is designed to be portable and stateless.

* **Scan Engine**: The actual engine that performs secret detection logic. This can be a regex-based engine (like Vectorscan), an AI-based engine, or a combination of multiple engines. The Scanner abstracts away engine specifics, allowing different engines to be plugged in seamlessly.

* **Processors**: Modular components that run before (`Pre-Processor`) or after (`Post-Processor`) the core scan operation to enhance the scanning process or refine findings.
  * **Pre-Processors**: Prepare input payloads or provide additional metadata for the scan engine (e.g., AST generation, sanitization).
  * **Post-Processors**: Refine scan results (e.g., false positive reduction, entropy matching).

* **Scanner Clients**: Scanner wrappers that allow caller applications to interact with the `Scanner` module. Examples include gRPC services and CLI applications.

* **Analyzers**: Applications that invoke the `Scanner` module to scan payloads. They transform payloads into scannable formats and provide scan configurations. Each Analyzer is responsible for a specific scan target type (e.g., Push Protection, Source Code CI, etc).

* **Configuration**: Defines Scanner operation parameters, including exclusions, custom rules, and timeout constraints. This is provided by the `Analyzer` to the `Scanner` via the `Scanner Client`.

#### Component Hierarchy

Here's a high-level view of how these components interact:

![Component Hierarchy](/images/engineering/architecture/design-documents/secret_detection/008_component_hierarchy.png "Component Hierarchy")

### High-level Design

Here's the comprehensive view of unified Secret Detection Scanner with the different Analyzers, each for a scan target type.

![High-level Design for unified Secret Detection Scanner](/images/engineering/architecture/design-documents/secret_detection/008_high_level_design.png "High-level Design for unified Secret Detection")

### Scanner Internals

As we plan to introduce additional scan engines (such as [AI-based (internal)](https://gitlab.com/groups/gitlab-org/-/epics/17886) or RE2 WASM-based for portability) to detect secrets, the Scanner API should decouple itself from a single underlying Scan Engine. This allows regex-based, AI-based, or multiple scan engines to be plugged in depending on the use case.

#### Processors

The Scan Engine component contains Processors that intercept before (`Pre-Processor`) and after (`Post-Processor`) the scan operation performed by the internal engine.

* `Pre-Processor`: Input payloads are passed through pre-processors to execute specific operations. They either return modified payloads or the original payloads with additional metadata helpful for the scan engine. Examples include AST generation for payloads, sanitizing payload data for inline exclusions, etc.

* `Post-Processor` processes findings detected by the internal engine(s) and returns either hints to discard certain detections due to false positives or runs domain-specific criteria to generate new metadata. Examples include AI-based false positive reduction, entropy matching, etc.

Processors can be either generic (running on every scan) or conditional (triggered by specific criteria such as engine type, customer tier, etc.). They can operate in a chained pipeline where one processor's output becomes another's input, or function independently. In both cases, the Scanner API collects the results from all processors to determine the final output.

Here's the illustration representing the internal components of the Scanner:

![Scanner Internals](/images/engineering/architecture/design-documents/secret_detection/008_scanner_internals.png "Scanner Internals")

### Distribution

Given there are different modes of using the scan engine and the Vectorscan regex engine being platform-dependent, it is important to identify how the engine is made accessible within GitLab applications.

* When using the engine as a distributed (grpc) service: The grpc service will be deployed using Runway making it accessible to internal GitLab.com applications. The clients can make remote scan request to the service via grpc endpoints.

* When using the engine as an Embedded module or Batch processing: Depending on the client application, the nature of embedding differs. The engine can be imported as a Go module for Go-based applications whereas for Non Go applications (mostly `Ruby` in GitLab Rails), there are two choices:

  1. Write Ruby C extension for a static library generated from Go's scan engine source
  2. Package Go binary executable within GitLab Rails using Omnibus

Considering the VectorScan engine scans on a chunk of data instead of a line-by-line basis (current approach), the [high heap usage problem](https://gitlab.com/gitlab-org/gitlab/-/issues/422574#note_1582015771) is no longer applicable. In addition, Vectorscan engine operates on a pre-allocated [scratch space](https://intel.github.io/hyperscan/dev-reference/runtime.html#scratch-space) contributing to the lower memory consumption of the overall scan. The throughput of calling Go methods via C-function adds barely any overhead when compared to IPC between Ruby <-> Go processes. All that said, the decision is leaning towards writing Ruby C extension. However, we are yet to conduct a spike to confirm this decision with a data-driven evidence.
