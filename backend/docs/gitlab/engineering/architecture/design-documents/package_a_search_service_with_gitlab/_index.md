---
title: Package a search service with GitLab
status: proposed
creation-date: "2025-04-18"
authors: [ "@terrichu" ]
coaches: [ "@DylanGriffith" ]
dris: [ "@terrichu", "@bvenker", "@changzhengliu" ]
owning-stage: "~devops::ai-powered"
participating-stages: []
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

## Summary

GitLab is proposing to package a search service with its distribution to solve multiple strategic challenges and unlock new capabilities. Currently, search and filtering functionality backed by PostgreSQL has significant limitations for larger instances and complex group hierarchies, which impacts feature delivery and user experience. While advanced search is adopted by a percentage of self-managed instances (with higher rates among larger customers), it remains an optional feature reliant upon infrastructure that requires separate installation and configuration.

By including a search service directly in GitLab packages, we aim to make it a standard component of the GitLab infrastructure. This change would improve database scalability by offloading text search operations, enable more powerful search and filtering capabilities, and provide a consistent platform for AI features that require vector storage for embeddings. For example, [AI Context Abstraction Layer](/handbook/engineering/architecture/design-documents/ai_context_abstraction_layer/).

This initiative will benefit existing and new customers by removing adoption barriers, improving performance, and enabling a consistent experience across GitLab.com and self-managed deployments. Implementation will follow a phased approach, beginning with optional installation but designed to eventually make a search service standard infrastructure for GitLab features.

## Motivation

The motivation for this work is to establish a standard data store for search, filtering, and vector operations across GitLab. By improving the percentage of instances with advanced search enabled, we give feature teams the best opportunity to reach the most self-managed customers with performant, feature-rich experiences.

### Problem Statement

GitLab features increasingly require scalable data storage solutions that go beyond PostgreSQL's capabilities, particularly for search and filtering, AI, and data-intensive operations. Despite numerous evaluations of potential solutions, we've reached a fragmented state where:

1. Not all self-managed users have advanced search enabled for GitLab (adoption averages increase for medium and large size customers)
2. Feature teams must limit functionality for medium and large instances using PostgreSQL
3. Database scalability remains a persistent challenge for growing instances

The consequence is a divided user experience where feature availability depends on infrastructure choices, creating adoption barriers and limiting GitLab's ability to deliver consistent functionality across all deployment types.

### Industry Context

Industry competitors like GitHub provide more integrated search experiences out-of-the-box, creating a competitive gap. GitHub Enterprise Server includes Elasticsearch as an integrated component of their product, demonstrating that this approach is viable and has precedent in the industry. This integration allows GitHub to provide consistent search experiences across all deployment types, while GitLab currently requires separate installation and configuration of Elasticsearch or OpenSearch. As vector embeddings become standard for AI-powered features, having a robust vector database is increasingly becoming table stakes in the developer platform market.

### Opportunities

1. **Improved User Experience**: More consistent feature availability across all deployment types
2. **Unified Feature Development**: Teams can build on a common foundation rather than maintaining compatibility with multiple data stores
3. **Reduced Database Load**: Offloading search and filtering operations from PostgreSQL
4. **Enhanced AI Capabilities**: Native support for vector embeddings enables next-generation AI features
5. **Competitive Parity**: Closing the gap with competitors who offer integrated search experiences
6. **Simplified Infrastructure**: Standardized components reduce operational and maintenance complexity

### Goals

- Increase adoption of advanced search on self managed instances

### Non-Goals

- Convert instances using Elasticsearch or OpenSearch to switch to the other
- Remove support for external Elasticsearch or OpenSearch configurations
- Replace other vector database solutions for specialized use cases
- Address all scaling challenges in PostgreSQL

## Proposal

We propose to [package a search service as an optional component](https://gitlab.com/groups/gitlab-org/-/epics/18396) in all GitLab installation methods through the following key initiatives:

1. **Search service selection**:
   -[Select a search service for packaging with GitLab](https://gitlab.com/gitlab-org/gitlab/-/issues/554626)

2. **Search service sizing and configuration**:
   - [Update reference architecture and documentation](https://gitlab.com/groups/gitlab-org/-/epics/18151) to include minimum and recommended system specifications for a search service. This includes:
     - Sizing guideline documentation
     - Configuration and performance optimizations
     - Resiliency and high availability
     - Upgrades
     - Backup and restore
     - Geo disaster recovery

3. **Improved configuration automation**:
   - [Automate index configuration](https://gitlab.com/gitlab-org/gitlab/-/issues/549311) for GitLab with sensible defaults
   - [Automate index maintenance tasks](https://gitlab.com/groups/gitlab-org/-/epics/15888)
   - Expand existing health checks and self-healing capabilities to include connectivity checks

## Design and Implementation Details

### Technical Approach

For the initial implementation, we will propose to include a search service with GitLab's distribution packages, with the following considerations:

1. **Packaging method considerations**:
   - For Omnibus: Include as a new optional [metapackage](https://gitlab.com/groups/gitlab-org/distribution/-/epics/74)
   - For Kubernetes: Build custom Helm charts, or consume community works
   - For Docker: Include in [gitlab/gitlab-ee Docker image](https://hub.docker.com/r/gitlab/gitlab-ee/)
   - For GET: Include as a configurable component (GET already supports OpenSearch)
   - Follow the progress of the [Self-Managed Basic and Advanced (SMB/SMA) blueprint](https://gitlab.com/gitlab-com/content-sites/handbook/-/merge_requests/14414) as it impacts the technical approach.

2. **Search service version**:
   - Version must not be deprecated
   - Version must support hybrid search capabilities. For Elasticsearch, version 8.12+. For OpenSearch, version 2.15+.
   - The latest versions have non-trivial cost savings and performance improvements for embeddings storage
   - **Implementation Note**: Elasticsearch Docker Hub images bundle both core and enterprise code, with the latter activated by default for a 30-day trial. As part of this work, we'll need to modify CI configurations to explicitly use only the core functionality.

3. **Configuration and resource allocation**:
   - Default to a minimal configuration suitable for small instances
   - Provide configuration templates for different instance sizes
   - Both Elasticsearch and OpenSearch are packaged with a JVM. In single server implementations, this requires careful resource allocation to prevent the search service from impacting GitLab performance.

### Evaluations and Evidence

Multiple teams across GitLab have invested significant resources evaluating potential solutions, but none has achieved majority adoption:

- [Package a search engine with GitLab](https://gitlab.com/groups/gitlab-org/-/epics/18389) - Original initiative to include Elasticsearch
- [Iteration plan for RAG](https://gitlab.com/gitlab-org/gitlab/-/issues/441110) - Comprehensive evaluation of data store options for Retrieval Augmented Generation
- [Documentation questions for Chat](https://gitlab.com/gitlab-org/gitlab/-/issues/451215) - Implementation using Vertex AI Search as a workaround
- [Spike on privacy-oriented embeddings](https://gitlab.com/gitlab-org/gitlab/-/issues/458770) - Investigation of embedding storage options for sensitive data
- [PgVector evaluation](https://gitlab.com/gitlab-org/gitlab/-/issues/438330#note_1780393655) - Assessment of PostgreSQL with PgVector extension

These explorations consistently highlight Elasticsearch as a preferred solution due to its:

- Hybrid search capabilities (combining keyword and vector search)
- Mature feature set for relevance ranking and filtering
- Existing integration with GitLab's advanced search
- Scalable architecture for large deployments
- Ability to handle embeddings for AI use cases

## References

- [Issue #438178: Package a search engine with GitLab](https://gitlab.com/gitlab-org/gitlab/-/issues/438178)
- [Issue #438330: Estimate timeline to deliver "Users can ask documentation questions on SM Chat"](https://gitlab.com/gitlab-org/gitlab/-/issues/438330)
- [Issue #441110: Iteration plan: RAG](https://gitlab.com/gitlab-org/gitlab/-/issues/441110)
- [Issue #451215: Solution implementation for "users can ask documentation questions on SM Chat"](https://gitlab.com/gitlab-org/gitlab/-/issues/451215)
- [Issue #458770: Spike: Investigate and Validate Path for Privacy-Oriented Embedding Models](https://gitlab.com/gitlab-org/gitlab/-/issues/458770)
- [Issue #514017: Include Elasticsearch into GitLab delivery packages](https://gitlab.com/gitlab-org/gitlab/-/issues/514017)
- [Merge Request #142787: RAG architecture blueprint](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/142787)
- [Issue #1048: Elasticsearch integration](https://gitlab.com/gitlab-org/omnibus-gitlab/-/issues/1048)
- [Issue #3857: Ship elasticsearch with omnibus packages by default](https://gitlab.com/gitlab-org/omnibus-gitlab/-/issues/3857)
- [Epic #14293: Use Advanced Search for Filtered Searches of Issues and Merge Requests](https://gitlab.com/groups/gitlab-org/-/epics/14293)
- [Epic #13510: Vulnerability Management utilizing ElasticSearch](https://gitlab.com/groups/gitlab-org/-/epics/13510)
