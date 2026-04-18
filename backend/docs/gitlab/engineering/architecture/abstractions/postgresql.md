---
title: "PostgreSQL - OLTP Database"
---

## Status

**Graduated** - Approved for use

## Owner

[To be assigned]

## Description

PostgreSQL is GitLab's approved relational database management system (RDBMS) for Online Transaction Processing (OLTP) workloads. It serves as the primary data store for transactional data, user information, project metadata, and other structured data requiring ACID compliance.

## Why This is a Key Abstraction

PostgreSQL has been selected as the standard OLTP database for GitLab because:

1. **Proven at Scale**: Successfully powers GitLab.com and large self-managed instances
2. **Feature Rich**: Provides robust support for complex queries, transactions, and data integrity
3. **Open Source**: Aligns with GitLab's values and eliminates vendor lock-in
4. **Strong Ecosystem**: Extensive tooling, community support, and operational expertise within GitLab
5. **Consistency**: Using a single RDBMS reduces operational complexity and allows teams to share knowledge and best practices
6. **Self-Managed Compatible**: Can be deployed in customer environments without licensing concerns

## Current State

GitLab.com is currently running PostgreSQL at scale coordinates `[1,1,0]` in the [Scale Cube](/handbook/engineering/architecture/practice/scalability/#the-scale-cube):

- **X-axis (Cloning)**: Multiple fully replicated instances (primary read-write with several read-only secondaries)
- **Y-axis (Componentization)**: We already partition all our data across 3 independent vertical domains called Main, CI and SEC. Each is it's own separate postgres database
- **Z-axis (Federation)**: Not currently implemented

## Use Cases

Use PostgreSQL for:

- Transactional data requiring ACID compliance
- Structured data with complex relationships
- Data requiring strong consistency guarantees
- Application state that needs to be queried flexibly
- User data, authentication, and authorization
- Project metadata, issues, merge requests, and other core GitLab entities

## Do Not Use For

Consider alternatives for:

- **Analytics / OLAP queries**: Use ClickHouse (candidate key abstraction)
- **Large binary data**: Use Object Storage
- **High-frequency event streaming**: Use NATS (candidate key abstraction)
- **Ephemeral cache data**: Use Redis
- **Full-text search**: Use Elasticsearch/OpenSearch
- **Code search**: Use GitLab Zoekt

### Best Practices

- Follow [Database Guidelines](/handbook/engineering/architecture/guidelines/database/)
- Read through our [database development docs](https://docs.gitlab.com/development/database/) for more information

## Related Key Abstractions

- **Redis**: For caching and reducing database load
- **Object Storage**: For binary data that would otherwise bloat the database
- **ClickHouse** (candidate): For analytical queries that would be inefficient in PostgreSQL

## Support and Resources

- [Database Guidelines](/handbook/engineering/architecture/guidelines/database/)
- [Scalability Practice](/handbook/engineering/architecture/practice/scalability/)
- [Internal database development documentation](https://docs.gitlab.com/development/database/) and [runbooks](https://gitlab.com/gitlab-com/runbooks/-/tree/master/docs/patroni)
- Database team office hours and Slack channels

## Questions or Issues

For questions about using PostgreSQL or proposing changes to this key abstraction:

1. Consult the Database Guidelines
2. Reach out to the Database Frameworks team
3. For architectural decisions, engage the Architecture Board through the [Architecture Workflow](/handbook/engineering/architecture/workflow/)
