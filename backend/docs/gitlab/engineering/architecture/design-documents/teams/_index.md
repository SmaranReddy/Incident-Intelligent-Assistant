---
# This is the title of your design document. Keep it short, simple, and descriptive. A
# good title can help communicate what the design document is and should be considered
# as part of any review.
title: Teams
status: proposed
creation-date: "2025-12-09"
authors: [ "@lohrc" ]
dris: [ "@lohrc" ]
owning-stage: "~devops::tenant scale"
participating-stages: []
# Hides this page in the left sidebar. Recommended so we don't pollute it.
toc_hide: true
---

## Summary

GitLab's current access control model conflates user management with project organization through the dual-purpose nature of groups. While the upcoming [Organizations](../organization/_index.md) will provide customer isolation and address some enterprise-scale challenges, it does not resolve the fundamental dual-purpose nature of groups within each Organization. This creates significant challenges for enterprise customers who need to separate organizational structure from resource management while maintaining security and compliance requirements within their customer boundaries.

This blueprint defines the architectural requirements for a Teams-based access control system that operates within Organization boundaries, separates user management from project organization, addresses critical customer pain points, and positions GitLab competitively in the enterprise DevOps market.

The Teams architecture aims to introduce a distinct entity for managing users' access to groups and projects within Organizations, providing clearer mental models, better inheritance behavior, and reduced administrative overhead. This separation addresses [the #1 vulnerability in the OWASP Top 10 (Broken Access Control)](https://owasp.org/Top10/A01_2021-Broken_Access_Control/) while reducing the cognitive load for users of the current system.

## Motivation

### Current State Analysis

In GitLab, groups serve dual purposes that create inherent tension:

1. **Project organization**: Hierarchical structure for managing code, issues, and CI/CD resources
2. **User management**: Organizing people and controlling access to resources

While the upcoming Organizations feature will provide customer isolation and address some enterprise-scale challenges, it does not resolve the fundamental dual-purpose nature of groups within each Organization. Organizations will contain groups, but those groups will still face the same access management and organizational modeling challenges that exist today.

This dual nature creates several critical problems identified through customer research and competitive analysis:

**Permission management complexity**: Nearly every customer interview mentioned challenges with GitLab's permission model, particularly around inheritance creating unexpected access levels when users have different roles at different hierarchy levels. While GitLab's roles provide fine-grained permission control with over 40 specific [custom permissions](https://docs.gitlab.com/user/custom_roles/abilities/), complexity arises from how these permissions interact with the dual-purpose nature of groups and the resulting inheritance patterns.

**Inconsistent sharing behavior**: Group sharing exhibits different inheritance patterns depending on the target (group vs project). Sharing a group with another group only shares direct members, while sharing a group with a project includes both direct and inherited members. This inconsistency violates user mental models and creates security risks.

**Cognitive load and mental model mismatch**: Users expect permissions to work like physical filing cabinets — explicit and visible. The current invisible inheritance model violates these expectations, leading to systematic errors and security vulnerabilities as users work around systems they don't understand.

### Relationship to Organizations and Architectural Approach

The Teams architecture operates within the [Organization](../organization/_index.md) framework currently under development. While Organizations provide customer isolation and administrative boundaries, they do not solve the fundamental access management challenges within each customer's boundary: complex permission inheritance, difficulty separating user management from project organization, and administrative overhead for cross-functional collaboration.
Teams complement Organizations by providing the "who does the work" layer while Organizations provide the "customer isolation" layer, enabling cross-group collaboration within customer boundaries without breaking isolation between customers.

#### Architectural Philosophy: Hybrid vs. Restructuring Approach

During design, significant discussion emerged about pursuing radical architectural restructuring (flattening GitLab's hierarchy entirely, separating groups into distinct systems for Access, Settings, Aggregation, Features, Relationships) versus a more incremental hybrid approach.
The radical restructuring approach would eliminate groups entirely, providing maximum architectural flexibility and clean separation of concerns. However, this presents substantial challenges:

- **Engineering risks**: Performance impact from removing hierarchical inheritance, migration complexity requiring vast code rewrites, database restructuring, and breaking thousands of existing integrations
- **Customer risks**: Namespace collisions, disruption to enterprise compliance boundaries, operational disruption for 100,000+ organizations, and retraining costs for millions of users

The hybrid approach adopted preserves groups for organizational hierarchy and namespace management while adding Teams specifically for access management and work coordination. This approach:

- Maintains backwards compatibility while solving immediate customer problems
- Provides clear separation: Organizations = "Customer isolation", Groups = "Where work lives", Teams = "Who does the work"
- Represents a "two-way door" decision enabling evolution based on real usage data rather than theoretical models

Industry evolution since 2023 validates this approach, with major platforms successfully implementing Teams as complementary layers rather than replacing existing organizational structures.

### Industry Context and Competitive Pressure

Research across the DevOps landscape reveals that access management inefficiencies cost US companies approximately $61 billion annually, with 94% of applications showing some form of broken access control. The competitive landscape has evolved significantly, with major platforms implementing sophisticated approaches to address enterprise access management challenges:

- **Team-based hierarchical models** with sophisticated nesting and clear inheritance patterns
- **Organizational structure separation** from project management through dedicated organizational modeling features
- **Project-level permission cascading** that eliminates individual resource permission management overhead
- **Clear separation patterns** between permanent user organization and temporary access mechanisms

These evolved approaches demonstrate industry recognition that traditional access control models are insufficient for modern enterprise DevOps requirements. GitLab's current dual-purpose group model increasingly appears dated compared to these specialized solutions, potentially limiting enterprise adoption and competitive positioning.

*Detailed competitive analysis including specific platform implementations and feature comparisons is available in the [DevOps Access Control Research Report](https://drive.google.com/drive/folders/1WeMK7PYvFhGtWqYFY8VPMUOxUVfay3vr?ths=true) for internal reference.*

### Goals

#### Primary Goals

1. **Separate user management from project organization within Organizations**
   - Create a distinct Teams entity that represents organizational structure
   - Enable independent evolution of user hierarchies and project hierarchies within an Organization boundary
   - Support enterprise identity integration without impacting project structure
   - Provide clear conceptual separation that aligns with user mental models

2. **Improve permission inheritance and organizational modeling within Organization boundaries**
   - Eliminate inconsistent sharing behavior between group-to-group and group-to-project scenarios
   - Make permission inheritance visible and auditable throughout the Organization
   - Provide "break inheritance" capabilities with clear impact warnings
   - Support both hierarchical organizational modeling and flat cross-functional collaboration
   - Integrate seamlessly with GitLab's existing roles system to provide organizational context for fine-grained permissions

3. **Address enterprise scale requirements within Organizations**
   - Support Organizations with 10,000+ users and thousands of projects
   - Provide comprehensive audit trails for compliance (NIST 800-53, ISO 27001, SOX)
   - Enable bulk operations for administrative efficiency
   - Support team lifecycle management

4. **Enable cross-functional program management and work coordination**
   - Support cross-functional work views: Teams can aggregate issues, MRs, and epics across multiple groups/projects within an Organization
   - Provide program-level dashboards: See all work for a team regardless of where it lives in the Organization hierarchy
   - Enable resource planning: Understand team capacity across different projects
   - Support delivery tracking: Monitor team velocity and burndown

#### Secondary Goals

1. **Enable advanced access patterns within Organizations**
   - Support just-in-time access for sensitive operations
   - Provide service account management with automated rotation
   - Enable temporary permission elevation
   - Support matrix organizations and multiple reporting structures

2. **Improve developer experience within Organizations**
   - Integrate access control directly into development workflows
   - Provide API-first design for automation and tooling integration
   - Support DevOps-specific permission patterns for CI/CD pipelines
   - Enable self-service access requests with appropriate approval workflows

### Relationship to Existing Systems

#### Integration with Roles

The existing roles system addresses permission granularity through up to 40+ specific permissions. Teams will complement this:

- Roles: Define what users can do (specific capabilities)
- Teams: Define which users should have access and how that access is organized within Organizations
- Combined System: Enable modeling organizational structure while maintaining precise permission control

#### Integration with Organizations

- Teams operate entirely within Organization boundaries, never crossing customer isolation
- Organization owners can manage Teams within their Organization
- Teams inherit the security and compliance characteristics of their containing Organization
- Enterprise identity integration occurs at the Organization level, with Teams providing internal structure

#### Integration with Groups and Projects

- Groups continue to provide namespace management, settings inheritance, and project organization
- Teams provide the access management and user organization layer
- Projects remain the fundamental unit of work, accessible through both Group hierarchy and Team membership

### Non-Goals

1. **Replacement of groups or radical architectural restructuring**
   - Teams are designed to complement groups, not replace them
   - Groups will continue to provide namespace management, settings inheritance, and project organization
   - We explicitly avoid a "one-way door" decision that would require rewriting vast portions of the system
   - Focus remains on solving access management problems while preserving organizational hierarchy benefits

2. **Replacement of existing permission systems**
   - Teams architecture should complement, not replace, the existing roles and permissions system
   - Teams should provide organizational structure and inheritance patterns for applying existing permission models

3. **Immediate disruption of existing workflows**
   - Current group functionality must remain fully supported during and after Teams implementation
   - Existing customer workflows cannot be disrupted without explicit migration path
   - API compatibility must be maintained to avoid breaking existing integrations

4. **Over-engineering organizational complexity**
   - Focus on common enterprise organizational patterns rather than edge cases within Organizations
   - Maintain philosophy of convention over configuration
   - Avoid creating complexity that contradicts the goal of reducing cognitive load

## Critical Requirements Analysis

### Customer Pain Points (Based on User Research)

**Permission Management and Inheritance Issues**

- Inheritance creates unexpected access levels with complex role interactions across group hierarchies
- Confusion around how permissions propagate through groups and subgroups, especially with custom permissions
- Difficulty visualizing who has access to what, especially in large organizations with multiple custom permissions
- Challenges with SAML/LDAP integration for mapping organizational groups to GitLab groups with appropriate roles
- Complex interactions between base roles, custom permissions, and inheritance patterns

**Organizational Structure Limitations**

- Need for organizational modeling that doesn't impact project structure
- Inflexibility for organizations that reorganize frequently
- Difficulties representing matrix organizations or multiple reporting lines
- Issues managing historical "legacy" structures as organizations evolve
- Inability to separate temporary project teams from permanent organizational structure

**Enterprise-Scale Management Challenges**

- Limited visibility into user activities and resource usage across thousands of projects
- Challenges with enforcing policies consistently across groups
- Need for better audit capabilities for compliance purposes
- Lack of tools for managing large-scale migrations or reorganizations
- Insufficient bulk operation capabilities for administrative efficiency
- Inability to assign user groups directly to specific resources (environments, branches) without first assigning them at the project level, forcing customers to use less explicit workarounds like role names that don't scale across thousands of projects

**Data Recovery and Deletion Concerns**

- Accidental deletion being too easy with cascading effects
- Recovery windows being too short for enterprise change management
- Inadequate safeguards for bulk operations
- Fear of data loss impacting willingness to restructure access

### Competitive Analysis Insights

**Hierarchical Team Models in the Market**

- Parent-child team relationships with automatic permission inheritance
- Integration of access control directly into development workflows through code ownership patterns
- Fine-grained access tokens with organization-level approval workflows
- Challenge: Deep nesting complexity and performance implications at scale

**Enterprise-Scale Organizational Integration**

- Multi-tier structures supporting thousands of organizational units and area paths
- Mapping capabilities for both structural hierarchies and temporal workflows
- Tight integration with enterprise directory services for seamless synchronization
- Challenge: Complexity can overwhelm smaller teams, creating vendor dependencies

**Cross-Product Permission Consistency**

- Project-level permissions that cascade to eliminate individual resource management
- Centralized administration providing consistent access across integrated product suites
- Simplified role models that balance flexibility with administrative simplicity
- Challenge: Maintaining consistency across diverse product capabilities

**Separation of Permanent and Temporary Access**

- Clear distinction between permanent user organization and temporary access mechanisms
- Permission boundaries that define maximum capabilities regardless of specific grants
- Just-in-time access patterns for sensitive operations
- Challenge: Implementation complexity and learning curve for traditional organizational models

*Detailed analysis of specific platform implementations is available in the [DevOps Access Control Research Report](https://drive.google.com/drive/folders/1WeMK7PYvFhGtWqYFY8VPMUOxUVfay3vr?ths=true) for comprehensive competitive insights.*

### Engineering Feasibility and Technical Considerations

**Performance and compatibility**

- Teams must integrate with existing hierarchical permission lookup patterns without performance degradation
- Database schema must follow established patterns (organization_id, namespace_id, project_id) and support existing partitioning strategies
- All existing API endpoints and CI/CD pipelines must continue functioning unchanged
- New Teams APIs must follow established authentication and authorization patterns

**Gradual adoption strategy**

- Teams functionality must be purely additive — overlaying on existing group hierarchies without requiring data migration
- Mixed environments (Teams + traditional groups) must be supported indefinitely
- Customers can adopt Teams incrementally without disrupting existing workflows
- Rollback capabilities must be available if Teams adoption creates issues

**Organization integration**

- All Teams functionality must respect Organization isolation boundaries — no cross-Organization Team assignment or visibility
- Organization owners must have full administrative control over Teams within their Organization
- Teams must integrate with Organization-level identity provider synchronization and audit trails
- Teams settings and policies must be configurable at the Organization level for compliance requirements

## Open Questions and Research Areas

### Teams and Organization Integration Patterns

**Organization-scoped Team management:**

- Should Teams be created at the Organization level and then assigned to groups/projects, or should there be another way?
- How should Team membership synchronization with external identity providers work within Organization boundaries?
- What administrative controls should Organization owners have over Teams created by Group owners within their Organization?

**Cross-group Team collaboration:**

- How should Teams spanning multiple groups within an Organization handle different group visibility levels (private/internal/public)?
- Should Teams inherit the most restrictive visibility settings from their constituent Groups, or should Teams have independent visibility controls?

### Hierarchical vs Flat Team Structure Decision

**Hierarchical approach benefits:**

- Natural mapping to organizational structure reduces cognitive load
- Inheritance patterns simplify administration at scale
- Aligns with existing GitLab group hierarchy mental models
- Supports enterprise compliance requirements for clear access paths

**Hierarchical approach challenges:**

- Deep nesting creates performance and usability problems (as seen in other enterprise platforms)
- Inheritance complexity can lead to unintended access (security risk)
- Organizational changes require structural modifications
- May not support matrix organizations or cross-functional teams effectively

**Research Questions:**

- Can a hybrid approach provide hierarchical benefits within Organizations without complex inheritance?
- Should Teams have parent-child relationships for organizational modeling without automatic permission inheritance?
- How do enterprise customers expect Teams to mirror their internal organizational structure?

### Integration with Existing Systems

**Roles and Teams interaction:**

- How should roles assigned to Teams interact with roles assigned directly to users?
- Should Teams be able to have different roles in different contexts (Group A vs Project B)?
- How do we handle conflicts when a user belongs to multiple Teams with different role assignments?

**Group sharing and Teams:**

- Should Teams be assigned to Groups in the same way Groups can be shared with other Groups?
- How should Teams assignment interact with existing Group sharing inheritance patterns?
- Should assigning a Team to a Group automatically grant the Team's role to all shared contexts?

## Alternative Solutions Analysis

### Option 1: Enhance Current Group Model

**Approach:** Improve existing groups to better separate user management concerns through enhanced sharing options and visibility improvements.

**Pros:**

- Minimal disruption to existing customers and workflows
- Leverages existing user familiarity with group concepts
- Lower development and migration costs
- Maintains current API compatibility

**Cons:**

- Doesn't address fundamental dual-purpose nature of groups
- Inheritance inconsistencies would persist
- Limited ability to implement enterprise-scale requirements
- Doesn't provide clear separation for compliance frameworks
- May not be sufficient to compete with evolved industry offerings

### Option 2: Complete Architectural Restructuring (Flat Structure)

**Approach:** Eliminate groups entirely within Organizations, implementing a flat structure with specialized systems for Access, Settings, Aggregation, Features, and Relationships.

**Pros:**

- Maximum architectural flexibility and clean separation of concerns
- Eliminate all inheritance complexity
- Align with theoretical ideals about system design
- Could provide ultimate solution to dual-purpose problems

**Cons:**

- Engineering complexity: Would require rewriting vast portions of permission-related code
- Performance concerns: Removing hierarchical inheritance could significantly impact permission lookup performance
- Migration risk: Enterprises have already built implementations around group hierarchies
- Namespace collisions: Flat structures reduce available naming cardinality significantly
- Enterprise disruption: Many organizations use group boundaries for SOX/GDPR compliance
- API compatibility: Would break thousands of existing integrations and CI/CD pipelines
- One-way door: Irreversible decision with significant risk if the approach proves problematic

### Option 3: Teams as Hybrid Solution Within Organizations

**Approach:** Implement Teams as a complementary layer within Organizations that provides access management and work coordination while preserving groups for organizational hierarchy.

**Pros:**

- Two-way door decision: Can gather customer feedback and evolve based on real usage patterns
- Backwards compatibility: Preserves existing workflows and integrations
- Clear mental models: Groups = "Where work lives", Teams = "Who does the work", Organizations = "Customer isolation"
- Engineering feasibility: Builds on existing architecture rather than requiring complete rewrite
- Customer value: Addresses immediate pain points while enabling future evolution
- Industry validation: Aligns with successful approaches taken by other major platforms

**Cons:**

- More complex than pure flat structure in some theoretical aspects
- Requires careful design to avoid adding confusion
- May not address every edge case in complex organizational structures

## Success Criteria and Metrics

### Quantitative Success Metrics

**Performance and scale:**

- Support 10,000+ users with <2 second permission resolution time
- Handle 1,000+ teams and 1,000+ projects without performance degradation
- Achieve 99.95% uptime for permission-related operations

**Administrative efficiency:**

- Reduce average time-to-grant-access from current baseline to <5 minutes
- Decrease organizational structure and inheritance-related support issues by 50% from current baseline
- Enable bulk operations for 100+ team-resource relationships simultaneously
- Improve role assignment efficiency through organizational context

**Security and compliance:**

- Provide 100% audit trail coverage for all permission changes
- Pass compliance audits for NIST 800-53, ISO 27001, and SOX requirements

### Qualitative Success Metrics

**User experience:**

- User research showing >80% of administrators can correctly predict organizational inheritance patterns
- Developer survey indicating >70% satisfaction with access request experience
- Support team reporting reduced complexity in organizational structure and inheritance-related escalations
- Improved understanding of how roles apply through organizational context

**Enterprise adoption:**

- Fortune 500 customers successfully implementing Teams architecture
- Migration completion from group-based to team-based access within 6 months

**Competitive position:**

- Feature parity or superiority compared to major DevOps platforms
- Sales team reporting Teams architecture as competitive advantage
- Customer retention improvement in enterprise segment

## Dependencies and Integration Points

### Internal GitLab Dependencies

**Organization Framework Integration:**

- Organization-level user management and identity provider synchronization
- Organization-scoped audit trails and compliance reporting
- Organization owner administrative capabilities for Teams management
- Organization isolation enforcement for all Teams functionality

**Feature Integration Requirements:**

- All existing access patterns in all DevOps stages must continue to work with Teams-based accesses

### External Integration Requirements

**Enterprise Identity Systems:**

- LDAP/enterprise directory group synchronization and mapping
- SAML assertion processing for team membership claims
- OIDC integration for just-in-time team provisioning
- Multi-factor authentication integration for sensitive team operations

### Migration and Compatibility Requirements

**Backward Compatibility:**

- Current group-based permissions must continue working during transition
- Existing API endpoints must maintain functionality with deprecation notices
- Customer automation scripts must continue working without modification
- Third-party integrations must not be disrupted during migration

**Migration Path Requirements:**

- Automated tools for converting group-based access to team-based access
- Rollback capabilities in case of migration issues
- Comprehensive validation tools to ensure permission equivalence
- Documentation and training materials for administrators managing transition

## Risk Assessment and Mitigation

### Performance and scalability risks

- Large-scale permission resolution could impact system performance
- Complex inheritance calculations may create database bottlenecks
- Migration process could cause temporary service degradation

**Mitigation strategies:**

- Implement comprehensive caching layers with cache invalidation strategies
- Design database schema optimized for permission resolution queries
- Plan migration in phases with rollback capabilities and performance monitoring

### Security risks

- Migration errors could inadvertently grant or remove access inappropriately
- New inheritance patterns could create unexpected privilege escalation paths
- Audit trail gaps during transition could impact compliance

**Mitigation strategies:**

- Extensive automated testing of permission resolution logic
- Comprehensive audit logging before, during, and after migration
- Security review of all inheritance patterns and edge cases

### Customer disruption risks

- Large enterprise customers may resist changes to access control patterns
- Learning curve for administrators could temporarily reduce productivity
- Integration with existing customer workflows may require significant adaptation

**Mitigation strategies:**

- Extensive customer research and feedback integration during design phase
- Comprehensive documentation, training, and migration support
- Gradual rollout with opt-in capabilities for early adopters

## Next Steps and Implementation Approach

### Research and Validation Phase

1. **Competitive feature analysis** (FY26Q2)
   - Detailed feature comparison with major DevOps platforms
   - Analysis of customer switching patterns and motivations
   - Evaluation of enterprise sales competitive positioning
   - Assessment of differentiation opportunities

2. **Detailed user research** (FY26Q3)
   - Conduct usability studies on inheritance pattern preferences
   - Interview enterprise customers about organizational modeling needs
   - Test mental model alignment through prototype evaluation
   - Validate success metrics through baseline measurement

3. **Technical architecture deep dive** (FY26Q4)
   - Design database schema for optimal permission resolution performance
   - Create proof-of-concept implementation for scalability testing
   - Evaluate integration patterns with existing GitLab architecture
   - Assess migration complexity and risk mitigation strategies

### Design and Prototyping Phase

1. **User experience design** (FY26Q4)
   - Wireframes and user flows for team creation and management
   - Permission inheritance visualization and audit trail interfaces
   - Migration tools and workflows for administrators
   - Progressive disclosure patterns for complex scenarios

2. **Architecture design documentation** (FY27Q1)
   - Detailed technical specifications for Teams entity and relationships
   - API design for team management, membership, and permission resolution
   - Database schema design optimized for performance and auditability
   - Integration specifications for external identity providers

3. **Implementation planning** (FY27Q2)
   - Phased delivery roadmap with clear milestones and dependencies
   - Resource allocation and team structure requirements
   - Risk mitigation plans and rollback strategies
   - Customer communication and support planning
