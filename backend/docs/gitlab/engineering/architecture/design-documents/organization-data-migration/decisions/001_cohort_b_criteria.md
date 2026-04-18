---
title: 'Organization Data Migration ADR 001: Cohort B Criteria'
owning-stage: "~devops::tenant scale"
toc_hide: true
---

## Context

Cohort B is the first cohort where we gain experience with real daily active users on Protocells. Unlike Cohort A (inactive free users focused on infrastructure validation), Cohort B introduces active workflows, real customer expectations, and the need for Security and Trust and Safety readiness. The goal is to learn from a small, curated, low-risk set of active organizations before scaling to Cohort C (top 1,000) and beyond.

## Decision

Cohort B eligibility for Organization data migrations to Protocells focuses on active, opt-in beta organizations. It establishes enrollment, profile, and trust and safety criteria to de-risk migrations before scaling to larger cohorts.

| Attribute | Value |
| --- | --- |
| Cohort name | Active Opt-In Beta |
| Target size | Up to 1,000 organizations (start with ~10 to 50 hand-picked, scale to 1,000) |
| Engagement model | Opt-in, guided (human-in-the-loop vetting) |
| Prerequisite | Cohort A migration validated; ability to transfer TLG to org with users' current feature set supported |
| Expected impact | Iteration on real-world adoption of Organizations and Cells. Tiny impact on WAL, LWLocks, and database size |

### 1. Enrollment and engagement criteria

#### 1.1 Opt-in with guided onboarding

Organizations must explicitly opt in through a curated enrollment process. There is no self-service sign-up for this cohort.

Tenant Scale, Protocells Steering Committee, and Security review each candidate before approval.

Customers must acknowledge the beta nature of the experience, including potential for brief read-only windows during migration, the one-way nature of the transfer, and the potential unavailability of certain features they have not yet adopted.

#### 1.2 Phased ramp

- Phase 1 (initial wave): 10 to 50 hand-picked organizations, heavily monitored with dedicated support.
- Phase 2 (expanded beta): Scale to up to 1,000 organizations as confidence builds from Phase 1 learnings.

### 2. Organization profile criteria

#### 2.1 Visibility and namespace structure

- Private namespaces only. Public namespaces are excluded to avoid complexity with cross-namespace visibility and public-facing feature surface area.
- Single root namespace per organization (single TLG to org transfer). Organizations with multiple TLGs requiring consolidation are deferred to later cohorts.
- No cross-namespace interactions. The organization's users should primarily operate within a single root namespace. Users who contribute significantly to other root namespaces outside the organization are excluded.

#### 2.2 License tier

- Free or paid (Premium or Ultimate) organizations are eligible, with different vetting levels.
- Free organizations: Lower bar for inclusion; useful for validating the full Organizations experience with active users who have lower expectations and no contractual SLAs.
- Paid organizations: Higher bar. Requires explicit Customer Success approval and a documented understanding of the beta program. Only include paid customers who have been vetted for feature usage compatibility (see Section 3 below).
- No trial subscriptions. Trial customers are in a transient state and unsuitable for a one-way migration.

#### 2.3 User count and activity level

- Active root namespace: At least one active user in the last 30 days (by login or Git or API activity).
- Small to medium user count: Target organizations with 500 or fewer users to bound the blast radius and migration complexity. Large enterprises (1,000 or more users) are deferred to Cohort C or D.
- Moderate activity level: Exclude the top 1,000 namespaces by database time (Cohort C). Target organizations in the long tail that are active but not disproportionately heavy on database load.

### 3. Trust and safety and security de-risking criteria

Based on the Security and Protocells sync discussions with Matt Coons and Ruby Nealon, the following criteria de-risk Cohort B from a security and trust perspective.

#### 3.1 Curated and vetted list

- Cohort B must be a curated list of organizations reviewed by Security Operations and Trust and Safety leadership before migration.
- Matt Coons stated: "If it was trusted orgs I'd be more confident" and "leadership buy-in might be easier if it's a curated list, we could then accept some coverage gaps."
- Each organization should be vetted against Trust and Safety criteria before enrollment.

#### 3.2 Account standing and abuse history

- No banned or flagged users within the organization.
- No history of abuse reports against the namespace or its members (verified via Omamori data on the legacy cell).
- No disposable email domains associated with the organization's members.
- Account age requirement: Organization owner account must be at least six months old to filter out ephemeral or bot-driven accounts.
- No recent Trust and Safety interventions in the past 90 days.

#### 3.3 No new user sign-ups on the cell

- New user registration must not be allowed on the Protocell. All new sign-ups continue on the legacy cell. This prevents unknown or untrusted users from landing on a cell without full Trust and Safety coverage.
- Users within migrated organizations access the cell through their existing authenticated sessions.
- Organization admins must explicitly invite or add new users to their organization.

## Consequences

- Cohort B is limited to a curated set of small, private, low-risk organizations, which bounds the blast radius but also limits the breadth of real-world validation.
- Every candidate organization requires manual vetting by Security Operations and Trust and Safety leadership before enrollment, adding operational overhead.
- No self-service sign-up is available; all enrollment is guided, which limits scale but increases control.
- Public namespaces and cross-namespace interactions are excluded, deferring complexity to later cohorts.
- The phased ramp (10-50 initially, then up to 1,000) provides incremental confidence but means Cohort B will take longer to fully execute.
- New user registration is blocked on Protocells, meaning all user growth continues on the legacy cell until Trust and Safety tooling coverage is sufficient.
- Paid customers require explicit Customer Success approval, adding a coordination dependency.
