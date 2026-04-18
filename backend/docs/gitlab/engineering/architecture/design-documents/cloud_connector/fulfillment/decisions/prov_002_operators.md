---
title: "Cloud Connector ADR PROV-002: Operator-Based Entitlements"
owning-stage: "~devops::fulfillment"
toc_hide: true
---

## Context

### Problems with the Previous Entitlement Architecture

#### Limitations and technical debt

The entitlement model treated **unit primitives (features)** as the *only* axis of evaluation. There was no notion of *operator-level entitlements* (who delivers the feature), which led to several limitations:

* **No way to combine add-ons per operator**
  In self-hosted scenarios, all AI features were forced to require `duo_enterprise`.
  It was impossible to support combinations such as:

  > "Self-Hosted + Duo Pro"

  Workarounds involved:

  * Overriding Cloud Connector configuration at runtime
  * Hardcoding rules like:

    > "All features now require Duo Enterprise under Self-Hosted"

  Similar problems arose with Amazon Q and future third-party operators.

* **No operator-level monetization**
  The same unit primitive could not be packaged or priced differently depending on who operated it.
  Example: `duo_chat` might be free under GitLab Cloud but paid under Amazon Q.

* **Workarounds increased technical debt**
  To handle missing operator-level entitlements, we relied on runtime overrides and hardcoded rules
  (e.g. “all features now require Duo Enterprise under Self-Hosted”).
  These workarounds duplicated logic outside Cloud Connector and left entitlement checks inconsistent and fragile.

## Decision

We will introduce **operator-level entitlements as a first-class dimension** alongside unit primitives.
This adds a new axis of flexibility, enabling us to package and monetize the same features differently depending on *who operates them*.

All operator keys will follow the naming convention:

* `<name>_operator`

### Operators

* `gitlab_cloud_operator` → GitLab operates the feature for the customer (cloud-based execution)
* `self_hosted_operator` → The customer operates the feature themselves (self-hosted execution)
* `amazon_q_operator` → Amazon operates the feature (Amazon Q integration)

## The New Architecture: Operator + Unit Primitive

### Two-dimensional entitlements

Instead of asking only:

> "Is feature X allowed?"

The new model evaluates:

> "Is operator Y allowed?"
> "Is feature X under operator Y allowed?"

### What is an operator?

An **operator** is the **execution authority** responsible for delivering a Unit Primitive (UP).

**Why it's separate from add-ons and UPs:**

* **Add-On** → commercial SKU the customer buys (pricing/packaging)
* **Operator** → *who runs the feature* (GitLab, Customer, Amazon, etc.)
* **Unit Primitive (UP)** → the feature capability itself

## Entitlement evaluation

* **Within a requirement list** (e.g. `add_ons` or `license_types` under an operator or a unit primitive):
  The entries are treated as an **OR list**. Any single matching add-on or license type is sufficient to satisfy that requirement.

* **Between operator and unit primitive**:
  Both must be satisfied. Final access is only granted if:

    1. **Operator requirements are met**
    2. **Unit primitive requirements are met**

Seat checks apply **only** to the specific add-on that satisfies a requirement if that add-on is seat-scoped.

**Example:**

* `duo_chat` supports: `duo_core` **or** `duo_pro` **or** `duo_enterprise`
* `gitlab_cloud_operator` is free → `duo_chat` is granted if the customer has any one of those add-ons.

  * An **unassigned** end user can access via `duo_core` because the operator is free and `duo_core` is instance-wide (non seat-based).

* `self_hosted_operator` requires: `duo_enterprise`

  * An **unassigned** end user with `duo_enterprise` + `duo_core` is **DENIED**: the operator's witness add-on is `duo_enterprise` (seat-scoped) and fails assignment.
  * An **assigned** end user with `duo_enterprise` + `duo_core` is **ALLOWED**: operator satisfied by assigned `duo_enterprise`; UP can be satisfied by `duo_core` (instance-wide) or by an assigned seat SKU.

## Implementation Details

### Example operator configuration

```yaml
---
name: self_hosted_operator
add_ons:
- duo_enterprise
license_types:
- premium
- ultimate
```

### Example unit primitive configuration

```yaml
---
name: duo_chat
add_ons:
- duo_core
- duo_pro
- duo_enterprise
license_types:
- premium
- ultimate
operators:
- gitlab_cloud_operator
- self_hosted_operator
- amazon_q_operator
```

## Migration Plan

1. Update Cloud Connector configs to introduce operators following the `_operator` naming convention:

   * `gitlab_cloud_operator`
   * `self_hosted_operator`
   * `amazon_q_operator`

2. Migrate entitlement checks to resolve entitlements in the order:
   > operator → unit primitive
3. Remove legacy runtime overrides (`self_hosted_models`) once all checks are migrated.

## Consequences

**Positive:**

* Enables different monetization models per operator
* No runtime overrides or patching required
* Flexible configuration in Cloud Connector

**Negative:**

* Increases complexity around entitlement logic and configuration
* Requires migration of existing entitlement checks to respect operators

## Useful Links

* [GitLab Cloud Connector - unit primitive config](https://gitlab.com/gitlab-org/cloud-connector/gitlab-cloud-connector)
* [Initial discussion on operator-based entitlements](https://gitlab.com/gitlab-org/gitlab/-/issues/502821)
* [Self-Hosted Models architectural constraints](https://gitlab.com/gitlab-org/gitlab/-/issues/503210)
