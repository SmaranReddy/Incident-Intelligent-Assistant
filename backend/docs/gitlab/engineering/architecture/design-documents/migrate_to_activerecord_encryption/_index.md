---
# This is the title of your design document. Keep it short, simple, and descriptive. A
# good title can help communicate what the design document is and should be considered
# as part of any review.
title: Migrate to ActiveRecord Encryption
status: postponed
creation-date: "2025-08-29"
authors: [ "@boconnor" ]
coaches: [ "@grzesiek" ]
# dris: [ "@product-manager", "@engineering-manager"
owning-stage: "~devops::runtime"
participating-stages: ["~Data Security Team", "~Department::Product Security" ]
# Hides this page in the left sidebar. Recommended so we don't pollute it.
toc_hide: true
---

<!-- Design Documents often contain forward-looking statements -->
<!-- vale gitlab.FutureTense = NO -->

<!-- This renders the design document header on the detail page, so don't remove it-->
{{< engineering/design-document-header >}}

> **Terminology note**
>
> Encrypting specific values inside a database can be called "row encryption," "column encryption," "field encryption," "value encryption," or other terms which are _nearly_, if not perfectly, synonymous. This document will refer to this technology as "intra-database encryption."

## Summary

Our use of encryption inside the monolith database has two major problems:

* We use a `db_key_base` element from the `config/secrets.yml` file which is used as the key for the majority of our encrypted fields. [We are not able to rotate this key like we do other encryption keys](https://gitlab.com/gitlab-org/gitlab/-/issues/25332) because we do not have tooling to handle doing so without substantial downtime for large installations of GitLab.
* Due to the way we use [`attr_encrypted`](https://github.com/attr-encrypted/attr_encrypted) in the monolith, an attacker who accesses a sufficient amount of encrypted data may be able to recover the `db_key_base`.

These two issues, taken together, motivate us to change out our intra-database encryption method entirely, changing our encryption library from [`attr_encrypted`](https://github.com/attr-encrypted/attr_encrypted) to [Active Record Encryption](https://guides.rubyonrails.org/active_record_encryption.html) and re-encrypting fields within the database.

This document draws extensively from, and deprecates, the [previous encryption key rotation blueprint](/handbook/engineering/architecture/design-documents/encryption_keys_rotation/).

## Motivation

GitLab encrypts specific sensitive data inside the database; in particular, GitLab encrypts credentials that it holds on behalf of users (e.g., deployment credentials and similar). Encryption key rotation is a [recommended best practice](https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-57pt1r5.pdf) (see section 5.3.6, subsection 6) for all encryption keys.

After [considering the status quo, threat models, and implementation choices for new intra-database encryption](https://gitlab.com/gitlab-com/gl-security/product-security/data-security/data-security-team/-/issues/101) (GitLab-internal link), a designated working group adopted the recommendation that we move to ActiveRecord Encryption for our intra-database encryption needs, specifically using envelope encryption with optional cloud-hosted key management systems and, where possible, allowing the use of [OpenBao](https://openbao.org/), part of [GitLab Secrets Manager](/handbook/engineering/architecture/design-documents/secret_manager/) (GLSM), to provide a unified access model for cloud KMS, cloud HSMs, and on-premises HSMs.

### Goals

* Stop using `attr_encrypted` in the monolith.
* Implement envelope encryption with Active Record Encryption (ARE).
* Support a cloud-based key management system (e.g., Google Cloud HSM, Google Cloud KMS, AWS KMS) as part of the envelope encryption system.
* Re-encrypt all encrypted data in .com, and (if necessary) provide migration tooling applicable to all types of GitLab installations.
  * Note that this includes the small number of ARE-encrypted fields currently in the database, as they do not use envelope encryption or cloud-backed encryption.
  * Handling the migration transparently during an upgrade would be preferable to requiring user interaction with custom tooling to perform the upgrade; however, a transparent migration may not be possible.

### Non-goals

* OpenBao/GLSM deployment and configuration: while we will hope to utilize GLSM if it is available, its details are best left in its [own blueprint](/handbook/engineering/architecture/design-documents/secret_manager/). Where GLSM is not available, we will provide a non-GLSM-facilitated source of key material.
* Enterprise Bring Your Own Key (BYOK) functionality: while BYOK for intra-database encryption is enabled at a technical level by using cloud-backed envelope encryption, building the feature into the product is outside the scope of this blueprint, and can be completed as a fully-separate task after this blueprint. (Realistically, it should happen after the GA of Organizations.)

## Proposal

We will migrate all encrypted fields in the monolith to [envelope encrypted](https://guides.rubyonrails.org/active_record_encryption.html#envelopeencryptionkeyprovider) fields using Active Record Encryption (ARE). We will use the functionality from the [`active_kms`](https://github.com/ankane/active_kms) library to back our envelope encryption with an appropriate cloud key management system (AWS KMS or Google Cloud KMS), where available; when this is not available, the primary KWK will be stored on disk.

If [OpenBao / GitLab Secrets Manager](/handbook/engineering/architecture/design-documents/secret_manager/) is available in an environment and has the ability to [function as a key management proxy](https://github.com/openbao/openbao/issues/1319), we will also use the [`transit` secrets engine](https://openbao.org/docs/secrets/transit/) as one way to implement this functionality.

We will intentionally disable ARE's [deterministic encryption](https://guides.rubyonrails.org/active_record_encryption.html#deterministic-and-non-deterministic-encryption) feature, which allows (a very limited form of) string matching on encrypted attributes. This will improve our security posture as described in the ARE documentation, and since key rotation is not supported by ARE's deterministic encryption, it prevents us being stuck in a situation in the future where key rotation again requires major engineering effort.

A complete discussion of the alternatives considered is available in [this issue](https://gitlab.com/gitlab-com/gl-security/product-security/data-security/data-security-team/-/issues/101) (GitLab-internal link).

### Definition: envelope encryption

[Envelope encryption](https://cloud.google.com/kms/docs/envelope-encryption) is one type of multilayered key management scheme. At its core, it uses two encryption keys:

* The **Data Encryption Key** (DEK) is a symmetric encryption key used to encrypt particular pieces of data.
* The **Key Wrapping Key** (KWK), also known as a Verpackungsschlüssel (source: @joernchen), is a symmetric or asymmetric key used to encrypt the DEK. (Some groups call this a Key Encrypting Key; however, KWK is a NIST term, and we will use it exclusively.)

Each blob of encrypted data includes an identifier for the DEK that encrypts it. This enables different DEKs (optionally, encrypted by different KWKs) to be used for different purposes, fields, or customers. During runtime, when a node first encounters a DEK it doesn't have cached, it will perform a KWK decryption operation to access the decrypted DEK, which it will then cache; subsequent DEK operations will not require accessing the KWK until the cache is purged.

Usually, the KWK is stored in a secure hardware element (like the [Apple Secure Enclave](https://support.apple.com/guide/security/secure-enclave-sec59b0b31ff/web), a [Hardware Security Module](https://en.wikipedia.org/wiki/Hardware_security_module), etc.). While a DEK could be stored in such an element, a local element would be extremely slow to use directly to decrypt data (orders of magnitude slower than a using key held in memory with CPU-accelerated cryptographic instructions), and a cloud element would be extremely costly (GCP Cloud KMS costs $0.03 per 10k operations, which at GitLab's scale would cause deleterious consequences for shareholder value). The DEK is stored encrypted elsewhere (for instance, in a database), and decrypted on demand.

Envelope encryption provides three key features:

* Accessing encrypted data ultimately requires access to a KWK stored in a secure element, _without_ requiring that _every_ decryption operation be performed by that secure element.
* Changing a KWK does not require re-encrypting every piece of data, but instead only re-encrypting the DEKs that are wrapped by that KWK.
  * One resource group (e.g., a Cell, Organization, or other; see the discussion below on the numbers of KWKs and DEKs in a system) might have multiple DEKs or even KWKs active at one time. There are situations in which, rather than re-encrypting data meant to be destroyed soon, we might choose to leave the data encrypted under an old key and archive it along with its key.
* Migrating encrypted data between systems (e.g., between [Cells](/handbook/engineering/infrastructure-platforms/tenant-scale/cells_and_organizations/)) does not require re-encrypting each piece of data, but instead only [re-encrypting the DEKs that protect that data](#workflow-migrating-data-between-gitlab-instances).

Choosing the number of DEKs and KEKs in a system may be guided by the following ideas.

Assuming that the numbers `n`, `d`, and `k` represent the numbers of encrypted fields, DEKs, and KEKs, respectively:

* The case where `n == d` means that each piece of data will have its own DEK. This will cause each data decryption to require a KWK operation (performed by a secure element), which is costly and undesirable. **Note: this is how ActiveRecord Encryption works in its default configuration.**
* The case where `d == k` means that every piece of data in a system (e.g., a Cell) is encrypted by the same DEK. This minimizes the number of KWK operations needed (by maximizing the ability to cache the decrypted DEK in memory), but maximizes the harm an attacker can do by accessing a single decrypted DEK (e.g., on a running node).
* [Organizations](/handbook/engineering/infrastructure-platforms/tenant-scale/cells_and_organizations/) MAY share DEKs. Large Organizations SHOULD NOT share DEKs with other Organizations, where possible, to facilitate migrations between Cells.

As this Blueprint is being made to memorialize our research and decisions at this time, with an unknown but nontrivial time between the Blueprint and implementation (potentially a year or more), we have not chosen values for `d` or `k` at this time, as it seems likely that there may be changes in the interim which would recommend new values.

Access permissions to the encrypted DEK will usually be granted in the same way as access to the ciphertext (that is, they're both stored in the database). The KWK can be controlled separately using the (necessarily) separate permissions scheme of the secure element. This opens the possibility to create multi-layered permissions schemes using one or more KWKs (a KWK can wrap another KWK which wraps a DEK, [and so on](https://en.wikipedia.org/wiki/Turtles_all_the_way_down)).

#### Extension: cloud-backed envelope encryption

When using envelope encryption, the KWK is often stored in a secure hardware element; that element can itself be cloud-based, using any of the major cloud platforms (e.g., [AWS KMS](https://aws.amazon.com/kms/), [GCP Cloud Key Management](https://cloud.google.com/security/products/security-key-management?hl=en), [Azure Key Vault](https://azure.microsoft.com/en-us/products/key-vault), etc.). This enables all the types of flexibility enhancements that come from cloud-based infrastructure (infrastructure-as-code deployment, flexible payment models, multiple locations, etc.), as applied to secure hardware elements. Some of these systems are FIPS 140-3 compliant, or have a mode or close parallel which is; this enables building FedRAMP solutions on top of these systems.

## Design and implementation details

Not all implementation details are fleshed out in this document at this time. This is intentional, as the timeline for beginning the implementation has not yet been set, and may be as late as Q4FY27 or Q1FY28 (February 2027).

### Key decisions to be made when preparing for implementation

* How should the monolith decide when to generate and use a new DEK? (The answer here may be as straightforward as "one DEK per Organization," or it might be more complex; one DEK per Organization provides a starting point for discussion.)

### Pre-work

* Integrate the necessary portion of the [`active_kms`](https://github.com/ankane/active_kms) library into the monolith.
* Create the configuration parameters necessary to designate which keys to use, for what purpose, and using which KWK storage mechanisms.
* Building on the [existing `.migrate_to_encrypts` method](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/191926) and ancillary work, update the method to use KWKs and generate DEKs as necessary (see the above discussion on how many DEKs to generate).

### Migrate columns

The workflow [proposed by Rémy to accomplish a re-encryption](https://gitlab.com/groups/gitlab-org/-/epics/15420#process), and [used for a PoC migration](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/189940) is (roughly):

1. Create a migration to add a new column `tmp_<field>`, and make appropriate changes to the model (utilizing tooling that will handle querying both columns during the transition).
2. Create a Batched Background Migration to do the decryption/encryption, and push to production.
3. After the BBM is complete, again make changes to the model to remove the transition tooling, rename the `tmp_<field>` to `<field>` (and mark the old `<field>` for removal), and mark ancillary fields (such as the IV and salt fields) for deprecation and removal. Push to production.
4. Remove the ancillary fields. Push to production.
5. Remove the deprecation/removal (ignore) flags. Push to production.

This process is straightforward, but intricate, with details that need to be checked against the runbook each time, and with the possibility of each field's having a few extra flags, configuration values, etc. that need to be migrated and removed alongside the field. Accordingly, it is an [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) problem. It will take a number of engineers a nontrivial amount of time to accomplish this for the 239 `attr_encrypted` fields, the 14 existing `ActiveRecord::Encryption` fields (which must now be rekeyed), and the few `Lockbox` and `TokenAuthenticatable` fields (for the latter, only when used as encryption rather than a simple token).

### Workflow: migrating data between GitLab instances

First, the question: is/are the DEK(s) shared between objects that are to be moved to the new location, and objects that are not moving?

* **If yes:** generating a new DEK for the objects to be moved and re-encrypting them in place (before moving the data) would be a best practice to avoid the same DEK being in use in two places, encrypted under two different KWKs. (This new DEK is what is referenced below.)
* **If no:** Proceed.

The actual move, then:

1. Decrypt the DEKs for the objects to be migrated.
2. Encrypt the DEKs under a KWK for the **new** environment (either an existing KWK or a new KWK), and store the newly-wrapped DEKs in the new environment.
3. Copy the encrypted data from the old environment to the new environment; re-encryption is not needed.

To facilitate this migration, we will need tooling to trigger the DEK re-encryption, in addition to existing tooling to copy data between databases.

## Alternative solutions

A complete discussion of the alternatives considered (including "do nothing") is available in [this issue](https://gitlab.com/gitlab-com/gl-security/product-security/data-security/data-security-team/-/issues/101) (GitLab-internal link).
