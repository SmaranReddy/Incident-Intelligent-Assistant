---
title: "ADR 004: Bundle cosign with GitLab Rails, perform attestation in GitLab Rails backend"
description: "Architecture Decision Record for changes in location for attestation and technical means for signing."
---

## Context

This ADR revises a decision made in [ADR 003](003_attest_sign_location.md) with
regards to the signing location for provenance statements. The decision to
perform signing in the glgo service is further documented in the linked ADR. As
a summary, glgo was chosen because it was a feasible mechanism to achieve SLSA
Level 3 compliance, it allowed delivery of attestation for gitlab.com customers
and required no additional services or infrastructure.

While discussing options for the integration between Rails and glgo, concerns
were raised about the [number of dependencies](https://gitlab.com/gitlab-org/gitlab/-/issues/551210#note_2611268806)
that would be added to glgo in the event that the cosign project was directly
imported.

An alternative approach was evaluated, which would perform signing within
Sidekiq workers. This ADR contains several possible options through which the
`cosign` binary can be embedded into them, which would allow for the
generation of software attestations.

## Options Considered

### 1. Including `cosign` in the GitLab distribution

`cosign` is a [Go project](https://github.com/sigstore/cosign) that is developed
as part of the sigstore project. It allows for signing OCI containers (and other
artifacts) using Sigstore. Cosign also has built-in support for [in-toto attestations](https://github.com/sigstore/cosign?tab=readme-ov-file#in-toto-attestations).
A proof of concept attestation program is available [in this merge request.](https://gitlab.com/darbyfrey/demodemo/-/merge_requests/4/diffs#717ea309f1716e0f92cc07090402218cd8d142cc)

**Pros:**

* The simplest of available options in terms of development effort.
* The cosign binary is feature-complete. This is in contrast to [sigstore-ruby](https://github.com/sigstore/sigstore-ruby), for example, which only supports verification.
* `cosign` is the most widely used mechanism through which to interact with sigstore, which means there will be documentation and support if we run into any issues.
* Shipping a binary means we can leverage existing tooling to maintain dependencies, such as [Renovate](/handbook/engineering/infrastructure-platforms/gitlab-delivery/build/maintenance/renovate/)
* Calling out to `cosign` will result in this being run in a separate process, so it will have a degree of separation from the dependencies and concerns of the Rails backend.

**Cons:**

* Even though distribution is feasible, it is not trivial. CNG and Linux packages [need to be created](https://gitlab.com/gitlab-org/gitlab/-/issues/554600#note_2618812171).

### 2. Creating a ruby gem that would build cosign for the target platform

**Cons:**

* It is effectively a workaround to potential difficulties in distribution. Since distributing a small, well-known go binary is [relatively easy](https://gitlab.com/gitlab-org/gitlab/-/issues/554600#note_2618812171) this is not necessary.
* It is more complex than the alternatives.
* Would need to deal with the intricacies of Ruby packaging.
* Would need to keep up with updates to the cosign binary and update it in the event there are security vulnerabilities.

### 3. Use a native Ruby implementation of cosign

**Pros:**

* Potentially a reduced number of dependencies.
* Does not introduce distribution complexity.

**Cons:**

* The current version of [sigstore ruby](https://docs.sigstore.dev/language_clients/ruby/) offers a very small subset of the features provided by cosign.
* It would require substantial development time.
* Requires non-trivial integration code with several third-party providers.
* Requires maintenance work to keep up with any changes to standards or third-party APIs.

## Decision

**We will move forward with Including cosign in the GitLab distribution.**

This option is the simplest in terms of infrastructure complexity, and provides
good support for all sigstore features through a widely used client.

Additionally, background workers are a good choice for this type of workload,
and avoiding interactions with an external service will mean that the points
where there could be a failure are significantly reduced. This will make the
implementation more available.

From a security perspective, we have a [well-established methodology](https://docs.gitlab.com/development/shell_commands/)
for dealing with interactions with external binaries in Rails, and we do not
have to ensure the integrity of the provenance statement is preserved between
the two services.

## Consequences

### Positive

* Reduced overall complexity.
* Because `cosign` is very widely used, we are likely to be able to support all required use cases. Additionally, we can use the exhaustive documentation available.
* Reduced distribution complexity as we remove the dependency on `glgo`. This service is not available for all types of GitLab installations at this stage.
* Existing tooling to maintain dependencies, such as [Renovate](/handbook/engineering/infrastructure-platforms/gitlab-delivery/build/maintenance/renovate/), can be used.

### Negative

* Interacting with a command-line tool is high-risk. Although initially the parameters we would pass to the tool are not user-controlled, this may change later depending on requirements. This negative consequence can be mitigated by adhering to well-established procedures documented in the [Shell command development guidelines page](https://docs.gitlab.com/development/shell_commands/).

## Performance & availability

In order to ensure high-levels of availability and robust performance, several
measures will be put in place by the `SSCS: Pipeline Security` team.

Firstly, in order to assess performance of `cosign` calls in production, we will
instrument the binary. This will allow for detailed observation of performance
of the binary at a high-level as well as detailed analysis of any potential
bottlenecks during execution.

Secondly, rollout of this feature will occur behind a feature flag so that we
can measure the impacts to performance as we scale up. Furthermore, while the
[architectural decision](https://gitlab.com/gitlab-org/gitlab/-/issues/547903#note_2574511236)
has not been finalised, at this stage this feature will only be enabled for
users who specifically opt-in.

Additionally, the Sidekiq job will be configured to retry [as appropriate](https://docs.gitlab.com/development/sidekiq/#retries) to ensure
that sporadic outages of sigstore infrastructure do not lead to persistent
failures in SLSA attestation generation.

### Benchmarking

The relative performance difference in signign between Golang and GitLab Rails
[was analysed](https://gitlab.com/gitlab-org/gitlab/-/issues/556202). In both cases the test
performs the signature and publishes the records to `Rekor` *two times*.

The analysis demonstrates no significant performance gains in Golang relative to
GitLab Rails.

#### GitLab Rails

```console
Command: ["cosign", "attest-blob", "--yes", "--new-bundle-format",
"--predicate", "-", "--bundle", "demodemo.gem.sigstore.json", "--oidc-issuer",
"https://gitlab.com", "--identity-token", "[MASKED]", "--hash",
"7a313044bd530eef848b8cce175073e90ef2287e4290ac805cedbb7d42bc580e",
"file://demodemo.gem"]

...

--- Execution Time ---
Wall clock time: 0.506 seconds
Monotonic time: 0.506 seconds
Duration (ms): 506.1 milliseconds
```

#### Golang

```console
--- Execution Time ---
Execution time: 0.491 seconds (490.6 ms)
Total time: 0.491 seconds (490.6 ms)

```

## Related Links

* [Evaluate options to bundle cosign with GitLab Rails](https://gitlab.com/gitlab-org/gitlab/-/issues/554600)
* [Evaluate options to limit dependencies or sandbox execution of cosign in glgo](https://gitlab.com/gitlab-org/gitlab/-/issues/554596)
