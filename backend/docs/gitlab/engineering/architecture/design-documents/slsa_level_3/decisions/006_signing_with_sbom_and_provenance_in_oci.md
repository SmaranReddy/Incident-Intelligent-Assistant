---
layout: handbook-page-toc
title: "ADR 006: Enable the creation of SLSA Level 3 Attestations for OCI images"
description: "This ADR establishes the technical mechanisms through which we can perform attestations of images within Open Container Initiative (OCI) registries within the control plane."
---

## Background

GitLab provides mechanisms through which our users can generate SLSA Level 2
attestations, as mentioned in the [Achieve SLSA Level 2 compliance with GitLab](https://about.gitlab.com/blog/achieve-slsa-level-2-compliance-with-gitlab/) blog post.
Additionally, GitLab is working towards providing ways for users to perform
Level 3 attestations, as documented in the [SLSA Level 3 Provenance Attestations](/handbook/engineering/architecture/design-documents/slsa_level_3/) epic.

To achieve this, it is a requirement of SLSA Level 3 that attestations are
generated outside the build, and within the control plane. Mechanisms for attesting artifacts in this way have already been released behind a
feature flag, as documented in the [Attestations API](https://docs.gitlab.com/api/attestations/)
and [GitLab SLSA](https://docs.gitlab.com/ci/pipeline_security/slsa/) pages.

This ADR establishes the technical mechanisms through which we can perform
attestations of images in Open Container Initiative (OCI) registries within
the control plane. Additionally, it documents why we are interested in
supporting this feature.

The scope of this ADR is restricted to the same constraints that our attestation
of artifacts. Specifically, only public projects using the [public good](https://openssf.org/blog/2023/10/03/running-sigstore-as-a-managed-service-a-tour-of-sigstores-public-good-instance/)
Sigstore infrastructure in GitLab.com are supported.

### Why is this change required?

Following on our work performing SLSA Level 3 attestations of artifacts, we can
leverage aspects of our previous integration to provide support for this very
common scenario. More information around the rationale for facilitating the
production of SLSA Level 3 artifacts and container images can
be found in the [epic page](/handbook/engineering/architecture/design-documents/slsa_level_3/#goals).

From a SLSA Level 3 compliance perspective, this change is required as
documented below.

## High-level overview

At a high-level, regardless of which OCI registry we use, the process of
attesting a registry image is as follows:

- A build produces an image and uploads it to a registry. For example, as documented in [Build and push container images to the container registry](https://docs.gitlab.com/user/packages/container_registry/build_and_push_images/).
- The build passes the required environment variables, specifically
  `ATTEST_CONTAINER_IMAGES`, `IMAGE_DIGEST`, and [SIGSTORE_ID_TOKEN](https://docs.gitlab.com/ci/yaml/signing_examples/#build-artifacts).
- `Ci::BuildFinishedWorker` is called and itself calls
  [`Ci::Slsa::PublishProvenanceService`](https://gitlab.com/gitlab-org/gitlab/-/blob/master/app/services/ci/slsa/publish_provenance_service.rb)
  if `SupplyChain::publish_provenance_for_build?` is true.
- Within this service, the configuration is checked, and if appropriate an
  attestation is performed.
- Information about this attestation is persisted in the `Ci::Slsa::Attestation`
  model.
- Users can verify this attestation using `glab`.

### UX Changes Required

The proposal [in the UX discussion](https://gitlab.com/gitlab-org/gitlab/-/work_items/547903#note_2748278333)
outlines a high-level approach. After making some changes based on the proof of
concept, the final UX will look as below:

```yaml
build-gitlab-registry:
  stage: build
  variables:
    ATTEST_CONTAINER_IMAGES: true
  id_tokens:
    SIGSTORE_ID_TOKEN:  # Used by Cosign to get certificate from Fulcio
      aud: sigstore
  script:
    - DOCKER_IMAGE_NAME=registry.test:5100/root/control-plane-container
    - echo "$CI_REGISTRY_PASSWORD" | docker login $CI_REGISTRY -u $CI_REGISTRY_USER --password-stdin
    - docker build -t $DOCKER_IMAGE_NAME .
    - docker push $DOCKER_IMAGE_NAME

    - IMAGE_DIGEST="$(docker inspect --format='{{index .Id}}' "$DOCKER_IMAGE_NAME") # returns a SHA256 has instead
    - echo "IMAGE_DIGEST=$IMAGE_DIGEST" >> build.env
  artifacts:
    reports:
      dotenv: build.env
```

The attestation will occur if the following conditions are met:

- ATTEST_CONTAINER_IMAGES = true
- IMAGE_DIGEST exist
- project.public? is true

### References

- [Open Container Initiative - Open Container Initiative](https://opencontainers.org/)
- [SLSA • Build: Requirements for producing artifacts](https://slsa.dev/spec/v1.2/build-requirements)
- [Registry Support - Sigstore](https://docs.sigstore.dev/cosign/system_config/registry_support/)
- [GitHub - google/go-containerregistry: Go library and CLIs for working with container registries](https://github.com/google/go-containerregistry)
- [How to Sign a Container with Cosign — Chainguard Academy](https://edu.chainguard.dev/open-source/sigstore/cosign/how-to-sign-a-container-with-cosign/)
- [Signing artifacts, attesting builds, and why you should do both](https://some-natalie.dev/blog/signing-attesting-builds/)
- [Calculate sha256 digest of artifact on PublishProvenanceService](https://gitlab.com/gitlab-org/gitlab/-/issues/559267)
- [[Discussion] UX to enable SLSA provenance generation (#547903)](https://gitlab.com/gitlab-org/gitlab/-/work_items/547903#note_2748278333)
- [Cosign Image Signatures. The protocol format explained!](https://medium.com/sigstore/cosign-image-signatures-77bab238a93)

## Technical background

### SLSA L3 Requirements

By performing attestations within the control plane, we are seeking to meet the [Provenance is Unforgeable](https://slsa.dev/spec/v1.2/build-requirements#provenance-unforgeable) requirement:

> Every field in the provenance MUST be generated or verified by the build
> platform in a trusted control plane. The user-controlled build steps MUST NOT
> be able to inject or alter the contents, except as noted in Provenance is
> Authentic.

More information on this topic is available in the [Architectural Decision: Where do we generate SLSA provenance? (#537049)](https://gitlab.com/gitlab-org/gitlab/-/work_items/537049)
ADR and [Clarify Provenance is Unforgeable requirements at Build L3 (#986)](https://github.com/slsa-framework/slsa/issues/986).

The control plane is [defined](https://github.com/slsa-framework/slsa/blob/56e4016f3c3332d64caf19ef629a1e06ccc7d407/docs/spec/v1.2/terminology.md) as a:

> Build platform component that orchestrates each independent build execution and produces provenance. The control plane is managed by an admin and trusted to be outside the tenant's control.

### Cross-registry support

Information on cross-registry support is available in the [Registry Support](https://docs.sigstore.dev/cosign/system_config/registry_support/)
page. `cosign` uses [`google/go-containerregistry`](https://github.com/google/go-containerregistry) for registry interactions, which has broad compatibility.

An example integration with the GitLab container registry and DockerHub is
available in [POC: Cosign artifact signing with SBOM and Provenance in OCI registries](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/219448).

## Key decision: Authentication

### Authenticating against the registry with `cosign attest`

As mentioned above, `cosign` uses `go-containerregistry` for integration with
registries. How that package handles authentication is documented in the [README file](https://github.com/google/go-containerregistry/blob/main/pkg/authn/README.md).

> The `DefaultKeychain` will use credentials as described in your Docker config
> file -- usually `~/.docker/config.json`, or `%USERPROFILE%\.docker\config.json` on
> Windows -- or the location described by the `DOCKER_CONFIG` environment
> variable, if set.

In practice, this means we can authenticate in the backend job as follows:

- Using the same techniques `docker login` uses, leveraging `cosign login`. By
  default, this way of authenticating stores credentials in a global location
  (specifically `~/.docker/config.json`). By overriding `DOCKER_CONFIG` as
  indicated above, we can override this behavior.
- Using the `cosign` command's `--registry*` parameters: `--registry-password`,
  `--registry-server-name`, `--registry-token`, `--registry-username`. A downside of
  this approach is that passing the password with a command-line parameter can
  lead to disclosure. See [Sending passwords securely with command line without being exposed in ps/wmic](https://stackoverflow.com/questions/50960822/sending-passwords-securely-via-command-line-without-being-exposed-in-ps-wmic-wi).

Of these two, overriding the `DOCKER_CONFIG` environment variable seems like the
most viable, in combination with temporary directories. An example of this can
be seen in the [container attestation proof of concept](https://gitlab.com/gitlab-org/gitlab/-/blob/0a81968df87d0f18e648b3c81db47ba8a7f68886/app/services/ci/slsa/publish_provenance_service.rb#L70).

Additional documentation regarding authentication:

- [Authenticate with the container registry](https://docs.gitlab.com/user/packages/container_registry/authenticate_with_container_registry/#use-gitlab-cicd-to-authenticate)
- [`docker login`](https://docs.docker.com/reference/cli/docker/login/)
- [Personal access tokens](https://docs.docker.com/security/access-tokens/#use-personal-access-tokens)
- [Private registry authentication in Amazon ECR - Amazon ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/registry_auth.html)

### Attestation without connection to the registry

`cosign attest` requires OCI Registry authentication for two reasons:

- Converting the OCI remote string to a SHA256 digest.
- Store the attestation in the remote OCI registry.

Because we are able to retrieve the SHA256 in the job itself, and because
storing the bundle ourselves aligns better with our database model, we are able
to attest containers as blobs. This has the advantage of not requiring access to
the registry and therefore being much simpler.

```shell
./cosign attest-blob --predicate ~/predicate.txt --type slsaprovenance1 --hash 127dfd758f2cf2ebab7ba7766bbf605694d7cc8ba68bc4f73386ac94e56d7eef --new-bundle-format --bundle container-blob.bundle index.docker.io/sroqueworcel/test-slsa-sbom
Using payload from: /Users/samroque-worcel/predicate.txt
Generating ephemeral keys...
Wrote bundle to file container-blob.bundle
~/code/cosign % ./cosign verify-blob --new-bundle-format --bundle container-blob.bundle --certificate-identity sroque-worcel@gitlab.com --certificate-oidc-issuer https://accounts.google.com /tmp/lol/blobs/sha256/127dfd758f2cf2ebab7ba7766bbf605694d7cc8ba68bc4f73386ac94e56d7eef
Verified OK

% cat container-blob.bundle | jq -r '.dsseEnvelope.payload' | base64 -d | jq  | head -n 15                                                                                                                                                                   {
  "_type": "https://in-toto.io/Statement/v0.1",
  "predicateType": "https://slsa.dev/provenance/v1",
  "subject": [
    {
      "name": "test-slsa-sbom",
      "digest": {
        "sha256": "127dfd758f2cf2ebab7ba7766bbf605694d7cc8ba68bc4f73386ac94e56d7eef"
      }
    }
  ],
  "predicate": {
    "buildDefinition": {
      "buildType": "https://docs.gitlab.com/ci/pipeline_security/slsa/provenance_v1",
      "externalParameters": {
```

This approach has several advantages:

- Does not require integration with any OCI registries. Attestations can be
  generated without reaching out to the registries.
- Does not require an upgrade of `cosign`.

### Attestation connecting to the registry

An example provenance attestation that reaches out to the OCI container (but
does not upload a file) looks as follows:

```shell
# Requires cosign v3.0.3 or later.
% ./cosign attest --predicate ~/predicate.txt --type slsaprovenance1 docker.io/sroqueworcel/test-slsa-sbom@sha256:127dfd758f2cf2ebab7ba7766bbf605694d7cc8ba68bc4f73386ac94e56d7eef --new-bundle-format --bundle container.bundle

% cat container.bundle | jq -r '.dsseEnvelope.payload' | base64 -d | jq | head -n 15
{
  "_type": "https://in-toto.io/Statement/v0.1",
  "predicateType": "https://slsa.dev/provenance/v1",
  "subject": [
    {
      "name": "index.docker.io/sroqueworcel/test-slsa-sbom",
      "digest": {
        "sha256": "127dfd758f2cf2ebab7ba7766bbf605694d7cc8ba68bc4f73386ac94e56d7eef"
      }
    }
  ],
  "predicate": {
    "buildDefinition": {
      "buildType": "https://docs.gitlab.com/ci/pipeline_security/slsa/provenance_v1",
      "externalParameters": {
```

Because attestations are matched exclusively by the SHA256, this attestation
can be verified in the same way as an attestation generated with the `cosign attest`
command.

### Integration with `Ci::Slsa::Attestation` and `glab`

The bundle files generated above are compatible with the `Ci::Slsa::Attestation`
model. This is because the way we attest artifacts also generates Sigstore
bundles.

By storing the contents of the bundle file above within that model, it
could then be retrieved in the same way as an artifact attestation is retrieved.
This approach will not require any changes to `glab` nor the model itself,
although usage will need to be documented:

```shell
% ./cosign save docker.io/sroqueworcel/test-slsa-sbom@sha256:127dfd758f2cf2ebab7ba7766bbf605694d7cc8ba68bc4f73386ac94e56d7eef --dir /tmp/dir
% glab verify attestation project/project /tmp/dir/blobs/sha256/127dfd758f2cf2ebab7ba7766bbf605694d7cc8ba68bc4f73386ac94e56d7eef
```

Verification using `cosign` requires the following code:

```shell
% ./cosign save docker.io/sroqueworcel/test-slsa-sbom@sha256:127dfd758f2cf2ebab7ba7766bbf605694d7cc8ba68bc4f73386ac94e56d7eef --dir /tmp/dir
% ./cosign verify-blob --new-bundle-format --bundle container.bundle --certificate-identity sroque-worcel@gitlab.com --certificate-oidc-issuer https://accounts.google.com /tmp/dir/blobs/sha256/127dfd758f2cf2ebab7ba7766bbf605694d7cc8ba68bc4f73386ac94e56d7eef
Verified OK
```

Alternatively, to avoid having to download the file to disk, the following
technique can be used:

```shell
./cosign verify-blob-attestation --bundle container.bundle --type=slsaprovenance1 --digest="127dfd758f2cf2ebab7ba7766bbf605694d7cc8ba68bc4f73386ac94e56d7eef" --digestAlg="sha256" --certificate-identity=sroque-worcel@gitlab.com --certificate-oidc-issuer=https://accounts.google.com
Verified OK
```

The above functionality requires v3.0.3 or later if using `attest`. This is
because support for the `--bundle` flag was introduced there. The
[changelog](https://github.com/sigstore/cosign/blob/main/CHANGELOG.md#v303) for
that release has the following note:

> Thank you for all of your feedback on Cosign v3! v3.0.3 fixes a number of bugs reported by
> the community along with adding compatibility for the new bundle format and attestation
> storage in OCI to additional commands. We're continuing to work on compatibility with
> the remaining commands and will have a new release shortly. If you run into any problems,
> please [file an issue](https://github.com/sigstore/cosign/issues)

It is possible we will be able to pass the `--bundle` flag directly to the
`verify-attestation` command in the near future, so users of `cosign` will be
able to verify attestations without downloading the image to disk as in the
example above.

### Decision

The pipeline security team decided to perform attestations without connecting to
the registry as discussed in [Attestation without connection to the registry](#attestation-without-connection-to-the-registry).

#### Positive Consequences

- Implementation time is significantly reduced.
- Reduced dependency on external infrastructure.
- We are not required to test authentication against all OCR Registries, and all
  authentication mechanisms.
- We do not have to address security concerns around handling credentials.
- The UX is simpler because we are not required to deal with registry usernames, passwords or
  tokens.

#### Negative Consequences

- This approach does not allow the uploading of attestations to the OCI
  Registries. Because we plan to store attestations in the `Ci::Slsa::Attestation`
  model, this does not affect us. In the future, if required, we can upload the attestations
  using `cosign attach attestation`.

## Key decision: New worker versus modifying the existing worker

In the [Cosign artifact signing with SBOM and Provenance in OCI registries](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/219448)
proof of concept, code that integrates with `cosign` to perform the attestation
of OCI images is written directly in `Ci::Slsa::PublishProvenanceService`. A new
worker and service was another option, but the modification of the current
worker was chosen because of these reasons:

- Ease of implementation: All the mechanisms required to execute `cosign` and
  retrieve the output already exist within this file.
- Simplicity of architecture: by avoiding having to create an additional worker,
  along with an associated service, we can have a simpler design.
- The naming of the worker is appropriate for attestation of multiple types of
  objects.
- There is a strong overlap between the two types of attestations. For example,
  both attestation types generate a bundle file and require the creation of an
  attestation based on that bundle.
- If a single build produces attestations of both OCI Containers and
  attestations, this simplifies the reporting of errors to the user as this is
  compatible with the approach taken in [Add SLSA Attestation Generation Error to CI build UI (#570341)](https://gitlab.com/gitlab-org/gitlab/-/work_items/570341).

This design has drawbacks: for each single build, the attestation of artifacts
and containers happens in the same background job. This lack of isolation could mean
that if attestation of OCI images fails, this failure can spread to the
attestation of artifacts within that same build. Other builds would not be
affected.

This can be mitigated by appropriate [exception handling](https://docs.gitlab.com/development/logging/#exception-handling)
 and [proper abstractions](https://docs.gitlab.com/development/reusing_abstractions/).

Proper abstractions and reusable code will be produced for the final
implementation.

## Testing, Performance and Availability

To ensure high-levels of availability and robust performance, several
measures will be put in place by the SSCS stage's Pipeline Security team:

- Manual testing of the attestation process in production behind a feature flag,
  to prevent accidental impacts to other GitLab users.
- End-to-end tests for our `cosign` integration will provide assurance
  for our `cosign` upgrade.
- We will implement extensive test coverage, particularly for scenarios
  involving registry failures, as well as Sigstore failures. This will ensure
  the appropriate measures are in place to allow for the graceful handling of
  outages.
- Sidekiq jobs will be configured to retry as appropriate to ensure that sporadic
  outages of Sigstore infrastructure or registries do not lead to persistent
  failures in SLSA attestation generation.
- This feature will only be enabled for users who specifically opt-in.
- We will use Sidekiq's default [retry configuration](https://docs.gitlab.com/development/sidekiq/#retries) for
  dealing with Sigstore transient failures. We have also flagged our worker as
  having [external dependencies](https://docs.gitlab.com/development/sidekiq/worker_attributes/#jobs-with-external-dependencies).
