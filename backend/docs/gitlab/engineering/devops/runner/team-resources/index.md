---
title: Runner Group - Team Resources
description: "The goal of this page is to document resources needed for day-to-day work within the Runner group."
---

## Overview

The goal of this page is to document resources needed for day-to-day work within the Runner group.

## Good bookmarks

- [Team Handbook](/handbook/engineering/devops/runner/)
- [Internal Engineering Handbook](https://internal.gitlab.com/handbook/engineering/)
- [Runner SaaS HQ issue](https://gitlab.com/groups/gitlab-org/-/epics/9969)
- [Public Runner Docs](https://docs.gitlab.com/runner/)
- [Public Development Docs](https://docs.gitlab.com/runner/development/)
- [Runner Runbooks](https://gitlab.com/gitlab-com/runbooks/-/tree/master/docs/ci-runners)
- [GitLab.com Triage](https://dashboards.gitlab.net/d/RZmbBr7mk/gitlab-triage?orgId=1) (for situational awareness)
- [Blueprints](https://gitlab.com/gitlab-org/gitlab/-/tree/master/doc/architecture/blueprints) (search for `runner`)

## Projects we maintain

As a team we maintain several projects. The <https://gitlab.com/gitlab-com/runner-maintainers> group
is added to each project with maintainer permission. We also try to align tools and versions used across them.

### Product projects

- [GitLab Runner](https://gitlab.com/gitlab-org/gitlab-runner)
- [GitLab Runner Operator for Kubernetes](https://gitlab.com/gitlab-org/gl-openshift/gitlab-runner-operator)
- [GitLab Runner Helm Chart](https://gitlab.com/gitlab-org/charts/gitlab-runner)
- [GitLab Runner UBI offline build](https://gitlab.com/gitlab-org/ci-cd/gitlab-runner-ubi-images)

### Runner component projects

- [Taskscaler](https://gitlab.com/gitlab-org/fleeting/taskscaler)
- [Fleeting](https://gitlab.com/gitlab-org/fleeting/fleeting)
- [Fleeting Plugin AWS](https://gitlab.com/gitlab-org/fleeting/fleeting-plugin-aws)
- [Fleeting Plugin Google Compute](https://gitlab.com/gitlab-org/fleeting/fleeting-plugin-googlecompute)
- [Fleeting Plugin Azure](https://gitlab.com/gitlab-org/fleeting/fleeting-plugin-azure)
- [Fleeting Plugin Static](https://gitlab.com/gitlab-org/fleeting/fleeting-plugin-static)
- [Nesting](https://gitlab.com/gitlab-org/fleeting/nesting)
- [Docker Machine (fork)](https://gitlab.com/gitlab-org/ci-cd/docker-machine)
- [Custom Executor Autoscaler](https://gitlab.com/gitlab-org/ci-cd/custom-executor-drivers/autoscaler)

### CI Steps projects

- [Step Runner](https://gitlab.com/gitlab-org/step-runner)
- [Action Runner](https://gitlab.com/gitlab-org/components/action-runner)

### Helper projects

- Linters
  - [Runner linters Docker images](https://gitlab.com/gitlab-org/ci-cd/runner-tools/runner-linters)
  - [goargs linter](https://gitlab.com/gitlab-org/language-tools/go/linters/goargs)
- Testing
  - [DinD image tests](https://gitlab.com/gitlab-org/ci-cd/tests/dind-image-tests)
  - [SaaS Runner Tests](https://gitlab.com/gitlab-org/ci-cd/tests/saas-runners-tests/)
- Release
  - [Process](https://gitlab.com/gitlab-org/ci-cd/runner-tools/releases)
  - [Release tool](https://gitlab.com/gitlab-org/ci-cd/runner-tools/releaser)
  - [GitLab Changelog](https://gitlab.com/gitlab-org/ci-cd/runner-tools/gitlab-changelog)
  - [Release index generator](https://gitlab.com/gitlab-org/ci-cd/runner-tools/release-index-generator)
- Maintenance
  - [Runner Pod Cleanup](https://gitlab.com/gitlab-org/ci-cd/gitlab-runner-pod-cleanup)

### GitLab projects that rely on Runner public-facing APIs

The following projects depend on the public Runner APIs, and should be taken
into consideration in the scope of any changes/deprecations to the public API surface:

| Project | API |
|---------| --- |
| [GitLab Terraform Provider](https://gitlab.com/gitlab-org/terraform-provider-gitlab) | [REST API](https://docs.gitlab.com/ee/api/api_resources.html) |
| [GitLab CLI](https://gitlab.com/gitlab-org/cli) | [REST API](https://docs.gitlab.com/ee/api/api_resources.html) |

## Security Process (Managing CVE vulnerability report issues)

Managing CVE vulnerability issues is part of GitLab's vulnerability management effort
([1](https://internal.gitlab.com/handbook/security/product_security/vulnerability_management/),
[2](/handbook/security/product-security/vulnerability-management/)), and is an important part of maintaining the
GitLab FedRAMP certification.

Using the [`container-scanners`](https://gitlab.com/gitlab-com/gl-security/appsec/container-scanners) project, GitLab
scans all images we produce to highlight CVE vulnerabilities. From those scans, the
[`vulnmapper`](https://gitlab.com/gitlab-com/gl-security/product-security/vulnerability-management/vulnerability-management-internal/vulnmapper)
project creates issues in the project that created the vulnerable image, including
[SLAs](/handbook/security/product-security/vulnerability-management/sla/) to which we must adhere.
The Runner team member assigned the `Support & Security Responder` role in the weekly team task should triage and
review the list of CVEs and address any issues as appropriate:

- `Critical` severity issues should be addressed immediately.
- `High`, `Medium`, and `Low` severity issues should be addressed in the priority order of the
  [remediation SLAs](/handbook/security/product-security/vulnerability-management/sla/).

The procedure for addressing CVE issues is as follows:

### Surfacing active vulnerability reports

- Use one of the following to surface active CVE issues assigned to our team:
  - [Issue search](https://gitlab.com/groups/gitlab-org/-/issues/?sort=created_date&state=opened&label_name%5B%5D=FedRAMP%3A%3AVulnerability&label_name%5B%5D=group%3A%3Arunner&not%5Blabel_name%5D%5B%5D=FedRAMP%3A%3ADR%20Status%3A%3AAccepted&not%5Blabel_name%5D%5B%5D=FedRAMP%3A%3ADR%20Status%3A%3AOpen&first_page_size=50)
  - The [`gitlab-dashboard cves`](https://gitlab.com/avonbertoldi/gitlab-dashboard) command.
  - The [`cver imageVulns`](https://gitlab.com/gitlab-org/ci-cd/runner-tools/cver.git) command.

     Many issues will reference the same CVE vulnerability report; it's best to group issues for the same vulnerability
    report and address them together.
- Focusing on CVE reports in priority order, start with `critical`, `high`, and `medium` severities first and proceed as
follows:
  1. For each group of common/related issues, confirm that the associated CVE is still valid. This can be done by
     scanning the `latest` version of the image(s) identified in the issue(s) with tools such as
     [`trivy`](https://trivy.dev/) and [`grype`](https://github.com/anchore/grype), and checking whether the CVE
     referenced in the issue appears in the `trivy` or `grype` scan.
  1. If the vulnerability is no longer reported in the `trivy` or `grype` scan of the relevant image(s), the issue(s)
     can be closed. Note that the `cver` internal tool mentioned above largely automates this task, including closing
     the relevant issues (see the documentation).
  1. If the vulnerability is **still** present in the relevant image(s), it must be addressed.

Note that issues that reference `ubi-fips` flavors of `gitlab-runner` or `gitlab-runner-helper` images take precedence
over other image flavors (like `alpine` or `ubuntu`) since the GitLab FedRAMP certification is contingent on `ubi-fips`
images only.

### Addressing active vulnerability reports

Vulnerabilities usually appear in one of three flavors (ordered in most to least frequency of occurrence):

- The vulnerability exists in a third-party OS package (like `git` or `git-lfs`).
- The vulnerability exists in `gitlab-runner` in one of its dependencies.
- The vulnerability exists in `gitlab-runner` in code we've written.

#### Third-party OS packages

In this case, the vulnerability:

- Has not been fixed upstream
- Has been fixed upstream but an OS package including the fix has not been created and published yet
- Will not be fixed upstream

The primary course of action here is to create a
[`deviation request issue`](https://gitlab.com/gitlab-com/gl-security/security-assurance/team-security-dedicated-compliance/poam-deviation-requests/-/issues)
(see
<https://handbook.gitlab.com/handbook/security/security-assurance/security-compliance/poam-deviation-request-procedure/>).
We generally create one deviation request issue per offending software module (e.g. `git-lfs` or `libcurl`). When
creating the issue, be sure to select `operational_requirement_template` as a template and complete the following
sections:

- Affected images
- Vulnerability details (one row for each relevant CVE report)
- Relevant `vulnmapper` issues
- Justification

Once the deviation request issue is created, add:

- A note to all the relevant `gitlab-runner` issues pointing to the deviation request issue
- The label `FedRAMP::DR Status::Open`
- The [most relevant label](/handbook/security/product-security/vulnerability-management/labels/) from this list:

  - `Vulnerability::Vendor Base Container::Fix Unavailable`
  - `Vulnerability::Vendor Base Container::Will Not Be Fixed`
  - `Vulnerability::Vendor Package::Fix Unavailable`
  - `Vulnerability::Vendor Package::Will Not Be Fixed`

Eventually, a fix in the offending package will make its way to the OS package manager, and then both the
`gitlab-runner` and deviation request issues can be closed.

#### `gitlab-runner` dependencies

The simplest course of action here is to update the dependency to the latest compatible version (or at least a version
that addresses the vulnerability). Once the MR with the dependency update is merged, the `gitlab-runner` issue can be
closed.

If the dependency does not address the vulnerability, possible courses of action are:

- If a fork of the dependency that addresses the vulnerability exists, use it with the Go module `replace` directive. In
this case, be sure to create a task to switch back to the upstream dependency when the vulnerability has been addressed
there.
- If possible, consider not using the dependency or replacing it with another similar dependency.
- Create a [deviation request issue](#third-party-os-packages).

#### `gitlab-runner` source

The only course of action here is to fix the vulnerable code. If the fix is not simple and will take time to implement
(and prevent us from meeting CVE SLAs), it might be necessary to create a [deviation request issue](#third-party-os-packages).

### Working with security forks

When issues are marked confidential, the MR that fixes the issue should be made in a project's security fork (see
[security-forks](https://gitlab.com/gitlab-org/security?filter=gitlab%20runner)). In general the process is identical to
crating and merging MRs in the canonical project repo, with a couple of notable differences.

Note that MRs in the security repo _must_ be reviewed/approved by a security counterpart in addition to a runner
code-owner.

The examples below are given for the [GitLab Runner](https://gitlab.com/gitlab-org/gitlab-runner) project, but apply
equally to all [runner-related projects with security forks](https://gitlab.com/gitlab-org/security?filter=gitlab%20runner).

#### Keeping the security fork up to date with its canonical repo

Security forks are configured to automatically synchronize with the canonical repo, but this can be disabled if changes
exist in the security fork's `main` branch that do not exists in the canonical repo's `main` branch. This usually
happens when a security MR is merged into the security fork's `main`, but not into the canonical repo's `main` branch.
In this event, it is necessary to manually synchronize the security fork against the canonical repo.

From a checked-out canonical repo:

```shell
git fetch # ensure you have the latest changes from the canonical repo.
git remote add security git@gitlab.com:gitlab-org/security/gitlab-runner.git # add the security repo as a remote, be sure to use the git url.
git fetch security # fetch the security fork repo references.
git checkout -b security-main security/main # checkout the security fork's main branch.
git rebase --rebase-merges origin/main # rebase the canoncial main onto the security main.
git log --color --topo-order --oneline # ensure the resulting history is sane.
git push --force # push the resulting local security main branch to the security remote repo.
```

Notes:

1. These steps will not fully synchronize the security and canonical repositories in both directions. They will only
   bring changes that are only the canonical repo, into the security repo. Synchronizing in the other direction is
   described below.
2. The security repos do/should not have force-push branch protection on the `main` branch, but if the one you are
   working with does, temporarily disable it so you can perform the last step.
3. If the security fork `main` branch becomes too out of date with the canonical repo `main` branch (specifically with
   changes that exist only in the security repo), merge conflicts are likely to occur when rebasing the canonical repo
   atop the security fork. You will have to resolve these.

#### Merging security MRs back into the canonical repo

When MRs created in the security repo are merged (into the security repo's `main` branch), the security and canonical
repo will become unsynchronized. Merging MRs from the security fork back into the canonical repo is a manual process.
Each MR in the security repo that a developer wants to incorporate into the canonical repo must be be done manually via
a new MR in the canonical repo. This procedure is manual so developers can control when these merges are done.

To merge an MR already merged in the security fork `main` branch into the canonical repo, follow these steps:

From a checked-out canonical repo:

```shell
git fetch # ensure you have the latest changes from the canonical repo.
git remote add security git@gitlab.com:gitlab-org/security/gitlab-runner.git # add the security repo as a remote, be sure to use the git url.
git fetch security # fetch the security fork repo references.
git checkout -b name-of-working-branch origin/main # create a new branch into which you'll cherry-pick commits from the security repo.
git cherry-pick sha-of-commit-in-security-repo # cherry-pick all commits from the relevant MR from the security repo into your branch in the canonical repo.
```

Repeat the final step for all commits in the relevant MR, in topographical order, _excluding the merge commit_. Do not
include the MR's merge commit in the cherry-picked commits.

Finally, create an MR in the canonical repo from this branch as usual.

Notes:

1. If the security fork becomes too out of date with the canonical repo, merge conflicts are likely when
cherry-picking the commits. You will have to resolve them.
2. You should manually synchronize the security repo as described above immediate after the MR is merged into the
   canonical main.
3. It is not the aim of these instruction to completely synchronize the security and canonical repos in both directions.
   Full synchronization will occur as a byproduct of merging all MRs from the security repo into the canonical repo. It
   is up to the developers' discretion when this happens for each MR.

## Metrics and logs

- Dashboards
  - [Runner Service Overview](https://dashboards.gitlab.net/d/ci-runners-main/ci-runners-overview?orgId=1)
  - Additional dashboards can be found in the dropdowns along the top bar:

![runner-dashboards](/images/engineering/devops/ops/verify/runner/team-resources/runner-dashboards.png)

- Metrics
  - [Runner Metrics](https://docs.gitlab.com/runner/monitoring/index.html)
- Logs
  - [Runner Logs](https://log.gprd.gitlab.net/goto/3d8891e0-2035-11ee-8afc-c9851e4645c0) (filter by shard)
  - You can find a list of shards in the dropdown along the top baf of any service dashboard:

![runner-shards](/images/engineering/devops/ops/verify/runner/team-resources/runner-shards.png)

## Internal tools

### Merge Request Bot

For
[`gitlab-org/gitlab-runner`](https://gitlab.com/gitlab-org/gitlab-runner)
we have the [Merge Request Bot](https://gitlab.com/merge-request-bot/merge-request-bot) enabled which posts
[comments for community contributions](https://gitlab.com/gitlab-org/gitlab-runner/-/merge_requests/2407#note_411098266).
This is configured via [Merge Request webhook events](https://gitlab.com/gitlab-org/gitlab-runner/hooks).

- [Appliation code](https://gitlab.com/merge-request-bot/merge-request-bot)
- [CloudRun deployment](https://gitlab.com/gitlab-org/ci-cd/merge-request-bot/infrastructure/cloud-run)
- [Logs](https://console.cloud.google.com/logs/query;query=resource.type%20%3D%20%22cloud_run_revision%22%0Aresource.labels.service_name%20%3D%20%22merge-request-bot%22%0Aresource.labels.location%20%3D%20%22europe-west4%22%0A%20severity%3E%3DDEFAULT%0Atimestamp%3E%3D%222020-09-11T10:25:17.532Z%22%20timestamp%3C%3D%222020-09-11T11:25:17.532Z%22;timeRange=PT1H;summaryFields=:true:32:beginning?customFacets=&scrollTimestamp=2020-09-11T11:25:01.157050000Z&project=group-verify-df9383)

## Developing / Testing for Windows

Our [development docs for Windows](https://docs.gitlab.com/runner/development/#developing-for-windows-on-a-non-windows-environment) suggest using Vagrant and Virtualbox.
But the easiest way to get started is just to create a Google Compute Engine Windows instance and RDP into it.
Create an instance from [this magical image](https://console.cloud.google.com/compute/imagesDetail/projects/group-verify-df9383/global/images/runners-windows-2004-core-containers-beta?project=group-verify-df9383).

### Supported versions

We support some pretty old versions of Windows because they are [LTSC](https://learn.microsoft.com/en-us/lifecycle/products/windows-10-enterprise-ltsc-2019)

## Third-party infrastructure

### Testing on IBM Z/OS

To facilitate testing the `s390x` architecture artifacts,
a Z/OS VM is available to GitLab team members.

#### Logging in

1. In [1Password](/handbook/security/corporate/systems/1password/),
   under the `Verify` vault, download the `zOS login - gitlabkey02.pem` file.
1. From the `zOS login` entry in the same vault, take note of the `user` and `address` fields.
1. SSH into the Z/OS VM:

    ```shell
    ssh -i "zOS login - gitlabkey02.pem" <user>@<address>
    ```

   Note: You'll be requested the password to unlock the .pem file. Enter the password attached
   to the `zOS login - gitlabkey02.pem` entry.

#### Testing helper image

Assuming you want to test a `prebuilt-s390x.tar.xz` image produced by a CI/CD pipeline,
and already have the .pem file from the [previous point](#logging-in),
the steps would be the following:

1. Copy the `prebuilt-s390x.tar.xz` file to the Z/OS VM:

    ```shell script
    scp -i "zOS login - gitlabkey02.pem" prebuilt-s390x.tar.xz <user>@<address>:/home/ubuntu/
    ```

   Note: You'll be requested the password to unlock the .pem file. Enter the password attached
   to the `zOS login - gitlabkey02.pem` entry.

1. SSH into the VM:

    ```shell
    ssh -i "zOS login - gitlabkey02.pem" <user>@<address>
    ```

1. Import the image and run it:

    ```shell
    sudo docker import ./prebuilt-s390x.tar.xz gitlab/gitlab-runner-helper:s390x-dev
    sudo docker run -it gitlab/gitlab-runner-helper:s390x-dev bash
    gitlab-runner-helper help
    ```

## Accessing Mac Runner AWS environments

GitLab SaaS Mac Runners are running on AWS.
We have production, staging, team sandbox and individual sandbox environments.
An individual sandbox can be created via [Hackystack(https://gitlabsandbox.cloud/cloud)].
Be sure to keep an eye on unused resources to reduce cost -- [oh-my-cost](https://gitlab.com/josephburnett/oh-my-cost) can help.
We also have a [team sandbox](https://gitlabsandbox.cloud/cloud/accounts/5442c67c-1673-4351-b85d-e366c328bfea) in Hackystack which is used to host our Mac Job Image builder instance.
Access to the team sandbox can be acquired via access request.
Within the team sandbox is also a role which has access to the staging and production Mac environments.

### Access Mac Runner Staging

From the team sandbox, activate a role named `eng_dev_verify_runner` with the account ID `251165465090` (staging).

### Access Mac Runner Production

From the team sandbox, activate a role named `eng_dev_verify_runner` with the account ID `215928322474` (production).

## Load Testing

The group [`gitlab-runner-stress`](https://gitlab.com/gitlab-org/ci-cd/gitlab-runner-stress) has a suite of tools for stress testing a GitLab and Runner instance.
Our canonical benchmark for Mac Runners is [`XcodeBenchmark`](https://gitlab.com/gitlab-org/ci-cd/tests/saas-runners-tests/macos-platform/XcodeBenchmark) (our fork).

## Runner Vending Machine (AWS Cloud Formation Templates)

The Partner Solution group maintains a curated collection of AWS Cloud Formation Templates for deploying Runner in AWS called the ["Runner Vending Machine"](https://gitlab.com/guided-explorations/aws/gitlab-runner-autoscaling-aws-asg#easy-buttons-provided).
We should keep them in the loop as we change how Runner works and is deployed so these templates can stay up-to-date.
Our point-of-contact is [DarwinJS](https://gitlab.slack.com/team/UPCBGABMK).

## Secrets

How we manage secrets for Runner and how they get into the right place is a whole thing.
This needs documenting: https://gitlab.com/gitlab-org/gitlab-runner/-/issues/29823.
