---
title: Self-Service Performance Regression Testing
description: Guide for teams to independently run performance Regression testing
---

## Overview

This guide enables teams to run their own performance regression testing without requiring Performance Enablement support.

**Who this is for**: Teams who understand what they're testing and why. If you need help defining what to test or how to test a new component, [escalate to Performance Enablement](#when-to-escalate-to-performance-enablement).

## When to Use This Guide

- Validating performance under load
- Testing major upgrades for performance regressions
- Validating performance before infrastructure changes

## Prerequisites

Before starting, ensure you have:

### Skills & Experience

- **GitLab Environment Toolkit (GET) experience**: You've deployed a reference architecture before (or can follow the [GET quick start guide](https://gitlab.com/gitlab-org/gitlab-environment-toolkit/-/blob/main/docs/environment_quick_start_guide.md))
- **GitLab Performance Tool (GPT) experience**: You've loaded data and run tests with [GPT](https://gitlab.com/gitlab-org/quality/performance) (or can follow the [GPT quick start guide](https://gitlab.com/gitlab-org/quality/performance/-/blob/main/docs/quick_start.md))
- **Understanding of the change**: You know what component/feature is being tested and why

### Access & Accounts

- Access to [GitLab Sandbox](https://gitlabsandbox.cloud/login)
- GitLab.com account with pull access to pre-release images
- GCP/AWS accounts within your sandbox account

**Not sure if you're ready?** Review the GET and GPT quick start guides above. If you get stuck, [escalate to Performance Enablement](#when-to-escalate-to-performance-enablement).

## Using This Guide in Your Work

This handbook page provides comprehensive implementation guidance for running performance regression tests and is paired with an issue template on the GitLab project to track progress. Create a work item using the `Performance Regression Testing` template to begin.

### How to use these resources together

1. **Read this handbook page** to understand the full process and implementation details
2. **Open a work item using the template** to track your progress and results
3. **Reference specific sections** of this handbook as you work through each phase
4. **Document your findings** in the work item for team visibility

The handbook page provides the "why" and "how" details, while the work item template provides the "what" checklist for tracking progress.

## Quick Decision Tree

```mermaid
flowchart TD
    %% Nodes
    Prep["Preparation<br/>(Define Test & Success, Choose Architecture)"]
    Setup["Setup<br/>(GCP, GET Config, Deploy, Seed Data)"]
    baseline{Need<br>Baseline?}
    Baseline_test["Baseline Test<br/>(Run Test, Capture Results)"]
    Upgrade_test["Upgrade & Test<br/>(Upgrade, Run Test, Capture Results)"]
    Analysis["Analysis<br/>(Compare, Document, Cleanup)"]
    Upgrade{Need<br>Upgrade Test?}

    %% Map
    Prep --> Setup
    Setup --> baseline
    baseline -->|No| Upgrade_test
    baseline -->|Yes| Baseline_test
    Baseline_test --> Upgrade
    Upgrade -- Yes -->Upgrade_test
    Upgrade -- No --> Analysis
    Upgrade_test --> Analysis

    %% Links
    click Prep "#test-preparation"
    click Setup "#setup-testing"
    click Baseline_test "#baseline-testing"
    click Upgrade_test "#upgrade-testing"
    click Analysis "#analysis-and-results"
```

---

## Test Preparation

```mermaid
flowchart LR
    %% Nodes
    Testing["Define<br/>Testing"]
    Success["Identify<br/>Success Criteria"]
    Architecture["Choose<br/>Reference Architecture"]
    Test_image["Identify<br/>Test Image"]

    %% Map
    Testing --> Success
    Success --> Architecture
    Architecture --> Test_image


    %% Links
    click Testing "#define-testing"
    click Success "#identify-success-criteria"
    click Architecture "#choose-reference-architecture"
    click Test_image "#identify-test-image"
```

### Define Testing

Your team should already understand what you're testing and why.

- What component/feature is being tested
- What change is being evaluated (e.g., Rails 7.2 upgrade, Ruby 3.3 upgrade)
- Expected impact areas (API response times, throughput, memory usage)

This step will also include deciding which tests to run and which test tooling you need to run it. This example will work off of GPT being selected as as the test tool, but we have others that can be used:

- [K6](https://grafana.com/docs/k6/latest/)
- [Component Performance Testing (CPT)](https://gitlab.com/gitlab-org/quality/component-performance-testing)
- [GitLab Browser Performance Tool (GBPT)](https://gitlab.com/gitlab-org/quality/performance-sitespeed)

If GPT is selected as the test tool, there are a [number of tests](https://gitlab.com/gitlab-org/quality/performance/-/tree/main/k6/tests) and [load levels pre defined](https://gitlab.com/gitlab-org/quality/performance/-/tree/main/k6/config/options) that are defined that can be used.

If there is not an existing test/load level that covers the desired use case, new will need to be written.

**If you're unsure about scope**: [Escalate to Performance Enablement](#when-to-escalate-to-performance-enablement) before proceeding.

### Identify Success Criteria

When running a performance test, having an idea of what success looks like can make the difference between a success or failure or wasted effort. You should know what success looks like to guide your efforts, or you may not capture the information you need and have to re-run the testing. Some examples:

- Performance didn't degrade from the baseline
- Performance has improved by X % in comparison to the baseline
- We've identified the performance characteristics of a new feature under load
- We've established a baseline for this component

Part of determining the Success Criteria is determining what the baseline to compare against is. Is there an existing baseline that can be used or should you run a new baseline? Or is establishing a baseline the goal?

Baseline testing produces valuable performance metrics even without subsequent comparison testing. Teams documenting current system performance can stop after running the baseline if that meets the goal.

#### Important Considerations

- **Avoid tunnel vision**: Going into a performance test with too specific of a goal (`We expect a 2 ms response on the Workhorse P95 metric`) can lead to tunnel vision, being too focused on a specific result that you miss interesting results that are not in that scope.
- **Test results don't translate 1:1 to production**: Performance improvements in test environments don't translate 1:1 to production. A 3 ms improvement in your test environment may result in an improvement in production, but the magnitude will differ due to different hardware, load patterns, and data characteristics.

### Choose Reference Architecture

**Use the [X Large](https://docs.gitlab.com/administration/reference_architectures/10k_users/) Reference Architecture** unless you have a specific reason not to.

The X Large architecture provides:

- Good balance between complexity and resource usage
- Well-documented and tested
- Aligns with published performance benchmarks
- Sufficient HA components for realistic testing

**Use a different size only if:**

- **Testing a component that doesn't need HA** → Use [Medium](https://docs.gitlab.com/administration/reference_architectures/3k_users/)
- **Testing at production scale** → Use [2X Large](https://docs.gitlab.com/administration/reference_architectures/25k_users/) or larger
- **Testing Geo or specialized features** → Use appropriate architecture (e.g., Geo setup)

You can use the [Reference Architecture sizing guide](https://docs.gitlab.com/administration/reference_architectures/#deciding-which-architecture-to-start-with) to help deciding on architecture. There are a couple of repos containing sample GET configs:

- [GitLab Environment Toolkit Example Configs](https://gitlab.com/gitlab-org/gitlab-environment-toolkit/-/tree/main/examples)
- [Reference Architecture Test Environments](https://gitlab.com/gitlab-com/gl-infra/software-delivery/operate/get-environments/ra-test-environments)
- [GitLab Environment Toolkit Test Configs](https://gitlab.com/gitlab-org/quality/gitlab-environment-toolkit-configs)

**If unsure which to use**: Default to [X Large](https://docs.gitlab.com/administration/reference_architectures/10k_users/) Reference Architecture.

### Identify Test Image

**Coordinate with the team responsible for the change to obtain the image URL.** They'll have the most up-to-date version and can provide any special considerations.

**Common image sources** (for reference):

- **MR/Branch artifacts**: Job artifact on the [MR/Branch pipeline](#capturing-an-image-url-from-a-branch-pipeline)
- **Nightly builds**: [packages.gitlab.com/gitlab](https://packages.gitlab.com/gitlab)
- **Master pipeline**: [Latest successful builds](#capturing-an-image-url-from-a-master-pipeline)
- **Release candidates**: Pre-release builds for upcoming versions
- **Docker Hub**: [hub.docker.com/u/gitlab](https://hub.docker.com/u/gitlab)
- **Dev Builds**: [dev.gitlab.org/gitlab/omnibus-gitlab](https://dev.gitlab.org/gitlab/omnibus-gitlab/-/pipelines)

**Note:** ⚠️ If you're getting the Image URL from the web pipeline artifacts page, you need to convert the URL to use the API endpoint

see [Capturing an image URL from a Master pipeline](#capturing-an-image-url-from-a-master-pipeline) or [Capturing an image URL from a branch pipeline](#capturing-an-image-url-from-a-branch-pipeline) to capture the URL and then convert:

```plaintext
https://<URL>/gitlab/omnibus-gitlab/-/jobs/<JOB_ID>/artifacts/file/pkg/ubuntu-noble/...
```

to

```plaintext
https://<URL>/api/v4/projects/gitlab%2Fomnibus-gitlab/jobs/<JOB_ID>/artifacts/pkg/ubuntu-noble/...
```

You can test the URL with

```bash
curl -H "PRIVATE-TOKEN: $PRIVATE_PROD_TOKEN" \
  "https://dev.gitlab.org/api/v4/projects/gitlab%2Fomnibus-gitlab/jobs/<JOB_ID>/artifacts/pkg/ubuntu-noble/<PACKAGE_NAME>.deb" \
  -o /tmp/test.deb -L

file /tmp/test.deb  # Should show "Debian binary package"
```

more information in the [Prepare GET Configuration](#prepare-get-configuration) section

**Document:**

- Image URL
- Version/commit information
- Source of the Image (in case you need to get a refreshed Image)

---

## Setup Testing

```mermaid
flowchart LR
    %% Nodes
    GET_prep["Prepare<br/>GET Configuration"]
    Test_infra["Setup<br/>Test Infrastructure"]

    %% Map
    GET_prep --> Test_infra

    %% Links
    click GET_prep "#prepare-get-configuration"
    click Test_infra "#setup-test-infrastructure"
 ```

### Prepare GET Configuration

Most of the effort is covered in the [GET Quick Start Guide](https://gitlab.com/gitlab-org/gitlab-environment-toolkit/-/blob/main/docs/environment_quick_start_guide.md)

The important section is [configuring `gitlab_deb_download_url`](https://gitlab.com/gitlab-org/gitlab-environment-toolkit/-/blob/main/docs/environment_configure.md#direct)

To have GET use the [test image](#identify-test-image) identified earlier add to the ansible `vars.yml` file:

```yaml
all:
  vars:
  # ... existing vars ...

    gitlab_repo_script_url: "https://packages.gitlab.com/install/repositories/gitlab/nightly-builds/script.deb.sh"
    gitlab_deb_download_url: "{{ lookup('env','GITLAB_UBUNTU_IMAGE') | default('https://gitlab.com/api/v4/projects/14588374/jobs/11423868576/artifacts/pkg/ubuntu-jammy/gitlab.deb', true)}}" # update to use latest image url
    gitlab_deb_download_url_headers: {
        "PRIVATE-TOKEN": "{{ lookup('env','PRIVATE_PROD_TOKEN')}}",
    } # use your .com token

  # ... rest of file ...
```

**For GPT testing**: Ensure required rate limits are disabled, custom Post Configure task is used https://gitlab.com/gitlab-org/quality/gitlab-environment-toolkit-configs/quality/-/blob/main/custom_task_files/gitlab_tasks/post_configure.yml#L1-35 applies it automatically (see https://gitlab.com/gitlab-org/quality/gitlab-environment-toolkit-configs/quality/-/blob/main/configs/reference_architectures/10k/ansible/inventory/vars.yml#L24-40 and https://gitlab.com/gitlab-org/gitlab-environment-toolkit/-/blob/main/docs/environment_advanced.md#custom-tasks).

### Setup Test Infrastructure

It is recommended that you run the load test from as close to the test environment as possible (unless you are looking to test including internet latency, which is normally outside our testing scope).

On GCP, a `n2-standard-2` vm will be sufficient. If you are intending on running a large number of tests, increasing the boot disk size to 100 GB would help prevent running out of disk space mid run.

- [Install GPT](https://gitlab.com/gitlab-org/quality/performance/-/blob/main/docs/quick_start.md) on the VM.
- Setup an [environment config file](https://gitlab.com/gitlab-org/quality/performance/-/blob/main/docs/quick_start.md#2-prepare-environment-config) for your environment

---

## Baseline Testing

```mermaid
flowchart LR
    %% Nodes
    deploy["Deploy<br/>Base Environment"]
    baseline_test["Run<br/>Baseline Test"]

    %% Map
    deploy --> baseline_test

    %% Links
    click deploy "#deploy-base-environment"
    click baseline_test "#run-baseline-test"
 ```

### Deploy Base Environment

This links into [Standing up the Environment](#stand-up-environment). To run a baseline, stand up a version of the environment that reflects how the system functions today, so you will probably want to comment out the `gitlab_deb_download_url` parameter.

### Run Baseline Test

Execute GPT against your baseline environment to establish comparison metrics.

**Steps**: [See "Running a Performance Test"](#running-a-performance-test)

**Specific to baseline**:

- Use a current image (not the custom image you [identified as your test image](#identify-test-image))
- Document baseline results with timestamp
- Save results as `baseline_metrics_YYYY-MM-DD.csv`

**Note**: If generating a baseline / documenting current system performance is the goal, the team can stop testing and go direct to [documenting the findings](#document-findings)

---

## Upgrade Testing

```mermaid
flowchart LR
    %% Nodes
    deploy["Upgrade<br/>Environment"]
    test["Run<br/>Post-Upgrade Test"]

    %% Map
    deploy --> test

    %% Links
    click deploy "#upgrade-environment"
    click test "#run-post-upgrade-test"
 ```

### Upgrade Environment

Re-run the [Standing up the Environment](#stand-up-environment) steps with the changes necessary to implement the change under test. Make sure you uncomment the `gitlab_deb_download_url` parameter if you commented it out for a baseline test.

### Run Post-Upgrade Test

Execute the same test configuration against your upgraded environment.

**Steps**: [See "Running a Performance Test"](#running-a-performance-test)

**Specific to post-upgrade**:

- Use upgraded image
- Document post-upgrade results with timestamp
- Save results as `post-upgrade_metrics_YYYY-MM-DD.csv`

## Analysis and Results

```mermaid
flowchart LR
    %% Nodes
    compare["Compare Results"]
    document["Document Findings"]
    cleanup["Cleanup"]

    %% Map
    compare --> document
    document --> cleanup

    %% Links
    click compare "#compare-results"
    click document "#document-findings"
    click cleanup "#cleanup"
 ```

### Compare Results

- Compare baseline vs post-upgrade metrics
- Check against [published thresholds](https://gitlab.com/gitlab-org/reference-architectures/-/wikis/Benchmarks/Latest/10k)
- Identify any performance regressions
- Review [Interpreting the Results](#interpreting-the-results)

### Document Findings

- Create summary of results
- Note any concerning metrics
- Determine if escalation needed

### Cleanup

- Destroy test environments and VMs
  - For GET (GitLab Environment Toolkit)

    ```bash
    # Ansible tear down
    cd <ansible_folder_in_GET>
    . ./get-python-env/bin/activate
    ansible-playbook -i environments/<ENV_NAME>/inventory playbooks/uninstall.yml

    # Terraform tear down
    cd <terraform_folder_in_GET>
    terraform destroy
    ```

  - You can leave the test load generator in an off state instead of destroying if you expect to run load tests in the future.
- Verify GCP resources are cleaned up
  - Pay particular to resources you created outside of GET, i.e. External IP addresses

---

## Stand up Environment

### Deploy Environment

[Provisioning the environment with GET](https://gitlab.com/gitlab-org/gitlab-environment-toolkit/-/blob/main/docs/environment_quick_start_guide.md#2c-provision) is as simple as

```bash
cd <terraform_folder_in_GET>
terraform apply
```

Once Terraform is complete, run the [Ansible configure](https://gitlab.com/gitlab-org/gitlab-environment-toolkit/-/blob/main/docs/environment_quick_start_guide.md#3c-configure)

```bash
cd <ansible_folder_in_GET>
. ./get-python-env/bin/activate
ansible-playbook -i environments/<ENV_NAME>/inventory playbooks/all.yml
```

**NOTE**: both of these tasks will take a while to run so be prepared for a wait

---

### Seed with Performance Data

To seed the environment, use the load generator you setup in [Setup Test Infrastructure](#setup-test-infrastructure). Ensure your [environment config file](https://gitlab.com/gitlab-org/quality/performance/-/blob/main/docs/quick_start.md#2-prepare-environment-config) is setup for the environment you are loading data into. More details can be found in the [GPT environment prep documentation](https://gitlab.com/gitlab-org/quality/performance/-/blob/main/docs/environment_prep.md). You will probably want to seed with horizontal and vertical data.

The test generator built into GPT [runs as](https://gitlab.com/gitlab-org/quality/performance/-/blob/main/docs/quick_start.md#3-generate-test-data):

```bash
docker run -it \
  -e ACCESS_TOKEN=your-access-token \
  -v $(pwd)/config:/config \
  -v $(pwd)/results:/results \
  gitlab/gpt-data-generator --environment my-env.json -u
```

the `-u` flag enables it to run unattended.

If you want to load a different repo, you can [include a tarball of the project](https://gitlab.com/gitlab-org/quality/performance/-/blob/main/docs/environment_prep.md#setup-custom-test-data-using-the-gpt-data-generator) to be used with the `--large-project-tarball=/home/user/<CUSTOM PROJECT TARBALL>.tar.gz` flag. [The Alternate tarballs that are available](https://gitlab.com/gitlab-org/quality/performance-data/-/tree/main/projects_export?ref_type=heads).

**NOTE**: Loading data can take 1-2 hours to run, it is recommended to run it in a [`screen` session](https://www.gnu.org/software/screen/manual/screen.html) so that if you get disconnected, the data loading continues.

---

## Running a Performance Test

With GPT [running the test](https://gitlab.com/gitlab-org/quality/performance/-/blob/main/docs/k6.md#docker-recommended) is as simple as running:

```bash
docker run -it \
  -e ACCESS_TOKEN=<TOKEN> \
  -v <HOST CONFIG FOLDER>:/config \
  -v <HOST TESTS FOLDER>:/tests \
  -v <HOST RESULTS FOLDER>:/results \
  gitlab/gitlab-performance-tool --environment <ENV FILE NAME>.json --options 60s_500rps.json --tests <TEST FILE>.js
```

**Before running a test**, verify that you have the system setup to capture the metrics you need (either in the results directly from the test or in the Observability).

**After the test completes**, capture and save the metrics for comparison:

- Test pass/failure statistics
- Response timings from the test tool
- System metrics from Observability

**Make sure you capture the metrics** relevant to your [success criteria](#identify-success-criteria).

---

## Capturing an image URL from a Master pipeline

In this example, I am grabbing from a test run in [e2e-run-master](https://gitlab.enterprise.slack.com/archives/CNV2N29DM)

![e2e-run-master slack channel](./images/slack.png)

Click on the `Pipeline` link.

![Pipeline view](./images/pipeline.png)

Expand the `GitLab` Upstream and find the `build-images` job:

![build-images stage](./images/build-images.png)

On the `build-gdk-image` job it shows the URL that was built:

![image url](./images/master-url.png)

the url is in:
`Built image 'registry.gitlab.com/gitlab-org/gitlab/gitlab-qa-gdk:2fe86491afa3db4d9b48c06302e295f038863c11'`

---

## Capturing an image URL from a branch pipeline

This example pulls an image from the [rails-next](https://gitlab.com/gitlab-org/gitlab/-/pipelines?page=1&scope=all&status=success&ref=rails-next) pipeline

![Rails Next Pipeline](./images/Rails-next-pipeline.png)

Expand the downstream pipelines

![Downstream pipelines](./images/downstream-job.png)

Open the `trigger-omnibus` job, this will open another pipeline that builds the images

![omnibus pipeline](./images/ubuntu-branch.png)

Choose the appropriate job in the `package` stage for your architecture (I used `Ubuntu-24.04-branch`) and view it's artifacts

![artifacts](./images/artifacts.png)

In this case I needed to navigate a couple of folders, `pkg` > `ubuntu-noble`.

Copy the URL for the  `*.deb` file and save it to be used as the `GITLAB_UBUNTU_IMAGE` environment variable for GET.

In this case the URL was:

```text
https://gitlab.com/gitlab-org/build/omnibus-gitlab-mirror/-/jobs/12607844542/artifacts/file/pkg/ubuntu-noble/gitlab-ee_18.7.0+rfbranch.2245624022.06bed6be-0_amd64.deb
```

**Note:** ⚠️ Job artifacts expire and are deleted within a day or two, so you will probably need to re-capture the image URL if you need to rebuild your test environment.

---

## Interpreting the Results

After running your baseline and post-upgrade tests, you need to understand what the metrics mean.

You can analyze the metrics several different ways:

- GPT generates a [summary comparing against thresholds](https://gitlab.com/gitlab-org/quality/performance/-/blob/main/docs/k6.md#test-output-and-results)
- Reviewing the raw output
  - Loading the output into a spreadsheet and comparing (like it was done in the [examples](#real-world-examples))
  - Writing a script to compare
  - Generating some graphs (script/other tool)
  - Manually comparing the results

Use this framework to evaluate your results and determine next steps.

### Green (On Track)

- Metrics within published thresholds
- No significant regressions vs baseline (< 5% difference)
- Performance stable or improved
- **Action**: Document results and close testing

### Yellow (Needs Attention)

- Metrics slightly above thresholds (5-10% over)
- Minor regressions vs baseline (5-10% slower)
- Investigate root cause, may need optimization
- **Action**: Determine if the performance trade-off is acceptable for your use case. If acceptable, document the changed baseline and proceed. If not, investigate optimization opportunities.

### Red (Rework Needed)

- Metrics significantly above thresholds (> 10% over)
- Major regressions vs baseline (> 10% slower)
- Functionality breaks under load
- **Action**: Investigate root cause and implement fixes. Re-run testing after changes to verify improvement.

### Mixed Results

If your results are mixed (some metrics Green, some Yellow or Red), focus on metrics most relevant to your change. Document trade-offs and fix the unacceptable problems.

### Example Metric Comparison

When comparing baseline vs post-upgrade results:

| Metric | Baseline | Post-Upgrade | Change | Status |
| ------ | -------- | ------------ | ------ | ------ |
| API Response Time (p95) | 250ms | 260ms | +4% | Green |
| Throughput (RPS) | 195 | 192 | -1.5% | Green |
| Error Rate | 0.1% | 0.15% | +0.05% | Green |
| Memory Usage | 4.2GB | 4.5GB | +7% | Yellow |
| CPU Utilization | 65% | 68% | +4.6% | Green |

In this example, memory usage shows a minor increase (Yellow), but other metrics are within acceptable ranges (Green). This would warrant investigation into the memory increase, but doesn't necessarily block the upgrade.

### Identifying Relevant Metrics

Not all metrics are relevant to every change. Focus on metrics related to your specific upgrade, some common metrics:

#### For Rails/Ruby upgrades

- API response times (p50, p95, p99)
- Throughput (requests per second)
- Error rates
- Memory usage
- CPU utilization

#### For database upgrades

- Query execution time
- Database connection pool usage
- Disk I/O metrics
- Memory usage

#### For infrastructure changes

- Network latency
- Disk throughput
- CPU utilization
- Memory usage

---

## Real-World Examples

### Rails 7.2 Upgrade Testing

The Rails 7.2 upgrade testing ([#579847](https://gitlab.com/gitlab-org/gitlab/-/issues/579847)) demonstrates the complete workflow:

**Setup**:

- Deployed 10k environment with Rails 7.2 omnibus package
- Used custom image from MR artifacts
- Configured Jammy (Ubuntu 22.04) for package compatibility
- Seeded with horizontal and vertical performance data

**Baseline Testing**:

- Ran 60s_200rps.json configuration against baseline environment
- Exported Grafana metrics for comparison
- Documented all results with timestamps

**Upgrade & Testing**:

- Applied Rails 7.2 upgrade to environment
- Verified new version running
- Ran same 60s_200rps.json configuration
- Captured the test results
- Exported post-upgrade Grafana metrics (screenshots of the graphs)

**Results**:

- Compared baseline vs post-upgrade metrics from the test run in a Google Sheet
- Visually compare the Graphana graph exports
- Checked against Reference Architecture benchmarks
- Identified any performance regressions
- Documented findings in issue

**Key Learnings**:

- Use `screen` command to prevent test interruption from unstable connections
- Ensure rate limits are disabled before testing
- Export Grafana metrics with clear timestamps for accurate comparison

See the full issue for detailed results and analysis: [#579847](https://gitlab.com/gitlab-org/gitlab/-/issues/579847)

### Ruby 3.3 Upgrade Testing

The Ruby 3.3 upgrade testing ([#516194](https://gitlab.com/gitlab-org/gitlab/-/issues/516194)) demonstrates baseline comparison methodology:

**Setup**:

- Deployed 10k environment with Ruby 3.3 omnibus package
- Created parallel environment with Ruby 3.2 for direct comparison
- Seeded both environments with identical performance data

**Baseline Testing**:

- Ran 60s_200rps.json configuration against both environments
- Documented all results with timestamps
- Captured the test results from each
- Exported post-upgrade Grafana metrics (screenshots of the graphs) from each

**Results Comparison**:

- Compared Ruby 3.3 vs Ruby 3.2 metrics in a Google Sheet
- Visually compare the Graphana graph exports
- Checked against Reference Architecture benchmarks
- Identified endpoints with performance differences
- Documented findings in comparison spreadsheet

**Key Findings**:

- Overall test score slightly better for Ruby 3.3 (88.81% vs 88.75%)
- Only a few endpoints showed degraded performance (api_v4_users, web_user, api_v4_groups_group, web_project_file_rendered)
- Memory utilization slightly higher for Ruby 3.3 but showed similar patterns
- Conclusion: Ruby 3.3 maintains acceptable performance levels

**Key Learnings**:

- Creating a parallel environment with the previous version enables direct comparison
- Spreadsheet-based results tracking helps identify specific endpoints with issues
- Memory usage patterns are as important as response times
- Document findings in a way that helps future decision-making

**Resources**:

- Full issue with detailed results: [#516194](https://gitlab.com/gitlab-org/gitlab/-/issues/516194)
- Runbook for future iterations: [Performance Environment Setup Runbook](https://gitlab.com/gitlab-org/quality/perf-toolkit/-/tree/main/docs/runbooks/performance_environment_setup)

---

## Common Issues & Troubleshooting

This section will be populated with concrete issues and solutions as they're encountered during testing. Check back here for troubleshooting guidance.

---

## When to Escalate to Performance Enablement

Contact Performance Enablement if:

- **Correct Test Tool choice**: Need to figure out what tool to use to test
- **Test framework changes**: Need to modify GPT for new components or test scenarios
- **Results interpretation**: Results don't match expectations and you need expert analysis
- **Custom testing**: Need performance testing for components not in standard suite

**How to escalate**: Create a RFH in the [Developer Experience RFH Project](https://gitlab.com/gitlab-org/quality/request-for-help) with:

- Link to your test environment/results
- Description of the issue
- What you've already tried
- Expected vs actual results

## Related Resources

- [Performance Tools Selection Guide](./performance-tools.md)
- [Reference Architectures Documentation](https://docs.gitlab.com/administration/reference_architectures/)
- [GitLab Environment Toolkit (GET)](https://gitlab.com/gitlab-org/gitlab-environment-toolkit)
- [GitLab Performance Tool (GPT)](https://gitlab.com/gitlab-org/quality/performance)
- [Reference Architecture Benchmarks](https://gitlab.com/gitlab-org/reference-architectures/-/wikis/Benchmarks/Latest)
- [Upgrade Example: Rails 7.2 Testing](https://gitlab.com/gitlab-org/gitlab/-/issues/579847)
- [Upgrade Example: Ruby 3.3 Testing](https://gitlab.com/gitlab-org/gitlab/-/issues/516194)

## Questions or Feedback?

If you have questions about this guide or encounter issues not covered here, please:

1. Check the [troubleshooting section](#common-issues--troubleshooting)
2. Review related resources above
3. Create an issue in [quality-engineering/team-tasks](https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks) if you need Performance Enablement support
