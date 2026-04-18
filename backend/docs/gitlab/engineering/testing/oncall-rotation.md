---
title: Pipeline DRI On-call Rotation
description: >-
  The Developer Experience Sub-Department has two on-call rotations: pipeline triage (SET-led) and incident management (EM-led).
---

## Developer Experience Sub-Department pipeline triage on-call rotation

This is a schedule to share the responsibility of debugging/analysing the failures
in the various scheduled pipelines that run on multiple environments.
Developer Experience sub-department's on-call does not include work outside GitLab's normal business hours. Weekends and [Family and Friends Days](/handbook/company/family-and-friends-day/) are excluded as well. SETs also have full autonomy to move their 1:1s and their attendance to department meetings asynchronous for the week of their oncall if they decide to do so but please communicate to your manager accordingly.

In the current iteration, we have a timezone based rotation and triage activities happen during each team member's working hours.

Please refer to the [Debugging Failing tests](https://docs.gitlab.com/development/testing_guide/end_to_end/debugging_end_to_end_test_failures/)
guidelines for an exhaustive [list of scheduled pipelines](/handbook/engineering/testing/end-to-end-pipeline-monitoring/)
and for [specific instructions on how to do an appropriate level of investigation and determine next steps for the failing test](/handbook/engineering/testing/pipeline-triage/).

### Responsibility

#### Triage, record, and unblock failures

- The Saturday before the new rotation begins, the handover bot will create an issue in the [pipeline-triage](https://gitlab.com/gitlab-org/quality/pipeline-triage/-/issues/) project.
- The handover bot also assigns the triage issue created to the DRIs for the upcoming week, according to the schedule below.
- During the scheduled dates, the support tasks related to the test runs become the Directly Responsible Individual's ([DRI]'s) highest priority.
- Reporting and analyzing the End to End test failures in [Production](https://ops.gitlab.net/gitlab-org/quality/production/pipelines), [Canary](https://ops.gitlab.net/gitlab-org/quality/canary/pipelines), and  [Staging](https://ops.gitlab.net/gitlab-org/quality/staging/pipelines) pipelines takes priority over [GitLab `master`](https://gitlab.com/gitlab-org/gitlab/pipelines), and [GitLab FOSS `master`](https://gitlab.com/gitlab-org/gitlab-foss/pipelines). Note this also takes priority over fixing tests.
- [Preprod](https://ops.gitlab.net/gitlab-org/quality/preprod/-/pipelines) pipelines have equal priority with `Production` and `Staging` pipelines during release candidate testing from the Monday through Thursday of the release week.
- If there is a time constraint, the DRI should report and analyze the failures in [Staging](https://ops.gitlab.net/gitlab-org/quality/staging/pipelines), [Canary](https://ops.gitlab.net/gitlab-org/quality/canary/pipelines) and [Production](https://ops.gitlab.net/gitlab-org/quality/production/pipelines) pipelines just enough to determine if it is an application or an infrastructure problem, and [escalate as appropriate](/handbook/engineering/workflow/development-processes/infra-dev-escalation/process/). All the reported failures in those pipelines should be treated as ~priority::1/~severity::1 until it's determined that they're not. That means they should be investigated ASAP, ideally within 2 hours of the report. If the DRI will not be able to do so, they should delegate any investigation they're unable to complete to the DRI from the following week. If neither DRI is available or will be able to complete the investigations, solicit help in the `#s_developer_experience` slack channel.
- Consider [blocking the release by creating an incident](/handbook/engineering/deployments-and-releases/deployments/#i-found-a-regression-what-do-i-do-next) if new ~severity::1 regressions are found in smoke specs.
- It is important that all other failure investigations are completed in a timely manner, ideally within 24 hours of the report. If the DRI is unable to investigate all the reported failures on [all the pipelines](/handbook/engineering/testing/end-to-end-pipeline-monitoring/) on time, they should solicit help in the `#s_developer_experience` slack channel.
- Cross-cutting issues such as https://gitlab.com/gitlab-org/quality/team-tasks/-/issues/530 are triaged by the on call DRI to determine the next action. Other team-members should notify the DRI if they come across such an issue. The DRI can inform the rest of the department via the `#s_developer_experience` channel, if necessary.
- Everyone in the Developer Experience sub-department should support the on-call DRI and be available to jump on a zoom call or offer help if needed.
  - If a change needs to be made to CI/CD variables in [gitlab-org/gitlab](https://gitlab.com/gitlab-org/gitlab/-/settings/ci_cd) and the DRI does not have access, they can ask for help in the `#dx_maintainers` and `#development` Slack channels. Any Quality maintainer should have sufficient access to be able to assist.
- Every day the APAC/EMEA/AMER DRI summarizes the failures found for that day in the triage issue like [this](https://gitlab.com/gitlab-org/quality/pipeline-triage/-/issues/45#note_408358615). This serves as an asynchronous handoff to the DRI in the other timezone.
- The DRI can make use of the [dri gem](https://gitlab.com/gitlab-org/ruby/gems/dri) to automate reporting handoff and surface useful triage information, such as newly created failures or feature flags enabled by environment.
- Be aware that failing smoke specs in staging/canary will block the deployer pipeline and may be treated as production incidents.

### Schedule

To view the pipeline triage rotation schedule, please visit the [pipeline-triage](https://gitlab.com/gitlab-org/quality/pipeline-triage#dri-weekly-rotation-schedule) project.

Or use `chatops` in Slack with command `/chatops run quality dri schedule`

### Responsibilities of the DRI for scheduled pipelines

- The DRI does the [triage](/handbook/engineering/testing/pipeline-triage/). They should solicit help in the `#s_developer_experience` slack channel if they need further assistance.
- The DRI makes the call whether to fix or quarantine the test.
- All fixes or quarnatines should be applied to all release branches to avoid release delays.
- The DRI reviews any [auto-generated quarantine MRs](https://gitlab.com/gitlab-org/gitlab/-/merge_requests/?sort=merged_at_desc&state=opened&label_name%5B%5D=automation%3Abot-authored&label_name%5B%5D=quarantine&label_name%5B%5D=group%3A%3Aunknown) for the tests with shared ownership (MRs labeled with `group::unknown`) and decides whether to merge or close.
- The DRI should periodically take a look at [the list of unassigned quarantined issues](https://gitlab.com/gitlab-org/gitlab/issues?state=opened&label_name%5B%5D=QA&label_name[]=type::bug) and work on them.
- If the DRI is not available during their scheduled dates (for more than 2 days), they can swap their schedule with another team member. If the DRI's unavailability during schedule is less than 2 days, the DRI in the same timezone the following week can cover.

### Responsibilities of the DRI for deployment pipelines

- The DRI helps the delivery team debug test failures that are affecting the release process.
- The DRI in the same timezone for the following week fills in when the DRI is not available.

### New hire on-call shadowing

To help a new hire prepare for their first on-call rotation, they should spend a few days before their first rotation shadowing the DRI. When the new hire is DRI, the DRI from the following week should pair with the new hire for a couple of days to answer questions and help triage/debug test failures.

### Supporting infrastructure and environment upgrades

The Developer Experience team provides support for planned upgrades to GitLab environments. This ensures changes are properly validated through appropriate test suites before and after deployment.

For teams requesting upgrade support (within or outside office hours):

1. Plan ahead: Use the [RFH (request for help) issue](https://gitlab.com/gitlab-org/quality/test-governance/request-for-help). Submit your request at least 2 weeks in advance.
1. Include key details:

    - Proposed upgrade date and time
    - Environment that is planned to be upgraded
    - Type of upgrade (infrastructure, database, service decomposition, etc.)
    - Request examples:
      - [Sec Decomposition GPRD Rollout](https://gitlab.com/gitlab-org/quality/test-governance/request-for-help/-/issues/2)
      - [CI Decomposition Rollout](https://ops.gitlab.net/gitlab-com/gl-infra/db-migration/-/blob/ae6240c4bdf94a7774f9ad844dcec26f936a2946/.gitlab/issue_templates/ci_decomposition.md)
      - [PostgreSQL 14 upgrade](https://ops.gitlab.net/gitlab-com/gl-infra/db-migration/-/blob/ae6240c4bdf94a7774f9ad844dcec26f936a2946/.gitlab/issue_templates/pg14_upgrade.md)
      - [Request for SET coverage during Database Upgrade](https://gitlab.com/gitlab-org/quality/test-governance/request-for-help/-/work_items/70)

1. What to expect after the request was submitted?

   - The request will be triaged and coverage schedule will be created.
     - Office hours:
       - The request will be taken care by [current Pipeline DRI](https://gitlab.com/gitlab-org/quality/pipeline-triage#dri-weekly-rotation-schedule)
     - Weekends:
       - The schedule with 4 hours shifts will be created to cover 24 hours.
   - The schedule and preferred way of communication will be shared with you no later than 1 week before expected upgrade.
   - Please add everyone who's on call to dedicated slack channel, so they can monitor the progress of the upgrade during their shift.

NOTE! We don't have many team members in APAC area, so sometimes there will be an empty window of 4 hours during which we kindly ask you to use our troubleshooting guides:

- [Using Duo to debug test failures](../testing/using-duo-to-debug-test-failures.md)
- [Guide to E2E test failure issues](../testing/guide-to-e2e-test-failure-issues.md)

For the Pipeline DRI:

1. Pre-upgrade preparation:

    - Review the upgrade plan with the requesting team
    - Identify and document the appropriate test suites to validate the changes
    - Ensure test environments are ready
    - Execute the test suites to get a baseline of test results

2. During the upgrade:

    - Monitor and review test results
    - Provide assistance with any testing-related issues
    - Document any problems or anomalies observed

3. Post-upgrade:

    - Create follow-up issues for any problems or improvements identified
    - Document lessons learned

4. If unavailable:

    - If you cannot be available during the planned upgrade (especially for off-hours upgrades), reach out to the team immediately to find coverage
    - Ensure the replacement DRI is briefed on the upgrade plan and testing requirements

## Developer Experience Sub-Department incident management on-call rotation

The EMs should share the responsibility of monitoring, responding to, and mitigating incidents.
Developer Experience Sub-Department's on-call does not include work outside GitLab's normal business hours. Weekends and [Family and Friends Days](/handbook/company/family-and-friends-day/) are excluded as well.
In the current iteration, incident management activities happen during each team member's working hours.

### Responsibility

- The Engineering Manager should ensure they have joined the Slack channel `#incidents`.
- The Engineering Manager should help with monitoring the incident management channel, tracking, directly helping, delegating, and raising awareness of incidents within the Developer Experience Sub-Department as appropriate.
- The current DRI should be clearly noted on the incident issue.
- If a corrective action is needed, the EM should make sure the DRI has created an issue and labeled it with ~'corrective action'.
- Everyone in the Developer Experience Sub-Department should support the on-call DRI and be available to jump on a Zoom call or offer help if needed.
