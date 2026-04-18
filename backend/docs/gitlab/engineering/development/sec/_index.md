---
title: Sec Section
description: >-
  The Sec Section is composed of development teams working on Secure
  and Software Supply Chain Security features of the GitLab DevOps Platform.
---

## Teams and Handbook Pages

The following teams comprise the sub-department:

- Software Supply Chain Security stage - [handbook](/handbook/engineering/development/sec/software-supply-chain-security/)
  - Anti-abuse group - [handbook](/handbook/engineering/development/sec/software-supply-chain-security/anti-abuse)
  - Authentication group - [handbook](/handbook/engineering/development/sec/software-supply-chain-security/authentication)
  - Authorization group - [handbook](/handbook/engineering/development/sec/software-supply-chain-security/authorization)
  - Compliance group - [handbook](software-supply-chain-security/compliance/)
- Application Security Testing stage - [handbook](/handbook/engineering/development/sec/secure/)
  - Composition Analysis group - [handbook](/handbook/engineering/development/sec/secure/composition-analysis/)
  - Dynamic Analysis group - [handbook](/handbook/engineering/development/sec/secure/dynamic-analysis/dynamic-analysis/)
  - Static Analysis group - [handbook](/handbook/engineering/development/sec/secure/static-analysis/)
  - Secret Detection group - [handbook](/handbook/engineering/development/sec/secure/secret-detection/)
  - Vulnerability Research group - [handbook](/handbook/engineering/development/sec/secure/vulnerability-research/)
  - API Security - [handbook](/handbook/engineering/development/sec/secure/dynamic-analysis/api-security/)
- Security Risk Management
  - Security Policies group - [handbook](/handbook/engineering/development/sec/security-risk-management/security-policies/)
  - Security Platform Management group
  - Security Insights group - [handbook](/handbook/engineering/development/sec/security-risk-management/security-insights/)
  - Security Infrastructure group - [handbook](/handbook/engineering/development/sec/security-risk-management/security-infrastructure/)

It is important to delineate who the EM and PM DRIs are for every functionality, especially where this may not be obvious. This is documented on a dedicated [delineation page](delineate-sec.html).

## AI Prompts

## Product Direction

Product direction can be found on the [Sec Section Product Direction](https://about.gitlab.com/direction/security/) handbook page.

## Sec Section Resources

The following resources provide guidance for common development patterns across Sec Section teams:

### Observability

- [Tutorial: Add observability metrics to a CI-based analyzer](/handbook/engineering/development/sec/secure/analyzer-observability-metrics/) -
  Step-by-step guide for implementing the decentralized events pattern in security analyzers.
- [Secret Detection metrics](/handbook/engineering/development/sec/secure/secret-detection/metrics/) -
  Guide for adding metrics to the Secret Detection analyzer and GitLab monolith.

## Project Setup

Keeping our projects organized is very important for productivity and maintainability.

- To setup a new project we follow the [company-wide Engineering guidelines](/handbook/engineering/workflow/gitlab-repositories/).
- Sec projects should be organized into one of
  - [https://gitlab.com/gitlab-org/secure](https://gitlab.com/gitlab-org/secure)
  - [https://gitlab.com/gitlab-org/software-supply-chain-security](https://gitlab.com/gitlab-org/software-supply-chain-security)
  - [https://gitlab.com/gitlab-org/security-products](https://gitlab.com/gitlab-org/security-products)

In general, we want to keep as few projects in `security-products` as necessary.

`security-products` should only contain :

- Source code for applications that will run as part of a customer install
- Demos
- Historical projects that are difficult to move.

`secure` and `software-supply-chain-security` should have projects for:

- End-to-end testing
- Benchmarks / Stats
- Tooling

There may be projects that should belong in `secure` or `software-supply-chain-security` but for technical reasons are much easier to have in `security-products`. In those cases, we can locate the project in `security-products` if reasonable efforts were made to get the project in `secure` or `software-supply-chain-security` but were unsuccessful.

### Recommended settings

When creating a new project, all settings should be left to the default options, except for the following which are specific to the secure stage:

1. Add a [CODEOWNERS](https://docs.gitlab.com/ee/user/project/codeowners/) file to the project, for example:

   ```shell
   [Maintainers]
   * @gitlab-org/maintainers/container-scanning

   ^[Reviewers]
   * @gitlab-org/secure/static-analysis
   ```

   We recommend creating a [dedicated group of maintainers](https://gitlab.com/groups/gitlab-org/maintainers) for use in the `CODEOWNERS` file.

1. Disable the project [issue tracker](https://docs.gitlab.com/ee/user/project/issues/).

   - `Settings -> General -> Visibility, project features, permissions -> Issues`
      - `Disabled`

   Issues should be created in the [groups/gitlab-org issue tracker](https://gitlab.com/groups/gitlab-org/-/issues) instead. See step `3.` below to configure this.

   Using a single, centralized issue tracker over per-project issue trackers has the following advantages:

      - It improves the visibility of issues and aligns with our value of [transparency](/handbook/values/#transparency).

        For example, it's very easy for community members to [filter the issues](https://gitlab.com/groups/gitlab-org/-/issues/?sort=due_date_desc&state=opened&label_name%5B%5D=quick%20win&first_page_size=100) in the `groups/gitlab-org` tracker to discover [GitLab issues seeking wider community contributions](/handbook/marketing/developer-relations/engineering/community-contributors-workflows/#seeking-wider-community-contributions).

      - It leverages existing tools and infrastructure, such as having `triage-ops` and other bots executed against issues, without any additional configuration.

      - It provides a more consistent experience, since all labels and issue templates will be the same.

      - It's easier to write automated scripts, such as using the [Security triage automation](https://gitlab.com/gitlab-org/secure/tools/security-triage-automation/) tool to create/modify vulnerabilities.

      - There are some issues that apply to multiple projects. If each project has their own issue tracker, we'd need to figure out which issue tracker should "own" an issue that applies to multiple projects.

   Having said that, there are currently some limitations related to using a single, centralized issue tracker, for example [resolving threads in new issues doesn't work](https://gitlab.com/gitlab-org/gitlab/-/issues/220535).

   Until this issue has been resolved, we may choose to leave the Issue tracker enabled in the new project.

   In these cases, please consider these to avoid abandoned issues:

   1. Make the tracker private.
   1. Add an issue template with instructions.
   1. Ensure there's a triage process in place.

1. Configure a [custom issue tracker](https://docs.gitlab.com/ee/user/project/integrations/custom_issue_tracker.html)

   - `Settings -> Integrations -> Custom issue tracker -> Configure`
      - `Enable integration`
         - `Active`
      - `Project URL`
         - `https://gitlab.com/gitlab-org/gitlab/issues`
      - `Issue URL`
         - `https://gitlab.com/gitlab-org/gitlab/issues/:id`
      - `New issue URL`
         - `https://gitlab.com/gitlab-org/gitlab/issues/new`

1. Configure the following [project features and permissions](https://docs.gitlab.com/ee/user/project/settings/) settings:

   - `Settings -> General -> Visibility, project features, permissions`
      - `Project visibility`
         - `Public`
      - `Additional options`
         - `Users can request access`
            - `Disabled`
      - `Container Registry`
         - `Only Project Members`
   - `Settings -> Repository -> Protected branches`
      - `Allowed to merge`
         - `Maintainers`
      - `Allowed to push and merge`
         - `No one`
      - `Allowed to force push`
         - `Disabled`
      - `Code owner approval`
         - `Enabled`
   - `Settings -> Repository -> Protected tags`
      - `Tag`
         - `v*`
      - `Allowed to create`
         - `Maintainers`
         - [GitLab Dev Service - Secure Stage - Analyzers Automation](https://gitlab.com/gl-service-dev-secure-analyzers-automation)
   - `Settings -> Merge Requests`
      - `Squash commits when merging`
         - `Require`
      - `Approval settings`
         - `Prevent approval by author`
         - `Prevent editing approval rules in merge requests`
         - `Remove approvals by Code Owners if their files changed`
      - `Merge request approvals -> Approval rules`
         - `Approvers`
            - `All eligible users`
         - `Target branch`
            - `All branches`
         - `Approvals required`
            - `1`
      - `Merge checks`
         - `All threads must be resolved`
         - `Pipelines must succeed`
      - `Merge commit message template`

         ```markdown
         Merge branch '%{source_branch}' into '%{target_branch}'

         %{title}

         %{issues}

         See merge request %{url}

         Merged-by: %{merged_by}
         %{approved_by}
         %{reviewed_by}
         %{co_authored_by}
         ```

      - `Default description template for merge requests`

         ```markdown
         ## What does this MR do?

         <!--
         Describe in detail what your merge request does, why it does that, etc.

         Please also keep this description up-to-date with any discussion that takes
         place so that reviewers can understand your intent. This is especially
         important if they didn't participate in the discussion.

         Make sure to remove this comment when you are done.
         -->

         ## What are the relevant issue numbers?

         ## Does this MR meet the acceptance criteria?

         - [ ] Changelog entry added
         - [ ] [Documentation created/updated for GitLab EE](https://docs.gitlab.com/ee/development/documentation/feature-change-workflow.html), if necessary
         - [ ] Documentation created/updated for this project, if necessary
         - [ ] Documentation reviewed by technical writer *or* follow-up review issue [created](https://gitlab.com/gitlab-org/gitlab-ee/issues/new?issuable_template=Doc%20Review)
         - [ ] [Tests added for this feature/bug](https://docs.gitlab.com/ee/development/testing_guide/index.html)
         - [ ] Job definition updated, if necessary
           - [ ] [Auto-DevOps template](https://gitlab.com/gitlab-org/gitlab-foss/tree/master/lib/gitlab/ci/templates)
           - [ ] [Job definition example](https://docs.gitlab.com/ee/ci/examples/sast.html)
           - [ ] [CI Templates](https://gitlab.com/gitlab-org/security-products/ci-templates/tree/master/includes)
         - [ ] Ensure the report version [matches the equivalent schema version](https://gitlab.com/gitlab-org/security-products/security-report-schemas/-/blob/master/CHANGELOG.md)
         - [ ] Conforms to the [code review guidelines](https://docs.gitlab.com/ee/development/code_review.html)
         - [ ] Conforms to the [Go guidelines](https://docs.gitlab.com/ee/development/go_guide/index.html)
         - [ ] Security reports checked/validated by reviewer

         /label ~"devops::secure" ~"Category:" ~"group::" ~"backend"
         ```

When configuring projects that are not part of the secure stage, please see the [GitLab Projects Baseline Requirements](/handbook/security/policies_and_standards/gitlab_projects_baseline_requirements) for more details.

## Performance Indicators

- [Sec Sub-department Performance Indicators](/handbook/product/groups/product-analysis/engineering/dashboards/)
- [Error Budgets](/handbook/engineering/error-budgets/) as Performance Indicators for stage groups

## Slack channels

- [#sec-section](https://gitlab.slack.com/archives/C02087FTL5V) - Sec Section discussions spanning the Software Supply Chain Security and Secure stages.
- [#sec-growth-datascience-people-leaders](https://gitlab.slack.com/archives/C033F69CQCB) - Engineering people leaders in Sec, Growth, and ModelOps.
- [🔒sec-growth-datascience-leadership-confidential](https://gitlab.slack.com/archives/GKWF00Y3E) - Private channel for engineering people leaders in Sec, Growth, and ModelOps.

## Calendars

We have three stage level calendars, [AST Stage Calendar](https://calendar.google.com/calendar/embed?src=gitlab.com_md0ao36gpvl5v1f4128erhnj6g%40group.calendar.google.com), [SRM Stage Calendar](https://calendar.google.com/calendar/embed?src=c_7c82f32669d585a441be28ca291a19611490d48bd9d27554dea93d9206612be1%40group.calendar.google.com&ctz=America%2FLos_Angeles), and [SCSS Stage Calendar](https://calendar.google.com/calendar/embed?src=gitlab.com_ed6207uel78de0j1849vjjnb3k%40group.calendar.google.com), where we host cross-group events such as:

1. Monthly retrospective
1. Coffee chats
1. Staff sync

Each group also has a calendar for team-based discussions, such as the our weekly group syncs.

We encourage utilizing our available [Google Groups](https://groups.google.com/my-groups) instead of including individuals as attendees when possible. Along with ensuring the event is represented on individual's calendars for visibility, new team members are automatically added to events (as well as removed when someone departs from a team).

### Google Groups

Google groups follow the convention [section]-[stage]-[group], separating multi-word names with `_` and are structured as the following:

- sec-section
- sec-software_supply_chain_security
- sec-security_risk_management
- sec-application_security_testing
- sec-security_risk_management-security_insights
- sec-security_risk_management-security_policies
- sec-security_risk_management-security_platform_management
- sec-security_risk_management-security_infrastructure
- sec-application_security_testing-static_analysis
- sec-application_security_testing-secret_detection
- sec-application_security_testing-dynamic_analysis
- sec-application_security_testing-composition_analysis
- sec-software_supply_chain_security-authentication
- sec-software_supply_chain_security-authorization
- sec-software_supply_chain_security-compliance
- sec-software_supply_chain_security-pipeline_security
- vulnerability-research

The members of each google group consists of stable counterparts and the correct `eng-dev-[stage]-[group]` group of engineers. When stable counterparts change, or team members onboard/offboard the appropriate group should be updated by the EM of the respective group.

## Staying Informed and Informing Team Members

- [Sec Week In Review Google Document](https://drive.google.com/drive/search?q=%22Sec%20Section%20Week%20In%20Review%22) - is an asynchronous weekly document of notables things happening in Sec. The document is inspired by the [Engineering Week In Review](/handbook/engineering/).
- Slack channels #s_secure and #s_software-supply-chain-security are informative since they are all part of Sec Section.

## Planning in the Section

In the vast majority of cases, work is scoped to individual groups within the section. However, there are times when the section needs to design
and execute solutions as a coordinated Section or risk creating poor and non-cohesive user experiences.

These initiatives will be orchestrated through epics and issues. Initiatives with the following labels are deemed to fall in this category of work.

- [`~section::sec` and `~group::not owned`](https://gitlab.com/groups/gitlab-org/-/issues/?sort=created_asc&state=opened&label_name%5B%5D=section%3A%3Asec&label_name%5B%5D=group%3A%3Anot_owned); or
- [`all Sec groups`](https://gitlab.com/groups/gitlab-org/-/issues/?sort=created_asc&state=opened&label_name%5B%5D=all%20Sec%20groups)

### Process for planning section-wide initiatives

At least once per milestone, Senior Engineering Managers in the section will do the following:

- In partnership with Product Management, initiatives 6 months or older will be evaluated to determine if they're still relevant.
- New initiatives will be triaged, checking their requirements for understandability and completeness. Further, the group most impacted will be identified.
  - In situations where most impacted group is not clear, technical leadership via #sec-section will be engaged to help discern which group that might be.
- Group most impacted will be declared DRI for that initiative and are expected to:
  - Produce a high-level implementation plan that will scale for the whole problem.
  - Create implementation issues that are broken down by feature category.
    - The original high-level implementation plan will be included, or at least directly linked, in the created issues.
    - Original issue where implementation plan was debated and created will also be linked to the generated issues.
  - Distribute implementation issues to the relevant groups.

Generated issues will be worked through normal prioritization processes as they are distributed to individual groups.

## Page Performance

Our team monitors [LCP](/handbook/engineering/development/performance-indicators/#largest-contentful-paint-lcp) (Largest Contentful Paint) to ensure performance is below our target (currently 2500ms).

[LCP Dashboard for Secure owned pages](https://dashboards.gitlab.net/d/sftijGFMz/sitespeed-lcp-leaderboard?from=now-90d&orgId=1&to=now&refresh=30s&var-namespace=sitespeed_io&var-path=desktop&var-testname=gitlab&var-domains=gitlab_com&var-pages=API_Fuzzing_Config_UI&var-pages=DAST_Profiles&var-pages=On_Demand_Scans&var-pages=SAST_Config_UI&var-pages=Secure_Dependency_List&var-pages=Secure_License_Compliance&var-pages=Secure_Security_Configuration&var-pages=DAST_Config_UI&var-browser=chrome&var-function=median&var-connectivity=cable)

## Working with Product Design

To streamline our workflow and ensure efficient collaboration between the Engineering and Product Design teams, we have established the following guidelines for UX involvement in merge request (MR) reviews:

**Merge Request UX Review Requirement:**

- A UX review is required only for work that has been explicitly designed by a Product Designer and should be reviewed by that Product Designer.
- MRs that do not involve work explicitly designed by a Product Designer can be labeled as `UX Tech Debt` and merged without a UX review.

**Handling High Priority UX Reviews:**

- If a high-priority task arises that requires a UX review but was not planned during the milestone planning process, it should be discussed with the [Product Design Manager for Sec](/handbook/product/categories/#sec-section).
- To accommodate this unexpected work, another task from the original milestone plan will need to be deprioritized or dropped.

**Exceptions:**

- These new guidelines do not apply to the Authentication, Authorization, and Pipeline Security groups, which will continue to operate under their current processes.

## Working with Customer Support

The Sec engineering teams do not provide support directly to customers. Instead engineers collaborate with our Customer Support Engineers via the [process on the Sec Sub-department support project](https://gitlab.com/gitlab-com/sec-sub-department/section-sec-request-for-help/).

## Working on security tooling requests

As GitLab grows, the Sec teams continue to build tooling that makes it easier to securely manage users for our largest customers. In doing so, managing team members on GitLab.com is an excellent use case where the features can be internally dogfooded before they are rolled out to our users. The Security and CorpSec teams can add the label `security tooling`, `section::sec` and the respective priority `priority::1/2/3` to tag an item that will help in such management of GitLab team members and needs to be added to the backlog. The [features page](/handbook/product/categories/features/) is handy in identifying where a particular functionality may belong, such that the correct EM/PM for the group can be tagged in the issue.

The backlog for these issues can viewed at [Sec Security Tooling - issue](https://gitlab.com/groups/gitlab-org/-/boards/9065128?label_name[]=security%20tooling&label_name[]=section%3A%3Asec) for individual issues. Each month, product and security counterparts will [review these requests](https://gitlab.com/gitlab-com/Product/-/issues/?sort=created_date&state=opened&label_name%5B%5D=security%20tooling&first_page_size=100) and ensure that the priority items are scheduled into the roadmap.

## How to work with the Quality team

### Frontend Responsibilities

1. Being able to identify what code changes would likely break E2E or System level tests and informing Quality.
1. Not to write E2E tests, but to catch potential failures and communicate gaps in coverage before landing to master.

### Identifying potential breakages

1. Look to see if issue you are working on [has existing test coverage](https://gitlab.com/gitlab-org/quality/team-tasks/-/issues/736). These are the tests likely to fail
1. If you are working around code that contains a selector like `data-qa-selector="&lt;name&gt;"`, then there is likely to be an existing E2E test. Tests can be found by searching our [E2E tests in Secure](https://gitlab.com/gitlab-org/gitlab/-/tree/master/qa/qa/specs/features/ee/browser_ui/secure).

### Communicating changes that may break tests

Ping the DRI for quality assigned to Secure. You can find the person on the [team page](/handbook/engineering/development/sec/secure/#team-members). If they are unavailable, then `#s_developer_experience` on Slack or the [triage DRI](https://gitlab.com/gitlab-org/quality/pipeline-triage#dri-weekly-rotation-schedule) dependent on severity.

## Section Retrospectives

In addition to our group retrospectives, we facilitate an async Sec Section level retrospective each month. The goal of the section wide retrospective is to review topics that bubbled-up from our group/team retrospectives. Additionally, we identify themes that could be discussed synchronously.  We use [this doc](https://docs.google.com/document/d/1g_FIMgr9r_Yf56xISxoI8B-1G-kbP3PQSeo7W-kKj24/edit#) and an [issue](https://gitlab.com/gitlab-com/www-gitlab-com/-/issues/?search=retrospective&sort=updated_desc&state=closed&label_name%5B%5D=section%3A%3Asec&first_page_size=20) created with [this template](https://gitlab.com/gitlab-com/www-gitlab-com/-/blob/master/.gitlab/issue_templates/sec-section-retro.md) to facilitate the section retrospective.

### Key Dates

1. The Monday after the monthly release - Group async retrospective issues are generated. Groups should start contributing topics.
1. The week the milestone ends - Groups hold their retrospectives. Team members bubble-up identified topics and follow-up items (outcomes) to the [section retrospective document](https://docs.google.com/document/d/1g_FIMgr9r_Yf56xISxoI8B-1G-kbP3PQSeo7W-kKj24/edit#).
1. The week of the release -  Section wide retrospective async review shared in the `#sec-section` Slack channel.

### DRI Responsibilities

The [DRI](/handbook/people-group/directly-responsible-individuals/) for Section-wide retrospectives will be the Senior Engineering Manager. The SEM will find a volunteer if it is needed on specific milestones. The following tasks are executed each milestone:

1. Prior to the async section retrospective, review bubble-up topics and identify 2-3 themes to support async discussion topics.
1. Ask everyone through Slack in `#sec-section` to review the [section retrospective document](https://docs.google.com/document/d/1g_FIMgr9r_Yf56xISxoI8B-1G-kbP3PQSeo7W-kKj24/edit#) and add comments.
1. Share a summary of the async discussions in Slack in `#sec-section`.
1. Follow up with groups on any identified improvements.
1. Promote, promote, promote!
