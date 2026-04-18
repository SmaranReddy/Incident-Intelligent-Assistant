---
title: "Vulnerability Research Group Reaction Rotation"
---

## Reaction Rotation

Every 2 weeks (cycle), two engineers in the team are assigned the role of Reaction Rotation, one as Primary and the other as Secondary. The assignments are in the [rotation schedule](https://gitlab.com/groups/gitlab-org/secure/vulnerability-research/-/wikis/home/Reaction-Rotation).

The Secondary role is to step-in when the Primary is unavailable or over capacity. In such instances, the Secondary assumes the same responsibilities as the Primary, but otherwise they work on tasks planned for the milestone.

### Responsibilities

#### Requests for Help

Issues are created in the [request-for-help](https://gitlab.com/gitlab-com/request-for-help) project. During Reaction Rotation, the assigned engineer must review the [open issues](https://gitlab.com/gitlab-com/request-for-help/-/issues/?sort=created_date&state=opened&or%5Blabel_name%5D%5B%5D=Help%20group%3A%3Avulnerability%20research&or%5Blabel_name%5D%5B%5D=group%3A%3Avulnerability%20research&first_page_size=20) and engage with each issue.

If a request for help highlights a bug or feature request, create an issue in the public tracker, link to it in the request for help issue, and close the latter.

#### Slack Questions

Check the [Vulnerability Research Slack Channel](https://gitlab.enterprise.slack.com/archives/CSELDERRN) and respond to any questions asked or delegate/ping a person that may know the answer. As with [Requests for Help](#requests-for-help), if the question concerns a bug or feature request, create an issue.

#### CNA Duties

The CNA duties can be found [in the group handbook page](cna.md)

#### GLAS Rules Release Cycle

Track and manage the GLAS rules release process (happens every milestone, so every 2 cycles):

1. Take the latest version of GLAS rules
2. Bump that version in the GLAS analyzer, releasing a new analyzer version
3. Communicate the release to stakeholders on Slack and relevant documentation

### Reaction Rotation Issue Tracking

To help track reaction rotation activities without unnecessary overhead, a [simple issue template](https://gitlab.com/gitlab-org/secure/general/-/blob/master/.gitlab/issue_templates/Reaction%20Rotation%20Vulnerability%20Research.md) is available for engineers to use. The template provides basic structure for documenting essential activities and includes helpful queries for monitoring key areas.

Engineers should use this template as a flexible tool to support their rotation work, adapting it as needed to fit their workflow while ensuring the core responsibilities outlined in this handbook are addressed.
