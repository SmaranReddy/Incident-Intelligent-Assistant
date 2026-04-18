---
title: Duo CLI
description: "We are building an intelligent command-line interface that brings GitLab Duo's AI capabilities directly to developers' terminals, enhancing productivity through natural language interactions with GitLab's DevSecOps platform."
---

## Team Members

This tool falls under the scope of [Editor Extension: Multi-Platform](/handbook/engineering/ai/editor-extensions-multi-platform/).

{{< group-by-slugs aelhusseiny andrei.zubov elwyn-gitlab aspringfield viktomas >}}

### Stable counterparts

The following people are [stable counterparts](/handbook/leadership/#stable-counterparts) of the Duo CLI group:

{{< group-by-slugs mcorren sam_reiss >}}

## Vision

The Duo CLI group is focused on developing GitLab Duo CLI as an AI-powered command-line interface that enables developers to interact with GitLab's DevSecOps platform using natural language. Our goal is to streamline developer workflows by bringing AI-powered assistance directly to the terminal, where developers spend much of their time.

## Internal Processes

### Development Phases

The Duo CLI is currently in the Dogfooding phase. Our development approach follows these phases:

| Phase | Description | Status |
| -------- | ------- | ------- |
| Ideation | Concept validation and initial research | ✅ Complete |
| MVC | Minimum Viable Concept development | ✅ Complete |
| Dogfooding | Internal Dogfooding | 📍 We are here |
| Beta | Beta Release | 📋 Planned |
| GA | General Availability | 📋 Planned |

## Active Work, Resources & Links

Find all the information, active work, and communication channels for the Duo CLI project:

### Development Resources

- **Repository**: [GitLab Language Server](https://gitlab.com/gitlab-org/editor-extensions/gitlab-lsp/-/tree/main/packages/cli?ref_type=heads)
- **Issue Tracker**: [Duo CLI Active Work](https://gitlab.com/groups/gitlab-org/-/boards/9839597?epic_id=3743089)
- **Epic**: [Duo CLI Development Epic](https://gitlab.com/groups/gitlab-org/-/epics/19070)

### Communication Channels

- **Slack**: `#f_duo_cli`

- **Leave us feedback in an issue**: [Duo CLI Feedback Epic](https://gitlab.com/groups/gitlab-org/-/epics/19806)

## Dogfooding

The Duo CLI is currently in the dogfooding phase, where GitLab team members are using and testing the tool internally to validate functionality and gather feedback before broader release.

### Dogfooding Duo CLI Epic

[Dogfooding Duo CLI Epic](https://gitlab.com/groups/gitlab-org/-/epics/19806)

### Getting Started

Install the latest version of Duo CLI with npm and run it using your GitLab personal access token:

**Note:** You'll need to generate an API-scope Personal Access Token:

1. Click your GitLab profile
2. Click **Preferences**
3. Click **Personal Access Tokens** in the navigation
4. Create a new Personal Access Token with `api` scope

### Installation

[Installation Guide](https://gitlab.com/groups/gitlab-org/-/epics/19806#installation)

**Run the following in any terminal:**

```bash
npm i -g @gitlab/duo-cli
duo
```

### Help us shape Duo CLI by Providing Feedback

We're collecting feedback to make the CLI effective for GitLab team members. Please test core development workflows like:

- **Code tasks**: Generation, explanation, refactoring, debugging, test writing
- **GitLab workflows**: MR reviews, issue analysis, CI/CD debugging, security reviews
- **Advanced scenarios**: Multi-file changes, architecture planning, large codebases

**Feedback Categories:**

- **Bugs**: Use bug template for crashes, errors, performance issues
- **Features**: Use feature template for missing capabilities or enhancements
- **Usability**: Report confusing workflows, unclear errors, documentation gaps
- **Success Stories**: Share what works well and productivity wins

---

*Note: This page documents the Duo CLI initiative which is currently in early development phases. Information will be updated as the project progresses.*
