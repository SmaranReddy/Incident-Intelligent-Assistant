---
title: "AI in Developer Experience"
description: "How the Developer Experience team uses AI tools and practices to improve engineering workflows"
---

The Developer Experience team actively uses AI to accelerate our own work, validate GitLab's AI capabilities from a developer perspective, and build institutional knowledge about effective AI-assisted engineering workflows. This page documents our tools, guidelines, and practices.

## Tools We Use

| Tool | Purpose | Primary Use Cases |
|------|---------|-------------------|
| **GitLab Duo** | GitLab's native AI assistant | Code suggestions, code review, merge request summaries, root cause analysis |
| **Claude (Anthropic)** | AI assistant for complex tasks | Writing, research, architecture discussions, long-form documentation |
| **gitlab-mcp** | MCP server connecting GitLab to AI tools | Giving AI assistants live access to GitLab projects, issues, MRs, and pipelines |

As GitLab's developer experience team, we prioritize GitLab Duo as our primary AI tool — both because it's our product and because using it as customer zero gives us direct feedback to improve it for all engineers.

## Guidelines and Principles

### What We Encourage

- **Dogfooding GitLab Duo first.** When an AI task can be done with GitLab Duo, use it. This generates real usage data, surfaces friction points, and creates feedback we can act on.
- **AI-assisted review, not AI-replaced review.** Use AI to help identify issues faster, not to skip human judgment on correctness or security.
- **Sharing what works.** When you find a useful prompt, workflow, or technique, document it here or share in our team channel so the whole team benefits.
- **Transparency with collaborators.** When AI meaningfully contributed to a deliverable (document, design, code), note it so collaborators understand the context.

### What We Avoid

- **Committing AI-generated code without review.** All code, AI-assisted or not, goes through standard review. AI output can be confidently wrong.
- **Sharing confidential data with external AI tools.** Do not paste internal customer data, unreleased feature details, or other confidential information into AI tools outside GitLab's boundaries. Follow [GitLab's AI acceptable use policy](/handbook/legal/acceptable-use-policy/).
- **Over-relying on AI for security-sensitive work.** Security decisions require human expertise. Use AI as a starting point, not the final word.

## Workflows

### Code Review Assistance

We use GitLab Duo Code Review to get an AI-generated summary and initial feedback before assigning human reviewers. This helps reviewers focus on higher-level concerns rather than mechanical issues.

**How we use it:**

1. Open a merge request and trigger Duo's code review summary
2. Address any straightforward issues flagged (typos, obvious style issues)
3. Assign human reviewers with the AI summary available for context

### Test Generation

For new features and bug fixes, we use AI to accelerate test scaffolding — generating initial test cases from a function signature or spec, which we then refine.

**Useful prompts:**

- "Generate RSpec unit tests for this method, covering edge cases for nil inputs and boundary values"
- "Given this failing test output, suggest what the root cause might be and how to fix it"

### Documentation Drafting

For handbook pages, runbooks, and design documents, we use AI to draft structure and boilerplate, then refine with team-specific knowledge and decisions.

### Incident Analysis

During and after incidents, we use GitLab Duo's root cause analysis feature on failing pipelines and use AI assistants to help parse large log outputs and identify patterns.

### Codifying Standards with MR Review Instructions

GitLab Duo's [custom MR review instructions](https://docs.gitlab.com/user/gitlab_duo/customize_duo/review_instructions/) let you define project-specific standards that Duo automatically enforces on every merge request review. Rather than relying on reviewers to catch the same patterns repeatedly, you write the rules once and Duo applies them consistently.

**How it works:**

Create a `.gitlab/duo/mr-review-instructions.yaml` file in your repository. Duo reads it whenever it reviews an MR and appends your custom rules to its standard review criteria. When it finds a violation, it comments: _"According to custom instructions in '[rule name]': [feedback]"_.

**Example — enforcing DevEx conventions:**

```yaml
instructions:
  - name: Ruby conventions
    fileFilters:
      - "**/*.rb"
      - "!spec/**/*.rb"
    instructions: |
      1. All public methods must have Sorbet type signatures
      2. Prefer keyword arguments for methods with 3 or more parameters
      3. Do not use `rescue Exception` — rescue specific error classes only

  - name: Test quality
    fileFilters:
      - "spec/**/*.rb"
    instructions: |
      1. Every new spec must have a description that reads as a sentence
      2. Avoid `allow_any_instance_of` — use proper doubles or dependency injection
      3. Flag any test that makes real network calls without stubbing

  - name: CI configuration
    fileFilters:
      - ".gitlab-ci.yml"
      - ".gitlab/ci/**/*.yml"
    instructions: |
      1. New jobs must define a `stage` explicitly
      2. Do not hardcode environment-specific values — use CI/CD variables
```

**What to codify:**

Good candidates for MR review instructions are standards that are:

- Frequently raised in human reviews but easy to miss
- Project-specific (not covered by linters or existing tooling)
- Clear enough to be checked mechanically (avoid vague rules like "write clean code")

**Tips:**

- Use `fileFilters` to scope rules to the right files — broad rules applied everywhere generate noise
- Number your instructions so Duo's feedback references them clearly
- Start with a small set of high-value rules and expand over time based on what human reviewers still catch
- Check the [GitLab Duo MR review instructions docs](https://docs.gitlab.com/user/gitlab_duo/customize_duo/review_instructions/) for the full YAML reference

### Baking Context into Repositories with CLAUDE.md and AGENTS.md

AI coding agents work best when they understand your project's conventions up front. Rather than explaining your codebase in every conversation, you can commit instruction files that agents automatically pick up whenever they work in your repository.

**CLAUDE.md** is read by Claude Code at the start of every session. Place one in the root of a repository and Claude will load it before doing any work — no prompting required. You can also add CLAUDE.md files in subdirectories; they load on demand when Claude reads files in that directory, which is useful in monorepos.

**AGENTS.md** is an [open standard](https://agents.md/) supported by multiple tools including OpenAI Codex, Cursor, and Windsurf. It serves the same purpose as CLAUDE.md but for a broader set of agents. If you want agent instructions that work across tools, AGENTS.md is the better choice.

Both files can coexist in the same repository without conflict.

**What to include:**

```markdown
# Commands
- Run tests: `bundle exec rspec`
- Lint: `bundle exec rubocop`

# Code Style
- Prefer keyword arguments for methods with 3+ parameters

# Repository Structure
- Feature code lives in `app/`, specs mirror this in `spec/`
- Shared test helpers go in `spec/support/`

# Workflow
- Branch naming: `<type>/<issue-id>-short-description`
- Always link MRs to an issue
- Never commit directly to the default branch (`main` or `master`)

# Off Limits
- Do not modify files in `db/migrate/` unless explicitly asked
- Do not change `.gitlab-ci.yml` without checking with the team
```

**Tips:**

- Keep files under ~200 lines — agents give less weight to long files
- Include the actual commands to run, not just tool names
- Use an "Off Limits" or "Never touch" section to prevent agents from modifying sensitive files (migrations, CI config, secrets)
- For monorepos, put a root-level file with shared conventions and per-package files for package-specific rules

**Generating a starting point with Claude Code:**

Run `/init` in Claude Code from the root of your repository and it will analyse your codebase and generate a CLAUDE.md draft you can refine.

### GitLab MCP Server

The [GitLab MCP server](https://docs.gitlab.com/user/gitlab_duo/model_context_protocol/mcp_server/) implements the [Model Context Protocol](https://modelcontextprotocol.io/) (MCP), a standard that lets AI assistants connect to external tools and data sources. With it configured, an AI tool like Claude Code can directly read issues, merge requests, pipelines, and project data from GitLab without you copying and pasting context manually.

**What it enables:**

- Ask your AI assistant about open issues or MR status without leaving your editor
- Have the AI create issues, comment on MRs, or trigger actions based on a conversation
- Provide the AI with live project context when debugging or planning work

**Setting it up**

GitLab's MCP server is available at `https://gitlab.com/api/v4/mcp` and uses OAuth for authentication. The HTTP transport is the recommended approach — no extra dependencies required.

_Claude Code:_

```shell
claude mcp add --transport http GitLab https://gitlab.com/api/v4/mcp
```

_Claude Desktop:_ Add the following to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "GitLab": {
      "type": "http",
      "url": "https://gitlab.com/api/v4/mcp"
    }
  }
}
```

After adding the server, you'll be prompted to authorise via browser-based OAuth on first use.

**glab mcp (alternative)**

The GitLab CLI also ships an experimental MCP server via `glab mcp serve`, which exposes similar GitLab functionality via the stdio transport. This is useful if you prefer running the server locally or need to connect to a self-managed instance. Note this feature is marked experimental and may change.

## Experimentation and Dogfooding

As a DevEx team, we have a unique responsibility to experience GitLab's AI features as a typical engineering team would. This informs both our own roadmap and our feedback to GitLab product teams.

### How We Dogfood

- We use GitLab Duo in our day-to-day work and file issues for friction points or missing functionality
- When a new Duo feature ships, we aim to try it on real work within the sprint and share findings in our team retrospective

## Resources

- [AGENTS.md open standard](https://agents.md/)
- [GitLab Duo documentation](https://docs.gitlab.com/ee/user/gitlab_duo/)
- [GitLab MCP server documentation](https://docs.gitlab.com/user/gitlab_duo/model_context_protocol/mcp_server/)
- [glab mcp documentation](https://docs.gitlab.com/cli/mcp/)
- [GitLab AI acceptable use policy](/handbook/legal/acceptable-use-policy/)
- [GitLab's approach to AI](/handbook/product/ai/)
