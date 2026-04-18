---
title: "Top Tips"
description: "Practical tips for getting the most out of AI tools in your day-to-day DevEx workflow"
---

Practical workflows the DevEx team has found genuinely useful. Add your own when you discover something worth sharing.

## Use Glean to Search Code and Prioritise Your Day

[Glean](https://www.glean.com/) is GitLab's enterprise search tool, connected to our internal systems including GitLab, Slack, Google Drive, and the handbook. Its AI assistant can answer questions across all of these sources in one place.

**Search for code and context:**

Rather than jumping between GitLab, Slack threads, and the handbook, ask Glean directly:

> "Where is the RSpec helper for mocking Gitaly calls?"
> "What was the decision behind how we structure CI job definitions in the main repo?"
> "Find the runbook for broken main branch incidents"

Glean searches across connected sources and surfaces the most relevant results with links.

**Ask for your top priorities:**

At the start of the day, ask Glean's AI assistant to summarise what needs your attention:

> "What are my top priorities today based on my open issues and recent Slack mentions?"
> "Summarise any unread mentions or threads from yesterday that I should follow up on"
> "What issues am I assigned to that have had recent activity?"

**Tips:**

- Glean works best for finding things you know exist but can't locate quickly — use it instead of searching GitLab and Slack separately
- Ask follow-up questions in the same conversation to refine results
- For handbook content, Glean is often faster than the built-in handbook search

## Create GitLab Issues from a Conversation

With the [GitLab MCP server](/handbook/engineering/infrastructure-platforms/developer-experience/ai/#gitlab-mcp-server) connected, you can ask Claude to create issues directly in GitLab without leaving your editor or browser.

**Example prompt:**

> "Create a GitLab issue in `gitlab-org/gitlab` titled 'Investigate flaky spec in pipeline_spec.rb'. Label it `type::bug` and `team::devex`. Set the description to include steps to reproduce and link to this pipeline run: [URL]."

Claude will call the MCP tool, create the issue, and return a link. Useful for capturing bugs, tasks, or follow-ups mid-flow without context-switching to the GitLab UI.

**Tips:**

- Specify the project path (`namespace/project`) explicitly — Claude won't guess it
- You can ask Claude to include structured content in the description (steps to reproduce, acceptance criteria, links) by describing it in your prompt
- Works well at the end of a debugging session: "Summarise what we found and create an issue to track the fix"

---

## Fetch MR Comments and Address Them in VS Code

When you have review feedback on a merge request, you can use Claude Code (with the GitLab MCP server) to pull the comments and work through them without leaving VS Code.

**Workflow:**

1. Open Claude Code in VS Code (`Ctrl+Shift+P` → `Claude Code`)
2. Ask Claude to fetch the MR comments:
   > "Fetch the unresolved review comments on MR !12345 in `gitlab-org/gitlab`"
3. Claude retrieves the comments and lists them in context
4. Work through them together:
   > "Address the comment on `app/models/pipeline.rb` line 42 — the reviewer is asking us to extract this into a separate method"
5. Claude makes the edit, you review the diff, and repeat for the next comment

**Tips:**

- Ask Claude to summarise the comments first before diving in: "Give me a grouped summary of the feedback by theme"
- For straightforward nits (style, naming, typos), you can batch them: "Address all the nit comments in one pass"
- For more complex feedback, work one comment at a time and verify each change before moving on
- Once done, you can ask Claude to post a reply on the MR confirming the comments have been addressed

---

## Enforce Team Standards Automatically with MR Review Instructions

Instead of leaving recurring review comments by hand, codify your standards in `.gitlab/duo/mr-review-instructions.yaml` and let GitLab Duo enforce them on every MR automatically.

**Quick setup:**

Create the file in your repository:

```yaml
instructions:
  - name: My team standards
    fileFilters:
      - "**/*.rb"
    instructions: |
      1. Public methods must have Sorbet type signatures
      2. Avoid `rescue Exception` — rescue specific error classes
      3. Do not leave debugging output (puts, pp, binding.pry) in production code
```

Duo will include these checks whenever it reviews an MR touching matching files, and will comment with a reference to the rule name when it finds a violation.

**Good rules to start with:**

- Patterns your team raises in almost every review ("don't forget the type sig")
- Things linters don't catch (semantic or architectural conventions)
- Common mistakes specific to your codebase

See the MR review instructions section of the [main AI page](../) for a fuller example and the complete YAML reference.

## Use AI for an Initial MR Review Before Human Review

Before assigning reviewers, run your MR through an AI review pass to catch issues early — logic gaps, missing tests, unclear naming, style inconsistencies. This makes the human review faster and more focused on higher-level concerns.

**Workflow in Claude Code:**

1. With your branch checked out, ask Claude to review the diff:
   > "Review the changes in this branch against main. Focus on correctness, test coverage, and anything a reviewer is likely to flag."
2. Claude reads the diff and provides structured feedback
3. Address any issues it raises before assigning human reviewers

Additionally, use [GitLab Duo Code Review](https://docs.gitlab.com/ee/user/project/merge_requests/duo_code_review.html) directly on the MR for an automated first-pass review comment.

**Tips:**

- Give Claude a focus area if relevant: "Pay particular attention to the database queries — I want to avoid N+1s"
- Ask it to check for missing edge cases: "Are there scenarios this change doesn't handle that the tests don't cover?"
- Treat the AI review as a checklist, not a gate — use your judgement on what to act on
- If the MR is large, ask Claude to prioritise: "List the three most important things a reviewer would raise"

---

## Fix Failing Pipelines with AI Assistance

When a pipeline fails, AI can help you move from "something is broken" to "here's the fix" faster by analysing logs, identifying root causes, and suggesting or applying fixes directly.

**Workflow in Claude Code:**

1. Grab the failing job log — either paste it directly or, with the GitLab MCP server connected, ask Claude to fetch it:
   > "Fetch the log for the failing job in pipeline #12345 in `gitlab-org/gitlab`"
2. Ask Claude to diagnose the failure:
   > "Analyse this CI log and identify the root cause of the failure"
3. If it's a code issue, ask Claude to fix it:
   > "Apply the fix to the relevant file"
4. If it's a flaky test or environment issue, ask Claude to explain and suggest next steps

**GitLab Duo Root Cause Analysis:**

For pipeline failures on GitLab.com, use [GitLab Duo Root Cause Analysis](https://docs.gitlab.com/ee/user/gitlab_duo/use_gitlab_duo_chat_in_cicd.html) directly in the pipeline UI. Click the failed job, then **"Root cause analysis"** to get an AI-generated explanation without leaving the browser.

**Tips:**

- For long logs, ask Claude to focus: "Look for the first error in this log and ignore subsequent failures that are likely downstream"
- If the fix isn't obvious, ask for hypotheses: "What are the three most likely causes of this failure?"
- For recurring failures, ask Claude to check git history for context: "Has this test file changed recently in a way that could explain this failure?"
- After fixing, ask Claude to check for similar patterns elsewhere: "Are there other tests in this file that could fail for the same reason?"

## Generate Well-Formed Commit Messages

Poorly written commit messages make `git log` and `git blame` much less useful. Use AI to generate commit messages that are accurate, consistently formatted, and follow [conventional commit](https://www.conventionalcommits.org/) style.

**With Claude Code**, you can ask directly:

> "Write a commit message for the staged changes. Use conventional commit format with a short subject line and a brief body explaining the why."

Claude reads your staged diff and produces a message grounded in what actually changed — no more "fix stuff" commits.

**Bake your commit format into AGENTS.md:**

Rather than specifying your format on every prompt, add it to your repository's `AGENTS.md` or `CLAUDE.md` so all agents pick it up automatically:

```markdown
# Commit Messages
- Use conventional commit format: `type(scope): short description`
- Keep the subject line under 72 characters
- Use imperative mood ("add feature", not "added feature")
- Include a body when the change needs context beyond the subject line
- Valid types: feat, fix, chore, docs, refactor, test, ci
- Always link to the relevant issue number in the body if one exists
```

Once this is in your repo, Claude Code will follow these rules without being asked — including when generating commit messages mid-conversation.

**Tips:**

- If the diff touches multiple concerns, ask Claude to flag it: "Does this diff represent a single logical change, or should it be split into separate commits?"
- For squash commits before merging, ask for a summary commit message: "Summarise all the commits on this branch into a single conventional commit message"
- Review the generated message — Claude bases it on the diff, but you know the intent better than it does
