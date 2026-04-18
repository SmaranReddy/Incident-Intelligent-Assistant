---
title: "Duo Agent Platform ADR 007: Support workspace Agent Skills"
status: proposed
creation-date: "2026-02-27"
authors: [ "@erran" ]
coach: [ ]
approvers: [ ]
owning-stage: "~devops::ai_powered"
participating-stages: []
toc_hide: true
---

## Context

The Agent Skills [(specification)](https://agentskills.io/specification) defines a reference structure for defining
skills that can be injected into an agent's context window through progressive disclosure.

Agent Skills are emerging as a lightweight industry standard for defining reusable agent functionality, particularly
for local development environments and CI/CD workflows.

Skills can be read from the current workspace, a remote repository, or through a remote skills catalog/marketplace.

## Problems

Without creating an official integration users must define custom instructions using `.gitlab/duo/chat-rules.md`
or `AGENTS.md` which instructs the LLM to read the entirety of `SKILL.md` files into a chat session's context window
through `read_file` or other tools to fetch remote skills.

Skills aren't consistently loaded into the context window so the experience varies of when the LLM is able to act on
them.

## Goals

- Support skills for up-to-date Duo CLI, Editor Extensions, and Remote Flows users for all Duo deployments.
- Preserve existing Duo security guardrails the reading SKILL.md files into context.
- Trigger attestation if trust policy and signed sigstore bundles are present for agent instructions.

### Backwards Compatibility

Customers should be able to use skills without requiring coordinated upgrades across:

- GitLab instance
- Duo Workflow Service
- AI Gateway
- Duo CLI or Editor Extensions

Self-managed and Duo self-hosted customers can opt out of skills without upgrading infrastructure.

This avoids introducing new interlock dependencies between GitLab components.

## Non-Goals

- Add skills to the AI catalog
- Add personal skills projects
- Add group-level skills projects
- Dynamic discovery of skills added after a session has started
- Provide enterprise-managed configuration — specifically a trust policy for agent instructions (handled in ADR-009)

## Decision

1. Support progressive disclosure of Agent Skills by traversing all workspace skills and including their SKILL.md YAML
   frontmatter as additional context.
1. Leverage existing tools to support existing GitLab Dedicated/Self-Managed and Duo self-hosted deployments.
1. Recommend mitigation of context poisoning risk through [Provenance](https://slsa.dev/spec/v0.2/provenance) and
   [Attestation](https://github.com/in-toto/attestation).
1. Support enterprise-managed policy for Duo CLI, IDE, and Remote Flows container images.

## Pros

- Support automatic and progressive disclosure of skills
- Leverages existing prompt and tool guardrails for `read_file` and `run_command`.
- Supports attestation of agent instructions through sigstore bundle(s)
- Removes the dependency on Duo Workflow Service and AI Gateway backend changes for a new additional context type

## Cons

- Introduces a small number of additional tokens on startup
- Skills content is loaded on-demand and relies on LLM deciding to load them based on agent skills metadata in context
- Provenance and attestation adds operational overhead for keyless signing in GitLab instances that aren't publicly
  accessible.
- Provenance support for workspace skills is not directly transferable to remote project skills

## Threat Model

Existing prompt injection detection and tool responses filtering for read_file and run_command mean we're already using
the existing human-in-the-loop approval experience available to Agentic Chat users.

SKILL.md files can already be read today if a user requests it manually or through other dynamically loaded
agent instructions files.

## Next Steps

1. Support managed trust policy for agent instructions for enterprise deployments.
1. Support fetching skills defined outside the current workspace.
1. Support updating skills additional context after a session has started.
