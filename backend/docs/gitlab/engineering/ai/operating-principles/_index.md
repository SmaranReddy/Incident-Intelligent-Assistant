---
title: Operating Principles
description: "The Operating Principles for the AI Engineering groups. These principles define how we work together to deliver reliable features under tight timelines without sacrificing long-term maintainability."
---

## Purpose

Improve stability, predictability, and quality of the DAP codebase while implementing new functionality in a timely manner. These principles define how we work together to deliver reliable features under tight timelines without sacrificing long-term maintainability.

<details markdown=1>
<summary>Short Summary of the Operating Principles</summary>

1. **Reviews and Merge Discipline**

    * Every MR gets two reviewers from the relevant specialty. No single-review merges unless it’s an S1-level situation.

2. **Quality Over Assumptions**

    * If something looks wrong or unclear, stop and ask. Pull in help rather than silently assuming it’s fine.

3. **Test Coverage Enforcement**

    * Coverage jobs must pass for FE and BE changes. No exceptions.

4. **No Pressure During the Reviewer Process**

    * Don’t rush or nudge reviewers with “we need this ASAP,” once the review process has started. If a deadline can’t hold, escalate instead of lowering review quality.

5. **Shift from Speed to Stability**

    * We’re shifting from rush to stability. Small, clean, minimal changes beat large, risky ones.

6. **Early Signal on Risks**

    * Raise delays, scope creep, unclear requirements, or blockers as soon as you spot them.

7. **Strict AI-Assisted Coding Discipline**

    * AI is a helper, not an authority. Keep implementations simple. Understand every line you merge.

8. **High-Bandwidth Communication**

    * If you foresee complexity, can’t review, or see improvements worth making, speak up immediately.

9. **Backwards Compatibility**

    * Changes outside of the GitLab Instance, e.g. in the AI gateway or editor extensions, should still follow the backwards compatibility guidelines.

</details>

## Principles

1. Reviews and Merge Discipline

    * Up until the code is considered stabilized (subject to another evaluation), all Merge Requests require at least two reviewers from the relevant specialty (BE or FE).
    * No single-review merges unless explicitly approved for high-severity (S1) fixes. Every MR has to have 2 approvals for the main functional discipline(s) of the MR (frontend and/or backend). Examples:
      * if an MR requires both frontend and UX reviews, it's expected to have 2 frontend reviews and 1 UX ([for MRs from teams that include a Product Designer](https://docs.gitlab.com/development/code_review/#reviewer-roulette))
      * if an MR requires both backend and frontend reviews it is expected to have 2 frontend and 2 backend reviews

2. Quality Over Assumptions

    * If something in an MR looks incorrect, unclear, or suspicious, stop and escalate. Ask questions, start a thread, or bring key product or engineering stakeholders into the conversation when in doubt.
    * Never merge code you don’t fully understand or can’t defend.
    * Test code you review locally at all times and test again if late changes were made.

3. Test Coverage Enforcement

    * Coverage jobs must run for every MR.

4. No Pressure During the Reviewer Process

    * Do not pressure reviewers to “review quickly,” “merge ASAP,” or “just approve so we can move forward” after they have started the review.
    * Time pressure is never a justification for lowering review quality.
    * If the timeline genuinely cannot hold, escalate through technical leadership or engineering management instead of placing that pressure on individual reviewers.
    * Reviews must happen at a pace that preserves clarity, safety, and correctness.
    * It is acceptable, though, to nudge an assigned reviewer/maintainer if the review hasn’t been started in a timely manner, as defined in our documentation.

5. Shift from Speed to Stability

    * The early phase prioritized shipping volume.
    * From now on, the priority is robustness, clarity, and correctness.
    * "Less is more" applies: small, focused, reliable changes are better than broad, risky ones.

6. Early Signal on Risks

    * Communicate risks as soon as they appear:
      * delays
      * scope creep
      * unclear requirements
      * dependencies blocking progress
    * Surfacing uncertainty early prevents compounding failures later.

7. Strict AI-Assisted Coding Discipline

    * AI is a tool, not an authority.
    * When using AI for code:
    * Ask for straightforward, explicit, simple implementations.
    * Validate every line produced.
    * Challenge unclear or magical logic.
    * Only push to a branch what you personally understand end-to-end and agree with.

8. High-Bandwidth Communication

    * If you foresee complexity in your task: speak up.
    * If you can’t review an MR promptly: speak up.
    * If you notice opportunities for simplification or improvement: speak up.
    * Silent uncertainty is more harmful than a wrong initial assumption.

9. Backwards compatibility

    * Changes outside of the GitLab Instance, e.g. in the AI gateway or editor extensions, still need to comply with our general [Backwards Compatibility guidelines](https://docs.gitlab.com/development/multi_version_compatibility/) for GA features
    * Beta features must be backwards compatible with at least the last 3 minor versions of GitLab
    * If in doubt, verify with an older version of a self-managed instance, that your change does not break backwards compatibility. You can get access to dedicated instances [here](https://gitlab.com/gitlab-com/gl-infra/gitlab-dedicated/sandbox/bmckitrick/dap-test-admin#using-it).
