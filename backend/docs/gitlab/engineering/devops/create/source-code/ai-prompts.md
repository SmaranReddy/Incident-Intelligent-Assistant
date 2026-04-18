---
title: "Create:Source Code AI prompts "
description: Common prompts we can use with Duo Chat to be more efficient
---

## From GitLab Discussion to Action

### Use Case / Impact

Generating a detailed issue from a thread/discussion.

Impact: summarizing detailed discussions and capturing them in follow-up issues favour the iteration and time to merge of each individual step.

(See [demo](https://levelup.edcast.com/insights/23921833) on levelup)

### Prompt

Identify the discussion you want to use as source and copy the URL to the top comment of the thread (right-click on the timestamp and copy link).

```markdown
Can you transform the following discussion TOP_COMMENT_LINK into an issue description? Keep it concise and useful for an engineer.
```

## Security vulnerability detection in MRs

### Use Case / Impact

When you're working on an MR and want to check if you missed any security vulnerability, you can use this prompt to identify them.

Impact: This can lead to faster review times and decrease vulnerabilities shipped in our code.

(See [demo](https://levelup.edcast.com/insights/23921833) on levelup)

### Prompt

While on the Merge Request page, drop this prompt in the Agentic Duo Chat:

```markdown
Can you detect any security concerns that are against OWASP's top 10 vulnerabilities in these MR changes? Provide examples.
```

## Decoding Complex MR Feedback

### Use Case / Impact

Sometimes you might get feedback that you don't immediately and fully understand. This prompt can help decode and
deconstruct that feedback to make it easier to understand and react to.

Impact: This can decrease review time by speeding up the discussion between author (you) and reviewer.

(See [demo](https://levelup.edcast.com/insights/23921833) on levelup)

### Prompt

* Grab the link to the comment you want to check.
* Write down your interpretation on what it means.

Use this information to compose a prompt such as:

```markdown
From the reviewers comment LINK_TO_COMMENT, I understand the following YOUR_INTERPRETATION. Does this make sense? Am I missing something?
```

## Debugging legacy code

### Use Case / Impact

One of the most challenging things for our group (group::source code) is we work with legacy code that might be drifted quite apart from our current coding standards. More often than not, fixing something involves touching utils that are shared across the codebase. This broader reach means it takes a lot longer to verify a fix to avoid a regression.

### Prompt

```markdown
Can you give me an overview of the code logic for [COMPONENT_NAME]?
```

```markdown
There is an potential xss issue, can you locate the problematic file/method?
```

```markdown
Can you tell me where we use this method and which product feature it covers? 
```

<!-- Use the template below, copy and paste to add your own:

## PROMPT_TITLE

### Use Case / Impact

REPLACE_WITH_DESCRIPTION

### Prompt

REPLACE_WITH_INTRO_STEPS

```markdown
REPLACE_WITH_PROMPT
```
-->
