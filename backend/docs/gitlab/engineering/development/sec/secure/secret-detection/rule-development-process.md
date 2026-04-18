---
title: Rule Development Process
---

## Overview

The Secret Detection rule development process primarily involves creating an appropriate regex pattern to identify a particular secret credential type. In addition, we also need to gather additional information such as rule keywords, validation regex, and remediation guides. This document walks you through the entire process.

## What does a rule contain?

A rule for a particular credential type is defined in a TOML file and contains the following properties:

- **`id`**: A unique identifier for the rule (e.g., `AWSSecretKey`).
- **`regex`**: The primary regular expression pattern used to detect the secret.
- **`keywords`**: (Optional) A list of keywords that help pre-filter content before applying the regex, improving performance.
- **`validationRegex`**: (Optional) A secondary regex used to validate the context around a detected secret, reducing false positives.
- **`examples`**: A list of up to five example strings that the `regex` should match.
- **`assignmentContext`**: (Optional) Configuration for auto-generating assignment-like regex patterns. This includes:
  - **`keywordGroup`**: A comma-separated string of keywords and groups of keywords to be used on the Left Hand Side (LHS) of an assignment.
  - **`operator`**: (Optional) The operator used in the assignment (e.g., `=`, `:`, ` `). Defaults to `[:=]`.
  - **`terminalRegex`**: (Optional) A regex pattern to match the end of the secret's usage in an assignment-like pattern.
- **`tags`**: (Optional) A list of tags to categorize the nature of the rule (e.g., `gitlab_blocking` for enabling rule for Secret Push Protection).

Here's an example of a rule definition for `Stripe Live Secret Key`:

```toml
[[rules]]
id = "StripeLiveSecretKey"
keywords = ["sk_live_"]
regex = '''\bsk_live_[A-Za-z0-9]{99}\b'''
validationRegex = "(?i)stripe"
examples = [
  "sk_live_...",
  ...
]
negativeExamples = [
  "sk_live_...",
  ...
]
tags = ["gitlab_blocking"]
```

Refer [rule schema](https://gitlab.com/gitlab-org/security-products/secret-detection/secret-detection-rules/-/blob/main/rules.schema.json?ref_type=heads) for more details about rule properties.

The directory structure inside [`secret-detection-rules/rules/mit`](https://gitlab.com/gitlab-org/security-products/secret-detection/secret-detection-rules/rules/mit?ref_type=heads) is arranged as `<company>/<company.toml>`, where the `<company>.toml` file contains all rule definitions for the corresponding credential types of that company. In the same company directory, remediation guides for all credential types of that company are placed.

## Development Process

At a high level, these are the typical steps for introducing a new rule for a secret credential type:

1. **Prerequisites:**
    - Confirm the secret credential type's existence and exposure risk
2. **Regex Generation:**
    - Develop an appropriate regex pattern matching the credential type
    - Verify the pattern's correctness in the public source code repositories
    - Identify false positive matches and take steps to reduce them
3. **Additional Rule Properties:**
    - Identify unique keywords in the regex pattern to set as rule keywords
    - Pull up to five matching samples from public sources to use as rule examples
    - Identify relevant keywords like company/product names associated with the rule to use in validation regex
4. **Remediation Guide:**
    - Generate a remediation guide for customers to follow in the event of a secret breach

Let's examine each step in detail.

### 1. Prerequisites

Before creating any new rule, it's essential to verify the credential's existence and actual presence in the wild. A typical method is to search for the credential in official documentation sources. For example, searching for "Google Cloud Service Account" in Google's official documentation leads to a page titled [Creating and managing service accounts](https://cloud.google.com/iam/docs/service-accounts-create), where you can view a sample Service Account key in JSON format.

Additionally, it's important to identify the credential's exposure risk, as this helps us define the rule severity. While the exposure risk isn't clearly defined in most documentation, you can infer it based on the credential's use case and security implications.

### 2. Regex Generation

This is the most crucial step in the entire process. A well-crafted regex pattern ensures high accuracy and minimizes false positives. Here are guidelines and tools to help you generate effective regex patterns:

- **Specificity is Key**: Aim for regex patterns that are as specific as possible to the secret credential type. Avoid overly broad patterns that might match common strings.
- **Regex Compatibility**: Ensure your regex is compatible with our regex engine ([Hyperscan](https://github.com/intel/hyperscan)). Review the [supported features](https://rbuckton.github.io/regexp-features/engines/hyperscan.html) when generating your regex.
- **Word Boundaries**: Use word boundaries (`\b`) where appropriate to ensure the regex matches the entire secret rather than a subset of a larger string. The word boundary character matches any character that is not alphanumeric or underscore (`_`), so appending it at the end of the regex usually helps. However, be aware that it may produce false negatives for some secret types because it might not consider punctuation like `:`, `,`, or other symbols.
- **Grouping**: Use non-capturing groups (`(?:)`) to logically group patterns and improve readability. There can be at most one capturing group in the regex.
- **Escaping Special Characters**: Escape special regex characters that might appear in the secret credential type.
- **Character Classes**: Always define explicit characters (`[a-zA-Z0-9]`) instead of using character classes (`\d`, `\w`, etc.) to match specific types of characters expected in the secret.
- **Quantifiers**: Define fixed or bounded range quantifiers (`{n}`, `?`, or `{n,m}`) instead of open-ended ones (`*`, `+`, `{n,}`) to specify the number of occurrences of a character or group. This helps avoid the risk of over-permissive matches.
- **Negative Lookaheads/Lookbehinds**: The use of lookaheads/lookbehinds is not allowed. Though these can be powerful for excluding specific patterns or contexts, they're not supported by our regex engine.
- **Common Secret Formats**: Many secrets follow common patterns (e.g., `prefix_alphanumeric_base64string`). Familiarize yourself with these to build robust patterns.

**Note**: Ensure your regex is compatible with the Hyperscan engine's [regex constraints](https://rbuckton.github.io/regexp-features/engines/hyperscan.html).

#### Tools for Regex Generation and Testing

1. **Regex101**: A popular online regex tester that provides detailed explanations of your regex, highlights matches, and allows you to test against sample text. It supports various regex flavors.
2. **Regex-based Code Search**:  Use [Grep App](https://grep.app) or other source code management systems with regex-based search capabilities to find patterns across repositories.

#### Assignment-Like Regex Pattern

Certain credential types have unique keywords in their pattern that help match the intended secret and minimize false positive matches. For example, GitLab PAT has `glpat-` at the beginning of the pattern.

However, many credential types lack such unique patterns, so their regex becomes generic or overlaps with standard formats like UUID or base32/64 encoded strings, matching random strings. In such cases, we leverage the context around the secret to confirm its existence. Currently, we check for an assignment-like pattern:

```plaintext
<Left Hand Side (LHS)><Operator><Right Hand Side (RHS)><Terminal regex>
```

- RHS should contain only the secret value (e.g., GitLab PAT pattern).
- LHS should contain a company/product name relevant to the credential type.
- Operator could be a colon (`:`), equal sign (`=`), etc.
- Terminal regex should be an appropriate pattern matching the end of the secret's usage to avoid false positive matches in assignment-like patterns.

Let's take the example of a `DataDog API Key` credential type. The regex pattern for this credential type could be: `[0-9a-f]{32}`, which is a generic hexadecimal string pattern. To ensure this pattern matches only DataDog's API Key, we check for an assignment-like pattern around this string with `dd_api_key`, `datadog_api_key`, and other relevant variable names and casing combinations on the LHS and `[0-9a-f]{32}` on the RHS. This results in increased rule accuracy.

#### Auto Assignment-Regex Generation

While `dd_api_key` or `datadog_api_key` cover only languages that adhere to this particular naming convention (i.e., lower snake case), several questions arise:

- What about naming conventions for other languages and configuration file formats?
- What about casing variants supporting other languages and configuration file formats?
- What if we want to improve the assignment pattern across all rules with minimal effort?

To simplify this process, we introduced a utility that auto-generates assignment-like regex patterns for a given set of inputs.

**Example**: For the [DataDog API Key](https://gitlab.com/gitlab-org/security-products/secret-detection/secret-detection-rules/-/blob/main/rules/mit/datadog/datadog.toml?ref_type=heads#L4) rule, all we need to do is define `assignmentContext.keywordGroup` in the rule definition:

```toml
# id = "DataDogAPIKey"
# regex = '''[0-9a-f]{32}\b'''
...
assignmentContext.keywordGroup = "datadog|dd,api,key"
...
```

The `assignmentContext.keywordGroup` value indicates there can be either `datadog` or `dd`, followed by `api` and `key` in the same order on the LHS. It will generate all possible naming and casing combinations of regex literals on the LHS, subsequently generating an entire assignment-like regex pattern like:

```plaintext
(?:DATADOG_API_KEY|datadog_api_key|datadogApiKey|datadog-api-key|datadog\.api\.key|DatadogApiKey|Datadog-Api-Key|DD_API_KEY|dd_api_key|ddApiKey|dd-api-key|dd\.api\.key|DdApiKey|Dd-Api-Key).{0,8}[:=].{0,8}([0-9a-f]{32}\b)
```

The rule's `regex` value will be replaced with the above generated regex when the ruleset package is distributed. There are more options for representing the assignment context; you can read more about them in the [rule schema documentation](https://gitlab.com/gitlab-org/security-products/secret-detection/secret-detection-rules/-/blob/main/rules.schema.json?ref_type=heads#L70).

> [!NOTE]
> While assignment-like regex patterns certainly reduce false positive matches, they also cause false negatives for cases where secrets appear directly or in formats other than assignment-like expressions, such as in comments or as function arguments. This is the trade-off when using assignment-like regex patterns.
>
> Use assignment-like regex patterns only when you're confident about the correctness of your regex and you notice many false positive matches in the Code Search search results.

### 3. Additional Rule Properties

Once you have a robust regex pattern, you'll need to gather additional information to complete the rule definition. These properties enhance the accuracy and usability of the secret detection rule.

- **Rule Keywords**: These are unique strings or patterns found within the secret that help identify it. They are used to quickly filter payloads that don't contain the secret, thereby avoiding regex pattern execution.
  - **How to identify**: Look for static prefixes, specific delimiters, or common phrases that are consistently part of the secret. For example, `glpat-` for GitLab Personal Access Tokens.
  - **Note**: Rule keywords are not mandatory for every rule. However, they can significantly improve performance when included. For assignment-like patterns, LHS keywords of the pattern are automatically set as rule keywords.

- **Rule Examples**: Providing concrete examples of the secret credential type is crucial for testing and documentation.
  - **How to gather**: From your Code Search results, select up to five distinct, real-world examples of the secret. Ensure these examples cover different variations or contexts if applicable.
  - **Usage**: These examples validate the rule's effectiveness and serve as clear illustrations for users and developers.

- **Validation Regex**: This regex is used in the rule development process to confirm the validity of a detected secret by matching the secret's context, such as the presence of a keyword in the file or filename. It confirms whether the generated regex pattern is relevant to its credential type by searching for the defined regex in code search results.
  - **How to identify**: Look for company/product names or concepts relevant to the credential type.
  - **Note**: Validation regex is not mandatory for every rule. It's particularly unnecessary for assignment-like patterns where the LHS keywords of the generated pattern represent the context around the secret, overlapping with validation regex logic.

- **Tags**: Currently, tags are mostly used to determine the rule's availablity for a Secret Detection analyzer. Tags are empty by default and every rule is enabled for Pipeline-based Secret Detection without having to specific. Adding `gitlab_blocking` tag enables the rule for Secret Push Protection.

### 4. Remediation Guide

A clear and concise remediation guide is essential for users to follow when a secret is detected. This guide should provide actionable steps to revoke the compromised secret and secure their systems.

- **Revocation Steps**: Detail how to revoke the specific secret type. This usually involves navigating to the service provider's dashboard, finding the credential, and initiating the revocation process.
  - **Example**: For an AWS Access Key, this involves going to the IAM console, selecting the user, and deactivating or deleting the access key.
- **Rotation Steps**: If applicable, provide instructions on how to rotate the secret (generate a new one) after revocation.
- **Best Practices**: Include general security best practices, such as using environment variables, secret management tools, or avoiding hardcoded secrets.

Information on the remediation process is generally available in the company's official documentation. For some credential types, it might be available in other public sources.

#### Remediation Guide Format

```markdown
# <Human-friendly name of the credential type, e.g., AWS Session Token>

## Description

<A brief description of the credential type>

## Remediation

For general guidance on handling security incidents regarding leaked keys, please see the GitLab
documentation on
[Credential exposure to the internet](https://docs.gitlab.com/ee/security/responding_to_security_incidents.html#credential-exposure-to-public-internet).

<Steps to remediate the credential type, e.g., token rotation>
```

#### Remediation Guide File Rules

- The remediation guide file should be in markdown format.
- There will be a dedicated remediation guide file for each rule.
- The remediation guide file should be created in the same directory where the rule file is present.
- The remediation guide for a particular rule should have the same filename as the rule's `id` value.

## Rule Validation and Testing

The secret-detection-rules repository implements a comprehensive testing and validation framework to ensure rule quality, accuracy, and performance. The testing process is automated through CI/CD pipelines and includes multiple validation layers.

### Automated Testing Framework

The repository uses Go-based testing with several test suites that validate different aspects of the rules:

#### Schema Validation Tests

- **JSON Schema Validation**: All rule files are validated against a comprehensive JSON schema that defines required properties (`id`, `regex`), optional properties (`keywords`, `validationRegex`, `examples`, `assignmentContext`, `tags`), and property constraints
- **TOML Format Validation**: Ensures all rule files are properly formatted TOML that can be parsed correctly

#### Core Rule Validation Tests

- **Rule ID Uniqueness**: Ensures no duplicate rule IDs exist across all rule files
- **Required Field Validation**: Verifies all rules have non-empty titles, descriptions, and remediation guides
- **Regex Compilation**: Tests that all `regex` patterns compile successfully with the RE2 engine and have at most one capturing group
- **Assignment Context Validation**: Ensures rules with `assignmentContext` don't have the `gitlab_blocking` tag and have properly defined `keywordGroup`

#### Example and Keyword Validation

- **Keyword Coverage Testing**: Generates random samples using the rule's regex pattern and verifies each contains at least one of the rule's keywords
- **Example Validation**: Tests that all provided examples match the rule's regex pattern and represent realistic secret formats

#### Scenario-Based Testing

The repository includes comprehensive scenario testing that validates rules work correctly in realistic code contexts:

- **Multi-Language Testing**: Tests secrets in Go, PHP, Rust, Shell, and XML contexts
- **Configuration Format Testing**: Validates detection in JSON, YAML, and TOML configuration files
- **Template-Based Generation**: Uses Go templates to generate realistic code scenarios with multiple naming conventions

### CI/CD Pipeline Validation

The pipeline includes:

- **Build and Verification**: Go unit tests, rule compilation, documentation generation, and artifact creation
- **Documentation Linting**: Style checking, link validation, spell checking, and prose quality validation
- **Performance Testing**: Blackbird testing, performance benchmarking, and memory usage validation

### Manual Testing Guidelines

Beyond automated testing, developers perform:

1. **Code Search Validation**: Manual review of search results to identify false positives and negatives
2. **Regex101 Testing**: Interactive testing of regex patterns with sample data

### Quality Gates

Rules must pass all validation criteria before being merged:

- All automated tests must pass
- Schema validation must succeed
- Precision threshold (70%) must be met for Blackbird testing
- Code review approval from team members
- Documentation completeness verification

## Reference

This [Merge Request](https://gitlab.com/gitlab-org/security-products/secret-detection/secret-detection-rules/-/merge_requests/155/) introduces multiple rules into the ruleset
covering different methods (unique prefix, assignment-based, jwt-based) along with
auto-generated remediation guides. You could use this for reference purposes.
