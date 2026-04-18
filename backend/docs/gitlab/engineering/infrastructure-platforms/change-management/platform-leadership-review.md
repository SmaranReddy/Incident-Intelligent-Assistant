---
title: "Platform Leadership Review Guidelines"
---

## Purpose

This page provides guidelines for Platform Leadership reviewers when approving C1 and C2 change requests. The goal is to ensure consistency in reviews and improve the quality of high-criticality changes.

## Who Can Approve

Platform Leadership approval (`platform_leadership_approved` label) can be provided by both ICs and EMs in Infrastructure Platforms. The members who are eligible are:

- All Engineering Managers EM+
- Staff+ ICs (SREs and backend engineers who are Staff+ including Principal)

The reviewer should be from a different team than the change author. This ensures an independent perspective and helps catch issues that may be overlooked by those close to the work.

The list of those who can approve are in the `@gitlab-org/saas-platforms/change-review-leadership` group.
Mention this group on the change request issue to get an approval.

## Review Guidelines

When reviewing a C1 or C2 change request, verify the following:

### 1. Pre-Production Validation

The motivation for why the change is needed should be clear from the description, with a link to an issue for further details.

The same change plan should have been executed in a non-production environment using the same steps.

- Confirm the change has been tested in staging or another non-production environment
- Verify the test environment execution used identical steps to those proposed for production
- Check that any issues discovered in non-production testing have been addressed

### 2. Rollback Plan Quality

The rollback plan should be documented and detailed enough to be executed by any SRE.

- Verify the rollback plan is explicitly documented in the change request
- Ensure rollback steps are specific and actionable (not vague statements like "revert the change")
- Confirm an SRE unfamiliar with the change could execute the rollback without additional context
- Check that rollback time estimates are provided and realistic

### 3. Monitoring and Validation

Monitoring links should be provided as well as explicit change steps for checking monitoring and what to validate if applicable.

- Verify relevant monitoring dashboard links are included
- Confirm the change plan includes explicit steps for checking monitoring during and after execution
- Check that success criteria are defined (what does "this change worked" look like?)
- Ensure there are clear indicators for when to trigger the rollback plan

## Applying the Label

Once you have verified the change request meets all the above criteria, apply the `platform_leadership_approved` label to the issue.

If the change request does not meet the criteria, provide specific feedback on what needs to be improved before approval can be granted.
