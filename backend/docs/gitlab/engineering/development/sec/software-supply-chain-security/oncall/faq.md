---
title: "Tier 2 On-Call FAQ"
description: "Frequently asked questions about the Software Supply Chain Security Tier 2 On-Call program"
---

## APAC Timezone Concerns

**Q: Why is APAC coverage marked as "best effort" only?**

A: The APAC time slot (02:00-10:00 UTC) starts very early for most APAC team members. For example, IST team members would need to be available starting at 4:30am. Given this challenge, APAC coverage is best effort, meaning if an engineer can answer the page they will, or they can follow up once their business day starts.

**Q: What happens if an APAC engineer cannot respond to a page at 4:30am?**

A: There is no penalty for not responding during off-hours. The expectation is that you respond when reasonably possible during your normal working hours. The escalation layer (covered by EMs 24x5) will help bridge any gaps.

**Q: Will APAC team members be expected to take more shifts than other regions?**

A: Currently, APAC team members are assigned approximately 6-7 weeks per person over the 52-week period due to the smaller team size (8 members). However, this is being actively addressed through:

- Onboarding new APAC team members to the rotation
- Exploring Sec cross-stage coverage options to better balance the load

**Q: Can APAC engineers adjust their shift times?**

A: Time slots are fixed and coordinated globally to ensure continuous coverage across all regions. However, the best-effort nature of APAC coverage provides flexibility in when you actually respond during your assigned week.

---

## Coverage for Medical Appointments

**Q: What should I do if I have a medical appointment during my on-call week?**

A: Follow the standard Incident.io override process:

1. Reach out in #sscs-tier2-rotation-coordination to your reporting manager or @adil.farrukh, @ajaythomasinc and @ken.mcdonald as rotation leaders.
2. Coordinate with them or a team member to cover that time slot
3. Notify your manager and the team in the above Slack channel.
4. Document the override in advance (ideally 48+ hours notice for non-emergencies)

**Q: What if I have a medical emergency during my shift?**

A: Your health comes first. If you're unable to respond:

1. If possible, notify your manager or the escalation layer (EMs) immediately
2. The escalation layer will handle coverage
3. Follow up with your manager when you're able to discuss coverage arrangements

**Q: Do I need to use PTO for medical appointments during on-call?**

A: No. Medical appointments during your normal working hours should be handled according to GitLab's standard PTO and flexibility policies. The on-call shift aligns with your regular work schedule, so treat appointments as you normally would.

**Q: What if I have a planned surgery or extended medical leave during my assigned week?**

A: Contact your manager as soon as you know about the conflict. They will:

- Work with the team to swap your week with another team member
- Update the schedule in incident.io
- Ensure there are no coverage gaps

---

## Escalation Procedures When Domain Expertise is Missing

**Q: What if I'm paged for an issue outside my domain expertise (e.g., I'm from Authentication but the issue is Pipeline Security)?**

A: This is expected and acknowledged in the rotation design. Follow this escalation path:

1. **Acknowledge the page** within 15 minutes to establish ownership
2. **Perform initial triage** using available runbooks and playbooks, and codebase investigation
3. **Loop in domain experts** by posting in the relevant team Slack channel
4. **Escalate to the EM layer** if you don't see a response to your slack reqeust in 5 mins
5. **Remain the DRI** (Directly Responsible Individual) for coordination, even if a domain expert takes over technical resolution

**Q: Who is in the escalation layer?**

A: The escalation layer consists of Engineering Managers from the Software Supply Chain Security teams, available 24x5. This includes @mmishaev for non-AMER hours escalation coverage.

**Q: When should I escalate to the EM layer?**

A: Escalate when:

- The issue requires domain expertise you don't have
- The incident severity is S1/S2 and you need additional resources. Please note that S1/S2 already do have IMOCs allocated to the incident.
- You're unsure about the appropriate response or mitigation
- You need help coordinating across multiple teams
- You've been working on an issue for 30+ minutes without progress

**Q: What resources are available to help me during an incident?**

A: You have access to:

- **Runbooks and playbooks** (being expanded as the program evolves)
- **Tier 2 On-Call training** in the Level Up Channel
- **EM escalation layer** (24x5 coverage)
- **Team Slack channels** for domain-specific questions
- **Incident.io documentation** and incident history
- **GitLab handbook** on-call procedures

**Q: Am I expected to fix issues outside my expertise?**

A: No. Your role as Tier 2 is to:

- Provide faster access to domain knowledge than SREs have
- Coordinate and own the incident response, and return to good state with various tools such as FF disablement, rollbacks, workarounds or code changes.
- Leverage your general SSCS knowledge to triage effectively
- Escalate appropriately when specialized expertise is needed

The program acknowledges that not all engineers across Authentication, Authorization, and Pipeline Security are familiar with each other's areas. Runbooks will help fill these gaps over time.

---

## How This Differs from Current Dev-On-Call

**Q: How is Tier 2 different from the current dev-on-call or IMOC?**

A: Key differences:

| Aspect | Current Dev-On-Call/IMOC | Tier 2 SME On-Call |
|--------|--------------------------|-------------------|
| **Scope** | Broad engineering coverage | Domain-specific (SSCS only) |
| **Primary responder** | SREs for GitLab.com | SREs/SIRT are first line (Tier 1) |
| **When you're paged** | For any incident | Only when domain expertise needed |
| **Response time** | 15 minutes | 15 minutes |
| **Coverage** | Mix of business hours pagerslack, and weekend dev-oncall for 24x7 coverage | 24x5 (Monday-Friday only) |
| **Rotation frequency** | Varies | 4-7 weeks per year depending on region |
| **Expertise expected** | General operational knowledge | Deep product/code knowledge |

**Q: Will I still be on IMOC or weekend dev-on-call?**

A: No. Once the Tier 2 rotation starts on January 5, 2026, you will be removed from IMOC or current dev-on-call schedules. You will not be expected to join more than one rotation (unless there is a critical coverage issue that can be discussed on an as-needed basis).

**Q: What types of incidents will I be paged for?**

A: You'll be paged when:

- SREs or SIRT need domain-specific expertise for SSCS-related incidents
- An incident involves Authentication, Authorization, or Pipeline Security components
- Tier 1 has completed initial triage but needs specialized knowledge to resolve
- Customer-impacting issues require deep product/code understanding

You will NOT be paged for:

- Routine operational issues that SREs can handle
- Infrastructure-only problems
- Issues outside the SSCS domain
- Non-critical support issues.

**Q: Do I need to be at my computer for the entire 8-hour shift?**

A: No. The expectation is that you:

- Can respond to pages within 15 minutes
- Have access to your laptop and a stable internet connection
- Are available during your assigned shift hours
- Can join incident calls if needed

This aligns with your normal work schedule, so you can continue your regular work while on-call.

**Q: What happens on weekends?**

A: The current 24x5 rotation does NOT include weekends. Weekend coverage will be addressed in future iterations, subject to Legal, HR, and Works Council approvals.

**Q: How will I know when I'm on-call?**

A: You will:

- See your assigned weeks in the incident.io schedule (created after confirmations)
- Receive incident.io notifications when your shift starts
- Setup calendar invites for your on-call periods
- Have visibility into upcoming rotations in the Incident.io app

**Q: What training is required before my first shift?**

A: Complete the training in the [Tier 2 On-Call Level Up Channel](https://levelup.edcast.com/channel/tier-2-on-call-support/home) as soon as your schedule allows. The training covers:

- Incident response procedures
- Using incident.io and PagerDuty
- Escalation paths
- Runbooks and playbooks (expanding over time)

All team members should complete this before the January 5, 2026 start date.

---

## Additional Questions

**Q: Can I swap shifts with a teammate?**

A: Yes. Coordinate directly with a teammate in your region, then:

1. Update the override in PagerDuty
2. Notify your manager
3. Document the swap in the team Slack channel

**Q: What if I'm on PTO during my assigned week?**

A: Contact your manager as soon as you know about the conflict. They will arrange coverage and update the schedule.

**Q: Who do I contact if I have questions not covered here?**

A: Reach out to:

- Your direct manager for schedule or coverage questions
- @adil.farrukh for program-related questions
- The EM escalation layer during an active incident
- Post in the team Slack channel for general questions

**Q: Will this FAQ be updated?**

A: Yes. As the program evolves and new questions arise, this FAQ will be expanded. Feedback and suggestions for additional topics are welcome.
