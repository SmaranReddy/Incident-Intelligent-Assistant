---
aliases:
  - /handbook/engineering/workflow/development-processes/hiring-contractors/
title: "Hiring Contingent Workers - Development Department"
description: "How the development department hires engineers as contingent workers"
---

GitLab normally prefers to hire full-time permanent engineers for its development teams using entities and [PEOs](/handbook/people-group/employment-solutions/#peo-professional-employer-organization-employer-of-record-and-not-a-gitlab-entity-or-branch). However, there are times when teams will have a need to hire temporary contingent workers to help with execution.

These are a few examples of situations where a contingent workers may be required:

- A team member is on long-term leave and the team needs to back-fill that position for a period of time.
- There is a one-off project that requires timely execution and other team members are busy with higher-level priorities.
- The team may lack very unique expertise or technical specialization and a consultant is needed to help.
- Provide financial support for contributions to an open-source project by an external party.

Whatever the reason, there are some unique challenges and processes related to hiring contracontingent workers, which makes it different from a regular hire.

In the next sections, you can find general information about the process. However, you should consult with your sub-department lead for any variations to this process. You should also read the GitLab [Contingent Worker Policy] (/handbook/finance/procurement/contingent-worker-policy/) as a pre-requisite.

## Planning and budget approval

Like with full-time hires, contingent workers will require planning ahead of time for the budget necessary to cover the time expenses according to your needs.

For this, you need to discuss with your sub-department leader the need and justification for a contingent workers. Plus, you need to work the numbers in terms of the level of experience required and the length of the contract. You also need to determine if the contingent workers will be working full-time hours or part-time.

When considering the budget, keep in mind that contingent workers work for salary only. There are no benefits for them. You should also consider if you will hire directly or if you will engage with a hiring agency to find them for you. The fees might be very different in these two scenarios. When working with a recruitment or team augmentation agency you will need to account for hiring fees or other administrative costs charged by the agency.

It is advisable to put together a concise proposal where you outline the objectives you are looking to meet by hiring a contingent workers, justifying clearly the reasons you can't use a full-time team member hire to meet the needs. Also include budget and timeline information in the proposal. [Here is one example](https://gitlab.com/gitlab-com/Finance-Division/procurement-team/procurement/-/issues/1020) of an issue that outlines the need and seeks budget approval.

Once you agree on the budget with your manager and get approval at the department head level, you can move on to the next phases of the hiring process.

## Candidate Sourcing

The first thing to be aware of about sourcing contingent workers candidates is that they are hired as professional services vendors through a [procurement process](/handbook/finance/procurement/). This means that as an engineering manager you get little to no support from talent acquisition recruiters at GitLab.

You should consider your strategy. You may reach out to individuals and hire them directly by using your professional networks or tools such as [LinkedIn Recruiter](/handbook/hiring). You can also engage with a recruitment agency or team augmentation agency for this purpose.

If you choose to hire them directly, you will have less administrative work to get the agency [vetted and approved](/handbook/finance/procurement/) by the procurement department. However, you will have to invest more effort in the sourcing work yourself.

One thing you can do is to ask GitLab recruiters to help you and be on the lookout for past candidates in Greenhouse that were not selected for similar jobs in your group or other groups, but who were still very strong candidates. They may make a great contingent workers for your needs and you can benefit from the context and the interview notes that will be available for those candidates as a starting point.

Going with a staff augmentation or recruitment agency can save you time on sourcing efforts, but you need to invest some additional overhead on vetting the agency itself. In this case, you may want to ask the Finance department and explore agencies already working with GitLab first. You will need to work with the selected agency to understand their policies and the way they work with companies to source and interview candidates.

In either case, you need a job description (JD). If you are hiring for a temporary backfill, you will want to have your JD be as close as possible to the [job description](/handbook/hiring/job-description-library/) of the role. Otherwise, be as specific about the technical and job requirements as possible. You also want to highlight the cultural norms and [values](/handbook/values) that make GitLab unique since contingent workers will still need to be aware of and adapt to these norms.

## Interview Process

For the interview, you will have a lot of agency to structure interviews as an engineering manager looking to make this hire. It is advisable to stay as close as possible to the company-level guidelines, especially as it relates to [diversity inclusion and belonging](/handbook/company/culture/inclusion/). You can also use your own sub-department or group guidelines as a basis for defining the interview process, including the diversity of interviewers and approach to technical vetting.

## Contracting and invoicing

Once you have identified the candidate that you want to hire (either directly or through an agency), the actual process of contracting is based on the professional services [procurement process](/handbook/finance/procurement/) outlined by the finance department.

You can read more in the handbook section above, but roughly the steps involved are as follows:

- Select a start and end date for the contract (Note: staff augmentation workers have a max of 24 months)
- Obtain (or draft yourself) a Statement of Work (SOW) outlining the work that the contingent workers will perform for GitLab and the terms and conditions as related to this contract. The SOW should specify the dates, the name of the individual (and/or agency) and the max total fees for the full length of the contract (even if the fees will be paid periodically over the extent of the contract).
- When creating the SOW, be as specific as possible. Clearly identify the scope of the work, the acceptance criteria and the deliverables. Also consider adding criteria related to how your team works. For example, do you expect the contingent workers to add daily async updates to their issues or expect merge requests to be under `x` lines? If so, consider adding it to the SOW.
- Have the contingent workers or vendor representative sign the SOW.
- Procurement process must be completed. How to submit a Zip request for a contractor
- PO approved in Coupa with a confirmed engagement end date. Note: if you don’t have access to Coupa, you can request access or ask somebody with access in your group to submit the request for you.
- Security and Legal Reviews must be complete in Zip.
- Background checks complete if accessing orange/red data
- If accessing orange/red data a GitLab a laptop is required for your contractor. The manager must have confirmed that IT is able to deliver a laptop to the contractor’s country/location.

- Look after the approval flow, optionally pinging individuals if they don't respond in a timely manner to the request.

Once the Coupa requisition is approved, the vendor/contingent workers will be able to invoice against the budget allocated at regular intervals as per the SOW. When they submit invoices, the Coupa requestor will be notified by Coupa to approve the payment.

## Onboarding

With the contracting out of the way, you will want to onboard the new contingent workers into the project or into your team. This can look very different from one contingent workers to another depending on the nature of the work, the team and the project.

Generally speaking, you should know that contingent workers are referred to as Temporary Service Providers. And there is specific documentation in the handbook about their [life cycle](https://internal.gitlab.com/handbook/security/corporate/end-user-services/access-request/temporary-service-providers/), how to create access requests for them and how to onboard and offboard them.

As a general guideline you would want them to:

- Learn about GitLab's values and culture
- Learn specifics about your team's process and rituals
- Learn about your team's technology stack
- Provision the necessary access to communication and productivity systems (project membership, ChatOps, Slack channels including `#production`.) For this, you can rely on access requests and asking questions in #it_help

### Announcements

It is important that not only your team knows about the contingent workers but there should be general awareness that Temporary Service Providers are working for GitLab.

It is encouraged to share information about new contingent workers joining in your group and stage Slack channel but you should also consider writing a short message into [#development](https://gitlab.slack.com/app_redirect?channel=development), [#engineering-fyi](https://gitlab.slack.com/app_redirect?channel=development) Slack channels or into the [Engineering Week In Review](https://docs.google.com/document/d/1JBdCl3MAOSdlgq3kzzRmtzTsFWsTIQ9iQg0RHhMht6E/edit?usp=sharing) document. For these company-wide channels it is enough to mention that some contingent workers will work for a particular group.

## Working with contingent workers

We treat contingent workers as regular team members when it comes to the contribution to issues or merge requests including the code review process.
