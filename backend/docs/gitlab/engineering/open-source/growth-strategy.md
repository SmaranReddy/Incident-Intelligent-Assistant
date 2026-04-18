---
title: "Open Source Growth Strategy"
---

## Dual Flywheels

GitLab has two flywheel strategies that reinforce each other: our open core flywheel and our development flywheel.
A flywheel strategy is [defined as](https://medium.com/evergreen-business-weekly/flywheel-effect-why-positive-feedback-loops-are-a-meta-competitive-advantage-6d0ed55b67c5) one that has positive feedback loops that build momentum, increasing the payoff of incremental effort.
You can visualize how the flywheels work in congruence via the diagram below.

In the open core flywheel, more features drive more users which in turn drive more revenue and more contributions which lead to more users.

The driving force behind the flywheel is that by using a DevSecOps platform to replace multiple point solutions, GitLab customers can achieve cost saving and efficiency gain. Therefore, when GitLab develops more features to improve the product maturity, it becomes easier to replace point solutions and GitLab will attract more users.

```mermaid
graph BT;
  id1(More Users)-->id2(More Revenue);
  id2(More Revenue)-->id3(More Features);
  id3(More Features)-.->id1(MoreUsers);
  id1(More Users)-.->id4(More Contributions);
  id3(More Features)-->id1(More Users);
  id4(More Contributions)-.->id3(More Features);
```

### KPIs and Responsible departments

| Part of flywheel | Key Performance Indicator (KPI) | Department |
|-----------------|---------------------------------|------------|
| More Users | Stage Monthly Active Users | Product |
| More Contributions | Unique Wider Community & Customer Contributors per month | Developer Relations and Engineering |
| More Features | Merge Requests per release per engineer in product development | Engineering and Product |
| More Revenue | IACV vs. plan | Sales and Marketing |

### Flywheel with two turbos

GitLab is a [complete platform](https://about.gitlab.com/platform/), with [contributions from the wider community](https://about.gitlab.com/community/contribute/).

Compared to other platforms, GitLab leverages unique turbos that boost the company:

1. [Advantages of a single application](/handbook/product/categories/gitlab-the-product/single-application/), leading to more [Stages per Organization](https://internal.gitlab.com/handbook/company/performance-indicators/product/#stages-per-organization-spo)
1. [Open Source stewardship](/handbook/company/stewardship/), leading to [Wider community & customer contributions](https://about.gitlab.com/community/contribute/)

In the development spend flywheel, we capture the relationship between merge requests (MRs), changes in ARR from one period to the next (Delta ARR), hyper growth R&D spend and the resulting impact on MRs. We see that more MRs merged, that check all our acceptance criteria, increase stage maturity which drives more monthly active users and stages per user which in turn drives more seats and more revenue which funds R&D spend and leads to more high quality MRs.

```mermaid
graph TB
  id2(R&D Wider MR Rate) --> id1
  id1(Wider community contributions Turbo) --> id3
  style id1 color:#ffffff, fill:#9B51E0
  id3(Stage maturity increase) --> id4
  id3 --> id9
  id4(MAU increase) --> id5
  id5(More Licensed users) --> id6
  id6(Net ARR) --> id12
  id12(ARR) --> id8
  id7(R&D spend ratio) --> id8
  id8(R&D investment) --> id11
  id11(Development Department Narrow MR Rate) --> id2
  id9(SpO increase Turbo)--> id5
  style id9 color:#ffffff, fill:#9B51E0
  id9 --> id10
  id10(Higher Revenue per licensed user) --> id6
  id6 --> id7
```

## Key Areas

We will be executing in 5 key areas in support of the company's broader dual-flywheels strategy to attain more contributions. The 5 key areas together are the building blocks of our contributor & contribution 10x acceleration strategy.

```mermaid
flowchart LR
  subgraph moreContributions["More Contributions"]
    contributorIncrease["Contributor Increase"]
    contributionIncrease["Contribution Increase"]
    bothIncrease["Contributor & Contribution Increase"]
    increaseValue("Increase Contribution Value")
    improveJourney("Improve Contributor Journey")
    fosterDiversity("Foster Diversity, Equity, and Inclusion")
    scaleCommunity("Scale the Community")
    expandOutreach("Expand Outreach")
    scaleCommunity-->improveJourney
    scaleCommunity-->increaseValue
    scaleCommunity-->fosterDiversity
    expandOutreach-->scaleCommunity
    increaseValue-->contributionIncrease
    improveJourney-->contributorIncrease
    fosterDiversity-->bothIncrease
  end
  style moreContributions fill:#FFF, stroke:#9370DB, stroke-dasharray: 5 5
  style contributionIncrease fill:#9370DB,stroke:#9370DB,stroke-width:10px
  style contributorIncrease fill:#9370DB,stroke:#9370DB,stroke-width:10px
  style bothIncrease fill:#9370DB,stroke:#9370DB,stroke-width:10px
  style improveJourney color:#6b4fbb, stroke:#9370DB
  style increaseValue color:#6b4fbb, stroke:#9370DB
  style fosterDiversity color:#6b4fbb, stroke:#9370DB
  style expandOutreach color:#6b4fbb, stroke:#9370DB
  style scaleCommunity color:#6b4fbb, stroke:#9370DB
  click improveJourney "./#improve-contributor-journey" _self
  click increaseValue "./#increase-contribution-value" _self
  click fosterDiversity "./#foster-diversity-equity-and-inclusion" _self
  click expandOutreach "./#expand-outreach" _self
  click scaleCommunity "./#scale-the-community" _self
 ```

### Current focus

We're prioritizing these 5 key items to maximize our impact within our current capacity. While all initiatives are valuable, this focused approach allows us to drive meaningful change more effectively.

* [Reduce Open Community MR Age](/handbook/engineering/open-source/growth-strategy/#reduce-open-community-mr-age)
* [Create a compelling contributor value proposition](/handbook/engineering/open-source/growth-strategy/#create-a-compelling-contributor-value-proposition)
* [Contributor advancement system](/handbook/engineering/open-source/growth-strategy/#contributor-recognition--advancement-system)
* [Returning & Frequent Contributors](/handbook/engineering/open-source/growth-strategy/#returning--frequent-contributors)
* [Non-code contributions](/handbook/engineering/open-source/growth-strategy/#non-code-contributions)

### Improve Contributor Journey

Provide an outstanding, highly efficient and fast contributor experience, from onboarding to getting a change merged. One of the tactics is to reduce contribution stumbling blocks to make the contributor journey more efficient. These stumbling blocks will be identified through gathering feedback from wider community contributors, product teams and GitLab Team Members on contribution friction.

#### Reduce Open Community MR Age

* **Why:** Improve the speed of contribution to production by reducing [Open community MR Age (OCMA)](/handbook/marketing/developer-relations/performance-indicators/#open-community-mr-age) & review time. We have identified product groups with the highest OCMA. Analysis and improvements are needed to address product groups with the biggest opportunity. In addition to improving MR review and gathering feedback.
* **DRI:** [Developer Relations Engineering](/handbook/marketing/developer-relations/engineering/)

#### Product & engineering alignment

* **Why:** Within [product groups](/handbook/company/structure/#product-groups), contribution submissions, backlog, and technology stack vary. Healthy community backlog alignment and the establishment of a common best practice for outreach are vital for contributor success. In addition, there should be a unified & known workflow for Wider Community Contributions.
* **DRI:** [Developer Advocacy team](/handbook/marketing/developer-relations/developer-advocacy/) & [Developer Relations Engineering team](/handbook/marketing/developer-relations/engineering/)

#### Simplify & improve contribution guides

* **Why:** Make contribution guides easy to navigate. Our current contribution guides are fragmented and can be hard for new contributors to navigate and understand.
* **DRI:** [Developer Relations team](/handbook/marketing/developer-relations/)

#### Improve contribution tooling

* **Why:** Provide fast and efficient contributor experience via our tooling. Our contributor tooling needs to be optimized for contributor productivity
* **DRI:** [Development Tooling team](/handbook/engineering/infrastructure-platforms/developer-experience/development-tooling/)

### Increase Contribution Value

Incentivize, attract and retain contributors by providing a compelling value and regular recognition of contributors for their work. Contributor career advancement materials and awards.

#### Create a compelling contributor value proposition

* **Why:** We need a clear definition of what drives people to contribute to GitLab and to present a compelling value proposition for increasing code contribution. One of the programs to drive this challenge is our [Co-Create program](https://about.gitlab.com/community/co-create/) for customers.
* **DRI:** [Developer Relations team](/handbook/marketing/developer-relations/)

#### Contributor recognition & advancement system

* **Why:** In open source projects, contributors are motivated not only by solving bugs or adding features but also by gaining experience and building their online presence. This can be solved through levelling-up, badging or other incentivizing systems. In addition, providing sustained and impactful recognition to recognize & retain our contributors. Increase frequency and targeted recognition to types and persona of contributors
* **DRI:** [Developer Relations Engineering team](/handbook/marketing/developer-relations/engineering/)

### Foster Diversity, Equity, and Inclusion

Center diversity, equity, and inclusion within our open source community relations and our wider community of contributors. Reach larger audiences of top contributors and offer more equitable opportunities to contribute.

#### Align with open source community DEI initiatives

* **Why:** Aligning with open source communities practicing DEI initiatives will offer GitLab more ideas on reaching new contributors and signal our commitment to DEI to potential contributors.
* **DRI:** [Developer Relations team](/handbook/marketing/developer-relations/) & [Developer Relations Engineering team](/handbook/marketing/developer-relations/engineering/)

#### Improve contributor inclusion

* **Why:** To improve the contributor experience for newcomers and returning contributors, and meet contributors at different experience levels. We can improve the changes of a contributor returning by creating a more inclusive experience for them.
* **DRI:** [Developer Relations Engineering team](/handbook/marketing/developer-relations/engineering/)

### Expand Outreach

Increase awareness with content and events to drive large amounts of contributors. Our outreach efforts so far have been limited. A proactive and focused effort to bring awareness and drive members from external communities will be required. Engineering to work alongside Developer Relations in expanded outreach events.

#### Increase contribution backlog exposure

* **Why:**  Contributing as a new member to a massive project can be overwhelming, which can lead to analysis paralysis and potentially losing contributors. We should offer a lens into a discoverable, sizable set of issues we can direct newcomers to. Consider using established 3rd party platforms.
* **DRI:** [Developer Relations team](/handbook/marketing/developer-relations/)

#### Scale contributor events

* **Why:** Build a sense of belonging, provide the social environment for contributors to have their voice, meet with their peers, share knowledge and celebrate.
* **DRI:** [Developer Relations team](/handbook/marketing/developer-relations/) & [Developer Relations Engineering team](/handbook/marketing/developer-relations/engineering/)

#### Community office hours & community pairing

* **Why:** We need to scale office hour calls and our community pairing sessions that have traditionally been a unique opportunity for product groups to provide support, guidance to code contributors as well as gather feedback.
* **DRI:** [Developer Relations team](/handbook/marketing/developer-relations/) & [Developer Relations Engineering team](/handbook/marketing/developer-relations/engineering/)

#### Increase social presence

* **Why:** Increase our social media presence beyond the currently limited mediums (Twitter, Discord), which will allow us to tap into existing developer communities.
* **DRI:** [Developer Relations team](/handbook/marketing/developer-relations/)

### Scale the Community

Leverage the full-time customer contributor model and create wider community teams for sustainable growth.

#### Returning & Frequent Contributors

* **Why:** Increased contribution by motivating recurring contributions from organizations that use or extend GitLab, such as customers, partners and OSS communities. Reward those that achieve that status with GitLab benefits that are valuable to the individual and the organization.
* **DRI:** [Developer Relations team](/handbook/marketing/developer-relations/) & [Developer Relations Engineering team](/handbook/marketing/developer-relations/engineering/)

#### Non-code contributions

* **Why:** Our userbase is made of a lot of different personas who each can make valuable contributions and improve GitLab. Creating design proposals, code reviews, triaging our issues or facilitating healthy discussions between all actors involved helps us achieve our goals faster. In addition, recognizing non-code contributions such as helping GitLab with promoting the product, our events or other ways to contribute should be recognized and celebrated.
* **DRI:** [Developer Relations Engineering team](/handbook/marketing/developer-relations/engineering/)

#### Foster more collaboration

* **Why:** We would like to depart from having contributors work single-handedly and create a team that can do more together.
* **DRI:** [Developer Relations Engineering team](/handbook/marketing/developer-relations/engineering/)
