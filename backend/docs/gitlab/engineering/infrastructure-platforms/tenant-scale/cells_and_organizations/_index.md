---
title: "Cells and Organizations"

description: "This is the handbook page for the Cells project. Cells is one of the top priorities for FY2025, with the goal of providing additional scalability for GitLab.com. This handbook page contains the project information such as the project plan, roadmap, workstreams, DRIs, stakeholders, and communication channels. It also has links to important documentation such as the Cells design blueprints."
---

## Intro

Cells is a new architecture for our software as a service platform. This architecture is horizontally scalable, resilient, and provides a more consistent user experience. It may also provide additional features in the future, such as data residency control (regions) and federated features.

For more information about the goals of Cells, see [goals](/handbook/engineering/architecture/design-documents/cells/goals.md).

## Requirements and Architecture

Cells overall architecture [design document](/handbook/engineering/architecture/design-documents/cells/).

A key component of Cells is [isolated Organizations](/handbook/engineering/architecture/design-documents/organization/).

## Roadmap, and DRIs

### Roadmap

* [Protocells](https://gitlab.com/groups/gitlab-com/gl-infra/-/epics/1616)

### DRIs and Stakeholders

| Stakeholder | Role | Responsibility |
| ----------  | ---  | -------------- |
| [Sabrina Farmer](https://gitlab.com/sabrinafarmer) | CTO | Executive Sponsor |
| [Gerardo Lopez-Fernandez](https://gitlab.com/glopezfernandez) | Engineering Director | Tenant Scale Engineering DRI |
| [Mark Wood](https://gitlab.com/mjwood) |  Group Product Manager | Tenant Scale Product DRI |
| [Darby Frey](https://gitlab.com/darbyfrey) | Staff Fullstack Engineer, Expansion | Sec and Monetization DRI |
| [Kerri Miller](https://gitlab.com/kerrizor) | Staff Backend Engineer, Create | Core Development DRI |

## Communication

### Slack Channels

* [#f_protocells (internal only)](https://gitlab.enterprise.slack.com/archives/C0609EXHX6F): Regular communication

### Meetings

* [Group Tenant Scale weekly standup (internal only)](https://docs.google.com/document/d/18V5rTAcD7mU8UfTRmhXbMLXFBZE39AuzTf2X1IAlk-0)

### Status updates

Status updates are automatically rolled up in the
[Protocell epic](https://gitlab.com/groups/gitlab-com/gl-infra/tenant-scale/cells-infrastructure/-/epics/1#hourglass-work-in-progress).

## Additional Information

### Group Tenant Scale Offsite

Group Tenant Scale held an offsite in Madrid between 2025-06-24 and 2025-06-26.
This offsite's major decision is to launch Protocells as soon as possible.

1. [Agenda, sessions, and Decisions (internal only)](https://docs.google.com/document/d/17j_0FNMhdSM4GACIcm9AznVqW27E1-Q7fLwO3B-ogZk/)

### Cells Fast Boot 2024

We held a Cells Fast Boot in Dublin, Ireland, between 2024-04-23 and 2024-04-24. Below are the artifacts from the event.

#### Agenda, Slides, and Videos

Please use the `Unfiltered` Google account to watch video recordings.

1. [Main agenda (internal only)](https://docs.google.com/document/d/1m5w8sVG5kCvZF0mg7h7HnKzVGREhTuH6e1FRhRlMjns/edit?usp=sharing)
1. Introductions, overview, and logistics: [Agenda (internal only)](https://docs.google.com/document/d/18gqKRORUE8BRULkqfBAv3FZk4yPpRw6BoqmGNUwKm50/edit?usp=sharing)
1. Cells Services - Global Service: [Agenda (internal only)](https://docs.google.com/document/d/1fTeiS6ksvhxJggui_DnCZ9tl5xIN23IZGrqgiqzB5JU/edit?usp=sharing), [Slides (internal only)](https://docs.google.com/presentation/d/12NlfOwolRf10DSLszQi9NjxFy0UUKc2XVC2kYW0HFGk), [Video (internal only)](https://www.youtube.com/watch?v=cNKsNda9Bkc)
1. Cells Services - Routing: [Agenda (internal only)](https://docs.google.com/document/d/1Z3fhilM5wYhLBhghqXeJJCLaO57YeTVdJROrTSgTuDg/edit?usp=sharing), [Slides (internal only)](https://docs.google.com/presentation/d/1wad5IOoXIPkLlVeJdzWnzGmpN_CvwRQi9cMesWj-kz0/edit?usp=sharing), [Video (internal only)](https://www.youtube.com/watch?v=1TgbM-qlQOM)
1. Application Readiness - Organizations and Users: [Agenda (internal only)](https://docs.google.com/document/d/18TH2FraEp2ISSlNOl4asBd3GYQv_homXl4jsLebveJ0/edit?usp=sharing)
1. Application Readiness - Dependencies and OKR alignments: [Agenda (internal only)](https://docs.google.com/document/d/1ySJDT2WQCsndCQikDFAN3VMDadvJwLg000qAI-u3M0I/edit?usp=sharing)
1. Deployment: [Agenda (internal only)](https://docs.google.com/document/d/1Pb280b90MkMNl7TU3i09hGZWAbBZmTUFATRoVFvssiE/edit?usp=sharing), [Slides (internal only)](https://docs.google.com/presentation/d/1rYQup9yc0UEBBrxPkVTDJ9ihzgtiLjmuGYOjZ68TZGs/edit?usp=sharing), [Video (internal only)](https://www.youtube.com/watch?v=Pf5BOMJ0A-0)
1. Provisioning: [Agenda (internal only)](https://docs.google.com/document/d/1Pb280b90MkMNl7TU3i09hGZWAbBZmTUFATRoVFvssiE/edit?usp=sharing)
1. Observability and Runners: [Agenda (internal only)](https://docs.google.com/document/d/1Pg7CKao-StGEbJQ-BEeF6x9xp-Ttux-NeiuvGmBstrQ/edit?usp=sharing)
1. Security: [Agenda (internal only)](https://docs.google.com/document/d/18a_q-g-l5RoUjwsMBp6rRHdvRT4wvD1_iBtwej8m4-M/edit?usp=sharing), [Slides (internal only)](https://docs.google.com/presentation/d/1ImP9KxwmMoZxUjYdoIz3pdPCbyUSTdhdn7JjWLK1tPE/edit?usp=drive_link), [Video (internal only)](https://drive.google.com/file/d/1qTQJZ3gEKTzEzhzb9yUygsqoy8WnQgM3/view?usp=drive_link)
1. Disaster Recovery: [Agenda (internal only)](https://docs.google.com/document/d/1eYFJAF3X_5kIgkBR-j3W_Ug3IMSK85vozCyrzF5gCao/edit?usp=sharing), [Slides (internal only)](https://docs.google.com/presentation/d/1JhWo_e6w_HYYwYCd2yN5vttntMS57aIDhPADEQD1aCw/edit?usp=sharing), [Video (internal only)](https://youtu.be/uDLDpwLYY3M)
1. Cells Mover and Isolation: [Agenda (internal only)](https://docs.google.com/document/d/1ySJDT2WQCsndCQikDFAN3VMDadvJwLg000qAI-u3M0I/edit?usp=sharing)
1. Scalability Headroom and Timeline: [Agenda (internal only)](https://docs.google.com/document/d/1XawTKX_MPM89Oyl_u3-9X1-A8QsOsZRlg8LB6ormP_8/edit?usp=sharing)

#### Decisions

1. No external customers on Cells 1.0, internal dogfooding only. Cells 1.x is the target to onboard new or existing external customers.

#### Artifacts

1. Day 1 recording: [Part 1 (internal only)](https://youtu.be/Y5mOBWqGCTQ), [Part 2 (internal only)](https://youtu.be/JtGoKK9NonE)
1. [Day 2 recording (internal only)](https://youtu.be/MM4Py3Ldnlg)
1. [Database breakout recording (internal only)](https://drive.google.com/drive/folders/13ZrWpPad_jg5ua7ocZvJ0b-XeWIp4uUy?usp=drive_link)
1. [Organizations breakout recording (internal only)](https://youtu.be/KqF1UARbMHA)
