---
title: "Organization: Frequently Asked Questions"
owning-stage: "~devops::tenant scale"
group: Tenant Scale
toc_hide: true
---

## Do we expect large SaaS customers to be licensed at the Organization level?

Yes. At the moment billing will move to the Organization level. We will continue to update [Organization and Billing](billing.md).

## Will Self Managed and Dedicated allow for multiple Organizations?

Yes, most likely. Initially our focus will be on GitLab.com but we understand there is demand for multiple Organizations on other platforms and anticipate rolling this feature out eventually.

## How will Organizations affect Self Managed?

Self Managed will remain relatively unchanged. The Organization will be exposed to Self Managed just like other platforms, and features may shift up or down to the Organization level.

Eventually we expect all platforms will provide the ability to operate multiple Organizations.

Our goal is to minimize disruption to Self Managed and Dedicated while we try to bring GitLab.com to parity with the other platforms.

In the longer term, we anticipate faster feature development by having Organizations as a common top level entity across all platforms.

## Can a user belong to multiple Organizations?

Currently, our model is designed for one user to belong to one Organization. This restriction was implemented because allowing users to belong to multiple Organizations created significant technical challenges and raised many product questions. We expect usage of multiple Organization membership to be relatively low, so this limitation will likely remain for the foreseeable future.

## Do we expect to be able to configure alternate GitLab domain names for Organizations (such as `customer.gitlab.com`)?

There is no plan at this point to allow configuration of alternate GitLab domain names.
GitLab Dedicated will be a much better fit for that at this moment.

## Do we expect Organizations to have visibility settings (public/private) of their own?

Visibility is documented on the main [Organization page](_index.md#organization-product-feature) and also [Organization Users](users.md#when-can-users-see-an-organization).

## What would the migration of a feature from the top-level Group to the Organization look like?

We will provide more information at a later date. It won't be possible to build a feature completely at the Organization level until the Organization feature is released.

However, data at the top-level group will still need to be linked back to an Organization for Organization data isolation purposes.

## How many Cells and Organizations do we expect to exist on GitLab.com?

At this time we can only make rough estimates of how many Organizations or Cells may exist.

As the Cells will be multi-tenanted, with many Organizations belonging to one Cell, there will inevitably be more Organizations than Cells.

There will likely be thousands of Organisations spread across the Cells.
Possibly even millions of Organizations, if most top-level groups get converted to their own Organization.

There will also likely be hundreds or even thousands of Cells.

At the start however, there will be far fewer of each, as we gain operational experience in production with customer data and decide how much to scale.
