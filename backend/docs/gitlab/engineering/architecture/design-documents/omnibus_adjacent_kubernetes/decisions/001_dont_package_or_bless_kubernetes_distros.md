---
title: "GitLab Omnibus-Adjacent Kubernetes ADR 001: Don't package or bless any kubernetes distributions"
owning-stage: "~devops::gitlab delivery"
toc_hide: true
---

## Context

Omnibus-Adjacent Kubernetes assumes a Kubernetes distribution running alongside Omnibus. One possibility could be
to package Kubernetes within Omnibus, or publish another metapackage that would include Kubernetes to support
seamless install of Kubernetes and complete automation via Omnibus cookbooks.

## Decision

We won't embed any Kubernetes distribution within Omnibus, nor package it in any form, and not bless any specific distribution in our docs.

By bundling Kubernetes, GitLab would effectively become Kubernetes distributors.
This creates ongoing maintenance responsibilities including testing Kubernetes upgrades, applying security patches of various severity, and ensuring compatibility across different deployment environments.
Meeting these requirements will demand staff with specialized Kubernetes expertise, and a heavy reliance on Product Security.
Further, the primary candidates for embedded Kubernetes do not provide any FIPS certifications, which would be of significant concern.

Kubernetes is a platform, as much as virtualization providers are a platform (AWS, GCP, VMware, ProxMox).
GitLab has never been responsible for the platform on which customers operate their instances.
This stands true with the Omnibus GitLab, in that we are not responsible for the physical or virtual machines on which this package operates.
Thus, we maintain the same stance in regards to Kubernetes.

## Consequences

Users will have the freedom to choose which distribution they would like to use, but they will have to fully maintain their
Kubernetes instance. They'll be responsible for installing and upgrading their instances.

What GitLab will provide, is a set of Omnibus configurations that, when set, will allow for the secure interconnection of
services started by Omnibus and services running from within the adjacent Kubernetes cluster.
