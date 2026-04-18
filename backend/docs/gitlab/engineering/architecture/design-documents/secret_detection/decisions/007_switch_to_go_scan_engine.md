---
title: "GitLab Secret Detection ADR 007: Switch to Vectorscan-based Go scan engine"
---

## Context

This ADR captures the [decision](https://gitlab.com/gitlab-org/gitlab/-/issues/526227) to switch from RE2-based(port) Ruby scan engine to Vectorscan-based(port) Go scan engine for Secret Detection scans.

Here's the historical context on why Ruby was chosen and the problems associated with it.

### Why Ruby was chosen earlier?

In the initial days of Secret Push Protection(SPP) feature development, we chose Ruby for running SD scans for two main reasons:

1. The expected team's velocity to build the SPP feature out-of-box was pretty high and adding tight time constraints in to the mix led us choosing an approach that had least resistance without compromising much on the quality. Given the team's familarity with Ruby and ease of integration with GitLab Rails, building the SPP feature within a Ruby Gem made sense.

2. Using RE2 engine ported in Ruby boosted our confidence in the decision as it performed much better in terms [latency](https://gitlab.com/gitlab-org/gitlab/-/issues/423832) and [memory consumption](https://gitlab.com/gitlab-org/gitlab/-/issues/422574#note_1582015771) than the other language+regex engine in the bucket i.e., Go with built-in regex engine.

Fast-forward to the technical discovery phase of building Secret Detection Service(SDS). We retained the decision of using the same language(Ruby) and engine(RE2 port) for a couple of reasons:

1. The initial scope of Secret Detection Service(SDS) was to contain _anything and everything_ pertaining to Secret Detection. That means, pulling the files from a particular source for scanning, updating metadata in the DB, running post-processing actions like revocation, to name a few. Given this idea, we could leverage and integrate with existing systems from GitLab Rails like workers, cache, DB etc. However, we later descoped to make it a stateless service responsible for just scanning the given payloads while Pre and Post-processing steps are handled at GitLab Rails side.

2. There was an active development of SPP feature by vendoring SD Ruby gem so the migration would've pushed the SPP feature release further. Given the strict release timeline we had, we couldn't have afforded it. We needed a non-disruptive approach.

3. Runway did not (and still doesn't) support deployment to Self-Managed and Dedicated environments. Cloud Connector was an effortful option to consider. Runway [had plans](https://gitlab.com/gitlab-com/gl-infra/platform/runway/team/-/issues/236) to support it so a temporary approach of reusing the on-going work (vendored gem) made more sense.

4. RE2 Regex performance was still _sufficiently_ better in terms [latency](https://gitlab.com/gitlab-org/gitlab/-/issues/423832) and [memory consumption](https://gitlab.com/gitlab-org/gitlab/-/issues/422574#note_1582015771) than built-in Go regex. We did not explore Go's alternative regex engines.

#### There are some quirks that come with choosing Ruby

1. We over-emphasized on raw regex performance of Ruby RE2 during the decision-making such that we skipped other crucial factors that contribute to the overall performance, particularly when serving the scan via HTTP request. Some of the missed out factors are Server throughput, request latency, memory consumption + GC, Cost-effectiveness due to Resource utilization etc., and Ruby underperforms in _all_ of these factors due to the following:

   1. Ruby generally has high-memory footprint, starting from its runtime VM to regex operations, and its GC is not optimised (like any other interpreted languages). We did consider ["running the scan within subprocess" approach](003_run_scan_within_subprocess.md) to address it, however, we had to drop it in SDS since GRPC doesn't support forking processes.

   2. Global Interpreter Lock(GIL) allows only one request at a time per-process-per-cpu, keeping the other requests blocked for longer periods that in turn could potentially block the customers on git push operation.

   3. No built-in support for concurrency. Even parallelism isn't easily achievable as Ruby doesn't effectively utilize multiple CPU cores and threads due to GIL.

2. Ruby lacks first-class support for GRPC, so it uses [C-based GRPC](https://github.com/grpc/grpc) library under the hood. C-Ruby bindings introduces some overhead in addition to missing language-specific optimizations, like for we have for [Go](https://grpc.io/blog/grpc-go-perf-improvements/).

3. Considering the plan of SDS as the sole platform of running scans across different scan target types, its high throughput and low latency becomes a mandatory requirement. Looking at the [metrics of GRPC servers](https://nexthink.com/blog/comparing-grpc-performance), Ruby is among the poorest performers in the list.

4. Missing first-class support for generating efficient binary executable makes it difficult to unify the Secret Scanning engine across the entire Secret Detection suite of features.

## Proposal

The proposal was explore other performant languages and regex engines suitable for running SD scans in different scan modes i.e, as a Distributed service and as an executable binary.

### Languages (and engines) considered for the decision

1. [Ruby](https://gitlab.com/vbhat161/regex-performance/-/tree/main/ruby) with RE2 Port

2. [Go](https://gitlab.com/vbhat161/regex-performance/-/tree/main/golang) with Vectorscan Port

3. [C++](https://gitlab.com/vbhat161/regex-performance/-/tree/main/cpp) with RE2 (native) and Vectorscan (native).

4. [Rust](https://gitlab.com/vbhat161/regex-performance/-/tree/main/rust) with Std Regex and Std Regex w/ parallelised rule processing.

### Conclusion

The [final conclusion](https://gitlab.com/gitlab-org/gitlab/-/issues/526227#note_2491660337) for the language and regex engine came out to be the combination of **Go and Vectorscan (Port)**.

Results for micro-benchmarking and grpc throughput testing can be found [here](https://gitlab.com/gitlab-org/gitlab/-/issues/526227#note_2460625471). The source code and dataset used for testing can be found [here](https://gitlab.com/vbhat161/regex-performance).

### Why Go will be suitable moving forward?

1. Go has a significant upper hand ([around 8-10x](https://nexthink.com/en/blog/comparing-grpc-performance)) over Ruby in terms of overall performance(mentioned above) such that it could compensate for its moderate regex performance.

2. We did not explore other Go regex alternatives during the Spike since we were convinced about using Ruby due to other constraints mentioned above. There could have been a possibility that a third-party Go regex libraries perform better.

3. First-class support for building executable binary makes Go suitable even in the areas where accessing SD service is not feasible. This also helps us with building a unified scanner engine as we could serve Secret Detection as a Service, Binary executable and CLI with a single Go source code.

4. Having [first-class GRPC support](https://github.com/grpc/grpc-go), we have better [tooling](https://github.com/grpc-ecosystem/awesome-grpc?tab=readme-ov-file#lang-go) for monitoring, testing, debugging and plethora of [middleware extensions](https://github.com/grpc-ecosystem/go-grpc-middleware). Go also has better alignment with gRPC's type system, so fewer unexpected type-related crashes

5. Given the critical role of SDS in the future, Go's maintainability, dependency management and its type system holds a significant advantage.

6. Adding a cherry on top, integration with [Cloud Connector for SDS](https://gitlab.com/gitlab-org/gitlab/-/work_items/525472) wouldn't have the blocker of a missing auth SDK, like we currently have for Ruby.

In addition to above, there are more positives of choosing Go over Ruby like efficient GC, low runtime resource consumption, smaller container size leading to faster deployment, efficient file handling, native go-groutines w/ channels for a clean concurrency code, compile-time checks, mature profiling/instrumentation tools, cloud native i.e, better suited for containerized and microservice architecture.

### Why now is the right time?

The scanner engine is currently stable with no significant feature planned in the immediate roadmap. Secret Detection Service is still in Beta with major features that actively use SDS in the planning phase. We are at an inflection point where delaying this decision further will exponentially increase the migration complexity and technical debt as SDS matures and becomes more deeply integrated. This is our only chance to establish the appropriate technical decision for the long-term.
