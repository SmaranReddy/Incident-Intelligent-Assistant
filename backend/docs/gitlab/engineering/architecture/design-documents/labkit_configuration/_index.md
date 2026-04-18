---
title: "LabKit Configuration Management"
status: proposed
creation-date: "2026-03-05"
authors: [ "@andrewn" ]
coaches: ["@stanhu"]
dris: []
owning-stage: "~devops::systems"
participating-stages: []
toc_hide: true
---

<!-- vale gitlab.FutureTense = NO -->

{{< engineering/design-document-header >}}

## Summary

This design document describes a standardized, protobuf-first configuration management module for [LabKit](https://gitlab.com/gitlab-org/labkit) (`v2/config`). The module provides a unified approach to configuration handling across GitLab's Go services, using Protocol Buffers as the single source of truth for configuration schemas. It includes format autodetection (YAML, JSON, TOML), strong validation via [protovalidate](https://protovalidate.com/), typed version migrations, and rollback safety through forwards-compatible parsing.

## Motivation

Configuration management is currently fragmented across GitLab's internal tooling. Each service defines its own configuration format, validation logic, and loading semantics, leading to:

- **Inconsistent formats**: Different tools use YAML, TOML, or JSON with varying conventions. Gitaly and Workhorse use TOML, which the Operate team has difficulty serializing when generating Helm chart configurations.
- **Duplicated validation**: Every tool reimplements validation checks, often incompletely.
- **No shared contract**: Downstream tooling (Helm charts, Omnibus) has no machine-readable specification to validate against in unit tests.
- **Poor developer experience**: Engineers must context-switch between different configuration idioms with no consistent mental model.
- **Documentation drift**: Without machine-readable schemas, documentation must be hand-authored and can drift out of sync with actual configuration structure.
- **No lifecycle stage enforcement**: GitLab has no mechanism to indicate whether a configuration setting is experimental, alpha, beta, or stable.

A centralized LabKit module addresses these concerns by providing convention-over-configuration for all LabKit-integrated tools.

### Goals

- Provide a single, consistent configuration format (YAML or JSON) for all LabKit-integrated tooling
- Use Protocol Buffers as the single source of truth for configuration schemas
- Enable downstream tooling to reference published schemas for automated validation
- Support all common configuration structures: scalars, maps, lists, nested objects, optional/required fields
- Apply guard-rails at load time: fail fast on schema violations, type mismatches, and constraint violations
- Be adoptable with minimal disruption: existing Go structs should be replaceable with generated protobuf types
- Lay groundwork for future capabilities: migration tooling, validation CLI commands, environment variable overrides

### Non-Goals

- This proposal does not define a runtime secrets management system. Secrets should continue to be injected via existing pathways and referenced by path or environment variable.
- This proposal does not replace infrastructure-level configuration (Kubernetes manifests, Terraform variables).
- This proposal does not mandate immediate migration of all existing tool configurations; adoption will be incremental.
- Environment variable overrides are explicitly deferred to a future proposal.
- Standardized defaults management is deferred until the module has been retrofitted to existing tools.

## Proposal

Introduce `v2/config` module in LabKit that provides:

1. **Protobuf-first schemas**: Each tool defines configuration in versioned `.proto` files using Protocol Buffers Edition 2023 syntax
2. **Format autodetection**: Automatically detect and parse YAML, JSON, or TOML based on file extension
3. **Strong validation**: Use protovalidate constraints defined inline in proto files for rich validation rules
4. **Typed migrations**: Support version upgrades via type-safe `func(source S) (T, error)` migration functions
5. **Rollback safety**: Unknown fields silently ignored by default to support safe rollbacks
6. **Strict mode**: Optional strict validation for CI pipelines to catch typos and obsolete config

### Configuration Format

Configurations are expressed in **YAML** (preferred) or **JSON** (alternative). YAML is preferred because:

- It's the dominant format across GitLab's infrastructure tooling
- Widely adopted in Kubernetes ecosystems (no learning curve)
- Type system aligns with Helm and existing configuration systems
- Tooling like `yq` enables easy manipulation in CI pipelines
- Many editors support JSON Schema validation/autocompletion for YAML

TOML is available as opt-in support for tools like Gitaly and Workhorse during migration.

Example configuration:

```yaml
# widget-service.config.yaml
version: 2
server:
  address: "0.0.0.0"
  port: 8080
  timeout: "30s"
logging:
  level: "info"
  format: "json"
feature_flags:
  enable_experimental_cache: false
```

The optional `version` field (integer, defaults to `1` if omitted) enables the migration system.

## Design and implementation details

### Protobuf-First Schema

Configuration schemas are defined in Protocol Buffers Edition 2023 at the well-known path:

```plaintext
<tool-repo-root>/proto/config/v<N>/config.proto
```

Example schema:

```proto
edition = "2023";
package myapp.config.v1;

option go_package = "myapp/gen/config/v1;configv1";
option features.field_presence = IMPLICIT;

import "buf/validate/validate.proto";

message Config {
  int32 version = 1;
  ServerConfig server = 2 [(buf.validate.field).required = true];
}

message ServerConfig {
  string address = 1 [(buf.validate.field).string.min_len = 1];
  uint32 port = 2 [(buf.validate.field).uint32 = {
    gte: 1
    lte: 65535
  }];
  google.protobuf.Duration timeout = 3;
}
```

**Benefits:**

- Code generation provides type safety
- Validation rules co-located with schema (cannot diverge)
- Cross-language interoperability
- Downstream tooling can parse `.proto` files for validation
- Rich constraints via protovalidate CEL expressions

### Validation

Validation uses [protovalidate](https://protovalidate.com) with constraints defined inline:

**Standard constraints:**

```proto
message ServerConfig {
  string host = 1 [(buf.validate.field).string = {
    min_len: 1
    max_len: 255
  }];

  uint32 port = 2 [(buf.validate.field).uint32 = {
    gte: 1
    lte: 65535
  }];
}
```

**Cross-field validation with CEL:**

```proto
message TLSConfig {
  string cert_path = 1;
  string key_path = 2;

  option (buf.validate.message).cel = {
    id: "tls_pair"
    message: "cert_path and key_path must both be set or both be empty"
    expression: "(this.cert_path == '') == (this.key_path == '')"
  };
}
```

### Migration System

The `version` field enables automatic migrations from version N-1 to N using typed, generic functions:

**v1 config:**

```proto
message Config {
  int32 version = 1;
  ServerConfig server = 2;
}

message ServerConfig {
  string host = 1;
  uint32 port = 2;
}
```

**v2 config (renamed field):**

```proto
message Config {
  int32 version = 1;
  ServerConfig server = 2;
}

message ServerConfig {
  string address = 1;  // renamed from "host"
  uint32 port = 2;
  google.protobuf.Duration timeout = 3;
}
```

**Typed migration function:**

```go
func migrateV1ToV2(source *configv1.Config) (*configv2.Config, error) {
    return &configv2.Config{
        Version: 2,
        Server: &configv2.ServerConfig{
            Address: source.Server.Host, // Type-safe rename
            Port:    source.Server.Port,
            Timeout: durationpb.New(30 * time.Second),
        },
    }, nil
}

loader, _ := config.New(config.WithMigration(migrateV1ToV2))
```

**Migration flow:**

1. Parse config file into target proto (v2) to detect version
2. Detect version mismatch (e.g., file has v1, binary expects v2)
3. Re-parse config file into source type (v1)
4. **Pre-migration validation**: Validate against v1 schema
5. Run typed migration function to transform v1 → v2
6. **Post-migration validation**: Validate against v2 schema

Only single-step migrations (N-1 → N) are supported. This is an explicit guard-rail to prevent accumulating configuration version debt.

### LabKit Module API

Minimal, ergonomic API:

```go
import "gitlab.com/gitlab-org/labkit/v2/config"

func main() {
    loader, err := config.New()
    if err != nil {
        log.Fatal(err)
    }

    var cfg configv1.Config
    if err := loader.Load("config.yaml", &cfg); err != nil {
        log.Fatal(err)
    }

    // cfg is loaded and validated
}
```

**With migration:**

```go
loader, _ := config.New(
    config.WithMigration(migrateV1ToV2),
)
```

**With strict mode (CI only):**

```go
loader, _ := config.New(
    config.WithStrictMode(),
)
```

**With custom format parser:**

```go
import "gitlab.com/gitlab-org/labkit/v2/config/toml"

loader, _ := config.New(
    config.WithParser(toml.NewTOMLParser()),
)
```

### Guard-Rails

| Guard-rail | Default | Rationale |
|------------|---------|-----------|
| Type mismatches fail fast | Enabled | Prevents runtime surprises from implicit coercion |
| Required fields enforced | Enabled | Via protovalidate `required` constraints |
| Version field defaults to `1` if absent | Enabled | Lowers adoption friction while preserving migration capability |
| Schema validation at load | Enabled | Single source of truth for what is valid |
| Unknown keys rejected (strict mode) | **Disabled** | Opt-in only — required for rollback safety |

### Rollback Safety and Strict Mode

Unknown fields are **silently ignored by default**. This is critical for safe rollbacks:

- If v2 deploys with a new config field, then rolls back to v1, the v1 binary must not crash on the unknown field
- With strict mode enabled, rollback would fail
- Strict mode should only be used in CI pipelines and pre-deployment validation

```go
// For CI validation only - NOT for production
loader, _ := config.New(config.WithStrictMode())
```

### Format Support

| Format | Availability | Notes |
|--------|-------------|-------|
| YAML (`.yaml`, `.yml`) | Always available | Recommended for human-authored configs |
| JSON (`.json`) | Always available | Recommended for machine-generated configs |
| TOML (`.toml`) | Opt-in via `WithParser` | For tools migrating from TOML |

### Error Handling

Errors include file, line, and column information:

**Parse error:**

```plaintext
config.yaml:5:3: invalid ServerConfig.port: value must be <= 65535 but got 99999
```

**Validation error:**

```plaintext
validation failed: invalid Config.server: embedded message failed validation |
  caused by: invalid ServerConfig.port: value must be greater than or equal to 1
```

**Migration error:**

```plaintext
migration from version 1 to 2 failed: server.host field is required
```

## Alternative Solutions

### Use an existing Go configuration library (e.g., Viper)

Viper is widely used but brings significant complexity, does not enforce well-known schema locations, and does not integrate with protobuf or protovalidate. It also has known limitations around strict unknown-key rejection. A focused LabKit module keeps the interface minimal and aligned with GitLab conventions.

### Use JSON Schema rather than protobuf

JSON Schema is reasonable for format validation but requires separate validation passes and does not provide code generation, type safety, or cross-language interoperability. Protobuf schemas support richer constraints via protovalidate and can encode lifecycle metadata via custom options.

### Use TOML as the primary format

TOML is used by Gitaly and Workhorse but has poor serialization support in Helm tooling used by the Operate team. Its type system doesn't align with YAML/JSON. It's supported as opt-in for migration but not the target format.

### Require JSON only (no YAML)

JSON is machine-friendly but harder for operators to author and read. Supporting YAML as primary and JSON as alternative provides both ergonomics and machine-friendliness.

### Map-based migrations over typed migrations

An earlier version considered migrations operating on `map[string]any`. The implemented approach uses typed generic functions (`func(source S) (T, error)`) instead, providing compiler type checking, IDE support, and easier testing.

## Adoption Path

1. **Phase 1 — Module implementation**: ✅ Implemented in [labkit!345](https://gitlab.com/gitlab-org/labkit/-/merge_requests/345)
2. **Phase 2 — Pilot integration**: Integrate into Donkey and Caproni to validate the API
3. **Phase 3 — Downstream schema validation**: Add proto-based validation to Helm Charts CI and Omnibus pipelines
4. **Phase 4 — Broader adoption**: Migrate additional tools incrementally; TOML opt-in eases Gitaly/Workhorse migration

## Future Work

### Standardized Validation Commands

Provide ready-made CLI subcommand for any LabKit-integrated tool:

```bash
my-tool config validate --file widget-service.config.yaml
```

### Environment Variable Overrides

Structured mechanism for environment variable overrides will be addressed in a dedicated proposal.

### Defaults Management

Standardized mechanism for declaring and applying defaults will be introduced once adoption experience informs the design.

### Documentation Assistance Tooling

Auto-generate configuration reference documentation from protobuf schemas, including field names, types, constraints, and lifecycle metadata.

### Lifecycle Stage Enforcement

Proto field options can encode lifecycle stage metadata (experimental, alpha, beta, stable, deprecated) in a machine-readable way, enabling tooling to warn when operators use settings not intended for general consumption.

### Multi-Language Support

Once the Go implementation is complete and proven through adoption across GitLab's Go services, we will extend this configuration management approach to other LabKit-supported ecosystems including Ruby, Rust, and other languages. The protobuf-first approach provides a natural foundation for cross-language consistency, as Protocol Buffers already support code generation for multiple languages and protovalidate implementations exist across different ecosystems.

## Related Work

- Original proposal: [gitlab#591894](https://gitlab.com/gitlab-org/gitlab/-/work_items/591894)
- Implementation: [labkit!345](https://gitlab.com/gitlab-org/labkit/-/merge_requests/345)
- Lifecycle stage exploration: [gitlab-org/charts/gitlab#6219](https://gitlab.com/gitlab-org/charts/gitlab/-/work_items/6219)
