---
title: ADR-001 Use JSONB for Classification Metadata Storage
description: 'Use JSONB for CDot Plan Classification Metadata Storage'
---

## Context

We needed a flexible storage solution for plan classification metadata that would allow us to add new fields without requiring database migrations for each addition.

## Decision

We will use a JSONB column named `custom_fields` on the `zuora_product_rate_plans` table to store all classification metadata.

## Alternatives Considered

1. **Plain Columns**
   - ✅ Strong schema validation and better query performance
   - ❌ Requires migrations for each new field, less flexible, more complex to maintain

2. **HSTORE**
   - ✅ Optimized for flat key-value structures with potentially better performance
   - ✅ No need for migrations when adding new metadata fields
   - ❌ Only supports string values, requiring type conversion
   - ❌ Requires enabling a PostgreSQL extension

3. **JSONB** (chosen)
   - ✅ Preserves native data types (booleans, numbers)
   - ✅ No extension required (built into PostgreSQL)
   - ✅ Team already familiar with JSONB
   - ✅ Provides flexibility for future needs
   - ❌ Potentially slightly slower than HSTORE for key-value lookups (not significant for our scale)

## Rationale

With our dataset being relatively small (~800 records) and our query patterns focused on simple filtering through scopes without complex ordering, the performance advantages of HSTORE were outweighed by JSONB's broader benefits. The familiarity of the team with JSONB, its native data type preservation, and not requiring a database extension made it the most practical choice.
