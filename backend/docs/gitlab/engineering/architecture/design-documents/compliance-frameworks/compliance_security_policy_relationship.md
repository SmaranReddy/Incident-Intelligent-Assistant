---
title: "Compliance Security Policies Relationship"
toc_hide: true
---

## Context

This documents the full relationship between Compliance Frameworks and Security Policies.

- Security Policy Project can be linked with either Project or Namespace (the record in `security_orchestration_policy_configurations` table is then created, with `security_policy_management_project_id` used to store information about selected Security Policy Project),
- Policies are defined in the Security Policy Project in the `policy.yml` file, and they are also represented in the security_policies table,
- A single policy can be scoped to multiple Compliance Frameworks (through the `compliance_framework_security_policies` join table), although you can leave the policy unscoped or scoped to a selected group or project; when the policy is unscoped, it affects all projects/namespaces linked to associated Security Policy Project.
  - For more details on Security Policy scoping refer the docs here https://docs.gitlab.com/ee/user/application_security/policies/#scope
- For a given Compliance Framework, you can define many Requirements (represented in the `compliance_requirements` table),
- A single Requirement can be associated with multiple Security Policies (through the `security_policy_requirements` join table), and a single Security Policy can be associated with multiple Requirements as well; the link between Requirement and Security Policy allows user to use selected Security Policy as the enforcement mechanism for selected Requirement

## Entity relationship diagram

```mermaid
erDiagram
    projects ||--o| security_orchestration_policy_configurations : "links with"
    namespaces ||--o| security_orchestration_policy_configurations : "links with"
    security_orchestration_policy_configurations ||--|| projects : "stores policies in"
    security_orchestration_policy_configurations ||--o{ security_policies : contains
    security_policies ||--|{ compliance_framework_security_policies : "links through"
    compliance_framework_security_policies }|--|| compliance_management_frameworks : "scopes to"
    compliance_management_frameworks ||--o{ compliance_requirements : defines
    compliance_requirements ||--o{ security_policy_requirements : "associates with"
    security_policy_requirements |o--o| security_policies : "associates with"
    
    projects {
        int id PK
        string name
        string path
    }

    namespaces {
        int id PK
        string name
        string path
    }

    compliance_management_frameworks {
        int id PK
        string name
        string description
    }

    compliance_framework_security_policies {
        int id PK
        int compliance_management_framework_id FK
        int security_policy_id FK
    }

    security_policies {
        int id PK
        string name
        string description
        int security_orchestration_policy_configuration_id FK
    }

    security_policy_requirements {
        int id PK
        int security_policy_id FK
        int compliance_requirement_id FK
    }

    compliance_requirements {
        int id PK
        int compliance_management_framework_id FK
        string description
    }

    security_orchestration_policy_configurations {
        int id PK
        int project_id FK "configuration can be linked either to project_id"
        int namespace_id FK "or namespace_id, but not both"
        int security_policy_management_project_id FK "defines project where we keep policy.yml file"
    }
```
