---
title: "Deployment Workflow"
---

## GitLab Deployment Workflow

The following diagram illustrates the complete deployment workflow from developer commit to production deployment on GitLab.com and self-managed releases.

```mermaid
graph TB
    subgraph "Developer Workflow"
        A[Developer commits to feature branch] --> B[Create Merge Request]
        B --> C[CI/CD Pipeline runs]
        C --> D[Code Review & Approval]
        D --> E[Merge to default branch]
    end
    
    subgraph "GitLab.com Deployment Path"
        E --> F[Deployer Pipeline Triggered]
        
        F --> G["Deploy to Staging Canary<br/>gstg-cny"]
        F -.Parallel.-> H["Deploy to Staging Ref<br/>gstg-ref<br/>Sandbox Testing"]
        
        G --> I["E2E Smoke Tests<br/>targeting gstg-cny (blocking)"]
        
        I --> J["Deploy to Production Canary<br/>gprd-cny"]
        
        J --> K["E2E Smoke Tests<br/>targeting gprd-cny (blocking)"]
        
        K --> L["Wait Period<br/>30 minutes"]
        
        L --> M[Manual Promotion]
        
        M --> N["Deploy to Staging<br/>gstg"]
        Q --> O["Deploy to Production<br/>gprd"]
        
        M --> Q[Production Checks]
        
        O --> R[Live on GitLab.com]
    end
    
    subgraph "Self-Managed Release Path"
        E --> S[Merged to default branch]
        S --> T["Monthly release cycle<br/>3rd Thursday of month"]
        T --> U["Create stable branch<br/>e.g., 17-2-stable-ee"]
        U --> V["Tag version<br/>e.g., v17.2.0"]
        V --> W["Build release packages<br/>Omnibus, Docker, etc."]
        W --> X[Publish to packages.gitlab.com]
        X --> Y["Available for Self-Managed<br/>installations"]
        
        V --> Z["Patch releases<br/>Twice monthly"]
        Z --> AA["Tag patch version<br/>e.g., v17.2.1"]
        AA --> W
    end
    
    subgraph "Documentation Updates"
        E --> AB[Docs updated on main]
        AB --> AC["Deployed to docs.gitlab.com<br/>within 1 hour"]
        U --> AD["Stable branch docs<br/>e.g., docs.gitlab.com/17.2/"]
    end

    style R fill:#1f77b4
    style Y fill:#ff7f0e
    style AC fill:#2ca02c
    style H fill:#9467bd
```

## Post-Deployment Migrations

Post-deployment migrations have their own dedicated workflow and are not guaranteed to run within a specific timeframe. The deployment team reserves the right to run them when deemed necessary. However, they are always executed prior to release management tasks.

For more details, see the [Post-Deploy Migration (PDM) Execution](/handbook/engineering/deployments-and-releases/deployments/#post-deploy-migration-pdm-execution) section in the deployments handbook.

## Related Documentation

For additional context on the deployment process, refer to the [Deploying Packages](/handbook/engineering/deployments-and-releases/deployments/#deploying-packages) section in the deployments and releases handbook.
