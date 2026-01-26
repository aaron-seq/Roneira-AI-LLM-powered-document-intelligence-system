# Deployment Standards & Pipeline

## 1. Core Concepts
*   **Frozen Production Build**: Production artifacts must have locked dependencies (no `latest` tags) and optimized assets. Ensure consistent behavior across envs.
*   **Reverse Proxy**: Always use **Nginx** or a cloud load balancer in front of the application. Never expose application servers (Node/Python) directly to the internet.
*   **SSL/HTTPS**: Mandatory for all production services. Use Let's Encrypt for automation.

## 2. Infrastructure & Networking
*   **SSH Tunneling**: Use SSH keys to manage access. Disable password login on VMs.
*   **Port Forwarding**: Tunnel remote internal ports (e.g., Mongo `27017`) to localhost for secure access tools.
*   **DNS**: Use A Records for IP mapping, CNAME for aliases.

## 3. CI/CD Pipeline (GitHub Flow)
1.  **Development**: Feature branch -> PR -> Review -> Merge to `main`.
2.  **CI (Automated)**:
    *   Linting (ESLint/Flake8).
    *   Testing (Unit/Integration).
    *   Build Docker Image.
3.  **CD (Automated/Manual)**:
    *   Push to Registry (Docker Hub/ECR).
    *   Deploy to Staging (Auto).
    *   Deploy to Production (Approval).

## 4. Environment Config
*   **Secrets**: Store in Vault/Secrets Manager or `.env` file (not committed).
*   **Config Separation**: Code is immutable; config varies by deployment (12-Factor App).

## 5. Observability
*   **Telemetry**: Do not expose stack traces or debug info in production.
*   **Logs**: Centralized logging (ELK/Loki).
