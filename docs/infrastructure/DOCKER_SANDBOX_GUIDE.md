# Docker Sandbox Guide (Docker Compose)

**What**: Run multi-container applications defined in `docker-compose.yml`.
**Why**: Ensures consistent environments for all developers.

## Quick Start

### 1. Build and Run Sandbox
```bash
docker-compose up --build
```
*   `--build`: Rebuilds images if Dockerfile changed.
*   Add `-d` to run in detached mode (background).

### 2. Verify Services
```bash
docker-compose ps
```
You should see:
*   `llm-doc-intelligence-app` (Backend)
*   `llm-doc-intelligence-frontend`
*   `postgres`
*   `redis`

### 3. Debugging (Python)
The backend is configured to listen for a debugger on port `5678`.
1.  Run the stack: `docker-compose up`
2.  Open VS Code.
3.  Go to "Run and Debug".
4.  Select **"Docker: Attach to Backend"**.
5.  Set breakpoints in your code.

## Troubleshooting

**Service failed to start?**
```bash
docker-compose logs [service_name]
# Example: docker-compose logs app
```

**Need a clean slate?**
```bash
docker-compose down -v
# -v removes volumes (database data). Use with caution!
```
