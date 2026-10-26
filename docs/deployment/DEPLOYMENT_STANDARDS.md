# Deployment Standards & Pipeline

The authoritative runbook is [DEPLOYMENT.md](../DEPLOYMENT.md): production
configuration/TLS, probes, scaling limits, observability, backups, migrations,
and rollback. Use locked dependencies and tested, identifiable build artifacts.

The [CI workflow](../../.github/workflows/ci.yml) validates pull requests;
it does not establish that staging or production deployment occurred.
Keep secrets outside source control and obtain authorization for production changes.
