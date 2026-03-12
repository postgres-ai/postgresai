# Security Audit Report

**Date**: 2026-03-12
**Scope**: Full codebase security review (CLI, reporter, Flask backend, Docker, Helm, Terraform)
**Auditor**: Automated security audit via Claude Code

---

## Executive Summary

The codebase demonstrates generally strong security practices. The recent SAST remediation (commit 990d865) successfully addressed command injection, SSRF, resource exhaustion, debug mode, and regex injection issues. SQL queries across all components use parameterized queries. Credential handling follows best practices with proper masking, file permissions, and Kubernetes secrets.

One high-severity PromQL injection vulnerability was found in the Flask backend and remediated during this audit. Several medium and low severity findings are documented below.

---

## Findings

### CRITICAL

_No critical findings._

### HIGH

#### H-01: PromQL Injection in Flask Backend (FIXED)

- **CWE**: CWE-74 (Injection)
- **File**: `monitoring_flask_backend/app.py`, lines 388-394, 700-714, 839-851
- **Description**: HTTP request parameters (`cluster_name`, `node_name`, `db_name`, `schemaname`, `tblname`, `idxname`) were interpolated directly into PromQL label matchers without sanitization. An attacker with access to the Flask API could inject arbitrary PromQL by including double quotes or curly braces in parameter values, potentially exfiltrating metric data from other tenants or causing denial of service.
- **Example**: `GET /pgss_metrics/csv?node_name="})+or+vector(1){__name__=~".+"` could escape the label matcher.
- **Fix Applied**: Added `_sanitize_promql_label()` function that escapes backslashes, double quotes, newlines, and strips curly braces. Applied to all 14 filter construction sites across 3 endpoints (`/pgss_metrics/csv`, `/btree_bloat/csv`, `/table_info/csv`).
- **Residual Risk**: The Flask backend is internal-only (not exposed via Docker port mapping). Risk is limited to attackers with access to the Docker network.

### MEDIUM

#### M-01: PromQL Label Values Not Sanitized in Reporter

- **CWE**: CWE-74 (Injection)
- **File**: `reporter/postgres_reports.py`, lines 396, 401, 481, 531-532, 673, 2893, 3053, 3363, 4321, 4733
- **Description**: The reporter module interpolates `cluster` and `node_name` values directly into PromQL queries without escaping. These values come from CLI arguments or environment variables, not HTTP requests, so exploitation requires control over the configuration.
- **Risk**: Low practical risk since values are operator-supplied. However, defense-in-depth suggests sanitizing them.
- **Recommendation**: Add PromQL label escaping to the reporter module matching the Flask backend pattern.

#### M-02: Docker Containers Run as Root

- **CWE**: CWE-250 (Execution with Unnecessary Privileges)
- **Files**: `reporter/Dockerfile`, `monitoring_flask_backend/Dockerfile`
- **Description**: Both the reporter and Flask backend Dockerfiles do not specify a non-root user. Containers run as root by default.
- **Risk**: If an attacker gains code execution inside the container, they have root privileges.
- **Recommendation**: Add `USER` directive to Dockerfiles to run as a non-privileged user.

#### M-03: cAdvisor Runs in Privileged Mode

- **File**: `docker-compose.yml`, line 258
- **Description**: The `self-cadvisor` container runs with `privileged: true`. This grants full host access.
- **Risk**: Container escape is trivial with privileged mode. This is a known cAdvisor requirement but should be documented as an accepted risk.
- **Recommendation**: Document this as an accepted risk and consider alternatives like using `--device` flags instead.

#### M-04: Hardcoded Default Credentials in Docker Compose

- **File**: `docker-compose.yml`, lines 39-40, 70-72, 166-167
- **Description**: Default passwords are set for demo/development: `POSTGRES_PASSWORD: postgres` (target-db, sink-postgres), `GF_SECURITY_ADMIN_PASSWORD: demo`. While `.env.example` encourages changing them, the defaults are weak.
- **Risk**: In a Docker Compose deployment without `.env` customization, services use weak passwords. The sink-postgres also sets `POSTGRES_HOST_AUTH_METHOD: trust`.
- **Mitigating Factor**: The sink-postgres `trust` auth is documented as intentional for internal Docker network use (comment at lines 60-63). PostgreSQL ports are not exposed externally.

#### M-05: Grafana Password Logged in Terraform User Data

- **File**: `terraform/aws/user_data.sh`, lines 157-158
- **Description**: The Grafana password is echoed to the console/log at the end of the user data script: `echo "Password: ${grafana_password}"`. User data logs persist on the EC2 instance at `/var/log/user-data.log`.
- **Risk**: Anyone with SSH access to the instance can read the Grafana password from the log file.
- **Recommendation**: Remove the password echo or redirect only to a file with restricted permissions.

#### M-06: Helm Chart Values Contain Placeholder Secrets

- **File**: `postgres_ai_helm/values.yaml`, lines 168-179
- **Description**: Default values include `CHANGE_ME_*` placeholder passwords. While `createFromValues` defaults to `false`, operators may accidentally enable it with these values.
- **Risk**: If deployed with defaults, predictable passwords are used.
- **Mitigating Factor**: `createFromValues: false` by default; documented warnings in comments.

### LOW

#### L-01: Self-Signed TLS Accepted by Default

- **File**: `cli/lib/init.ts`, lines 51-53, 397, 453
- **Description**: The CLI defaults to `{ rejectUnauthorized: false }` for SSL connections (mimicking `sslmode=prefer`). This is a deliberate design choice documented in code, matching libpq's default behavior.
- **Risk**: MITM attacks are possible if the network is untrusted.
- **Mitigating Factor**: Users can set `PGSSLMODE=verify-full` to enforce certificate validation.

#### L-02: API Key Sent Over HTTP Warns But Allows

- **File**: `cli/lib/storage.ts`, lines 93-95, 185-187
- **Description**: When the storage URL uses HTTP, the CLI warns but still sends the API key. This is intentional for localhost development.
- **Risk**: API key exposed in plaintext if used over non-localhost HTTP.
- **Mitigating Factor**: Warning is displayed; production URLs use HTTPS.

#### L-03: Pre-commit Hooks Only Include Gitleaks

- **File**: `.pre-commit-config.yaml`
- **Description**: Only gitleaks (v8.30.0) is configured. No static analysis, linting, or other security hooks.
- **Recommendation**: Consider adding hooks for Python (bandit), TypeScript (eslint-plugin-security), and Dockerfile linting (hadolint).

#### L-04: Flask Error Responses May Leak Internal Details

- **File**: `monitoring_flask_backend/app.py`, lines 341, 500, 650, etc.
- **Description**: Several endpoints return `str(e)` in error responses (`jsonify({"error": str(e)})`). Exception messages may contain internal paths, hostnames, or connection details.
- **Risk**: Information disclosure to API consumers.
- **Mitigating Factor**: Flask backend is internal-only (not externally exposed).

#### L-05: Terraform Private Key Path Hardcoded

- **File**: `terraform/aws/main.tf`, line 254
- **Description**: SSH private key path is hardcoded as `file("~/.ssh/${var.ssh_key_name}.pem")`. This assumes a specific key naming convention.
- **Risk**: Low. Operational concern rather than security vulnerability.

---

## SAST Remediation Verification (Commit 990d865)

| CWE | Fix | Verification |
|-----|-----|-------------|
| CWE-78 (Command Injection) | Replaced `execPromise()` (uses `child_process.exec` with shell) with `execFilePromise()` (uses `child_process.execFile` without shell) across 7 call sites | PASS - `execPromise` function fully removed. All process execution uses `execFilePromise` (no shell) or `spawn` (with explicit args array). No `exec()` calls remain. |
| CWE-918 (SSRF) | Added `ALLOWED_HOSTS` allowlist to `embed-checkup-dictionary.ts`; Added `isValidProjectRef()` validation in `supabase.ts` before URL construction | PASS - Both fixes verified. Supabase client validates project ref format in constructor AND in `fetchPoolerDatabaseUrl`. |
| CWE-770 (Resource Exhaustion) | Added `timeout=30` to all `requests.get/post` calls in reporter | PASS - All 3 `requests` calls in reporter now have `timeout=30`. Flask backend uses `PrometheusConnect` which has its own timeout handling. |
| CWE-489 (Debug) | Changed `debug=True` to `debug=False` in Flask `__main__` block | PASS - `app.run(debug=False)` confirmed. Production uses gunicorn (Dockerfile CMD). |
| CWE-185 (Regex) | Escaped `instanceName` in regex construction | PASS - `instanceName.replace(/[.*+?^${}()\|[\]\\]/g, "\\$&")` applied before `new RegExp()`. |

**Overall SAST Assessment**: All 5 CWE remediations are complete and correct. No regressions found.

---

## SQL Injection Review

All SQL queries across all components use parameterized queries or static SQL. No SQL injection vulnerabilities found.

| Component | Files Reviewed | Method | Result |
|-----------|---------------|--------|--------|
| CLI (TypeScript) | `checkup.ts`, `init.ts` | `pg` client with `$1` params and static inline SQL | PASS |
| Reporter (Python) | `postgres_reports.py` | `psycopg2` with `%s` params | PASS |
| Flask Backend (Python) | `app.py` | `psycopg2` with `%s` params | PASS |
| CLI Init SQL | `sql/*.sql` | Template variables use `quoteIdent()`/`quoteLiteral()` with null-byte rejection | PASS |

---

## Credential Handling Review

| Area | Assessment |
|------|-----------|
| Config file permissions | `config.ts`: directory created with `0o700`, files with `0o600` |
| Connection string masking | `init.ts`: `maskConnectionString()` replaces password with `*****` |
| SQL password redaction | `init.ts`: `redactPasswordsInSql()` redacts PASSWORD literals in verbose output |
| API key masking | `util.ts`: `maskSecret()` shows partial key in debug output |
| Terraform sensitive vars | `variables.tf`: `grafana_password`, `postgres_ai_api_key`, `vm_auth_password` marked `sensitive = true` |
| Terraform config files | `user_data.sh`: `.pgwatch-config` and `.env` created with `chmod 600` |
| Helm secrets | K8s secrets used by default; `createFromValues: false` prevents values.yaml secrets |
| .gitignore | `.env`, `.pgwatch-config` properly gitignored |
| Gitleaks | Pre-commit hook configured (v8.30.0) |

---

## VictoriaMetrics Basic Auth Review (Commit 46ed2f3)

The implementation correctly:
- Passes `VM_AUTH_USERNAME` and `VM_AUTH_PASSWORD` via environment variables to the VM container
- Conditionally enables auth only when both username and password are set (shell conditional in docker-compose.yml entrypoint)
- Propagates credentials to reporter, Flask backend, and Grafana containers
- Reporter and Flask backend both check `VM_AUTH_USERNAME`/`VM_AUTH_PASSWORD` env vars and configure HTTP Basic Auth
- Helm chart stores VM auth password in Kubernetes Secret (key: `vm-auth-password`)
- Terraform marks `vm_auth_password` as `sensitive = true`

**Assessment**: Implementation is correct and complete.

---

## Dependency Review

### Python (reporter/requirements.txt)
| Package | Version | Notes |
|---------|---------|-------|
| requests | 2.32.5 | Current |
| psycopg2-binary | 2.9.11 | Current |
| jsonschema | 4.23.0 | Current |
| requests-aws4auth | 1.2.3 | Current |
| boto3 | 1.34.69 | Slightly older; no known security issues |

### Python (monitoring_flask_backend/requirements.txt)
| Package | Version | Notes |
|---------|---------|-------|
| Flask | 3.0.0 | Current major version |
| gunicorn | 23.0.0 | Current |
| requests | 2.32.4 | Minor version behind reporter (2.32.5) |
| psycopg2-binary | 2.9.9 | Slightly behind reporter (2.9.11) |
| pytest | 8.3.4 | Dev dependency in production image |

**Recommendation**: Remove `pytest` from `monitoring_flask_backend/requirements.txt` -- it should only be in dev requirements. Pin versions consistently across components.

---

## SECURITY.md Review

The `SECURITY.md` file at repository root:
- Provides clear reporting instructions (email: security@postgres.ai)
- Specifies 3 working day acknowledgment SLA
- Describes disclosure timeline (typically 7 days)
- Covers when to report and when not to

**Assessment**: Clear and complete.

---

## Docker Security Review

| Container | Runs as Root | Exposed Ports | Secrets in Env |
|-----------|-------------|---------------|----------------|
| reporter | Yes | None | VM_AUTH_* |
| flask-backend | Yes | None (internal only) | VM_AUTH_*, POSTGRES_SINK_URL |
| sink-prometheus | No (VM default) | 59090 (configurable bind host) | VM_AUTH_* |
| grafana | No (Grafana default) | 3000 (configurable bind host) | GF_SECURITY_ADMIN_PASSWORD, OAuth secrets, VM_AUTH_* |
| target-db | No (postgres default) | None | POSTGRES_PASSWORD |
| sink-postgres | No (postgres default) | None | POSTGRES_PASSWORD (trust auth) |
| self-cadvisor | Root + privileged | None | None |

**Key Concerns**:
1. Reporter and Flask backend run as root (M-02)
2. cAdvisor runs privileged (M-03)
3. Credentials passed via environment variables (standard Docker practice, but visible via `docker inspect`)

---

## Summary of Actions Taken

1. **FIXED** (H-01): Added `_sanitize_promql_label()` to Flask backend and applied it to all 14 PromQL filter construction sites across 3 endpoints.

## Recommendations (Not Fixed)

1. (M-01) Add PromQL label sanitization to the reporter module
2. (M-02) Add non-root USER to reporter and Flask Dockerfiles
3. (M-05) Remove password echo from Terraform user_data.sh
4. (L-03) Add additional pre-commit security hooks (bandit, eslint-plugin-security)
5. Remove `pytest` from Flask backend production requirements
6. Pin dependency versions consistently across Python components
