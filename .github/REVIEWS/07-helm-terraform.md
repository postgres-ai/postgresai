# Review 07: Helm Chart & Terraform Configuration

**Scope**: `postgres_ai_helm/` (Helm chart) and `terraform/aws/` (Terraform module)
**Date**: 2026-03-12

---

## Summary

The Helm chart and Terraform module are generally well-structured with good patterns
(secret management options, configmap checksums, init containers for dependency
ordering). Several security and correctness issues were identified and fixed.

---

## Findings: Helm Chart (`postgres_ai_helm/`)

### FIXED: Sink PostgreSQL allows unauthenticated remote connections (Critical)

**File**: `templates/sink-postgres-configmap.yaml`, `templates/sink-postgres-statefulset.yaml`

The sink PostgreSQL was configured with `POSTGRES_HOST_AUTH_METHOD: trust` and
`pg_hba.conf` allowing `host all all 0.0.0.0/0 trust`. This means any pod in
the cluster (or any network-reachable client) could connect without a password,
bypassing the secret-based password management entirely.

**Fix applied**:
- Changed `POSTGRES_HOST_AUTH_METHOD` from `trust` to `md5`
- Restricted `pg_hba.conf` remote access to RFC 1918 private ranges with `md5` auth
- Local and loopback connections remain `trust` (needed for init scripts)

### FIXED: Invalid SQL syntax in init.sql (Critical)

**File**: `templates/sink-postgres-configmap.yaml`

The init script used MySQL syntax (`CREATE DATABASE IF NOT EXISTS`,
`CREATE USER IF NOT EXISTS`) which is not valid PostgreSQL. This would cause
init script failures on fresh deployments.

**Fix applied**: Replaced with valid PostgreSQL equivalents using `\gexec` and
`DO $$ ... $$` blocks.

### FIXED: Chart version not SemVer compliant (Low)

**File**: `Chart.yaml`

Version was `0.12` instead of `0.12.0`. Helm expects SemVer format.

**Fix applied**: Changed to `0.12.0`.

### NOT FIXED: Grafana datasource ConfigMap leaks secrets (High)

**File**: `templates/grafana-datasources.yaml`

Lines 40 and 26 embed `secrets.postgres.password` and `secrets.vmAuth.password`
directly into a ConfigMap. ConfigMaps are not encrypted and are readable by anyone
with RBAC access to read configmaps in the namespace.

This is a known limitation of the Grafana sidecar datasource provisioning model.
Proper fix would require either:
- Using Grafana's `secureJsonData` with a secret-mounted file
- Creating the datasource ConfigMap as a Secret instead
- Using Grafana Terraform provider or API to configure datasources post-deploy

**Recommendation**: Convert `grafana-datasources.yaml` to render a Secret instead
of a ConfigMap, and configure the Grafana sidecar to read from secrets.

### NOT FIXED: No default resource requests/limits on any container (Medium)

**File**: `values.yaml`

All containers have `resources: {}` as default. In a shared cluster, this means:
- No guaranteed CPU/memory (pods can be evicted under pressure)
- No memory limits (a single component can OOM the node)
- Pods will have `BestEffort` QoS class

**Recommendation**: Add sensible defaults, for example:
```yaml
sinkPostgres:
  resources:
    requests: { cpu: 100m, memory: 256Mi }
    limits: { memory: 1Gi }
victoriaMetrics:
  resources:
    requests: { cpu: 100m, memory: 256Mi }
    limits: { memory: 2Gi }
```

### NOT FIXED: No securityContext on most containers (Medium)

Most containers run as root by default. Only cadvisor explicitly sets
`securityContext: { privileged: true }` (required for its function).

**Recommendation**: Add `securityContext` to pods/containers:
```yaml
securityContext:
  runAsNonRoot: true
  readOnlyRootFilesystem: true  # where feasible
  allowPrivilegeEscalation: false
```

### NOT FIXED: No PodDisruptionBudgets (Low)

No PDBs are defined. During node drains or cluster upgrades, all pods can be
evicted simultaneously causing monitoring downtime.

### NOT FIXED: No NetworkPolicies (Low)

No NetworkPolicies restrict inter-pod communication. Any pod in the namespace
(or cluster, depending on CNI) can reach the sink PostgreSQL, VictoriaMetrics,
and other internal services.

### Ingress defaults to enabled without TLS (Informational)

**File**: `values.yaml`

`ingress.enabled: true` with `tls: []` means Grafana is exposed over HTTP by
default. The example hostname `postgres-ai-monitoring.example.com` is fine for
documentation but users should be aware TLS is not configured out of the box.

### RBAC review: Acceptable (OK)

The ClusterRole grants read-only access to nodes, pods, services, endpoints,
and configmaps, plus `/metrics` non-resource URL. This is the minimum required
for Prometheus service discovery. The ClusterRoleBinding correctly scopes to the
chart's ServiceAccount and namespace.

### Chart.yaml review (OK)

- Dependencies properly declared with condition toggle
- Chart.lock is present and matches
- Metadata (home, sources, maintainers) is populated

---

## Findings: Terraform (`terraform/aws/`)

### FIXED: user_data.sh prints Grafana password to log file (High)

**File**: `terraform/aws/user_data.sh`

The script logged the Grafana password in plaintext to `/var/log/user-data.log`
at the end of installation. This log is readable by any user who can SSH to the
instance.

**Fix applied**:
- Suppressed `grafana-cli admin reset-admin-password` output
- Removed password from echo statements
- Fixed IMDS metadata call to use IMDSv2 (instance requires it per `main.tf`
  metadata_options, but the script was using IMDSv1 curl)

### NOT FIXED: No remote state backend configured (Medium)

**File**: `terraform/aws/main.tf`

State is stored locally by default. The `terraform.tfstate` file contains all
secrets in plaintext (Grafana password, API keys, database connection strings).

The README documents this trade-off and recommends S3+KMS for production. This
is acceptable for the current scope but a `backend "s3"` block (commented out)
would be helpful.

### NOT FIXED: Connection strings stored in Terraform state (Medium)

**File**: `terraform/aws/variables.tf`

`monitoring_instances` includes `conn_str` with embedded passwords. These end up
in `terraform.tfstate` and are passed through `user_data` (which is also stored
in state and AWS API). The README acknowledges this and suggests manual
configuration as an alternative.

### NOT FIXED: Docker installed via convenience script (Low)

**File**: `terraform/aws/user_data.sh`

`curl -fsSL https://get.docker.com | sh` downloads and executes arbitrary code.
While this is Docker's official script, pinning to a specific version or using
apt repository directly would be more reproducible.

### NOT FIXED: No input validation on variables (Low)

**File**: `terraform/aws/variables.tf`

Variables like `grafana_password`, `data_volume_size`, `instance_type` lack
`validation` blocks. For example:
- `grafana_password` could enforce minimum length
- `data_volume_size` could enforce `>= 20`
- `data_volume_type` could be constrained to `["gp3", "gp2", "st1", "sc1"]`

### Security group review (OK)

- SSH restricted to `allowed_ssh_cidr` (no default value forces explicit config)
- Grafana port only opened when `allowed_cidr_blocks` is non-empty
- Egress is fully open (typical for monitoring that needs to reach external DBs)

### EC2 instance configuration (OK)

- IMDSv2 enforced (`http_tokens = "required"`)
- Root and data volumes encrypted
- EBS volume properly attached and formatted
- `lifecycle { ignore_changes = [user_data] }` prevents recreation on config changes

### Outputs review (OK)

- `grafana_credentials` and `deployment_info` are marked `sensitive = true`
- `grafana_access_hint` and `next_steps` provide context-aware instructions
- No raw passwords in non-sensitive outputs

### Documentation review (OK)

- README.md is comprehensive with sizing guidance, security recommendations,
  troubleshooting, and management procedures
- QUICKSTART.md provides a clear step-by-step flow
- terraform.tfvars.example has good comments and warns against `0.0.0.0/0`

### outputs.tf references non-existent output (Minor)

**File**: `terraform/aws/README.md` line 310

References `terraform output -raw security_group_id` but this output is not
defined in `outputs.tf`.

---

## Changes Applied

| File | Change |
|------|--------|
| `postgres_ai_helm/Chart.yaml` | Version `0.12` to `0.12.0` (SemVer) |
| `postgres_ai_helm/templates/sink-postgres-configmap.yaml` | Fixed pg_hba.conf to use `md5` instead of `trust` for remote connections; fixed invalid SQL syntax |
| `postgres_ai_helm/templates/sink-postgres-statefulset.yaml` | Changed `POSTGRES_HOST_AUTH_METHOD` from `trust` to `md5` |
| `terraform/aws/user_data.sh` | Removed password from log output; suppressed grafana-cli output; fixed IMDS to use v2 |
