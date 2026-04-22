# PostgresAI monitoring helm chart

Kubernetes helm chart for deploying the PostgresAI monitoring stack, including PGWatch, VictoriaMetrics, Grafana, and automated reporting.

## Quick start

```bash
# Install with default values
helm install postgres-ai-monitoring ./postgres_ai_helm

# Install with custom values
helm install postgres-ai-monitoring ./postgres_ai_helm -f custom-values.yaml

# Upgrade existing installation
helm upgrade postgres-ai-monitoring ./postgres_ai_helm
```

## Documentation

- **[INSTALLATION_GUIDE.md](./INSTALLATION_GUIDE.md)**: Complete installation and configuration guide
- **[RELEASE.md](./RELEASE.md)**: Release process and versioning guide
- **[values.yaml](./values.yaml)**: Default configuration values

## Components

This helm chart deploys the following components:

- **PGWatch**: Monitors Postgres databases and collects metrics
- **VictoriaMetrics**: Time-series database for storing metrics
- **Grafana**: Visualization and dashboards, including Dashboard 14 I/O statistics powered by `pg_stat_io` for PostgreSQL 16+
- **Node exporter**: System-level metrics
- **cAdvisor**: Container metrics
- **Reporter**: Automated health check reports

## Configuration

The chart can be customized via `values.yaml`. Key configuration options:

```yaml
# Enable/disable components
grafana:
  enabled: true

victoriametrics:
  enabled: true

# Configure database targets
pgwatch:
  databases:
    - host: postgres.example.com
      port: 5432
      dbname: mydb

# Resource limits
resources:
  limits:
    cpu: 2000m
    memory: 4Gi
```

See [values.yaml](./values.yaml) for all available options.

## Requirements

- Kubernetes 1.19+
- Helm 3.0+
- PersistentVolume provisioner (for VictoriaMetrics and Grafana storage)

## Installing from releases

Download a specific release from [GitLab releases](https://gitlab.com/postgres-ai/postgresai/-/releases):

```bash
# Download the chart (replace <VERSION> with the desired release, e.g. 0.14.0)
curl -LO https://gitlab.com/postgres-ai/postgresai/-/releases/helm-v<VERSION>/downloads/postgres-ai-monitoring-chart.tgz

# Install
helm install postgres-ai-monitoring postgres-ai-monitoring-chart.tgz
```

## Development

### Testing locally

```bash
# Test release script logic (semver validation, sed injection, git operations)
./test-release-logic.sh

# Full helm chart validation (dependencies, linting, templates, packaging)
./test-release.sh

# Or test individual components:
# Update dependencies
helm dependency update

# Lint the chart
helm lint .

# Render templates
helm template test-release .

# Dry-run installation
helm install test-release . --dry-run --debug
```

### Creating a release

To create a new release of the helm chart, see [RELEASE.md](./RELEASE.md).

Quick release:

```bash
./release.sh <VERSION>   # e.g.: ./release.sh 0.14.0
```

This will automatically:
- Update Chart.yaml
- Create a git tag
- Trigger the release workflow
- Package and publish the chart

## Automated workflows

The chart includes GitLab CI/CD pipelines for:

- **validate-helm-chart**: Runs `test-release-logic.sh` unit/integration tests then validates chart on merge requests
- **release-helm-chart**: Automated releases when tags are pushed

## Support

- **Issues**: https://gitlab.com/postgres-ai/postgresai/-/issues
- **Documentation**: https://postgres.ai/docs
- **Community**: https://postgres.ai/community

## License

See [LICENSE](../LICENSE) file for details.



