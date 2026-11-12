# old/: the pgwatch + VictoriaMetrics stack

pgai-mon (`mon/`) replaced this stack. What is here is kept for reference only:
CI does not build its images or run its tests, and nothing in it is shipped.

- `pgwatch/`: the patched pgwatch image (collector)
- `monitoring_flask_backend/`: the Flask API Grafana and the reporter used for CSV and query texts
- `rds-host-stats/`: the RDS/Aurora host metrics pusher
- `postgres_ai_helm/`: the Helm chart, which deploys this stack
- `old-stack.docker-compose.yml`: the last Compose file of this stack
- `config/`: sink-postgres init, pgwatch-postgres `metrics.yml`, the Prometheus/VictoriaMetrics
  scrape config and vmalert host rules, the pgwatch sources generator, the PGWatch Grafana
  datasources, and the PromQL Grafana dashboards
- `scripts/rotate-vm-auth.sh`: VictoriaMetrics credential rotation
- `tests/`: the tests of the above, as they were (paths inside them are the pre-move ones)

Still read from here by live code: `mon/grafana/compile.py` compiles the pgai-mon dashboards
from `config/grafana/dashboards/`, and `scripts/sync-wait-event-colors.py` and
`cli/test/config-consistency.test.ts` read those dashboards too; a few CLI and reporter tests
read `config/pgwatch-postgres/metrics.yml`.
