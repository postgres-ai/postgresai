# Preview environment runbook

Operations guide for the PostgresAI monitoring preview environments.

## Architecture

- **VM Host:** Hetzner VM (cpx31, 4 vCPU / 8GB)
- **Reverse Proxy:** Traefik with DNS-01 wildcard SSL via Cloudflare
- **DNS:** Dynamic A records created per preview via Cloudflare API
- **Max Concurrent:** 2 previews

## Access

### SSH to the preview VM

```bash
ssh -i ~/.ssh/preview-deploy deploy@<PREVIEW_VM_HOST>
```

### Traefik dashboard (via SSH tunnel)

```bash
ssh -L 8080:localhost:8080 -i ~/.ssh/preview-deploy deploy@<PREVIEW_VM_HOST>
# Then open http://localhost:8080
```

## Common operations

### List running previews

```bash
ssh deploy@<VM> "docker ps --filter 'label=pgai.preview=true' --format '{{.Names}}' | grep grafana"
```

### View preview credentials

```bash
ssh deploy@<VM> "cat /opt/postgres-ai-previews/previews/{BRANCH_SLUG}/.env"
```

### View preview state

```bash
ssh deploy@<VM> "cat /opt/postgres-ai-previews/previews/{BRANCH_SLUG}/state.json"
```

### Manually deploy a preview

```bash
ssh deploy@<VM> "BRANCH_SLUG=my-branch COMMIT_SHA=abc123 /opt/postgres-ai-previews/manager/deploy.sh"
```

### Manually destroy a preview

```bash
ssh deploy@<VM> "BRANCH_SLUG=my-branch /opt/postgres-ai-previews/manager/destroy.sh"
```

### View cleanup logs

```bash
ssh deploy@<VM> "tail -50 /opt/postgres-ai-previews/manager/cleanup.log"
```

### Run cleanup manually

```bash
ssh deploy@<VM> "/opt/postgres-ai-previews/manager/cleanup-ttl.sh"
```

## Troubleshooting

### Preview deployment fails

1. Check quota:
   ```bash
   ssh deploy@<VM> "docker ps --filter 'label=pgai.preview=true' | wc -l"
   ```
   Should be less than 18 (2 previews x 9 containers each)

2. Check disk space:
   ```bash
   ssh deploy@<VM> "df -h /opt/postgres-ai-previews"
   ```

3. Check memory:
   ```bash
   ssh deploy@<VM> "free -m"
   ```

4. Check deploy logs:
   ```bash
   ssh deploy@<VM> "ls -la /opt/postgres-ai-previews/previews/{BRANCH_SLUG}/"
   ```

### Grafana not accessible

1. Check if container is running:
   ```bash
   ssh deploy@<VM> "docker ps | grep preview-{BRANCH_SLUG}-grafana"
   ```

2. Check container health:
   ```bash
   ssh deploy@<VM> "docker inspect --format='{{.State.Health.Status}}' preview-{BRANCH_SLUG}-grafana-1"
   ```

3. Check Grafana logs:
   ```bash
   ssh deploy@<VM> "docker logs preview-{BRANCH_SLUG}-grafana-1 --tail=50"
   ```

### DNS not resolving

1. Check Cloudflare DNS record:
   ```bash
   dig preview-{BRANCH_SLUG}.pgai.watch A
   ```

2. Manually create DNS record:
   ```bash
   ssh deploy@<VM> "source /opt/postgres-ai-previews/.env && /opt/postgres-ai-previews/scripts/cloudflare-dns.sh create preview-{BRANCH_SLUG}"
   ```

### SSL certificate issues

1. Check Traefik logs:
   ```bash
   ssh deploy@<VM> "docker logs traefik --tail=100 | grep -i 'acme\|cert\|error'"
   ```

2. Check acme.json permissions:
   ```bash
   ssh deploy@<VM> "ls -la /opt/postgres-ai-previews/traefik/acme.json"
   ```

### Metrics not showing in Grafana

1. Check pgwatch is running:
   ```bash
   ssh deploy@<VM> "docker ps | grep preview-{BRANCH_SLUG}-pgwatch"
   ```

2. Check pgwatch logs:
   ```bash
   ssh deploy@<VM> "docker logs preview-{BRANCH_SLUG}-pgwatch-1 --tail=50"
   ```

3. Check VictoriaMetrics is receiving data:
   ```bash
   ssh deploy@<VM> "docker exec preview-{BRANCH_SLUG}-sink-prometheus-1 wget -qO- 'http://localhost:9090/api/v1/query?query=up'"
   ```

## Recovery procedures

### Full preview stack restart

```bash
ssh deploy@<VM> "cd /opt/postgres-ai-previews/previews/{BRANCH_SLUG} && docker compose -p preview-{BRANCH_SLUG} down && docker compose -p preview-{BRANCH_SLUG} up -d"
```

### Traefik restart

```bash
ssh deploy@<VM> "cd /opt/postgres-ai-previews/traefik && docker compose restart traefik"
```

### Force cleanup all previews (emergency)

```bash
ssh deploy@<VM> "docker stop \$(docker ps -q --filter 'label=pgai.preview=true') && docker rm \$(docker ps -aq --filter 'label=pgai.preview=true') && docker volume prune -f && rm -rf /opt/postgres-ai-previews/previews/*"
```

## Monitoring

### VM health check

The cleanup cron runs every 30 minutes and logs to `/opt/postgres-ai-previews/manager/cleanup.log`.

### Disk usage alert

The cleanup script triggers Docker image prune when disk usage exceeds 80%.

## GitLab CI variables

Required CI variables for preview deployments:

| Variable | Type | Description |
|----------|------|-------------|
| `PREVIEW_SSH_PRIVATE_KEY` | File | SSH private key for deploy user |
| `PREVIEW_VM_HOST` | Variable | IP address of preview VM |

## Cloudflare configuration

Required environment variables on the VM (in `/opt/postgres-ai-previews/.env`):

| Variable | Description |
|----------|-------------|
| `CF_DNS_API_TOKEN` | Cloudflare API token with DNS edit permissions |
| `CF_ZONE_ID` | Cloudflare zone ID for pgai.watch |
| `VM_PUBLIC_IP` | Public IP of the preview VM |
