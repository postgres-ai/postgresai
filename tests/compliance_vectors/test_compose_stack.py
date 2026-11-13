"""Contract tests for docker-compose.yml (the pgai-mon stack) and .env.example.

What the CLI, terraform, the reporter, instance-jobs and Grafana rely on: the
service set, pgai-mon published on 127.0.0.1 only and behind a required token,
its liveness probe, mounts and owner, resource defaults that .env.example
documents, Grafana's bind and opt-out, and no trace of the old stack.

Rendered with ``docker compose config`` (the substitution users hit at
runtime); the tests that need it skip without Docker, the raw-YAML ones always
run.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.yml"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

REQUIRED_ENV = {
    "PGAI_TAG": "test",
    "PGAI_MON_TOKEN": "test-token-0123456789",
    "REPLICATOR_PASSWORD": "test-replicator-password",
    "GF_SECURITY_ADMIN_PASSWORD": "test-grafana-password",
}

# Per-service limits: variable and default (bytes / CPUs). .env.example documents each.
SERVICE_LIMITS: dict[str, tuple[str, str, str, str]] = {
    "target-db": ("TARGET_DB_CPUS", "0.2", "TARGET_DB_MEM", str(768 * 1024**2)),
    "target-standby": ("TARGET_STANDBY_CPUS", "0.2", "TARGET_STANDBY_MEM", str(768 * 1024**2)),
    "pgai-mon": ("PGAI_MON_CPUS", "1.0", "PGAI_MON_MEM", str(1536 * 1024**2)),
    "grafana": ("GRAFANA_CPUS", "0.5", "GRAFANA_MEM", str(512 * 1024**2)),
    "postgres-reports": ("POSTGRES_REPORTS_CPUS", "1.0", "POSTGRES_REPORTS_MEM", str(1792 * 1024**2)),
    "instance-jobs": ("INSTANCE_JOBS_CPUS", "0.25", "INSTANCE_JOBS_MEM", str(256 * 1024**2)),
}

OLD_STACK_SERVICES = {
    "sink-postgres", "sink-prometheus", "vmalert", "pgwatch-postgres", "pgwatch-prometheus",
    "sources-generator", "monitoring_flask_backend", "rds-host-stats",
    "self-cadvisor", "self-node-exporter", "self-postgres-exporter",
}

PGAI_MON_URL = "http://pgai-mon:8765"


def _docker_compose() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "compose", "version"], capture_output=True).returncode == 0


needs_compose = pytest.mark.skipif(not _docker_compose(), reason="docker compose not available")

_RESOURCE_VARS = {v for spec in SERVICE_LIMITS.values() for v in (spec[0], spec[2])}
_STRIPPED = _RESOURCE_VARS | {"GRAFANA_BIND_HOST", "GRAFANA_SCALE", "PGAI_MON_USER", "INSTANCE_JOBS_USER",
                              "PGAI_MON_STORE", "COMPOSE_PROFILES", "COMPOSE_FILE", "COMPOSE_PROJECT_NAME",
                              "PGAI_MON_RETENTION_DAYS", "PGAI_MON_TEXTS_RETENTION_DAYS",
                              "VM_RETENTION_PERIOD", "QUERYID_RETENTION_HOURS"}


def render(extra: dict[str, str] | None = None, drop: tuple[str, ...] = (), profiles: tuple[str, ...] = ("demo", "instance-jobs")) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in _STRIPPED}
    env.update(REQUIRED_ENV)
    env.update(extra or {})
    for k in drop:
        env.pop(k, None)
    args = ["docker", "compose", "-f", str(COMPOSE_PATH), "--project-directory", str(PROJECT_ROOT)]
    # An .env in the checkout must not leak into the rendering: point compose at an empty one.
    args += ["--env-file", os.devnull]
    for p in profiles:
        args += ["--profile", p]
    return subprocess.run(args + ["config", "--format", "yaml"], capture_output=True, text=True, env=env)


def rendered(extra: dict[str, str] | None = None, **kw: Any) -> dict[str, Any]:
    r = render(extra, **kw)
    assert r.returncode == 0, f"docker compose config failed:\n{r.stderr}"
    return yaml.safe_load(r.stdout)


def raw() -> dict[str, Any]:
    return yaml.safe_load(COMPOSE_PATH.read_text())


def env_of(service: dict[str, Any]) -> dict[str, str]:
    e = service.get("environment") or {}
    if isinstance(e, list):
        return dict(item.split("=", 1) for item in e)
    return {k: "" if v is None else str(v) for k, v in e.items()}


# --- raw YAML (always runs) -------------------------------------------------------------------


def test_service_set():
    services = set(raw()["services"])
    assert services == {"config-init", "target-db", "target-standby", "pgai-mon", "grafana",
                        "postgres-reports", "instance-jobs"}
    assert not services & OLD_STACK_SERVICES


def test_no_old_stack_references():
    text = COMPOSE_PATH.read_text()
    for needle in ("sink-prometheus", "sink-postgres", "VM_AUTH", "PROMETHEUS_URL", "pgwatch-prometheus",
                   "victoria", ":9090"):
        assert needle not in text, f"docker-compose.yml still mentions {needle}"


def test_pgai_mon_service_contract():
    svc = raw()["services"]["pgai-mon"]
    assert svc["image"] == "${PGAI_REGISTRY:-postgresai}/pgai-mon:${PGAI_TAG:?PGAI_TAG is required}"
    assert svc["ports"] == ["127.0.0.1:8765:8765"]
    assert "/livez" in " ".join(svc["healthcheck"]["test"])
    assert svc["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in svc["security_opt"]
    vols = svc["volumes"]
    assert "pgai_mon_data:/var/lib/pgai-mon" in vols
    assert "./instances.yml:/etc/pgai-mon/instances.yml:ro" in vols
    assert "./host-metrics:/etc/pgai/host-metrics:ro" in vols
    assert svc["depends_on"]["config-init"]["condition"] == "service_completed_successfully"


def test_own_images_follow_registry_and_tag():
    services = raw()["services"]
    for name, image in (("config-init", "postgres-ai-configs"), ("pgai-mon", "pgai-mon"),
                        ("postgres-reports", "reporter"), ("instance-jobs", "instance-jobs")):
        assert services[name]["image"] == f"${{PGAI_REGISTRY:-postgresai}}/{image}:${{PGAI_TAG:?PGAI_TAG is required}}", name


def test_terraform_generates_the_token_or_stops():
    """The Terraform VM writes PGAI_MON_TOKEN into .env itself; a failed generation is fatal."""
    text = (PROJECT_ROOT / "terraform" / "aws" / "user_data.sh").read_text()
    assert re.search(r"for secret_name in [^;\n]*\bPGAI_MON_TOKEN\b", text)
    assert "openssl rand -hex 32" in text and "FATAL: cannot generate $secret_name" in text


def test_consumers_point_at_pgai_mon():
    services = raw()["services"]
    for name in ("postgres-reports", "instance-jobs"):
        e = env_of(services[name])
        assert e["PGAI_MON_URL"] == PGAI_MON_URL, name
        assert e["PGAI_MON_TOKEN"].startswith("${PGAI_MON_READ_TOKEN:-${PGAI_MON_TOKEN:?"), name
        assert services[name]["depends_on"]["pgai-mon"]["condition"] == "service_healthy", name
    assert env_of(services["grafana"])["PGAI_MON_TOKEN"].startswith("${PGAI_MON_READ_TOKEN:-${PGAI_MON_TOKEN:?")
    assert env_of(services["postgres-reports"])["REPORTER_SOURCES_PATH"] == "/app/instances.yml"
    assert "./instances.yml:/app/instances.yml:ro" in services["postgres-reports"]["volumes"]


@pytest.mark.parametrize("service,spec", sorted(SERVICE_LIMITS.items()), ids=sorted(SERVICE_LIMITS))
def test_resource_limit_indirection(service, spec):
    cpus_var, cpus, mem_var, mem = spec
    svc = raw()["services"][service]
    assert svc["cpus"] == f"${{{cpus_var}:-{cpus}}}"
    assert svc["mem_limit"] == f"${{{mem_var}:-{mem}}}"


def test_env_example_documents_the_stack():
    text = ENV_EXAMPLE.read_text()
    assert re.search(r"^PGAI_MON_TOKEN=$", text, re.M), "PGAI_MON_TOKEN must be an empty required placeholder"
    for var in (
        "PGAI_MON_STORE",
        "PGAI_MON_RETENTION_DAYS",
        "PGAI_MON_TEXTS_RETENTION_DAYS",
        "PGAI_MON_USER",
        "GRAFANA_BIND_HOST",
        "GRAFANA_SCALE",
        "RDS_DB_INSTANCE_IDENTIFIER",
    ):
        assert re.search(rf"^#\s*{var}=", text, re.M), f".env.example must document {var}"
    documented = dict(re.findall(r"^#\s*([A-Z0-9_]+)=([^\s#]+)", text, re.M))
    for cpus_var, cpus, mem_var, mem in SERVICE_LIMITS.values():
        assert documented.get(cpus_var) == cpus, cpus_var
        assert documented.get(mem_var) == mem, mem_var
    # VM_RETENTION_PERIOD and QUERYID_RETENTION_HOURS stay: pgai-mon still reads them
    for gone in ("VM_AUTH", "SINK_", "PGWATCH_", "FLASK_", "CADVISOR_"):
        assert gone not in text, f".env.example still mentions {gone}"


# --- rendered (docker compose config) --------------------------------------------------------


@needs_compose
def test_token_tag_and_grafana_password_are_required():
    for var in ("PGAI_MON_TOKEN", "PGAI_TAG", "GF_SECURITY_ADMIN_PASSWORD"):
        r = render(drop=(var,))
        assert r.returncode != 0 and var in r.stderr, f"compose must refuse to start without {var}"


@needs_compose
@pytest.mark.parametrize("service,spec", sorted(SERVICE_LIMITS.items()), ids=sorted(SERVICE_LIMITS))
def test_resource_defaults_render(service, spec):
    svc = rendered()["services"][service]
    assert float(svc["cpus"]) == pytest.approx(float(spec[1]))
    assert int(svc["mem_limit"]) == int(spec[3])


@needs_compose
def test_resource_override_renders():
    svc = rendered({"PGAI_MON_MEM": str(3 * 1024**3)})["services"]["pgai-mon"]
    assert int(svc["mem_limit"]) == 3 * 1024**3


@needs_compose
def test_only_loopback_ports_by_default():
    for name, svc in rendered()["services"].items():
        for port in svc.get("ports", []):
            assert port.get("host_ip") == "127.0.0.1", f"{name} publishes {port} beyond 127.0.0.1"


@needs_compose
def test_grafana_bind_host_empty_means_all_interfaces():
    """terraform's grafana_bind_host = "" and the CLI's explicit empty value open Grafana."""
    port = rendered({"GRAFANA_BIND_HOST": ""})["services"]["grafana"]["ports"][0]
    assert "host_ip" not in port and str(port["published"]) == "3000"
    pgai = rendered({"GRAFANA_BIND_HOST": ""})["services"]["pgai-mon"]["ports"][0]
    assert pgai["host_ip"] == "127.0.0.1"


@needs_compose
def test_grafana_can_be_left_out():
    assert rendered()["services"]["grafana"].get("scale", 1) == 1
    assert rendered({"GRAFANA_SCALE": "0"})["services"]["grafana"]["scale"] == 0


@needs_compose
def test_pgai_mon_runs_as_the_install_owner():
    services = rendered()["services"]
    assert services["pgai-mon"]["user"] == "10001:10001"
    assert env_of(services["config-init"])["PGAI_MON_OWNER"] == "10001:10001"
    services = rendered({"INSTANCE_JOBS_USER": "1000:1000"})["services"]
    assert services["pgai-mon"]["user"] == "1000:1000"
    assert env_of(services["config-init"])["PGAI_MON_OWNER"] == "1000:1000"
    services = rendered({"INSTANCE_JOBS_USER": "1000:1000", "PGAI_MON_USER": "1001:1001"})["services"]
    assert services["pgai-mon"]["user"] == "1001:1001"


@needs_compose
def test_store_defaults_to_the_local_volume():
    services = rendered()["services"]
    assert env_of(services["pgai-mon"])["PGAI_MON_STORE"] == ""
    store = "s3://bucket/pgai-mon/x"
    assert env_of(rendered({"PGAI_MON_STORE": store})["services"]["pgai-mon"])["PGAI_MON_STORE"] == store


@needs_compose
def test_retention_settings_reach_pgai_mon():
    """Unset: empty, so the image's config decides (14 and 30 days). The old stack's
    variables are passed too: pgai-mon honours them when the new ones are unset."""
    names = ("PGAI_MON_RETENTION_DAYS", "PGAI_MON_TEXTS_RETENTION_DAYS", "VM_RETENTION_PERIOD", "QUERYID_RETENTION_HOURS")
    env = env_of(rendered()["services"]["pgai-mon"])
    assert [env[n] for n in names] == ["", "", "", ""]
    values = dict(zip(names, ("60", "90", "13w", "9600")))
    env = env_of(rendered(values)["services"]["pgai-mon"])
    assert {n: env[n] for n in names} == values


@needs_compose
def test_secrets_are_not_on_any_command_line():
    services = rendered()["services"]
    for name, svc in services.items():
        cmd = " ".join(map(str, (svc.get("command") or []) + (svc.get("entrypoint") or [])))
        assert REQUIRED_ENV["PGAI_MON_TOKEN"] not in cmd, name


def test_the_grafana_plugin_is_pinned():
    """Grafana downloads the plugin at start: without a version every restart may run new code
    that holds the read token."""
    text = COMPOSE_PATH.read_text()
    plugins = re.search(r"^\s*GF_INSTALL_PLUGINS:\s*(.+)$", text, re.M).group(1)
    for plugin in plugins.split(","):
        assert re.fullmatch(r"[a-z0-9-]+ \d+\.\d+\.\d+", plugin.strip()), plugin
