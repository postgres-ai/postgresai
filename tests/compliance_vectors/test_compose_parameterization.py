"""Contract tests for env-var driven resource limits in docker-compose.yml.

These tests verify that:

1. Every per-service ``cpus`` / ``mem_limit`` and the parameterized
   VictoriaMetrics tuning flags resolve to the **documented default** when no
   override env var is set. This guards the "no behavior change for laptop-dev
   users" promise from issue #176.
2. Two representative services prove the env-var override path is real and
   not a typo; raw-YAML assertions exhaustively cover the rest so the tests
   stay fast on hosts without Docker.

The tests prefer ``docker compose config`` because it exercises the same
substitution path that real users hit at runtime. If ``docker`` is not
available the test falls back to a string match against the raw YAML so the
tests still run in minimal CI environments and on laptops without Docker.

Note: ``VM_RETENTION_PERIOD`` is parameterized in main (MR !259) and is
covered by separate tests; this module focuses on the per-service mem/cpu
vars and the additional VictoriaMetrics tuning flags introduced by MR !252.
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


# Documented defaults per service. Memory values are bytes (matches the
# convention introduced by !238 / !248). CPUs are floats. The defaults must
# match the current effective values on main so unset env vars produce no
# behavioral change for laptop-dev users.
SERVICE_DEFAULTS: dict[str, dict[str, Any]] = {
    "target-db": {
        "cpus_var": "TARGET_DB_CPUS",
        "cpus_default": "0.2",
        "mem_var": "TARGET_DB_MEM",
        "mem_default": str(768 * 1024 * 1024),  # 768 MiB
    },
    "target-standby": {
        "cpus_var": "TARGET_STANDBY_CPUS",
        "cpus_default": "0.2",
        "mem_var": "TARGET_STANDBY_MEM",
        "mem_default": str(768 * 1024 * 1024),  # 768 MiB
    },
    "sink-postgres": {
        "cpus_var": "SINK_POSTGRES_CPUS",
        "cpus_default": "0.4",
        "mem_var": "SINK_POSTGRES_MEM",
        "mem_default": str(1024 * 1024 * 1024),  # 1 GiB
    },
    "sink-prometheus": {
        "cpus_var": "SINK_PROMETHEUS_CPUS",
        "cpus_default": "0.75",
        "mem_var": "SINK_PROMETHEUS_MEM",
        "mem_default": str(1536 * 1024 * 1024),  # 1.5 GiB
    },
    "pgwatch-postgres": {
        "cpus_var": "PGWATCH_POSTGRES_CPUS",
        "cpus_default": "0.35",
        "mem_var": "PGWATCH_POSTGRES_MEM",
        "mem_default": str(512 * 1024 * 1024),  # 512 MiB
    },
    "pgwatch-prometheus": {
        "cpus_var": "PGWATCH_PROMETHEUS_CPUS",
        "cpus_default": "0.5",
        "mem_var": "PGWATCH_PROMETHEUS_MEM",
        "mem_default": str(512 * 1024 * 1024),  # 512 MiB
    },
    "grafana": {
        "cpus_var": "GRAFANA_CPUS",
        "cpus_default": "0.5",
        "mem_var": "GRAFANA_MEM",
        "mem_default": str(512 * 1024 * 1024),  # 512 MiB
    },
    "monitoring_flask_backend": {
        "cpus_var": "FLASK_CPUS",
        "cpus_default": "0.5",
        "mem_var": "FLASK_MEM",
        "mem_default": str(1024 * 1024 * 1024),  # 1 GiB
    },
    "postgres-reports": {
        "cpus_var": "POSTGRES_REPORTS_CPUS",
        "cpus_default": "1.0",
        "mem_var": "POSTGRES_REPORTS_MEM",
        "mem_default": str(1792 * 1024 * 1024),  # 1.75 GiB
    },
    "self-cadvisor": {
        "cpus_var": "CADVISOR_CPUS",
        "cpus_default": "0.25",
        "mem_var": "CADVISOR_MEM",
        "mem_default": str(384 * 1024 * 1024),  # 384 MiB
    },
    "self-node-exporter": {
        "cpus_var": "NODE_EXPORTER_CPUS",
        "cpus_default": "0.05",
        "mem_var": "NODE_EXPORTER_MEM",
        "mem_default": str(96 * 1024 * 1024),  # 96 MiB
    },
    "self-postgres-exporter": {
        "cpus_var": "POSTGRES_EXPORTER_CPUS",
        "cpus_default": "0.1",
        "mem_var": "POSTGRES_EXPORTER_MEM",
        "mem_default": str(128 * 1024 * 1024),  # 128 MiB
    },
}


# VictoriaMetrics tuning flags injected into the sink-prometheus entrypoint
# shell. Defaults preserve current main behavior (16 concurrent requests,
# 30s max query duration). Only flags that already existed in main's
# sink-prometheus command are parameterized; flags that would otherwise be
# added new (e.g. ``-memory.allowedPercent``, ``-search.maxQueueDuration``)
# are intentionally out of scope to keep this MR a no-op by default.
#
# ``VM_RETENTION_PERIOD`` is parameterized on main (MR !259) and asserted
# elsewhere; not duplicated here.
VM_FLAG_DEFAULTS: dict[str, dict[str, str]] = {
    "VM_MAX_CONCURRENT_REQUESTS": {
        "flag": "-search.maxConcurrentRequests=",
        "default": "16",
    },
    "VM_QUERY_DURATION": {
        "flag": "-search.maxQueryDuration=",
        "default": "30s",
    },
}


RESOURCE_VARS = frozenset(
    item
    for spec in SERVICE_DEFAULTS.values()
    for item in (spec["cpus_var"], spec["mem_var"])
) | frozenset(VM_FLAG_DEFAULTS)


# Env-var values required to pass schema validation when invoking
# ``docker compose config``. None of them affect the assertions below.
COMPOSE_REQUIRED_ENV: dict[str, str] = {
    "PGAI_TAG": "test",
    "REPLICATOR_PASSWORD": "test-replicator-password",
    "VM_AUTH_USERNAME": "test-vmauth",
    "VM_AUTH_PASSWORD": "test-vmauth-password",
}


def _expected_env_defaults() -> dict[str, str]:
    defaults = {
        spec["cpus_var"]: spec["cpus_default"]
        for spec in SERVICE_DEFAULTS.values()
    }
    defaults.update(
        {
            spec["mem_var"]: spec["mem_default"]
            for spec in SERVICE_DEFAULTS.values()
        }
    )
    defaults.update(
        {var: info["default"] for var, info in VM_FLAG_DEFAULTS.items()}
    )
    return defaults


def _docker_compose_available() -> bool:
    """Return True if ``docker compose config`` can run on this host."""
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "compose", "version"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _render_compose_config(extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run ``docker compose config`` and return the rendered YAML as a dict."""
    env = {**os.environ, **COMPOSE_REQUIRED_ENV}
    if extra_env:
        env.update(extra_env)
    # Strip variables this test never sets so the rendered output reflects the
    # in-file defaults (the user's shell may have FLASK_CPUS exported).
    for key in list(env):
        if key in RESOURCE_VARS and (not extra_env or key not in extra_env):
            env.pop(key, None)

    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_PATH), "config"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        pytest.fail(
            "docker compose config failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return yaml.safe_load(result.stdout)


def _raw_compose_text() -> str:
    return COMPOSE_PATH.read_text()


def _service_environment(service: dict[str, Any]) -> dict[str, str]:
    environment = service.get("environment", {})
    if isinstance(environment, dict):
        return {str(key): str(value) for key, value in environment.items()}
    if isinstance(environment, list):
        return {
            item.split("=", 1)[0]: item.split("=", 1)[1]
            for item in environment
            if isinstance(item, str) and "=" in item
        }
    raise AssertionError(f"unsupported environment type: {type(environment).__name__}")


def _bytes_from_compose_value(value: Any) -> int:
    """Normalize a compose ``mem_limit`` value to bytes.

    docker compose v2 renders ``mem_limit`` as an integer (bytes). Older
    output forms may use a suffix string like ``1g``, so accept both.
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.fullmatch(r"(\d+)([kmgKMG]?)", value.strip())
        if match is None:
            raise AssertionError(f"unrecognised mem value: {value!r}")
        amount = int(match.group(1))
        suffix = match.group(2).lower()
        multiplier = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[suffix]
        return amount * multiplier
    raise AssertionError(f"unsupported mem value type: {type(value).__name__}")


# ---------------------------------------------------------------------------
# Path A: docker compose config available -- exercise real substitution
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _docker_compose_available(),
    reason="docker compose not available; covered by raw-yaml fallback below",
)
@pytest.mark.parametrize(
    "service_name,spec",
    sorted(SERVICE_DEFAULTS.items()),
    ids=sorted(SERVICE_DEFAULTS),
)
def test_service_resource_defaults_render_via_compose_config(service_name, spec):
    rendered = _render_compose_config()
    service = rendered["services"][service_name]

    assert float(service["cpus"]) == pytest.approx(float(spec["cpus_default"]))
    assert _bytes_from_compose_value(service["mem_limit"]) == int(spec["mem_default"])


@pytest.mark.skipif(
    not _docker_compose_available(),
    reason="docker compose not available; covered by raw-yaml fallback below",
)
def test_sink_prometheus_default_vm_flags_via_compose_config():
    rendered = _render_compose_config()
    sink_prometheus = rendered["services"]["sink-prometheus"]
    environment = _service_environment(sink_prometheus)
    cmd_block = sink_prometheus.get("command")
    # docker compose may render command as a list of strings or a single
    # string; normalize to one string for substring searches.
    if isinstance(cmd_block, list):
        cmd_text = " ".join(cmd_block)
    else:
        cmd_text = str(cmd_block)

    for var, info in VM_FLAG_DEFAULTS.items():
        assert environment[var] == info["default"]
        assert f"{info['flag']}\"$${{{var}}}\"" in cmd_text, (
            f"sink-prometheus command must read {var} from the runtime "
            "environment with shell quoting"
        )


@pytest.mark.skipif(
    not _docker_compose_available(),
    reason="docker compose not available; covered by raw-yaml fallback below",
)
@pytest.mark.parametrize(
    "service_name,override_var,override_value,expected_bytes",
    [
        (
            "sink-prometheus",
            "SINK_PROMETHEUS_MEM",
            str(2 * 1024 * 1024 * 1024),
            2 * 1024 * 1024 * 1024,
        ),
        (
            "monitoring_flask_backend",
            "FLASK_MEM",
            str(3 * 1024 * 1024 * 1024),
            3 * 1024 * 1024 * 1024,
        ),
    ],
    ids=["sink_prometheus_mem_override", "flask_mem_override"],
)
def test_representative_mem_overrides_propagate_via_compose_config(
    service_name, override_var, override_value, expected_bytes
):
    rendered = _render_compose_config(extra_env={override_var: override_value})
    service = rendered["services"][service_name]
    assert _bytes_from_compose_value(service["mem_limit"]) == expected_bytes


# ---------------------------------------------------------------------------
# Path B: raw-YAML fallback. Always runs. These guard the in-file pattern
# even on hosts without docker, and they catch typos in the indirection
# itself (wrong var name, missing default).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "service_name,spec",
    sorted(SERVICE_DEFAULTS.items()),
    ids=sorted(SERVICE_DEFAULTS),
)
def test_service_resource_indirection_in_raw_yaml(service_name, spec):
    text = _raw_compose_text()
    expected_cpus = f"cpus: ${{{spec['cpus_var']}:-{spec['cpus_default']}}}"
    expected_mem = f"mem_limit: ${{{spec['mem_var']}:-{spec['mem_default']}}}"
    assert expected_cpus in text, (
        f"{service_name}: missing or wrong cpus indirection. Expected line "
        f"contents: {expected_cpus!r}"
    )
    assert expected_mem in text, (
        f"{service_name}: missing or wrong mem_limit indirection. Expected "
        f"line contents: {expected_mem!r}"
    )


@pytest.mark.parametrize(
    "var,info",
    sorted(VM_FLAG_DEFAULTS.items()),
    ids=sorted(VM_FLAG_DEFAULTS),
)
def test_vm_flag_indirection_in_raw_yaml(var, info):
    text = _raw_compose_text()
    expected_env = f"- {var}=${{{var}:-{info['default']}}}"
    expected_flag = f"{info['flag']}\"$${{{var}}}\""
    assert expected_env in text, (
        "missing VictoriaMetrics env default indirection: expected substring "
        f"{expected_env!r}"
    )
    assert expected_flag in text, (
        "missing VictoriaMetrics runtime flag expansion: expected substring "
        f"{expected_flag!r}"
    )


def test_vm_retention_period_remains_parameterized():
    """``VM_RETENTION_PERIOD`` is owned by MR !259 / main; this MR must not
    drop or rename it. Spot-check the env block to keep the rebase honest.
    """
    text = _raw_compose_text()
    assert "- VM_RETENTION_PERIOD=${VM_RETENTION_PERIOD:-336h}" in text


def test_env_example_documents_every_new_variable_with_default_values():
    env_text = (PROJECT_ROOT / ".env.example").read_text()
    expected_defaults = _expected_env_defaults()
    documented_defaults = {
        match.group(1): match.group(2)
        for match in re.finditer(
            r"^#\s*([A-Z0-9_]+)=([^\s#]+)",
            env_text,
            flags=re.MULTILINE,
        )
    }

    missing = [v for v in expected_defaults if v not in documented_defaults]
    assert not missing, (
        ".env.example must mention every new tuning variable so users can "
        f"discover overrides. Missing: {missing}"
    )

    wrong_defaults = {
        var: {"expected": expected, "actual": documented_defaults[var]}
        for var, expected in expected_defaults.items()
        if var in documented_defaults and documented_defaults[var] != expected
    }
    assert not wrong_defaults, (
        ".env.example must document the same defaults used by compose. "
        f"Mismatches: {wrong_defaults}"
    )
