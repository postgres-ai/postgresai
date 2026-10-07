"""Tests for config/scripts/postgres-reports.sh daemon-loop branches.

These tests exercise the real shell script with a stubbed ``python`` on PATH.
The stub records the arguments it was invoked with and exits non-zero, which
(under ``set -e``) terminates the daemon loop after exactly one cycle — no
real sleeps, no real reporter run.

Covered branches:

1. api_key + project_name  -> upload mode (--api-url/--project-name/--token)
2. api_key, no project_name -> warning + local-only generation (--no-upload)
3. no api_key               -> local-only generation (--no-upload)

Branch 2 is the regression fixed by MR !341: previously the loop logged
"skipping upload this cycle" but skipped the whole cycle, generating no
reports at all. Against that version, the stub is never invoked and the
script sleeps for REPORTER_INTERVAL_SECONDS, so the test times out and fails.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "config" / "scripts" / "postgres-reports.sh"

# The stub exits with this code; ``set -e`` propagates it as the script's
# exit code, proving the reporter was invoked exactly once.
STUB_EXIT_CODE = 7

# Generous cap for one no-sleep cycle; only reached when the script hangs
# (i.e. a cycle that never invokes the reporter — the pre-fix bug).
TIMEOUT_SECONDS = 15


def run_one_cycle(tmp_path: Path, config_content=None):
    """Run one daemon cycle of postgres-reports.sh with a stubbed python.

    Returns (completed_process, recorded_args) where recorded_args is the
    list of arguments the stubbed ``python`` was invoked with, or None if it
    was never invoked.
    """
    stub_dir = tmp_path / "stub-bin"
    stub_dir.mkdir()
    args_file = tmp_path / "python-args.txt"

    stub = stub_dir / "python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'printf \'%s\\n\' "$@" > "$STUB_ARGS_FILE"\n'
        f"exit {STUB_EXIT_CODE}\n"
    )
    stub.chmod(0o755)

    config_path = tmp_path / ".pgwatch-config"
    if config_content is not None:
        config_path.write_text(config_content)

    env = os.environ.copy()
    env.pop("REPORTER_PROJECT_NAME", None)
    env.pop("USE_CURRENT_TIME", None)
    env.update(
        {
            "PATH": f"{stub_dir}:{env['PATH']}",
            "STUB_ARGS_FILE": str(args_file),
            "REPORTER_PGWATCH_CONFIG_PATH": str(config_path),
            "REPORTER_INITIAL_DELAY_SECONDS": "0",
            # Large on purpose: if a cycle ever finishes without invoking the
            # reporter (the pre-fix bug), the loop sleeps and the test fails
            # via timeout instead of silently passing on a later cycle.
            "REPORTER_INTERVAL_SECONDS": "86400",
            "REPORTER_OUTPUT_TEMPLATE": str(tmp_path / "all_reports_%Y%m%d_%H%M%S.json"),
        }
    )

    try:
        proc = subprocess.run(
            ["bash", str(SCRIPT)],
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "postgres-reports.sh completed a daemon cycle without invoking the "
            "reporter (python stub never ran) — no reports would be generated"
        )

    recorded = None
    if args_file.exists():
        recorded = args_file.read_text().splitlines()
    return proc, recorded


@pytest.mark.unit
def test_api_key_and_project_name_uploads(tmp_path):
    """With api_key + project_name the reporter is invoked in upload mode."""
    proc, args = run_one_cycle(
        tmp_path,
        "api_key=secret-token-123\nproject_name=my-project\n",
    )

    assert proc.returncode == STUB_EXIT_CODE, proc.stderr
    assert args is not None, "reporter was never invoked"
    assert args[:2] == ["-m", "reporter.postgres_reports"]
    assert "--api-url" in args
    assert "--project-name" in args
    assert args[args.index("--project-name") + 1] == "my-project"
    assert "--token" in args
    assert args[args.index("--token") + 1] == "secret-token-123"
    assert "--no-upload" not in args
    assert "generating reports (upload enabled)" in proc.stdout


@pytest.mark.unit
def test_api_key_without_project_name_generates_local_reports(tmp_path):
    """With api_key but no project_name: warn, then still generate locally.

    Regression test for MR !341 — before the fix this branch skipped the
    whole cycle (no reporter invocation at all), so this test fails against
    the old script.
    """
    proc, args = run_one_cycle(tmp_path, "api_key=secret-token-123\n")

    assert "project name is required for upload" in proc.stderr
    assert proc.returncode == STUB_EXIT_CODE, proc.stderr
    assert args is not None, "reporter was never invoked"
    assert args[:2] == ["-m", "reporter.postgres_reports"]
    assert "--no-upload" in args
    # Credentials must not leak into the local-only invocation.
    assert "--project-name" not in args
    assert "--token" not in args
    assert "generating reports (no upload)" in proc.stdout


@pytest.mark.unit
def test_no_api_key_generates_local_reports(tmp_path):
    """Without an api_key the reporter runs in local-only mode."""
    proc, args = run_one_cycle(tmp_path, config_content=None)

    assert proc.returncode == STUB_EXIT_CODE, proc.stderr
    assert args is not None, "reporter was never invoked"
    assert args[:2] == ["-m", "reporter.postgres_reports"]
    assert "--no-upload" in args
    assert "--token" not in args
    assert "generating reports (no upload)" in proc.stdout


def run_one_cycle_logging_calls(tmp_path: Path, config_content: str, env_override=None):
    """Like run_one_cycle, but the stub logs every call: its argv and stdin.

    The stub fails every call; a call the script tolerates is followed by the
    next one, and the reporter's failure ends the cycle.
    """
    stub_dir = tmp_path / "stub-bin"
    stub_dir.mkdir()
    calls_dir = tmp_path / "calls"
    calls_dir.mkdir()
    stub = stub_dir / "python"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'n=$(ls "$STUB_CALLS_DIR"/*.args 2>/dev/null | wc -l | tr -d " ")\n'
        'printf \'%s\\n\' "$@" > "$STUB_CALLS_DIR/$n.args"\n'
        'printf \'%s\' "${REPORTER_API_URL:-}" > "$STUB_CALLS_DIR/$n.url"\n'
        'if [ -t 0 ]; then : > "$STUB_CALLS_DIR/$n.stdin"; else cat > "$STUB_CALLS_DIR/$n.stdin"; fi\n'
        f"exit {STUB_EXIT_CODE}\n"
    )
    stub.chmod(0o755)
    config_path = tmp_path / ".pgwatch-config"
    config_path.write_text(config_content)
    env = os.environ.copy()
    env.pop("REPORTER_PROJECT_NAME", None)
    env.update(
        {
            "PATH": f"{stub_dir}:{env['PATH']}",
            "STUB_CALLS_DIR": str(calls_dir),
            "REPORTER_PGWATCH_CONFIG_PATH": str(config_path),
            "REPORTER_API_URL": "https://api.example.test/api/general",
            "REPORTER_INITIAL_DELAY_SECONDS": "0",
            "REPORTER_INTERVAL_SECONDS": "86400",
            "REPORTER_OUTPUT_TEMPLATE": str(tmp_path / "all_reports_%Y%m%d_%H%M%S.json"),
        }
    )
    for key, value in (env_override or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    proc = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
        stdin=subprocess.DEVNULL, timeout=TIMEOUT_SECONDS,
    )
    calls = []
    for i in range(len(list(calls_dir.glob("*.args")))):
        calls.append(((calls_dir / f"{i}.args").read_text().splitlines(),
                      (calls_dir / f"{i}.stdin").read_text()))
    return proc, calls


@pytest.mark.unit
def test_api_key_is_renewed_each_cycle_before_the_reports(tmp_path):
    """A box token lives 90 days (internal#354): each cycle renews it first.

    The token goes on stdin, never argv; a failed renewal does not stop the
    cycle's reports.
    """
    proc, calls = run_one_cycle_logging_calls(
        tmp_path, "api_key=secret-token-123\nproject_name=my-project\n"
    )

    assert proc.returncode == STUB_EXIT_CODE, proc.stderr
    assert len(calls) == 2, calls
    renew_args, renew_stdin = calls[0]
    assert renew_args == ["-m", "reporter.token_renew", "https://api.example.test/api/general"]
    assert renew_stdin == "secret-token-123\n"
    assert calls[1][0][:2] == ["-m", "reporter.postgres_reports"]


@pytest.mark.unit
def test_no_api_key_renews_nothing(tmp_path):
    proc, calls = run_one_cycle_logging_calls(tmp_path, "project_name=my-project\n")

    assert proc.returncode == STUB_EXIT_CODE, proc.stderr
    assert [args[:2] for args, _ in calls] == [["-m", "reporter.postgres_reports"]]


@pytest.mark.unit
@pytest.mark.parametrize("env_url", ["", None])
@pytest.mark.parametrize("config_url", [
    "https://preview.example.test/api/general",
    "https://preview.example.test/api/general/",
    None,
])
def test_api_url_falls_back_to_config_then_production(tmp_path, env_url, config_url):
    config = "api_key=test-key\nproject_name=my-project\n"
    if config_url is not None:
        config += f"api_url={config_url}\n"
    proc, calls = run_one_cycle_logging_calls(
        tmp_path, config, {"REPORTER_API_URL": env_url}
    )

    assert proc.returncode == STUB_EXIT_CODE, proc.stderr
    assert len(calls) == 2
    expected = config_url.rstrip("/") if config_url else "https://postgres.ai/api/general"
    assert calls[0][0] == ["-m", "reporter.token_renew", expected]
    for i in range(2):
        assert (tmp_path / "calls" / f"{i}.url").read_text() == expected
    args = calls[1][0]
    assert args[:2] == ["-m", "reporter.postgres_reports"]
    assert args[args.index("--api-url") + 1] == expected


@pytest.mark.unit
def test_nonempty_api_url_env_wins_over_config(tmp_path):
    expected = "https://override.example.test/api/general"
    proc, calls = run_one_cycle_logging_calls(
        tmp_path,
        "api_key=test-key\nproject_name=my-project\napi_url=https://preview.example.test/api/general\n",
        {"REPORTER_API_URL": expected},
    )

    assert proc.returncode == STUB_EXIT_CODE, proc.stderr
    assert len(calls) == 2
    assert calls[0][0] == ["-m", "reporter.token_renew", expected]
    for i in range(2):
        assert (tmp_path / "calls" / f"{i}.url").read_text() == expected
    args = calls[1][0]
    assert args[:2] == ["-m", "reporter.postgres_reports"]
    assert args[args.index("--api-url") + 1] == expected


@pytest.mark.unit
def test_projects_file_enables_upload_without_legacy_project_name(tmp_path):
    projects = tmp_path / ".pgai-report-projects.json"
    projects.write_text('{"projects": ["app", "app2"]}')
    proc, calls = run_one_cycle_logging_calls(tmp_path, "api_key=synthetic-token\n")
    assert proc.returncode == STUB_EXIT_CODE
    args = calls[-1][0]
    assert "--no-upload" not in args
    assert "--token" in args
    assert "project name is required" not in proc.stderr
