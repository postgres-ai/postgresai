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
