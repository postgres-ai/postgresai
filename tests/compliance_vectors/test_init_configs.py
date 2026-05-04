"""Contract tests for config/init-configs.sh idempotency.

The config-init service runs `init-configs.sh` on every `docker compose up`
because it is a `service_completed_successfully` dependency of multiple
long-running services. The script must therefore be idempotent: it MUST NOT
clobber user-edited config files (e.g. pgwatch metrics.yml) on subsequent
invocations at the same image version. See issue #175.
"""

import os
import shutil
import subprocess
import tempfile

import pytest


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SCRIPT_PATH = os.path.join(PROJECT_ROOT, 'config', 'init-configs.sh')
VERSION_MARKER = '.pgai-configs-version'


def _make_source_tree(source_dir):
    """Populate a fake $SOURCE_DIR with a couple of nested config files."""
    os.makedirs(os.path.join(source_dir, 'pgwatch-prometheus'))
    os.makedirs(os.path.join(source_dir, 'prometheus'))
    with open(os.path.join(source_dir, 'pgwatch-prometheus', 'metrics.yml'), 'w') as f:
        f.write('# image default metrics.yml\n')
    with open(os.path.join(source_dir, 'prometheus', 'prometheus.yml'), 'w') as f:
        f.write('# image default prometheus.yml\n')


def _run_script(source_dir, target_dir, version_file, version='1.2.3'):
    """Run init-configs.sh with overridden SOURCE_DIR/TARGET_DIR/VERSION_FILE."""
    with open(version_file, 'w') as f:
        f.write(version)
    # BUILD_TS is also read by the script; provide a sibling file.
    build_ts_file = os.path.join(os.path.dirname(version_file), 'BUILD_TS')
    with open(build_ts_file, 'w') as f:
        f.write('2026-04-29T00:00:00Z')
    env = {
        **os.environ,
        'SOURCE_DIR': source_dir,
        'TARGET_DIR': target_dir,
        'VERSION_FILE': version_file,
        'BUILD_TS_FILE': build_ts_file,
    }
    return subprocess.run(
        ['sh', SCRIPT_PATH],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def test_empty_target_copies_files_and_writes_version_marker():
    """First install: empty $TARGET_DIR -> all files copied, marker written."""
    with tempfile.TemporaryDirectory() as tmp:
        source_dir = os.path.join(tmp, 'src')
        target_dir = os.path.join(tmp, 'tgt')
        version_file = os.path.join(tmp, 'VERSION')
        os.makedirs(source_dir)
        os.makedirs(target_dir)
        _make_source_tree(source_dir)

        _run_script(source_dir, target_dir, version_file, version='9.9.9')

        # All source files are copied verbatim.
        copied = os.path.join(target_dir, 'pgwatch-prometheus', 'metrics.yml')
        assert os.path.isfile(copied)
        with open(copied) as f:
            assert f.read() == '# image default metrics.yml\n'
        assert os.path.isfile(os.path.join(target_dir, 'prometheus', 'prometheus.yml'))

        # Version marker reflects the source /VERSION value.
        marker = os.path.join(target_dir, VERSION_MARKER)
        assert os.path.isfile(marker)
        with open(marker) as f:
            assert f.read().strip() == '9.9.9'


def test_matching_version_marker_preserves_user_edits():
    """Re-run at same version: user-edited file MUST survive untouched."""
    with tempfile.TemporaryDirectory() as tmp:
        source_dir = os.path.join(tmp, 'src')
        target_dir = os.path.join(tmp, 'tgt')
        version_file = os.path.join(tmp, 'VERSION')
        os.makedirs(source_dir)
        os.makedirs(target_dir)
        _make_source_tree(source_dir)

        # Simulate state after a successful first run.
        os.makedirs(os.path.join(target_dir, 'pgwatch-prometheus'))
        user_edited = os.path.join(target_dir, 'pgwatch-prometheus', 'metrics.yml')
        user_content = '# user-tuned: calls >= 3 AND exec_time_total >= 1000\n'
        with open(user_edited, 'w') as f:
            f.write(user_content)
        with open(os.path.join(target_dir, VERSION_MARKER), 'w') as f:
            f.write('9.9.9')

        result = _run_script(source_dir, target_dir, version_file, version='9.9.9')

        # User edit preserved verbatim.
        with open(user_edited) as f:
            assert f.read() == user_content
        # Source defaults NOT copied over the user file.
        assert 'image default' not in user_content
        # Script announces the skip.
        assert 'skip' in result.stdout.lower()


def test_mismatched_version_marker_overwrites_target():
    """Image upgrade (different /VERSION) must reseed and update marker."""
    with tempfile.TemporaryDirectory() as tmp:
        source_dir = os.path.join(tmp, 'src')
        target_dir = os.path.join(tmp, 'tgt')
        version_file = os.path.join(tmp, 'VERSION')
        os.makedirs(source_dir)
        os.makedirs(target_dir)
        _make_source_tree(source_dir)

        # Pre-existing volume state at OLD version with a stale user edit.
        os.makedirs(os.path.join(target_dir, 'pgwatch-prometheus'))
        stale_path = os.path.join(target_dir, 'pgwatch-prometheus', 'metrics.yml')
        with open(stale_path, 'w') as f:
            f.write('# stale user edit at v1.0.0\n')
        with open(os.path.join(target_dir, VERSION_MARKER), 'w') as f:
            f.write('1.0.0')

        _run_script(source_dir, target_dir, version_file, version='2.0.0')

        # File overwritten with new image defaults.
        with open(stale_path) as f:
            assert f.read() == '# image default metrics.yml\n'
        # Marker bumped to new version.
        with open(os.path.join(target_dir, VERSION_MARKER)) as f:
            assert f.read().strip() == '2.0.0'


def test_missing_version_marker_on_nonempty_target_overwrites():
    """Pre-existing volume from old image (no marker) must be reseeded once."""
    with tempfile.TemporaryDirectory() as tmp:
        source_dir = os.path.join(tmp, 'src')
        target_dir = os.path.join(tmp, 'tgt')
        version_file = os.path.join(tmp, 'VERSION')
        os.makedirs(source_dir)
        os.makedirs(target_dir)
        _make_source_tree(source_dir)

        # Pre-existing files but no marker (e.g. upgraded from pre-fix image).
        os.makedirs(os.path.join(target_dir, 'pgwatch-prometheus'))
        legacy_path = os.path.join(target_dir, 'pgwatch-prometheus', 'metrics.yml')
        with open(legacy_path, 'w') as f:
            f.write('# legacy pre-fix content\n')

        _run_script(source_dir, target_dir, version_file, version='2.0.0')

        with open(legacy_path) as f:
            assert f.read() == '# image default metrics.yml\n'
        marker = os.path.join(target_dir, VERSION_MARKER)
        assert os.path.isfile(marker)
        with open(marker) as f:
            assert f.read().strip() == '2.0.0'
