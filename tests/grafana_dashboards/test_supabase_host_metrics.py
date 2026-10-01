"""The Supabase host metrics scrape contract. The Host row is in test_host_row.py."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_supabase_scrape_is_internal() -> None:
    config = yaml.safe_load((ROOT / 'config/prometheus/prometheus.yml').read_text())
    job = next(j for j in config['scrape_configs'] if j['job_name'] == 'supabase-host-metrics')
    assert job['scrape_interval'] == '60s'
    assert job['scrape_timeout'] == '20s'
    assert job['metrics_path'] == '/supabase/metrics'
    # The target and its cluster/node_name labels come from the file that
    # sources-generator writes from instances.yml, as pgwatch's labels do.
    assert job['file_sd_configs'] == [{'files': ['/postgres_ai_configs/prometheus/supabase-host-metrics.json']}]
    assert 'static_configs' not in job
    assert 'http_sd_configs' not in job
    # Only a keep/drop gate on the feature flag; nothing may rewrite the address.
    assert job['relabel_configs'] == [
        {'target_label': '__tmp_enabled', 'replacement': '%{PGAI_SUPABASE_HOST_METRICS}'},
        {'source_labels': ['__tmp_enabled'], 'regex': '(?i)\\s*true\\s*', 'action': 'keep'},
    ]
    assert job['metric_relabel_configs'] == [{'source_labels': ['__name__'], 'regex': 'node_cpu_seconds_total|node_memory_.*|node_disk_.*|node_network_.*_bytes_total|node_filesystem_.*|node_load.*', 'action': 'keep'}]
    compose = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())
    assert 'PGAI_SUPABASE_HOST_METRICS=${PGAI_SUPABASE_HOST_METRICS:-false}' in compose['services']['sink-prometheus']['environment']
    service = compose['services']['instance-jobs']
    assert not service.get('ports')
    assert not any('instances.yml' in volume for volume in service['volumes'])


def _active_jobs(flag: str, tmp: Path) -> list[dict]:
    """Start the pinned VictoriaMetrics with our scrape config; return active target jobs."""
    import shutil, subprocess, time, urllib.request, uuid
    import pytest
    if not shutil.which('docker') or subprocess.run(['docker', 'info'], capture_output=True, timeout=30).returncode:
        pytest.skip('docker unavailable')
    name = f'pgai-test-vm-{uuid.uuid4().hex[:8]}'
    try:
        targets = tmp / 'supabase-host-metrics.json'
        targets.write_text(json.dumps([{'targets': ['instance-jobs:9188'], 'labels': {'cluster': 'c', 'node_name': 'n'}}]))
        subprocess.run(['docker', 'run', '-d', '--rm', '--name', name, '-p', '127.0.0.1::8428',
                        '-v', f'{ROOT}/config/prometheus/prometheus.yml:/p.yml:ro',
                        '-v', f'{targets}:/postgres_ai_configs/prometheus/supabase-host-metrics.json:ro',
                        '-e', 'VM_AUTH_USERNAME=u', '-e', 'VM_AUTH_PASSWORD=p', '-e', f'PGAI_SUPABASE_HOST_METRICS={flag}',
                        'victoriametrics/victoria-metrics:v1.140.0',
                        '-promscrape.config=/p.yml', '-promscrape.config.strictParse=false'],
                       check=True, capture_output=True, timeout=120)
        port = subprocess.run(['docker', 'port', name, '8428/tcp'], check=True, capture_output=True,
                              text=True, timeout=30).stdout.split()[0].rsplit(':', 1)[1]
        deadline = time.time() + 30
        while True:
            try:
                data = json.load(urllib.request.urlopen(f'http://127.0.0.1:{port}/api/v1/targets', timeout=2))['data']
                if data['activeTargets']:
                    return [t['labels'] for t in data['activeTargets']]
            except OSError:
                pass
            if time.time() > deadline:
                raise AssertionError('VictoriaMetrics did not report targets')
            time.sleep(0.5)
    finally:
        subprocess.run(['docker', 'rm', '-f', name], capture_output=True, timeout=60)


def test_supabase_target_only_when_enabled(tmp_path: Path) -> None:
    assert 'supabase-host-metrics' not in [t['job'] for t in _active_jobs('false', tmp_path)]
    supabase = [t for t in _active_jobs('true', tmp_path) if t['job'] == 'supabase-host-metrics']
    assert len(supabase) == 1
    assert (supabase[0]['cluster'], supabase[0]['node_name']) == ('c', 'n')


GENERATOR = ROOT / 'config/scripts/generate-pgwatch-sources.sh'
TARGET = 'instance-jobs:9188'


def _generate(tmp: Path, instances: str) -> tuple[list, str]:
    import subprocess
    (tmp / 'instances.yml').write_text(instances)
    run = subprocess.run(['bash', str(GENERATOR)], capture_output=True, text=True, timeout=30,
                         env={'PATH': '/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin', 'INSTANCES_PATH': str(tmp / 'instances.yml'), 'CONFIGS_DIR': str(tmp / 'out')})
    assert run.returncode == 0, run.stderr
    return json.loads((tmp / 'out/prometheus/supabase-host-metrics.json').read_text()), run.stderr


def _entry(name: str, conn: str, cluster: str, node: str, enabled: str = 'true') -> str:
    return (f'- name: {name}\n  conn_str: {conn}\n  preset_metrics: full\n  custom_metrics:\n  is_enabled: {enabled}\n'
            f'  group: default\n  custom_tags:\n    env: production\n    cluster: {cluster}\n    node_name: {node}\n    sink_type: ~sink_type~\n')


def test_generator_labels_supabase_target_like_pgwatch(tmp_path: Path) -> None:
    other = _entry('other', 'postgresql://u:p@10.0.0.5:5432/app', 'self', 'box')
    direct = _entry('main', 'postgresql://monitor:secret@db.abcdefghijklmnopqrst.supabase.co:5432/postgres', 'prod', 'main-db')
    assert _generate(tmp_path, other + direct)[0] == [{'targets': [TARGET], 'labels': {'cluster': 'prod', 'node_name': 'main-db'}}]
    # Session pooler form; quoted YAML scalars keep their value, and a quote
    # inside a value must still produce valid JSON.
    pooler = _entry('main', "'postgresql://postgres.abcdefghijklmnopqrst:secret@aws-0-us-east-1.pooler.supabase.com:5432/postgres'", '"prod"', "'node \"1\"'")
    assert _generate(tmp_path, other + pooler)[0] == [{'targets': [TARGET], 'labels': {'cluster': 'prod', 'node_name': 'node "1"'}}]


def test_generator_leaves_supabase_target_unlabelled_unless_one_match(tmp_path: Path) -> None:
    other = _entry('other', 'postgresql://u:p@10.0.0.5:5432/app', 'self', 'box')
    assert _generate(tmp_path, other)[0] == [{'targets': [TARGET], 'labels': {}}]
    off = _entry('old', 'postgresql://m:s@db.bbbbbbbbbbbbbbbbbbbb.supabase.co:5432/postgres', 'x', 'y', enabled='false')
    one = _entry('main', 'postgresql://m:s@db.abcdefghijklmnopqrst.supabase.co:5432/postgres', 'prod', 'main-db')
    assert _generate(tmp_path, off + one)[0] == [{'targets': [TARGET], 'labels': {'cluster': 'prod', 'node_name': 'main-db'}}]
    # Two Supabase targets: the relay serves one project, and we cannot tell
    # which, so the series stay out of the cluster/node views.
    two = _entry('second', 'postgresql://m:s@db.cccccccccccccccccccc.supabase.co:5432/postgres', 'prod2', 'n2')
    targets, stderr = _generate(tmp_path, one + two)
    assert targets == [{'targets': [TARGET], 'labels': {}}]
    assert '2 Supabase targets' in stderr
    assert 'secret' not in stderr and 'm:s@' not in stderr


# instances.yml exactly as the CLI's js-yaml writes it: a conn_str longer than
# 80 columns is a folded block scalar, and a plain scalar keeps backslashes and
# quotes as they are.
CLI_FIXTURE = ROOT / 'tests/fixtures/instances-supabase-cli.yml'
CLI_EXPECTED = [{'targets': [TARGET], 'labels': {'cluster': 'prod \\ "east" \'1\'', 'node_name': 'supabase-main'}}]


def test_generator_reads_cli_serialized_instances(tmp_path: Path) -> None:
    assert _generate(tmp_path, CLI_FIXTURE.read_text())[0] == CLI_EXPECTED


def test_generator_in_the_shipped_image(tmp_path: Path) -> None:
    """sources-generator runs in bash:5.2, whose BusyBox awk escapes differently."""
    import shutil, subprocess
    import pytest
    if not shutil.which('docker') or subprocess.run(['docker', 'info'], capture_output=True, timeout=30).returncode:
        pytest.skip('docker unavailable')
    compose = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())
    image = compose['services']['sources-generator']['image']
    (tmp_path / 'instances.yml').write_text(CLI_FIXTURE.read_text())
    subprocess.run(['docker', 'run', '--rm', '-v', f'{tmp_path}:/w', '-v', f'{GENERATOR.parent}:/s:ro',
                    '-e', 'INSTANCES_PATH=/w/instances.yml', '-e', 'CONFIGS_DIR=/w/out', image, 'bash', '/s/generate-pgwatch-sources.sh'],
                   check=True, capture_output=True, timeout=120)
    assert json.loads((tmp_path / 'out/prometheus/supabase-host-metrics.json').read_text()) == CLI_EXPECTED


def test_generator_keeps_block_scalar_semantics(tmp_path: Path) -> None:
    # The label must equal what pgwatch reads from the same YAML: a literal
    # block keeps its newlines (clip keeps one at the end, - strips it), a
    # folded block joins lines with spaces.
    conn = 'postgresql://m:s@db.abcdefghijklmnopqrst.supabase.co:5432/postgres'
    literal = f'- name: main\n  conn_str: {conn}\n  is_enabled: true\n  custom_tags:\n    cluster: |\n      prod\n      east\n    node_name: >-\n      main\n      db\n'
    assert _generate(tmp_path, literal)[0] == [{'targets': [TARGET], 'labels': {'cluster': 'prod\neast\n', 'node_name': 'main db'}}]
    assert yaml.safe_load(literal)[0]['custom_tags'] == {'cluster': 'prod\neast\n', 'node_name': 'main db'}
    stripped = literal.replace('cluster: |', 'cluster: |-')
    assert _generate(tmp_path, stripped)[0][0]['labels']['cluster'] == 'prod\neast'
