"""Supabase host panels and the internal scrape contract."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_supabase_host_panels():
    dashboard = json.loads((ROOT / 'config/grafana/dashboards/Dashboard_1_Node_performance_overview.json').read_text())
    row = next(p for p in dashboard['panels'] if p['title'] == 'Host (Supabase)')
    assert row['collapsed'] is True
    assert len(row['panels']) == 5
    assert [p['fieldConfig']['defaults']['unit'] for p in row['panels']] == ['percent', 'bytes', 'bytes', 'ops', 'binBps']
    for panel in row['panels']:
        assert panel['datasource']['uid'] == 'P7A0D6631BB10B34F'
        for target in panel['targets']:
            assert target.get('interval') == '60s'
            assert 'cluster=' not in target['expr']
            assert 'node_name=' not in target['expr']
            assert 'service_type="db"' in target['expr']
            assert '{{supabase_project_ref}}' in target['legendFormat']
            assert 'job="supabase-host-metrics"' in target['expr']

    cpu, memory, filesystem, disk, network = row['panels']
    assert 'avg by (supabase_project_ref)' in cpu['targets'][0]['expr']
    assert filesystem['title'] == 'Filesystem available'
    assert 'node_filesystem_avail_bytes{' in filesystem['targets'][0]['expr']


def test_supabase_scrape_is_internal():
    config = yaml.safe_load((ROOT / 'config/prometheus/prometheus.yml').read_text())
    job = next(j for j in config['scrape_configs'] if j['job_name'] == 'supabase-host-metrics')
    assert job['scrape_interval'] == '60s'
    assert job['scrape_timeout'] == '20s'
    assert job['metrics_path'] == '/supabase/metrics'
    assert job['static_configs'] == [{'targets': ['instance-jobs:9188']}]
    assert 'http_sd_configs' not in job
    assert 'relabel_configs' not in job
    assert job['metric_relabel_configs'] == [{'source_labels': ['__name__'], 'regex': 'node_cpu_seconds_total|node_memory_.*|node_disk_.*|node_network_.*_bytes_total|node_filesystem_.*|node_load.*', 'action': 'keep'}]
    compose = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())
    service = compose['services']['instance-jobs']
    assert not service.get('ports')
    assert not any('instances.yml' in volume for volume in service['volumes'])
