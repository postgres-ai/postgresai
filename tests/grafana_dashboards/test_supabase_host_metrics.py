"""The Supabase relay stays on the compose network. Its scrape job is written by
`mon targets add` (cli/test/host-metrics-providers.test.ts); the Host row is in
test_host_row.py."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_supabase_relay_is_internal() -> None:
    service = yaml.safe_load((ROOT / 'docker-compose.yml').read_text())['services']['instance-jobs']
    assert not service.get('ports')
    assert not any('instances.yml' in volume for volume in service['volumes'])
