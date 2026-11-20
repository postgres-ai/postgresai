"""Each confirmed primary gets its own files, upload, and physical standbys (#406)."""
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from reporter import postgres_reports
from reporter.postgres_reports import PostgresReportGenerator
from reporter.mon_client import MonClient


@pytest.fixture
def topology(monkeypatch, generator):
    def configure(roles, ids):
        def query(q):
            if 'pgwatch_settings_configured' in q:
                rows = [{'metric': {'node_name': n}} for n in reversed(roles)]
            else:
                n = next(n for n in roles if f'node_name="{n}"' in q)
                values = ids.get(n, [None])
                rows = [{'metric': {'sys_id': value}, 'value': [1, str(roles[n])]} for value in values] if roles[n] is not None else []
            return {'status': 'success', 'data': {'result': rows}}
        monkeypatch.setattr(generator, 'query_instant', query)
        return generator
    return configure


def test_two_unrelated_primaries(topology):
    gen = topology({'p1': 0, 'p2': 0}, {'p1': ['7312345678901234560'], 'p2': ['7312345678901234561']})
    assert gen.get_reporting_units('default') == [
        {'cluster': 'default/p1', 'nodes': {'primary': 'p1', 'standbys': []}},
        {'cluster': 'default/p2', 'nodes': {'primary': 'p2', 'standbys': []}},
    ]


def test_warning_names_no_unassigned_nodes_when_every_node_is_placed(topology, monkeypatch):
    gen = topology({'p1': 0, 'p2': 0, 's1': 1},
                   {'p1': ['7312345678901234560'], 'p2': ['7312345678901234561'], 's1': ['7312345678901234560']})
    warning = Mock()
    monkeypatch.setattr(postgres_reports.logger, 'warning', warning)
    gen.get_reporting_units('default')
    message = warning.call_args.args[0]
    assert "one report per primary" in message and '--cluster' in message
    assert 'without a unique primary' not in message and '[]' not in message


@pytest.mark.parametrize('unassigned_ids', [[None], [''], ['unrelated'], ['one', 'two']])
def test_standbys_follow_unique_system_identifier(topology, monkeypatch, unassigned_ids):
    gen = topology({'p1': 0, 'p2': 0, 's1': 1, 's2': 1, 'orphan': 1, 'unknown': None},
                   {'p1': ['7312345678901234560'], 's1': ['7312345678901234560'],
                    'p2': ['7312345678901234561'], 's2': ['7312345678901234561'],
                    'orphan': unassigned_ids, 'unknown': ['7312345678901234561']})
    warning = Mock()
    monkeypatch.setattr(postgres_reports.logger, 'warning', warning)
    assert gen.get_reporting_units('default') == [
        {'cluster': 'default/p1', 'nodes': {'primary': 'p1', 'standbys': ['orphan', 's1', 'unknown']}},
        {'cluster': 'default/p2', 'nodes': {'primary': 'p2', 'standbys': ['s2']}},
    ]
    message = warning.call_args.args[0]
    assert "['orphan', 'unknown']" in message and "first primary 'p1'" in message
    assert '--cluster' in message


def test_cloned_primaries_are_ambiguous(topology):
    gen = topology({'p1': 0, 'p2': 0, 's': 1}, {n: ['same'] for n in ('p1', 'p2', 's')})
    units = gen.get_reporting_units('default')
    assert units[0]['nodes']['standbys'] == ['s']
    assert units[1]['nodes']['standbys'] == []


def test_system_identifier_survives_the_sql_query_path(generator, monkeypatch):
    generator.mon = MonClient('http://mon.test')
    def sql(query, what):
        assert "['cluster', '=', 'default']" in query
        if 'pgwatch_settings_configured' in query:
            nodes = [('p1', 0, '1'), ('p2', 0, '2'), ('s', 1, '2')]
        else:
            nodes = [('p1', 0, '1'), ('p2', 0, '2'), ('s', 1, '2')]
            nodes = [row for row in nodes if f"['node_name', '=', '{row[0]}']" in query]
        return [{'name': None, 'labels': {'node_name': n, 'sys_id': sid}, 't': [1000], 'value': [role]}
                for n, role, sid in nodes]
    monkeypatch.setattr(generator.mon, 'sql', sql)
    assert generator.get_reporting_units('default') == [
        {'cluster': 'default/p1', 'nodes': {'primary': 'p1', 'standbys': []}},
        {'cluster': 'default/p2', 'nodes': {'primary': 'p2', 'standbys': ['s']}},
    ]


def _fixed_builders(monkeypatch, generator, clusters=('default',)):
    """Exercise the real report assembly with fixed, nonempty per-check facts."""
    class Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 11, 18, tzinfo=timezone.utc)
    monkeypatch.setattr(postgres_reports, 'datetime', Frozen)
    generator._build_metadata = {'version': 'test', 'build_ts': 'fixed'}
    for name, method in inspect.getmembers(generator, inspect.ismethod):
        if not name.startswith('generate_') or len(name.split('_')[1]) != 4:
            continue
        check = name.split('_')[1].upper()
        signature = inspect.signature(method)
        def builder(*args, _signature=signature, _check=check, **kwargs):
            bound = _signature.bind(*args, **kwargs)
            assert bound.arguments.get('cluster', 'default') in clusters
            node = bound.arguments['node_name']
            return generator.format_report_data(_check, {'fact': node}, node)
        monkeypatch.setattr(generator, name, builder)


def test_single_primary_reports_are_byte_identical(topology, monkeypatch):
    gen = topology({'p': 0, 's': 1, 'unknown': None}, {'p': ['1'], 's': ['1']})
    _fixed_builders(monkeypatch, gen)
    unit, = gen.get_reporting_units('default')
    assert unit == {'cluster': 'default', 'nodes': {'primary': 'p', 'standbys': ['s', 'unknown']}}
    reports = gen.generate_all_reports('default', reporting_nodes=unit['nodes'])
    # Recorded from the pre-split implementation with the same clock, metadata, and facts.
    assert hashlib.sha256(json.dumps(reports, indent=2).encode()).hexdigest() == '7d00f52dfb420c904b4be989cab8edd3f88cda5a9fb7ed7987e7de182494ec3e'


@pytest.mark.parametrize('extra_clusters', [(), ('default%2Fp1',), ('default%2Fp1', 'default%2Fp1_2')])
@pytest.mark.parametrize('reverse', [False, True])
def test_main_writes_and_uploads_each_unit(topology, monkeypatch, tmp_path, extra_clusters, reverse):
    gen = topology({'p1': 0, 'p2': 0, 's': 1}, {'p1': ['1'], 'p2': ['2'], 's': ['2']})
    clusters = ['default', *extra_clusters]
    if reverse:
        clusters.reverse()
    _fixed_builders(monkeypatch, gen, clusters)
    discover = gen.get_reporting_units
    monkeypatch.setattr(gen, 'get_all_clusters', lambda: clusters)
    monkeypatch.setattr(gen, 'get_reporting_units', lambda c: discover(c) if c == 'default' else [
        {'cluster': c, 'nodes': {'primary': 'ordinary', 'standbys': []}},
    ])
    monkeypatch.setattr(gen, 'test_connection', lambda: True)
    monkeypatch.setattr(gen, 'extract_queryids_from_reports', lambda reports: {'db': {'42'}})
    text_nodes = []
    def texts(*args, cluster, node_name, **kwargs):
        assert cluster in clusters
        text_nodes.append(node_name)
        return {'db': {'42': f'select {node_name}'}}
    monkeypatch.setattr(gen, 'get_queryid_queries_from_sink', texts)
    queried = []
    def metrics(cluster, node, db, qid, **kwargs):
        queried.append((cluster, node))
        return {'calls': 1}
    monkeypatch.setattr(gen, 'get_query_metrics_from_prometheus', metrics)
    calls = []
    report_ids = iter(range(101, 103 + len(extra_clusters)))
    def request(api, endpoint, data):
        calls.append((endpoint, data))
        if endpoint == '/rpc/checkup_report_create':
            return {'report_id': next(report_ids)}
        if endpoint == '/rpc/checkup_report_file_post':
            return {'report_chunk_id': 1}
        return {'report_id': data['report_id'], 'status': data['status']}
    monkeypatch.setattr(postgres_reports, 'make_request', request)
    factory = Mock(return_value=gen, DEFAULT_EXCLUDED_DATABASES=gen.DEFAULT_EXCLUDED_DATABASES)
    monkeypatch.setattr(postgres_reports, 'PostgresReportGenerator', factory)
    monkeypatch.setattr(sys, 'argv', ['reporter', '--project-name', 'proj', '--token', 'test-token'])
    monkeypatch.chdir(tmp_path)
    postgres_reports.main()
    uploads = [data for endpoint, data in calls if endpoint == '/rpc/checkup_report_file_post']
    expected_units = []
    for cluster in clusters:
        if cluster == 'default':
            suffix = f'_{len(extra_clusters) + 1}' if extra_clusters else ''
            expected_units.extend([
                ('default%2Fp1' + suffix, 'default/p1', 'p1', []),
                ('default%2Fp2', 'default/p2', 'p2', ['s']),
            ])
        else:
            expected_units.append((cluster, cluster, 'ordinary', []))
    assert len(list(tmp_path.glob('*.json'))) == 30 * len(expected_units)
    for report_id, (prefix, output_cluster, primary, standbys) in enumerate(expected_units, 101):
        paths = [tmp_path / u['filename'] for u in uploads if u['checkup_report_id'] == report_id]
        assert len(paths) == 30  # 29 checks and one query file
        for path in paths:
            data = json.loads(path.read_text())
            assert data['nodes'] == {'primary': primary, 'standbys': standbys}
            assert set(data['results']) == {primary, *standbys}
            upload, = [u for u in uploads if u['checkup_report_id'] == report_id and u['filename'] == path.name]
            assert json.loads(upload['data']) == data
            assert upload['check_id'] == data.get('checkId', '')
            assert upload['generate_issue'] == ('checkId' in data)
        query = json.loads((tmp_path / f'{prefix}_query_42.json').read_text())
        assert query['cluster_id'] == output_cluster
        assert query['query_text'] == f'select {primary if primary != "ordinary" else None}'
        assert all(path.name.startswith(prefix + '_') for path in paths)
    assert [node for node in text_nodes if node is not None] == ['p1', 'p2', 's']
    assert text_nodes.count(None) == len(extra_clusters)
    assert [entry for entry in queried if entry[0] == 'default'] == [('default', 'p1'), ('default', 'p2'), ('default', 's')]
    assert sum(endpoint == '/rpc/checkup_report_create' for endpoint, _ in calls) == len(expected_units)
    statuses = [(data['report_id'], data['status']) for endpoint, data in calls
                if endpoint == '/rpc/checkup_report_status_update']
    assert statuses == [(report_id, 'completed') for report_id in range(101, 101 + len(expected_units))]
