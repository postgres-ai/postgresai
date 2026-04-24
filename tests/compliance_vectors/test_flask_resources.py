"""Resource contract tests for the monitoring Flask backend."""

import os
import subprocess

import pytest
import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXPECTED_MEMORY_BYTES = 1073741824
EXPECTED_QUERY_INFO_ENV = {
    'QUERYID_ACTIVE_MINUTES': 10,
    'QUERYID_RETENTION_HOURS': 720,
    'QUERYID_RETENTION_BATCH_SIZE': 10000,
    'QUERYID_RETENTION_MAX_ITERATIONS': 10,
}


def _load_yaml(*parts):
    with open(os.path.join(PROJECT_ROOT, *parts)) as f:
        return yaml.safe_load(f)


def test_docker_compose_flask_resources_match_expected_limits():
    compose = _load_yaml('docker-compose.yml')
    flask = compose['services']['monitoring_flask_backend']

    assert flask['cpus'] == 0.5
    assert flask['mem_limit'] == EXPECTED_MEMORY_BYTES


def test_helm_flask_resources_match_expected_requests_and_limits():
    values = _load_yaml('postgres_ai_helm', 'values.yaml')
    resources = values['flask']['resources']

    assert resources['requests'] == {
        'cpu': '500m',
        'memory': '256Mi',
    }
    assert resources['limits'] == {
        'cpu': '500m',
        'memory': '1Gi',
    }


def test_helm_flask_query_info_retention_env_defaults_are_explicit():
    values = _load_yaml('postgres_ai_helm', 'values.yaml')
    query_info = values['flask']['queryInfo']

    assert query_info == {
        'activeMinutes': EXPECTED_QUERY_INFO_ENV['QUERYID_ACTIVE_MINUTES'],
        'retentionHours': EXPECTED_QUERY_INFO_ENV['QUERYID_RETENTION_HOURS'],
        'retentionBatchSize': EXPECTED_QUERY_INFO_ENV['QUERYID_RETENTION_BATCH_SIZE'],
        'retentionMaxIterations': EXPECTED_QUERY_INFO_ENV['QUERYID_RETENTION_MAX_ITERATIONS'],
    }


def test_helm_flask_deployment_exposes_query_info_retention_env_vars():
    with open(os.path.join(PROJECT_ROOT, 'postgres_ai_helm', 'templates', 'flask-deployment.yaml')) as f:
        template = f.read()

    for env_name in EXPECTED_QUERY_INFO_ENV:
        assert f'name: {env_name}' in template
    assert '.Values.flask.queryInfo.activeMinutes' in template
    assert '.Values.flask.queryInfo.retentionHours' in template
    assert '.Values.flask.queryInfo.retentionBatchSize' in template
    assert '.Values.flask.queryInfo.retentionMaxIterations' in template



@pytest.fixture
def rendered_helm_docs():
    try:
        subprocess.run(['helm', 'version', '--short'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip('helm not available')

    chart_path = os.path.join(PROJECT_ROOT, 'postgres_ai_helm')
    subprocess.run(['helm', 'dependency', 'build', chart_path], capture_output=True, check=False)
    result = subprocess.run(
        ['helm', 'template', 'test', chart_path, '--set', 'secrets.createFromValues=true'],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f'helm template failed: {result.stderr}')

    return [doc for doc in yaml.safe_load_all(result.stdout) if isinstance(doc, dict)]


def test_helm_template_renders_flask_resources(rendered_helm_docs):
    flask_deployments = [
        doc for doc in rendered_helm_docs
        if doc.get('kind') == 'Deployment'
        and doc.get('metadata', {}).get('name', '').endswith('-flask')
    ]
    assert len(flask_deployments) == 1

    container = flask_deployments[0]['spec']['template']['spec']['containers'][0]
    assert container['resources'] == {
        'requests': {'cpu': '500m', 'memory': '256Mi'},
        'limits': {'cpu': '500m', 'memory': '1Gi'},
    }
