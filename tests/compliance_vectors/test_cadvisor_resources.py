"""Resource contract tests for the self-cadvisor container."""

import os

import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXPECTED_MEMORY_BYTES = 402653184  # 384 MiB


def _load_yaml(*parts):
    with open(os.path.join(PROJECT_ROOT, *parts)) as f:
        return yaml.safe_load(f)


def test_docker_compose_cadvisor_resources_match_expected_limits():
    """Pin the cAdvisor defaults introduced in !248 and parameterized in !252.

    After the MR for issue #176 the compose file reads ``cpus`` / ``mem_limit``
    from env vars (``CADVISOR_CPUS`` / ``CADVISOR_MEM``) with the historical
    values as defaults; ``yaml.safe_load`` returns the raw ``${VAR:-default}``
    template string unexpanded, so we assert the template form directly. The
    rendered value is exercised end-to-end by
    ``test_compose_parameterization.py``.
    """
    compose = _load_yaml('docker-compose.yml')
    cadvisor = compose['services']['self-cadvisor']

    assert cadvisor['cpus'] == '${CADVISOR_CPUS:-0.25}'
    assert cadvisor['mem_limit'] == f'${{CADVISOR_MEM:-{EXPECTED_MEMORY_BYTES}}}'


def test_helm_cadvisor_resources_match_expected_requests_and_limits():
    values = _load_yaml('postgres_ai_helm', 'values.yaml')
    resources = values['cadvisor']['resources']

    assert resources['requests'] == {
        'cpu': '100m',
        'memory': '192Mi',
    }
    assert resources['limits'] == {
        'cpu': '250m',
        'memory': '384Mi',
    }
