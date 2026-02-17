"""Tests for VictoriaMetrics Basic Auth configuration across deployment targets."""

import os
import subprocess
import pytest
import yaml


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


class TestDockerComposeVMAuth:
    """Verify docker-compose.yml has VM auth configuration."""

    @pytest.fixture(autouse=True)
    def load_compose(self):
        compose_path = os.path.join(PROJECT_ROOT, 'docker-compose.yml')
        with open(compose_path) as f:
            self.compose = yaml.safe_load(f)

    def test_sink_prometheus_has_auth_flags(self):
        """VM service command script should have -httpAuth.username and -httpAuth.password flags."""
        command = self.compose['services']['sink-prometheus']['command']
        # Command is now a shell script string (single element list or string)
        cmd_str = command if isinstance(command, str) else ' '.join(command)
        assert '-httpAuth.username' in cmd_str, "Missing -httpAuth.username in VM command"
        assert '-httpAuth.password' in cmd_str, "Missing -httpAuth.password in VM command"

    def test_flask_has_vm_auth_env(self):
        """Flask backend should have VM_AUTH_USERNAME and VM_AUTH_PASSWORD env vars."""
        env = self.compose['services']['monitoring_flask_backend']['environment']
        env_keys = [e.split('=')[0] if isinstance(e, str) else e for e in env]
        assert 'VM_AUTH_USERNAME' in str(env), "Flask missing VM_AUTH_USERNAME"
        assert 'VM_AUTH_PASSWORD' in str(env), "Flask missing VM_AUTH_PASSWORD"

    def test_reporter_has_vm_auth_env(self):
        """Reporter should have VM_AUTH_USERNAME and VM_AUTH_PASSWORD env vars."""
        env = self.compose['services']['postgres-reports']['environment']
        assert 'VM_AUTH_USERNAME' in str(env), "Reporter missing VM_AUTH_USERNAME"
        assert 'VM_AUTH_PASSWORD' in str(env), "Reporter missing VM_AUTH_PASSWORD"

    def test_grafana_has_vm_auth_env(self):
        """Grafana should have VM_AUTH_USERNAME and VM_AUTH_PASSWORD env vars for datasource provisioning."""
        env = self.compose['services']['grafana']['environment']
        assert 'VM_AUTH_USERNAME' in str(env), "Grafana missing VM_AUTH_USERNAME"
        assert 'VM_AUTH_PASSWORD' in str(env), "Grafana missing VM_AUTH_PASSWORD"

    def test_auth_conditionally_applied(self):
        """Auth flags should only be applied when env vars are non-empty (conditional check)."""
        command = self.compose['services']['sink-prometheus']['command']
        cmd_str = command if isinstance(command, str) else ' '.join(command)
        # The shell script should have a conditional check for non-empty vars
        assert 'VM_AUTH_USERNAME' in cmd_str
        assert 'VM_AUTH_PASSWORD' in cmd_str
        # Should have a conditional (if/then or [ -n ]) to avoid passing empty flags
        assert '-n' in cmd_str or 'if' in cmd_str, \
            "Command should conditionally apply auth flags"


class TestGrafanaDatasourceVMAuth:
    """Verify Grafana datasource has basic auth configuration."""

    @pytest.fixture(autouse=True)
    def load_datasource(self):
        ds_path = os.path.join(
            PROJECT_ROOT, 'config', 'grafana', 'provisioning',
            'datasources', 'datasources.yml'
        )
        with open(ds_path) as f:
            self.config = yaml.safe_load(f)

    def test_prometheus_datasource_has_basic_auth(self):
        """PGWatch-Prometheus datasource should have basicAuth enabled."""
        datasources = self.config['datasources']
        prom_ds = next(ds for ds in datasources if ds['name'] == 'PGWatch-Prometheus')
        assert prom_ds.get('basicAuth') is True
        assert 'basicAuthUser' in prom_ds
        assert 'secureJsonData' in prom_ds
        assert 'basicAuthPassword' in prom_ds['secureJsonData']


class TestPrometheusConfigVMAuth:
    """Verify prometheus.yml has auth for self-scrape."""

    @pytest.fixture(autouse=True)
    def load_config(self):
        config_path = os.path.join(PROJECT_ROOT, 'config', 'prometheus', 'prometheus.yml')
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def test_victoriametrics_job_has_basic_auth(self):
        """victoriametrics scrape job should have basic_auth configured."""
        scrape_configs = self.config['scrape_configs']
        vm_job = next(j for j in scrape_configs if j['job_name'] == 'victoriametrics')
        assert 'basic_auth' in vm_job, "victoriametrics job missing basic_auth"
        assert 'username' in vm_job['basic_auth']
        assert 'password' in vm_job['basic_auth']

    def test_self_scrape_uses_env_var_syntax(self):
        """Self-scrape auth should use %{ENV_VAR} syntax for VictoriaMetrics."""
        scrape_configs = self.config['scrape_configs']
        vm_job = next(j for j in scrape_configs if j['job_name'] == 'victoriametrics')
        username = vm_job['basic_auth']['username']
        password = vm_job['basic_auth']['password']
        assert '%{VM_AUTH_USERNAME}' in username
        assert '%{VM_AUTH_PASSWORD}' in password


class TestHelmVMAuth:
    """Verify Helm chart has VM auth configuration."""

    @pytest.fixture(autouse=True)
    def check_helm(self):
        """Skip if helm is not available."""
        try:
            subprocess.run(['helm', 'version', '--short'], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("helm not available")

    @pytest.fixture
    def helm_chart_path(self):
        chart_path = os.path.join(PROJECT_ROOT, 'postgres_ai_helm')
        # Ensure chart dependencies are built
        subprocess.run(
            ['helm', 'dependency', 'build', chart_path],
            capture_output=True, check=False
        )
        return chart_path

    def _render_template(self, chart_path, set_values=None):
        """Render helm chart and return YAML docs."""
        cmd = ['helm', 'template', 'test', chart_path,
               '--set', 'secrets.createFromValues=true']
        if set_values:
            for k, v in set_values.items():
                cmd.extend(['--set', f'{k}={v}'])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            pytest.fail(f"helm template failed: {result.stderr}")

        docs = []
        for doc in yaml.safe_load_all(result.stdout):
            if doc and isinstance(doc, dict):
                docs.append(doc)
        return docs

    def _find_doc(self, docs, kind, name_contains):
        """Find a k8s resource by kind and name substring."""
        for doc in docs:
            if (doc.get('kind') == kind and
                    name_contains in doc.get('metadata', {}).get('name', '')):
                return doc
        return None

    def test_values_has_auth_config(self):
        """values.yaml should have victoriaMetrics.auth section."""
        values_path = os.path.join(PROJECT_ROOT, 'postgres_ai_helm', 'values.yaml')
        with open(values_path) as f:
            values = yaml.safe_load(f)
        assert 'auth' in values['victoriaMetrics']
        assert 'enabled' in values['victoriaMetrics']['auth']
        assert 'username' in values['victoriaMetrics']['auth']

    def test_auth_disabled_no_httpauth_args(self, helm_chart_path):
        """With auth disabled, VM statefulset should not have -httpAuth args."""
        docs = self._render_template(helm_chart_path)
        vm_ss = self._find_doc(docs, 'StatefulSet', 'victoriametrics')
        assert vm_ss is not None, "VictoriaMetrics StatefulSet not found"

        container = vm_ss['spec']['template']['spec']['containers'][0]
        args_str = ' '.join(container.get('args', []))
        assert '-httpAuth' not in args_str

    def test_auth_disabled_no_vm_env_in_flask(self, helm_chart_path):
        """With auth disabled, flask should not have VM_AUTH env vars."""
        docs = self._render_template(helm_chart_path)
        flask_dep = self._find_doc(docs, 'Deployment', 'flask')
        assert flask_dep is not None, "Flask Deployment not found"

        container = flask_dep['spec']['template']['spec']['containers'][0]
        env_names = [e['name'] for e in container.get('env', [])]
        assert 'VM_AUTH_USERNAME' not in env_names
        assert 'VM_AUTH_PASSWORD' not in env_names

    def test_auth_enabled_vm_has_httpauth_args(self, helm_chart_path):
        """With auth enabled, VM should have -httpAuth args."""
        docs = self._render_template(helm_chart_path,
                                     {'victoriaMetrics.auth.enabled': 'true'})
        vm_ss = self._find_doc(docs, 'StatefulSet', 'victoriametrics')
        container = vm_ss['spec']['template']['spec']['containers'][0]
        args_str = ' '.join(container.get('args', []))
        assert '-httpAuth.username' in args_str
        assert '-httpAuth.password' in args_str

    def test_auth_enabled_vm_has_env_vars(self, helm_chart_path):
        """With auth enabled, VM should have VM_AUTH env vars."""
        docs = self._render_template(helm_chart_path,
                                     {'victoriaMetrics.auth.enabled': 'true'})
        vm_ss = self._find_doc(docs, 'StatefulSet', 'victoriametrics')
        container = vm_ss['spec']['template']['spec']['containers'][0]
        env_names = [e['name'] for e in container.get('env', [])]
        assert 'VM_AUTH_USERNAME' in env_names
        assert 'VM_AUTH_PASSWORD' in env_names

    def test_auth_enabled_flask_has_env_vars(self, helm_chart_path):
        """With auth enabled, flask should have VM_AUTH env vars."""
        docs = self._render_template(helm_chart_path,
                                     {'victoriaMetrics.auth.enabled': 'true'})
        flask_dep = self._find_doc(docs, 'Deployment', 'flask')
        container = flask_dep['spec']['template']['spec']['containers'][0]
        env_names = [e['name'] for e in container.get('env', [])]
        assert 'VM_AUTH_USERNAME' in env_names
        assert 'VM_AUTH_PASSWORD' in env_names

    def test_auth_enabled_reporter_has_env_vars(self, helm_chart_path):
        """With auth enabled, reporter should have VM_AUTH env vars."""
        docs = self._render_template(helm_chart_path,
                                     {'victoriaMetrics.auth.enabled': 'true'})
        reporter_cj = self._find_doc(docs, 'CronJob', 'reporter')
        assert reporter_cj is not None, "Reporter CronJob not found"
        container = reporter_cj['spec']['jobTemplate']['spec']['template']['spec']['containers'][0]
        env_names = [e['name'] for e in container.get('env', [])]
        assert 'VM_AUTH_USERNAME' in env_names
        assert 'VM_AUTH_PASSWORD' in env_names

    def test_auth_enabled_secret_has_vm_password(self, helm_chart_path):
        """With auth enabled, secret should have vm-auth-password key."""
        docs = self._render_template(helm_chart_path,
                                     {'victoriaMetrics.auth.enabled': 'true'})
        secret = self._find_doc(docs, 'Secret', 'secrets')
        assert secret is not None, "Secret not found"
        assert 'vm-auth-password' in secret.get('stringData', {})

    def test_auth_enabled_grafana_datasource_has_basic_auth(self, helm_chart_path):
        """With auth enabled, grafana datasource should have basicAuth."""
        docs = self._render_template(helm_chart_path,
                                     {'victoriaMetrics.auth.enabled': 'true'})
        # Find the grafana datasource configmap
        ds_cm = self._find_doc(docs, 'ConfigMap', 'grafana-datasources')
        assert ds_cm is not None, "Grafana datasources ConfigMap not found"

        ds_yaml = ds_cm['data']['datasources.yaml']
        assert 'basicAuth: true' in ds_yaml
        assert 'basicAuthUser' in ds_yaml

    def test_vm_password_from_secret_ref(self, helm_chart_path):
        """VM_AUTH_PASSWORD env var should come from secretKeyRef, not plaintext."""
        docs = self._render_template(helm_chart_path,
                                     {'victoriaMetrics.auth.enabled': 'true'})
        vm_ss = self._find_doc(docs, 'StatefulSet', 'victoriametrics')
        container = vm_ss['spec']['template']['spec']['containers'][0]

        vm_pass_env = next(e for e in container['env'] if e['name'] == 'VM_AUTH_PASSWORD')
        assert 'valueFrom' in vm_pass_env
        assert 'secretKeyRef' in vm_pass_env['valueFrom']
        assert vm_pass_env['valueFrom']['secretKeyRef']['key'] == 'vm-auth-password'


class TestTerraformVMAuth:
    """Verify Terraform files have VM auth configuration."""

    def test_variables_has_vm_auth_username(self):
        """variables.tf should define vm_auth_username."""
        var_path = os.path.join(PROJECT_ROOT, 'terraform', 'aws', 'variables.tf')
        with open(var_path) as f:
            content = f.read()
        assert 'variable "vm_auth_username"' in content

    def test_variables_has_vm_auth_password(self):
        """variables.tf should define vm_auth_password as sensitive."""
        var_path = os.path.join(PROJECT_ROOT, 'terraform', 'aws', 'variables.tf')
        with open(var_path) as f:
            content = f.read()
        assert 'variable "vm_auth_password"' in content
        assert 'sensitive' in content

    def test_main_passes_vm_auth_to_template(self):
        """main.tf should pass vm_auth vars to user_data template."""
        main_path = os.path.join(PROJECT_ROOT, 'terraform', 'aws', 'main.tf')
        with open(main_path) as f:
            content = f.read()
        assert 'vm_auth_username' in content
        assert 'vm_auth_password' in content

    def test_user_data_writes_vm_auth_to_env(self):
        """user_data.sh should write VM_AUTH vars to .env."""
        ud_path = os.path.join(PROJECT_ROOT, 'terraform', 'aws', 'user_data.sh')
        with open(ud_path) as f:
            content = f.read()
        assert 'VM_AUTH_USERNAME' in content
        assert 'VM_AUTH_PASSWORD' in content
