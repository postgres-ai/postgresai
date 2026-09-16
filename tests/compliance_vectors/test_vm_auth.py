"""Tests for VictoriaMetrics Basic Auth configuration across deployment targets."""

import os
import re
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



class TestVictoriaMetricsAdminAuthKeys:
    """postgresai#359: VictoriaMetrics admin endpoints must require a key.

    Grafana's datasource proxy filters POST but forwards every GET, and
    VictoriaMetrics demands no key while ``-deleteAuthKey`` /
    ``-snapshotAuthKey`` / ``-forceMergeAuthKey`` are unset. A Grafana Viewer
    token could therefore GET ``/api/v1/admin/tsdb/delete_series`` through the
    proxy and wipe the whole metrics history.

    ``-pprofAuthKey`` belongs to the same fix: ``/debug/pprof/*`` is reachable
    with the credentials the proxy already holds, and heap and goroutine dumps
    are both a disclosure and a DoS lever. It is also what made the first cut of
    this fix bypassable, when the keys were flag values and
    ``/debug/pprof/cmdline`` served them verbatim.
    """

    # env var -> (VictoriaMetrics flag, key file basename)
    ADMIN_KEYS = {
        'VM_DELETE_AUTH_KEY': ('-deleteAuthKey', 'delete'),
        'VM_SNAPSHOT_AUTH_KEY': ('-snapshotAuthKey', 'snapshot'),
        'VM_FORCE_MERGE_AUTH_KEY': ('-forceMergeAuthKey', 'force_merge'),
        'VM_PPROF_AUTH_KEY': ('-pprofAuthKey', 'pprof'),
    }

    # Everything that talks to VictoriaMetrics to *read* metrics. None of them
    # may be handed an admin key: the whole point is that the credential a
    # query path holds cannot also destroy the store.
    METRIC_READERS = ('grafana', 'monitoring_flask_backend', 'postgres-reports')

    @pytest.fixture(autouse=True)
    def load_compose(self):
        compose_path = os.path.join(PROJECT_ROOT, 'docker-compose.yml')
        with open(compose_path) as f:
            self.compose = yaml.safe_load(f)
        command = self.compose['services']['sink-prometheus']['command']
        self.cmd = command if isinstance(command, str) else '\n'.join(command)

    @pytest.mark.parametrize('env_var,spec', sorted(ADMIN_KEYS.items()))
    def test_flag_is_on_the_command_line(self, env_var, spec):
        flag, basename = spec
        assert f'{flag}=file://$$KEYS/{basename}' in self.cmd, (
            f'sink-prometheus must pass {flag}; without it VictoriaMetrics '
            'accepts the endpoint from anyone who can reach it'
        )

    @pytest.mark.parametrize('env_var', sorted(ADMIN_KEYS))
    def test_env_var_is_declared_with_an_empty_default(self, env_var):
        env = self.compose['services']['sink-prometheus']['environment']
        assert f'{env_var}=${{{env_var}:-}}' in [str(e) for e in env], (
            f'{env_var} must be declared on sink-prometheus so an operator or '
            'the CLI can supply a persistent key'
        )

    @pytest.mark.parametrize('env_var', sorted(ADMIN_KEYS))
    def test_empty_key_is_replaced_before_exec(self, env_var):
        """An empty value is the hole itself, so the entrypoint must mint one.

        Guards the fallback that keeps a plain ``docker compose up`` against a
        pre-#359 .env from starting VictoriaMetrics wide open, and the refusal
        that fires if minting itself ever fails.
        """
        basename = self.ADMIN_KEYS[env_var][1]
        assert f'put_key {basename} "$${env_var}"' in self.cmd
        assert 'rand_key()' in self.cmd
        # The guard must test the FILE, not the value we meant to write: an
        # empty key file is the one fail-open case, while a missing one is
        # refused. See test_entrypoint_behaviour_end_to_end.
        assert '[ ! -s "$$KEYS/$$1" ]' in self.cmd
        assert 'refusing to start' in self.cmd
        assert 'exit 1' in self.cmd
        assert 'set -e' in self.cmd

    def test_no_secret_is_passed_on_the_command_line(self):
        """argv is readable: /debug/pprof/cmdline and the host process table.

        Every secret must reach VictoriaMetrics through ``file://``, never as a
        literal flag value. This is what made the first cut of #359 bypassable.
        """
        for env_var in self.ADMIN_KEYS:
            assert f'-{env_var}' not in self.cmd
            assert f'AuthKey="$${{{env_var}}}"' not in self.cmd, (
                f'{env_var} must not be expanded into argv; use file://'
            )
        assert '-httpAuth.password=file://' in self.cmd, (
            'the basic-auth password must not sit in argv either'
        )
        assert '-httpAuth.password=$$VM_AUTH_PASSWORD' not in self.cmd

    def test_admin_flags_cannot_be_disarmed_by_extra_args(self):
        """Go's flag package lets a flag repeat and the last one wins.

        ``VM_EXTRA_ARGS=-deleteAuthKey=`` silently disabled the control while
        every "is the flag present" assertion stayed green, so the admin flags
        must come after VM_EXTRA_ARGS on the exec line.
        """
        exec_line = next(
            line for line in self.cmd.splitlines() if line.strip().startswith('exec ')
        )
        extra_at = exec_line.index('$$VM_EXTRA_ARGS')
        admin_at = exec_line.index('$$ADMIN_ARGS')
        assert admin_at > extra_at, (
            'ADMIN_ARGS must be expanded after VM_EXTRA_ARGS so free-form '
            'tuning cannot override the admin keys'
        )
        for env_var, (flag, _) in self.ADMIN_KEYS.items():
            assert f'{flag}=' in self.cmd

    @pytest.mark.parametrize('service', METRIC_READERS)
    def test_metric_readers_never_receive_an_admin_key(self, service):
        env = str(self.compose['services'][service].get('environment', ''))
        for env_var in self.ADMIN_KEYS:
            assert env_var not in env, (
                f'{service} reads metrics; giving it {env_var} would let a '
                'leak of its credentials destroy the store'
            )

    def test_no_service_pulls_the_whole_env_file(self):
        """``env_file: .env`` would hand a service every admin key at once.

        The per-service check above only reads ``environment:``, so without this
        the likeliest regression would slip straight past it.
        """
        for name, service in self.compose['services'].items():
            assert 'env_file' not in service, (
                f'{name} declares env_file, which would inject the admin keys; '
                'list the variables it actually needs under environment: instead'
            )

    def test_env_example_documents_the_keys(self):
        env_text = open(os.path.join(PROJECT_ROOT, '.env.example')).read()
        for env_var in self.ADMIN_KEYS:
            assert f'{env_var}=' in env_text


class TestTerraformVMAdminAuthKeys:
    """The Terraform path writes the same keys, unconditionally (#359).

    Nesting this back inside the ``vm_auth_password != ""`` conditional would
    silently disable it whenever basic auth is left off, and would otherwise
    pass every other test in this file.
    """

    @pytest.fixture(autouse=True)
    def load_user_data(self):
        path = os.path.join(PROJECT_ROOT, 'terraform', 'aws', 'user_data.sh')
        with open(path) as f:
            self.content = f.read()

    @pytest.mark.parametrize('env_var', sorted(
        TestVictoriaMetricsAdminAuthKeys.ADMIN_KEYS
    ))
    def test_user_data_writes_the_key(self, env_var):
        assert env_var in self.content

    def test_generation_failure_is_fatal(self):
        assert 'openssl rand -hex 32' in self.content
        assert 'FATAL: cannot generate' in self.content

    def test_block_is_not_gated_on_basic_auth(self):
        """It must sit after the `%{ endif ~}` that closes the VM_AUTH block."""
        endif_at = self.content.index('%{ endif ~}')
        keys_at = self.content.index('VM_DELETE_AUTH_KEY')
        assert keys_at > endif_at, (
            'the admin-key block must not be inside the vm_auth_password '
            'conditional, or it is skipped whenever basic auth is unset'
        )


class TestVictoriaMetricsEntrypointBehaviour:
    """Actually RUN the sink-prometheus entrypoint, don't just grep it.

    Every other assertion in this file is a substring match against the compose
    text, so a ``put_key`` that ignored its arguments would keep them all green.
    These tests execute the real shell with ``exec`` stubbed out and inspect
    what it produced.
    """

    ENV_VARS = (
        'VM_DELETE_AUTH_KEY',
        'VM_SNAPSHOT_AUTH_KEY',
        'VM_FORCE_MERGE_AUTH_KEY',
        'VM_PPROF_AUTH_KEY',
    )

    @staticmethod
    def _script(keys_dir, extra_replacements=()):
        """The shipped command, with compose's $$ -> $ applied and exec stubbed.

        ``exec`` becomes ``echo`` so the script ends by printing the argv the
        real VictoriaMetrics would have received.
        """
        compose_path = os.path.join(PROJECT_ROOT, 'docker-compose.yml')
        with open(compose_path) as f:
            compose = yaml.safe_load(f)
        command = compose['services']['sink-prometheus']['command']
        script = command if isinstance(command, str) else '\n'.join(command)
        script = script.replace('$$', '$')
        script = script.replace('KEYS=/tmp/vm-keys', f'KEYS={keys_dir}')
        script = script.replace('exec /victoria-metrics-prod', 'echo ARGV: /victoria-metrics-prod')
        for old, new in extra_replacements:
            assert old in script, f'entrypoint no longer contains {old!r}'
            script = script.replace(old, new)
        return script

    @staticmethod
    def _run(script, env):
        base = {
            'VM_AUTH_USERNAME': '', 'VM_AUTH_PASSWORD': '',
            'VM_RETENTION_PERIOD': '336h', 'VM_QUERY_DURATION': '30s',
            'VM_MAX_CONCURRENT_REQUESTS': '16', 'VM_MAX_MEMORY_PER_QUERY': '512MiB',
            'VM_MAX_UNIQUE_TIMESERIES': '20000', 'VM_MEMORY_ALLOWED_PERCENT': '60',
            'VM_EXTRA_ARGS': '',
            'VM_DELETE_AUTH_KEY': '', 'VM_SNAPSHOT_AUTH_KEY': '',
            'VM_FORCE_MERGE_AUTH_KEY': '', 'VM_PPROF_AUTH_KEY': '',
            'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
        }
        base.update(env)
        return subprocess.run(
            ['sh', '-c', script], capture_output=True, text=True, env=base,
        )

    def test_blank_env_mints_four_distinct_usable_keys(self, tmp_path):
        keys = tmp_path / 'keys'
        result = self._run(self._script(str(keys)), {})

        assert result.returncode == 0, result.stderr
        minted = {}
        for basename in ('delete', 'snapshot', 'force_merge', 'pprof'):
            value = (keys / basename).read_text()
            assert re.fullmatch(r'[0-9a-f]{64}', value), (
                f'{basename} key is not 32 random bytes of hex: {value!r}'
            )
            minted[basename] = value
        assert len(set(minted.values())) == 4, 'minted keys must not repeat'

    def test_configured_keys_are_used_verbatim(self, tmp_path):
        keys = tmp_path / 'keys'
        result = self._run(
            self._script(str(keys)),
            {v: f'configured-{v}' for v in self.ENV_VARS},
        )

        assert result.returncode == 0, result.stderr
        assert (keys / 'delete').read_text() == 'configured-VM_DELETE_AUTH_KEY'
        assert (keys / 'pprof').read_text() == 'configured-VM_PPROF_AUTH_KEY'
        # No trailing newline: VictoriaMetrics compares the file's bytes.
        assert not (keys / 'delete').read_text().endswith('\n')

    def test_key_files_are_not_readable_by_others(self, tmp_path):
        keys = tmp_path / 'keys'
        # The basic-auth password goes through the same helper, so it must get
        # the same mode - it was 0644 for one round because a umask reset sat
        # between the admin keys and it.
        assert self._run(
            self._script(str(keys)),
            {'VM_AUTH_USERNAME': 'vmauth', 'VM_AUTH_PASSWORD': 'pw'},
        ).returncode == 0
        for name in ('delete', 'snapshot', 'force_merge', 'pprof', 'http_password'):
            assert oct((keys / name).stat().st_mode & 0o777) == '0o600', name
        assert oct(keys.stat().st_mode & 0o777) == '0o700'

    def test_refuses_to_start_when_minting_returns_nothing(self, tmp_path):
        """The fail-OPEN case: an empty key file means no key is required."""
        keys = tmp_path / 'keys'
        result = self._run(
            self._script(str(keys), [("rand_key() { head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \\n'; }",
                                      'rand_key() { echo ""; }')]),
            {},
        )

        assert result.returncode != 0, 'must not exec with an empty key'
        assert 'refusing to start' in result.stderr
        assert 'ARGV:' not in result.stdout

    def test_umask_is_restored_before_exec(self, tmp_path):
        """VictoriaMetrics inherits the umask, and it outlives the container.

        077 here would silently tighten the metrics volume's permissions.
        """
        keys = tmp_path / 'keys'
        result = self._run(
            # YAML strips the block indent, so the line is flush-left here.
            self._script(str(keys), [('\numask 022\n', '\numask 022\numask\n')]),
            {},
        )
        assert result.returncode == 0, result.stderr
        assert '0022' in result.stdout, result.stdout

    def test_an_existing_loose_key_file_is_tightened(self, tmp_path):
        """A pre-existing file keeps its own mode, so umask alone is not enough.

        `docker compose restart` on a container built from an older image can
        leave one behind.
        """
        keys = tmp_path / 'keys'
        keys.mkdir(parents=True)
        stale = keys / 'delete'
        stale.write_text('stale')
        stale.chmod(0o644)

        assert self._run(self._script(str(keys)), {}).returncode == 0
        assert oct(stale.stat().st_mode & 0o777) == '0o600'

    def test_admin_flags_survive_a_disarming_extra_args(self, tmp_path):
        """Go's flag package takes the LAST occurrence of a repeated flag."""
        keys = tmp_path / 'keys'
        result = self._run(
            self._script(str(keys)),
            {'VM_EXTRA_ARGS': '-deleteAuthKey='},
        )

        assert result.returncode == 0, result.stderr
        # Go takes the last occurrence, so inspect the last -deleteAuthKey token
        # rather than a substring position: the disarming value is a prefix of
        # the real one.
        occurrences = [
            token for token in result.stdout.split() if token.startswith('-deleteAuthKey=')
        ]
        assert len(occurrences) == 2, occurrences
        assert occurrences[0] == '-deleteAuthKey=', 'the injected value should come first'
        assert occurrences[-1] == f'-deleteAuthKey=file://{keys}/delete', (
            'the real -deleteAuthKey must come after anything VM_EXTRA_ARGS injects'
        )

    def test_no_secret_value_reaches_argv(self, tmp_path):
        keys = tmp_path / 'keys'
        secrets = {v: f'SECRET-{v}' for v in self.ENV_VARS}
        secrets['VM_AUTH_USERNAME'] = 'vmauth'
        secrets['VM_AUTH_PASSWORD'] = 'SECRET-HTTP-PASSWORD'
        result = self._run(self._script(str(keys)), secrets)

        assert result.returncode == 0, result.stderr
        for name, value in secrets.items():
            if name == 'VM_AUTH_USERNAME':
                continue  # a username is not a secret and VM takes it inline
            assert value not in result.stdout, (
                f'{name} reached the command line; it must be passed as file://'
            )
        assert (keys / 'http_password').read_text() == 'SECRET-HTTP-PASSWORD'
