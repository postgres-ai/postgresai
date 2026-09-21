package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeConfig(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), ".pgwatch-config")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("INSTANCE_JOBS_CONFIG_PATH", path)
	// The instance id now falls back to the environment, so a stray value on the
	// developer's machine must not decide a test's outcome.
	t.Setenv("PGAI_INSTANCE_ID", "")
	t.Setenv("PGAI_MONITORING_INSTANCE_ID", "")
	t.Setenv("PGAI_API_BASE_URL", "")
	t.Setenv("PROMETHEUS_URL", "")
	t.Setenv("VM_AUTH_USERNAME", "")
	t.Setenv("VM_AUTH_PASSWORD", "")
	return path
}

func TestLoadReadsTheReporterConfigFormat(t *testing.T) {
	// The value is everything after the FIRST `=`, CRLF survives, and a comment
	// line is ignored -- the format the CLI writes and the reporter wrapper greps.
	writeConfig(t, "# managed by postgresai\r\napi_key=tok=en/with=equals\r\ninstance_id=11111111-1111-1111-1111-111111111111\napi_base_url=https://example.com/api/general\nproject_name=p\n")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.APIToken != "tok=en/with=equals" {
		t.Fatalf("api_key = %q", cfg.APIToken)
	}
	if cfg.InstanceID != "11111111-1111-1111-1111-111111111111" {
		t.Fatalf("instance_id = %q", cfg.InstanceID)
	}
	if cfg.APIBaseURL != "https://example.com/api/general" {
		t.Fatalf("api_base_url = %q", cfg.APIBaseURL)
	}
	if p := cfg.Problem(); p != "" {
		t.Fatalf("Problem() = %q with everything present", p)
	}
}

// The file wins over the environment: it is what the install wrote for THIS
// instance, while the env var is inherited from whatever ran `compose up`.
func TestFileBaseURLWinsOverTheEnvironment(t *testing.T) {
	// After writeConfig, which clears it: the harness blanks the variables that
	// decide these outcomes so the developer's machine cannot.
	writeConfig(t, "api_key=k\ninstance_id=i\napi_base_url=https://written.example.com\n")
	t.Setenv("PGAI_API_BASE_URL", "https://inherited.example.com")
	cfg, _ := Load()
	if cfg.APIBaseURL != "https://written.example.com" {
		t.Fatalf("api_base_url = %q", cfg.APIBaseURL)
	}

	writeConfig(t, "api_key=k\ninstance_id=i\n")
	t.Setenv("PGAI_API_BASE_URL", "https://inherited.example.com")
	cfg, _ = Load()
	if cfg.APIBaseURL != "https://inherited.example.com" {
		t.Fatalf("without a written value the environment should be used, got %q", cfg.APIBaseURL)
	}
}

// A missing file is the un-provisioned state, not an error: the process idles
// through it rather than crash-looping.
func TestMissingFileIsNotAnError(t *testing.T) {
	// writeConfig is the only place the decisive environment is blanked, and
	// this test does not use it, so do the same here.
	writeConfig(t, "")
	t.Setenv("INSTANCE_JOBS_CONFIG_PATH", filepath.Join(t.TempDir(), "absent"))
	cfg, err := Load()
	if err != nil {
		t.Fatalf("Load = %v, want no error", err)
	}
	if got, want := cfg.Problem(), "api_key, instance_id missing"; got != want {
		t.Fatalf("Problem() = %q, want %q", got, want)
	}
}

func TestInstanceIdAloneIsNotConfigured(t *testing.T) {
	writeConfig(t, "api_key=k\n")
	cfg, _ := Load()
	if got, want := cfg.Problem(), "instance_id missing"; got != want {
		t.Fatalf("Problem() = %q, want %q", got, want)
	}
}

func TestStoreDefaultsToTheComposeNetwork(t *testing.T) {
	writeConfig(t, "api_key=k\ninstance_id=i\n")
	cfg, _ := Load()
	if cfg.StoreURL != DefaultStoreURL {
		t.Fatalf("StoreURL = %q, want %q", cfg.StoreURL, DefaultStoreURL)
	}
}

// Nothing writes the instance id into .pgwatch-config yet, so until the install
// does, the environment is the only place it exists on a box: PGAI_INSTANCE_ID
// is what the provisioning flow passes to the CLI, PGAI_MONITORING_INSTANCE_ID
// what the telemetry service reads.
func TestInstanceIDFallsBackToTheEnvironment(t *testing.T) {
	writeConfig(t, "api_key=k\n")
	t.Setenv("PGAI_INSTANCE_ID", "from-provisioning")
	cfg, _ := Load()
	if cfg.InstanceID != "from-provisioning" {
		t.Fatalf("instance_id = %q, want the PGAI_INSTANCE_ID value", cfg.InstanceID)
	}
	if p := cfg.Problem(); p != "" {
		t.Fatalf("Problem() = %q with an id in the environment", p)
	}

	t.Setenv("PGAI_INSTANCE_ID", "")
	t.Setenv("PGAI_MONITORING_INSTANCE_ID", "from-telemetry")
	cfg, _ = Load()
	if cfg.InstanceID != "from-telemetry" {
		t.Fatalf("instance_id = %q, want the PGAI_MONITORING_INSTANCE_ID value", cfg.InstanceID)
	}

	// Both set and disagreeing: the provisioning flow's name wins, or a box
	// could adopt another instance's id and post its results there.
	t.Setenv("PGAI_INSTANCE_ID", "from-provisioning")
	cfg, _ = Load()
	if cfg.InstanceID != "from-provisioning" {
		t.Fatalf("instance_id = %q, want PGAI_INSTANCE_ID to win", cfg.InstanceID)
	}

	// Whitespace is not a value: an id that is blank-but-set would read as
	// configured, and a padded one would match no instance on the platform.
	t.Setenv("PGAI_INSTANCE_ID", "   ")
	t.Setenv("PGAI_MONITORING_INSTANCE_ID", "  padded-id  ")
	cfg, _ = Load()
	if cfg.InstanceID != "padded-id" {
		t.Fatalf("instance_id = %q, want the trimmed fallback", cfg.InstanceID)
	}
}

func TestTheFileWinsOverTheEnvironmentForTheInstanceID(t *testing.T) {
	writeConfig(t, "api_key=k\ninstance_id=from-file\n")
	t.Setenv("PGAI_INSTANCE_ID", "from-env")
	cfg, _ := Load()
	if cfg.InstanceID != "from-file" {
		t.Fatalf("instance_id = %q, want the written value", cfg.InstanceID)
	}
}

// The org token travels in a request header, so a plain-http platform URL would
// put it on the wire in clear either way. Loopback stays allowed: that is what a
// local rig and this suite use.
func TestPlainHTTPPlatformURLIsRefusedUnlessLoopback(t *testing.T) {
	cases := map[string]bool{
		"https://postgres.ai/api/general": true,
		"https://example.com":             true,
		"http://127.0.0.1:3000":           true,
		"http://localhost:3000/api":       true,
		"http://[::1]:3000":               true,
		"http://platform.example.com":     false,
		// Parses cleanly with an empty host, and would then fail every poll as
		// a transport error rather than be reported as the config problem it is.
		"https:///rpc":         false,
		"https://":             false,
		"http://10.0.0.5:3000": false,
		"ftp://example.com":    false,
		"not a url at all":     false,
		"":                     false,
	}
	for raw, wantOK := range cases {
		writeConfig(t, "api_key=k\ninstance_id=i\napi_base_url="+raw+"\n")
		cfg, _ := Load()
		if raw == "" {
			// An empty value in the file leaves the default in place -- assert
			// the default itself, not merely that nothing complained.
			if cfg.APIBaseURL != DefaultAPIBaseURL {
				t.Errorf("an empty api_base_url gave %q, want the default %q",
					cfg.APIBaseURL, DefaultAPIBaseURL)
			}
			if cfg.Problem() != "" {
				t.Errorf("the default api_base_url is not usable: %q", cfg.Problem())
			}
			continue
		}
		if gotOK := cfg.Problem() == ""; gotOK != wantOK {
			t.Errorf("api_base_url %q: usable = %v, want %v (problem: %q)",
				raw, gotOK, wantOK, cfg.Problem())
		}
	}
}

// A config file that exists but cannot be read is a real error, not the
// un-provisioned state: swallowing it would idle silently on a machine whose
// credential file is simply mis-permissioned.
func TestAnUnreadableConfigFileIsAnError(t *testing.T) {
	dir := t.TempDir()
	// A directory where the file should be: readable as a path, unreadable as a
	// file, and reproducible without depending on the test running as non-root.
	path := filepath.Join(dir, ".pgwatch-config")
	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("INSTANCE_JOBS_CONFIG_PATH", path)
	if _, err := Load(); err == nil {
		t.Fatal("Load() returned no error for an unreadable config file")
	}
}

// The store's basic-auth pair comes from the environment compose wires in.
func TestStoreCredentialsComeFromTheEnvironment(t *testing.T) {
	writeConfig(t, "api_key=k\ninstance_id=i\n")
	t.Setenv("VM_AUTH_USERNAME", "vmauth")
	t.Setenv("VM_AUTH_PASSWORD", "s3cret")
	cfg, _ := Load()
	if cfg.StoreUsername != "vmauth" || cfg.StorePassword != "s3cret" {
		t.Fatalf("store credentials = %q/%q, want the environment's", cfg.StoreUsername, cfg.StorePassword)
	}
}

// The reporter wrapper greps this same file with `^api_key=`, so the two
// readers have to agree about what a line is.
func TestParseFileMatchesTheReportersReading(t *testing.T) {
	cases := []struct {
		name    string
		content string
		want    map[string]string
	}{
		{"plain", "api_key=k\n", map[string]string{"api_key": "k"}},
		{"crlf", "api_key=k\r\n", map[string]string{"api_key": "k"}},
		{"value keeps its equals", "api_key=a=b\n", map[string]string{"api_key": "a=b"}},
		{"value is trimmed", "api_key=  k  \n", map[string]string{"api_key": "k"}},
		// FIRST wins, like the reporter's `grep ... | head -n 1`. Last-wins is
		// the shell convention, but it made the two readers use different
		// lines as the credential: the reporter uploaded with the fresh key
		// while this side polled with the stale one and backed off on PT401.
		{"first assignment wins, as the reporter reads it", "api_key=first\napi_key=second\n",
			map[string]string{"api_key": "first"}},
		{"comments are skipped", "# api_key=nope\napi_key=k\n", map[string]string{"api_key": "k"}},
		{"a line with no equals is skipped", "nonsense\napi_key=k\n",
			map[string]string{"api_key": "k"}},
		// An indented key is ignored, exactly as `grep ^api_key=` ignores it.
		{"an indented key is not a key", "  api_key=k\n", map[string]string{"  api_key": "k"}},
		// A byte-order mark would otherwise make the first key unmatchable.
		{"a byte-order mark is stripped", "\ufeffapi_key=k\n", map[string]string{"api_key": "k"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), ".pgwatch-config")
			if err := os.WriteFile(path, []byte(tc.content), 0o600); err != nil {
				t.Fatal(err)
			}
			got, err := parseFile(path)
			if err != nil {
				t.Fatal(err)
			}
			for k, want := range tc.want {
				if got[k] != want {
					t.Errorf("parseFile[%q] = %q, want %q (all: %v)", k, got[k], want, got)
				}
			}
		})
	}
}

// These three are a contract with docker-compose.yml, not just defaults.
func TestTheDefaultsMatchTheComposeService(t *testing.T) {
	compose, err := os.ReadFile(filepath.Join("..", "..", "..", "docker-compose.yml"))
	if err != nil {
		t.Fatalf("compose file not readable: %v", err)
	}
	// The instance-jobs stanza ALONE: both strings also appear under
	// postgres-reports, so a whole-file search passes with this service deleted.
	service := serviceStanza(t, string(compose), "instance-jobs")
	for _, want := range []string{DefaultPath, DefaultStoreURL} {
		if !strings.Contains(service, want) {
			t.Errorf("the instance-jobs service does not mention %q; the default and "+
				"the service have drifted apart", want)
		}
	}
	// The service is read_only with one tmpfs, so anything the process writes
	// has to live under it or every write fails in production.
	if !strings.Contains(service, "read_only: true") {
		t.Error("the instance-jobs service is no longer read_only; the health path " +
			"assumption below no longer holds")
	}
	if !strings.Contains(service, "tmpfs:") || !strings.Contains(service, "- /tmp:") {
		t.Error("the instance-jobs service has no /tmp tmpfs; the health file has " +
			"nowhere to go")
	}
	if DefaultAPIBaseURL != "https://postgres.ai/api/general" {
		t.Errorf("DefaultAPIBaseURL = %q, want the production api root", DefaultAPIBaseURL)
	}
}

// serviceStanza returns one compose service's own lines, from its key to the
// next top-level key at the same indentation.
func serviceStanza(t *testing.T, compose, name string) string {
	t.Helper()
	lines := strings.Split(compose, "\n")
	start := -1
	for i, line := range lines {
		if line == "  "+name+":" {
			start = i
			break
		}
	}
	if start < 0 {
		t.Fatalf("docker-compose.yml has no %q service", name)
	}
	for i := start + 1; i < len(lines); i++ {
		l := lines[i]
		if l != "" && !strings.HasPrefix(l, "   ") && !strings.HasPrefix(l, "  #") {
			return strings.Join(lines[start:i], "\n")
		}
	}
	return strings.Join(lines[start:], "\n")
}

// The file a fresh install leaves behind, in the shape updatePgwatchConfig
// produces it: `key=value` lines, no spaces, trailing newline, 0600.
//
// The install and this container are written in different languages and agree
// only on the bytes of this file. That seam is what broke -- a self-registering
// box got a config with no instance_id and the container idled forever -- and
// nothing on either side asserted its shape. The fixture is hardcoded rather
// than generated from the CLI, so this pins what the loader accepts, not what
// the CLI emits; an end-to-end check that the two still agree needs a rig.
func TestTheFileTheInstallWritesIsLoadable(t *testing.T) {
	const asWritten = "api_key=pai-token\n" +
		"project_name=7\n" +
		"instance_id=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\n"

	writeConfig(t, asWritten)

	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if p := cfg.Problem(); p != "" {
		t.Fatalf("the file the install writes is not usable: %s", p)
	}
	if cfg.InstanceID != "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" {
		t.Fatalf("instance_id = %q", cfg.InstanceID)
	}
	if cfg.APIToken != "pai-token" {
		t.Fatalf("api_key = %q", cfg.APIToken)
	}
	// project_name is in the file but not in Config: the CLI writes keys this
	// container does not read, so an unknown key must be ignored, not fatal.
}
