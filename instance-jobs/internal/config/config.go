// Package config reads the instance's own settings.
//
// Everything identifying the instance comes from .pgwatch-config, the file the
// reporter container already mounts -- no new credential and no new file. It is
// re-read on every tick, so a file written after the container started (the
// install writes the instance id) is picked up without a restart.
package config

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"strings"
)

// DefaultPath is where docker-compose mounts .pgwatch-config.
const DefaultPath = "/app/.pgwatch-config"

// DefaultAPIBaseURL is the production API root, the same default the reporter
// wrapper uses when the environment says nothing.
const DefaultAPIBaseURL = "https://postgres.ai/api/general"

// DefaultStoreURL is the metric store on the compose network.
const DefaultStoreURL = "http://sink-prometheus:9090"

// DefaultDBLabURL is the engine's address on a DBLab box. The engine listens on
// 2345 and this process runs beside it, so loopback is the address -- the
// platform never supplies one (platform-all#805).
const DefaultDBLabURL = "http://127.0.0.1:2345"

// DefaultJoeURL is Joe's address on a Joe box, for the same reason and with the
// same shape: Joe listens on 2400 (JOE_APP_PORT's default) and this process runs
// beside it, so the platform supplies no host here either (#398).
const DefaultJoeURL = "http://127.0.0.1:2400"

// Config is the resolved runtime configuration. Nothing here is ever logged.
//
// ONE BOX SERVES ONE CHANNEL, and there are now three of them: InstanceID names
// a monitoring instance, DBLabToken stands for a DBLab engine and JoeToken for a
// Joe -- neither of the latter two carries an instance id, by design, because
// the token identifies the box. Exactly ONE may be set: the loop polls one rpc
// per tick, so a config naming two is an ambiguity to report rather than a
// preference to resolve silently (platform-all#805, #398). A box that runs both
// an engine and a Joe runs a SECOND CONTAINER of this image, one channel each.
type Config struct {
	SupabaseHostMetrics   bool
	SupabaseMetricsListen string
	Path                  string
	APIToken              string
	InstanceID            string
	// InstanceSecret is the instance's own secret that provisioning writes to
	// .pgwatch-config (platform-all!884). File only, never the environment.
	InstanceSecret string
	APIBaseURL     string
	StoreURL       string
	StoreUsername  string
	StorePassword  string
	// DBLabToken is the engine's OWN per-instance platform token, issued by
	// v1.dblab_instance_register on first registration. There is NO instance id
	// here and there is not meant to be: the token identifies the engine, so the
	// box names nothing and cannot name another (platform-all#805).
	DBLabToken       string
	DBLabURL         string
	DBLabVerifyToken string
	// JoeToken is Joe's OWN per-instance platform token, the mirror of
	// DBLabToken and carrying no instance id for the same reason (#398).
	JoeToken string
	JoeURL   string
	// JoeVerifyToken is NOT a bearer token despite the name it shares with the
	// engine's: it is the HMAC KEY every call to Joe is signed with, and its
	// value is the box's joe_communication_signing_secret (Joe's own
	// `signingSecret`). It never goes on the wire -- see internal/joe.
	JoeVerifyToken string
}

// IsDBLab reports whether this box serves the DBLab channel.
func (c Config) IsDBLab() bool { return c.DBLabToken != "" }

// IsJoe reports whether this box serves the Joe channel.
func (c Config) IsJoe() bool { return c.JoeToken != "" }

// Problem returns why this instance cannot poll yet, or "" when it can. The
// text names what is wrong, never a value, and is safe to log and to put in the
// health file.
func (c Config) Problem() string {
	// Named first, or a box carrying two channels would be reported as fully
	// configured and would then serve whichever the code happened to prefer.
	if p := channelConflict(c); p != "" {
		return p
	}

	var missing []string
	switch {
	case c.IsDBLab():
		// No api_key: a DBLab box has none. Its credential IS dblab_token, which
		// is set by definition here.
		if c.DBLabVerifyToken == "" {
			missing = append(missing, "dblab_verify_token")
		}
	case c.IsJoe():
		// Same shape, and joe_verify_token is required for a sharper reason: Joe
		// refuses an unsigned call with a 403, so a box without the key would
		// fail every job with what looks like a permissions problem.
		if c.JoeVerifyToken == "" {
			missing = append(missing, "joe_verify_token")
		}
	default:
		if c.APIToken == "" {
			missing = append(missing, "api_key")
		}
		if c.InstanceID == "" {
			missing = append(missing, "instance_id")
		}
	}

	if len(missing) > 0 {
		return strings.Join(missing, ", ") + " missing"
	}
	switch {
	case c.IsDBLab():
		if p := localURLProblem("dblab_url", c.DBLabURL); p != "" {
			return p
		}
	case c.IsJoe():
		if p := localURLProblem("joe_url", c.JoeURL); p != "" {
			return p
		}
	}
	return baseURLProblem(c.APIBaseURL)
}

// channelConflict refuses a box that names more than one channel. The whole
// point is that it is reported rather than resolved: two credentials on one box
// mean the install wrote a key it should not have, and picking a winner here
// would leave the other channel looking dead with nothing saying why.
func channelConflict(c Config) string {
	var named []string
	if c.InstanceID != "" {
		named = append(named, "instance_id")
	}
	if c.IsDBLab() {
		named = append(named, "dblab_token")
	}
	if c.IsJoe() {
		named = append(named, "joe_token")
	}
	if len(named) < 2 {
		return ""
	}
	// "both" for two keeps the exact message the DBLab arm already emits, so
	// adding a third channel does not reword an existing box's health file.
	quantifier := "are both set"
	if len(named) > 2 {
		quantifier = "are all set"
	}
	return strings.Join(named[:len(named)-1], ", ") + " and " + named[len(named)-1] +
		" " + quantifier + "; one box serves one channel"
}

// localURLProblem rejects an address on this box that this process cannot call,
// naming the key it came from. Plain http is fine and is the norm: the engine and
// Joe both run BESIDE this process, so the request does not leave the box --
// which is the point of the inversion. What is rejected is a value that is not an
// address at all, because the alternative is every job failing as a transport
// error rather than being reported here as the configuration problem it is.
func localURLProblem(key, raw string) string {
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return key + " is not a url"
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return key + " scheme is not http(s)"
	}
	return ""
}

// baseURLProblem rejects a platform URL that would put the org token on the
// wire in clear. Plain http is allowed only for a loopback host, which is what
// a local rig and the test suite use.
func baseURLProblem(raw string) string {
	// Host, not just a parse: `https:///rpc` parses cleanly with an empty host
	// and would then fail every poll as a transport error rather than be
	// reported here as the configuration problem it is.
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return "api_base_url is not a url"
	}
	switch u.Scheme {
	case "https":
		return ""
	case "http":
		if isLoopback(u.Hostname()) {
			return ""
		}
		return "api_base_url must be https (http only for a loopback host); " +
			"the api key is sent as a request header"
	default:
		return "api_base_url scheme is not http(s)"
	}
}

func isLoopback(host string) bool {
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

// Load resolves the configuration. A missing config file is not an error: it is
// the un-provisioned state, and the process idles through it.
func Load() (Config, error) {
	cfg := Config{
		SupabaseHostMetrics:   strings.EqualFold(strings.TrimSpace(os.Getenv("PGAI_SUPABASE_HOST_METRICS")), "true"),
		SupabaseMetricsListen: envOr("PGAI_SUPABASE_METRICS_LISTEN", ":9188"),
		Path:                  envOr("INSTANCE_JOBS_CONFIG_PATH", DefaultPath),
		APIBaseURL:            envOr("PGAI_API_BASE_URL", DefaultAPIBaseURL),
		StoreURL:              envOr("PROMETHEUS_URL", DefaultStoreURL),
		StoreUsername:         os.Getenv("VM_AUTH_USERNAME"),
		StorePassword:         os.Getenv("VM_AUTH_PASSWORD"),
		DBLabURL:              envOr("PGAI_DBLAB_URL", DefaultDBLabURL),
		JoeURL:                envOr("PGAI_JOE_URL", DefaultJoeURL),
	}

	values, err := parseFile(cfg.Path)
	if err != nil {
		return cfg, err
	}
	cfg.APIToken = values["api_key"]
	cfg.InstanceSecret = values["instance_secret"]
	// The file is the source of truth once the install writes the id there. Until
	// it does, the environment is the only place the id exists on a box:
	// PGAI_INSTANCE_ID is what the provisioning flow already passes to the CLI,
	// PGAI_MONITORING_INSTANCE_ID what the telemetry service reads.
	cfg.InstanceID = firstNonEmpty(values["instance_id"],
		os.Getenv("PGAI_INSTANCE_ID"), os.Getenv("PGAI_MONITORING_INSTANCE_ID"))
	// The file wins over the environment: it is what the install wrote for THIS
	// instance, while the env var is inherited from whatever ran `compose up`.
	if v := values["api_base_url"]; v != "" {
		cfg.APIBaseURL = v
	}

	// The DBLab channel (platform-all#805). Same file, same precedence: an
	// install writes the keys, the environment is what a hand-started container
	// inherits. A box with none of these is a monitoring instance and nothing
	// below changes for it.
	cfg.DBLabToken = firstNonEmpty(values["dblab_token"],
		os.Getenv("PGAI_DBLAB_TOKEN"))
	cfg.DBLabVerifyToken = firstNonEmpty(values["dblab_verify_token"],
		os.Getenv("PGAI_DBLAB_VERIFY_TOKEN"))
	if v := values["dblab_url"]; v != "" {
		cfg.DBLabURL = v
	}

	// The Joe channel (#398), read exactly like the DBLab keys above. A box with
	// none of these is a monitoring instance and nothing here changes for it.
	cfg.JoeToken = firstNonEmpty(values["joe_token"], os.Getenv("PGAI_JOE_TOKEN"))
	cfg.JoeVerifyToken = firstNonEmpty(values["joe_verify_token"],
		os.Getenv("PGAI_JOE_VERIFY_TOKEN"))
	if v := values["joe_url"]; v != "" {
		cfg.JoeURL = v
	}
	return cfg, nil
}

// parseFile reads `key=value` lines, the format the CLI writes and the reporter
// wrapper greps. The value is everything after the FIRST `=`, so a value
// containing `=` survives, and it is trimmed, which also drops the `\r` of a
// CRLF file. Keys are NOT trimmed: the reporter greps `^api_key=` and ignores
// an indented line, and accepting one here would have the two readers disagree
// about which line is the credential.
//
// On a duplicate key this takes the FIRST, matching the reporter's
// `grep ... | head -n 1`. The shell convention would be last-wins, but the two
// readers must not disagree about which line is the credential: a file with a
// stale `api_key=` above a fresh one had the reporter uploading happily while
// this side polled with the old token and backed off on PT401 for ten minutes
// at a time. Appending a key is a documented step (CONTRIBUTING.md), so a
// duplicate is reachable in a file the CLI never wrote.
//
// Where the two still differ, this being the more forgiving reader: the
// reporter keeps surrounding whitespace in a value and this trims it, and a
// leading byte-order mark makes the reporter's grep miss the key entirely
// while this strips it.
func parseFile(path string) (map[string]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]string{}, nil
		}
		return nil, fmt.Errorf("reading %s: %w", path, err)
	}
	values := make(map[string]string)
	for _, line := range strings.Split(strings.TrimPrefix(string(raw), "\ufeff"), "\n") {
		line = strings.TrimSuffix(line, "\r")
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		key, value, found := strings.Cut(line, "=")
		if !found {
			continue
		}
		if _, seen := values[key]; seen {
			continue // first wins, as above
		}
		values[key] = strings.TrimSpace(value)
	}
	return values, nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v = strings.TrimSpace(v); v != "" {
			return v
		}
	}
	return ""
}

func envOr(name, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return fallback
}
