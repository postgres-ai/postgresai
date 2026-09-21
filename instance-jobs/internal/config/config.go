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

// Config is the resolved runtime configuration. Nothing here is ever logged.
type Config struct {
	Path          string
	APIToken      string
	InstanceID    string
	APIBaseURL    string
	StoreURL      string
	StoreUsername string
	StorePassword string
}

// Problem returns why this instance cannot poll yet, or "" when it can. The
// text names what is wrong, never a value, and is safe to log and to put in the
// health file.
func (c Config) Problem() string {
	var missing []string
	if c.APIToken == "" {
		missing = append(missing, "api_key")
	}
	if c.InstanceID == "" {
		missing = append(missing, "instance_id")
	}
	if len(missing) > 0 {
		return strings.Join(missing, ", ") + " missing"
	}
	return baseURLProblem(c.APIBaseURL)
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
		Path:          envOr("INSTANCE_JOBS_CONFIG_PATH", DefaultPath),
		APIBaseURL:    envOr("PGAI_API_BASE_URL", DefaultAPIBaseURL),
		StoreURL:      envOr("PROMETHEUS_URL", DefaultStoreURL),
		StoreUsername: os.Getenv("VM_AUTH_USERNAME"),
		StorePassword: os.Getenv("VM_AUTH_PASSWORD"),
	}

	values, err := parseFile(cfg.Path)
	if err != nil {
		return cfg, err
	}
	cfg.APIToken = values["api_key"]
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
