package config

import (
	"strings"
	"testing"
)

// A DBLab box is configured by the same file, with its own keys
// (platform-all#805). Nothing about a monitoring box changes.
func TestLoadReadsTheDBLabKeys(t *testing.T) {
	writeConfig(t, "dblab_token=tok-for-one-engine\ndblab_url=http://127.0.0.1:2345\ndblab_verify_token=secret\napi_base_url=https://example.com/api/general\n")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.IsDBLab() {
		t.Fatal("IsDBLab() = false for a box with dblab_token")
	}
	if cfg.DBLabToken != "tok-for-one-engine" || cfg.DBLabURL != "http://127.0.0.1:2345" ||
		cfg.DBLabVerifyToken != "secret" {
		t.Fatalf("dblab config = %+v", cfg)
	}
	if p := cfg.Problem(); p != "" {
		t.Fatalf("Problem() = %q, want configured", p)
	}
}

// A DBLab box has NO org api_key, and requiring one would make every such box
// permanently unconfigured. Its credential IS the per-instance token.
func TestADBLabBoxNeedsNoApiKey(t *testing.T) {
	writeConfig(t, "dblab_token=tok\ndblab_verify_token=s\n")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.APIToken != "" {
		t.Fatalf("APIToken = %q, want empty", cfg.APIToken)
	}
	if p := cfg.Problem(); p != "" {
		t.Fatalf("Problem() = %q, want configured without an api_key", p)
	}
}

// The engine runs BESIDE this process, so the address is loopback and nothing
// has to be configured for the common case.
func TestTheDBLabUrlDefaultsToTheLocalEngine(t *testing.T) {
	writeConfig(t, "dblab_token=tok\ndblab_verify_token=s\n")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	// The literal, not DefaultDBLabURL: 2345 is the engine's own listen port,
	// so the default is a contract with a service in another repository rather
	// than a value this package is free to pick. Read as the constant, this
	// test asserts only that the default flows through and passes against any
	// address at all.
	if want := "http://127.0.0.1:2345"; cfg.DBLabURL != want {
		t.Fatalf("DBLabURL = %q, want %q", cfg.DBLabURL, want)
	}
}

// ONE BOX SERVES ONE CHANNEL. The loop polls one rpc per tick, so a config
// naming both targets is an ambiguity to report -- resolving it silently would
// have the box serve whichever channel the code happened to prefer, and the
// other one would look dead with nothing anywhere saying why.
func TestABoxNamingBothChannelsIsRefused(t *testing.T) {
	writeConfig(t, "api_key=tok\ninstance_id=11111111-1111-1111-1111-111111111111\ndblab_token=dbltok\ndblab_verify_token=s\n")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	p := cfg.Problem()
	if !strings.Contains(p, "one box serves one channel") {
		t.Fatalf("Problem() = %q, want it to name the ambiguity", p)
	}
}

// The engine requires its verification token on every call, so a box without
// one would fail every job with an engine 401 that looks like a platform
// problem. Reported here instead, where it names the missing key.
func TestADBLabBoxWithoutAVerifyTokenIsNotConfigured(t *testing.T) {
	writeConfig(t, "dblab_token=tok\n")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if p := cfg.Problem(); !strings.Contains(p, "dblab_verify_token") {
		t.Fatalf("Problem() = %q, want the missing token named", p)
	}
}

// A monitoring box is untouched by any of this: no dblab_* key, same two
// requirements as before.
func TestAMonitoringBoxIsUnaffected(t *testing.T) {
	writeConfig(t, "api_key=tok\ninstance_id=11111111-1111-1111-1111-111111111111\n")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if cfg.IsDBLab() {
		t.Fatal("a monitoring box reads as DBLab")
	}
	if p := cfg.Problem(); p != "" {
		t.Fatalf("Problem() = %q", p)
	}

	writeConfig(t, "instance_id=11111111-1111-1111-1111-111111111111\n")
	cfg, _ = Load()
	if p := cfg.Problem(); !strings.Contains(p, "api_key") {
		t.Fatalf("a monitoring box without api_key: Problem() = %q", p)
	}
}

// Plain http is FINE here and is the norm: the engine is on this box, so the
// request never leaves it. What is refused is a value that is not an address,
// which would otherwise fail every job as a transport error instead of being
// reported as the configuration problem it is.
func TestTheEngineUrlIsCheckedForBeingAnAddressNotForTls(t *testing.T) {
	writeConfig(t, "dblab_token=tok\ndblab_verify_token=s\ndblab_url=http://dblab.internal:2345\n")
	cfg, _ := Load()
	if p := cfg.Problem(); p != "" {
		t.Fatalf("plain http to a local engine: Problem() = %q", p)
	}

	for _, bad := range []string{"not a url", "ftp://x/y", "http:///status"} {
		writeConfig(t, "dblab_token=tok\ndblab_verify_token=s\ndblab_url="+bad+"\n")
		cfg, _ := Load()
		if p := cfg.Problem(); !strings.Contains(p, "dblab_url") {
			t.Errorf("dblab_url %q: Problem() = %q, want it named", bad, p)
		}
	}
}
