package config

import (
	"strings"
	"testing"
)

// A Joe box is configured by the same file with its own keys (#398), exactly as
// a DBLab box is. Nothing about either other channel changes.
func TestLoadReadsTheJoeKeys(t *testing.T) {
	writeConfig(t, "joe_token=tok-for-one-joe\njoe_url=http://127.0.0.1:2400\njoe_verify_token=signing-secret\napi_base_url=https://example.com/api/general\n")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.IsJoe() {
		t.Fatal("IsJoe() = false for a box with joe_token")
	}
	if cfg.IsDBLab() {
		t.Fatal("a Joe box reads as DBLab")
	}
	if cfg.JoeToken != "tok-for-one-joe" || cfg.JoeURL != "http://127.0.0.1:2400" ||
		cfg.JoeVerifyToken != "signing-secret" {
		t.Fatalf("joe config = %+v", cfg)
	}
	if p := cfg.Problem(); p != "" {
		t.Fatalf("Problem() = %q, want configured", p)
	}
}

// A Joe box has NO org api_key either: its credential IS joe_token, so requiring
// one would leave every such box permanently unconfigured.
func TestAJoeBoxNeedsNoApiKey(t *testing.T) {
	writeConfig(t, "joe_token=tok\njoe_verify_token=s\n")
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

// Joe runs BESIDE this process, so the address is loopback and nothing has to be
// configured for the common case.
func TestTheJoeUrlDefaultsToTheLocalJoe(t *testing.T) {
	writeConfig(t, "joe_token=tok\njoe_verify_token=s\n")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	// The literal, not DefaultJoeURL: 2400 is Joe's own listen port, so the
	// default is a contract with a service in another repository rather than a
	// value this package may pick. Read as the constant this would assert only
	// that some default flows through, and pass against any address at all.
	if want := "http://127.0.0.1:2400"; cfg.JoeURL != want {
		t.Fatalf("JoeURL = %q, want %q", cfg.JoeURL, want)
	}
}

// Joe answers an unsigned call with a 403, so a box without the signing key
// would fail every job with what reads as a permissions problem. Reported here
// instead, where it names the missing key.
func TestAJoeBoxWithoutAVerifyTokenIsNotConfigured(t *testing.T) {
	writeConfig(t, "joe_token=tok\n")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if p := cfg.Problem(); !strings.Contains(p, "joe_verify_token") {
		t.Fatalf("Problem() = %q, want the missing key named", p)
	}
}

// ONE BOX SERVES ONE CHANNEL, and with three channels that is three pairs plus
// the all-three case. Every one of them must be refused and must NAME both keys:
// resolving the ambiguity silently would serve whichever channel the code
// happened to prefer and leave the other looking dead with nothing saying why.
func TestAnyTwoChannelsOnOneBoxAreRefused(t *testing.T) {
	const monitoring = "api_key=tok\ninstance_id=11111111-1111-1111-1111-111111111111\n"
	const dblab = "dblab_token=dbltok\ndblab_verify_token=s\n"
	const joe = "joe_token=joetok\njoe_verify_token=s\n"

	for _, tc := range []struct {
		name    string
		content string
		want    []string
	}{
		{"monitoring and dblab", monitoring + dblab, []string{"instance_id", "dblab_token"}},
		{"monitoring and joe", monitoring + joe, []string{"instance_id", "joe_token"}},
		{"dblab and joe", dblab + joe, []string{"dblab_token", "joe_token"}},
		{"all three", monitoring + dblab + joe, []string{"instance_id", "dblab_token", "joe_token"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			writeConfig(t, tc.content)
			cfg, err := Load()
			if err != nil {
				t.Fatal(err)
			}
			p := cfg.Problem()
			if !strings.Contains(p, "one box serves one channel") {
				t.Fatalf("Problem() = %q, want it to name the ambiguity", p)
			}
			// Both keys, not just the refusal: the message is what an operator
			// reads out of the health file, and one that named only the conflict
			// would not say which line to delete.
			for _, key := range tc.want {
				if !strings.Contains(p, key) {
					t.Errorf("Problem() = %q, want it to name %q", p, key)
				}
			}
		})
	}
}

// A conflict is refused BEFORE the missing-key checks, or a box carrying two
// credentials with one of them incomplete would be told to add the very key that
// is the problem.
func TestAChannelConflictIsReportedBeforeAMissingKey(t *testing.T) {
	writeConfig(t, "dblab_token=dbltok\ndblab_verify_token=s\njoe_token=joetok\n")
	cfg, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	p := cfg.Problem()
	if !strings.Contains(p, "one box serves one channel") {
		t.Fatalf("Problem() = %q, want the conflict reported first", p)
	}
	if strings.Contains(p, "missing") {
		t.Fatalf("Problem() = %q, want it not to ask for the key that is the problem", p)
	}
}

// Plain http is FINE and is the norm: Joe is on this box, so the request never
// leaves it. What is refused is a value that is not an address, which would
// otherwise fail every job as a transport error.
func TestTheJoeUrlIsCheckedForBeingAnAddressNotForTls(t *testing.T) {
	writeConfig(t, "joe_token=tok\njoe_verify_token=s\njoe_url=http://joe.internal:2400\n")
	cfg, _ := Load()
	if p := cfg.Problem(); p != "" {
		t.Fatalf("plain http to a local Joe: Problem() = %q", p)
	}

	for _, bad := range []string{"not a url", "ftp://x/y", "http:///webui/channels"} {
		writeConfig(t, "joe_token=tok\njoe_verify_token=s\njoe_url="+bad+"\n")
		cfg, _ := Load()
		if p := cfg.Problem(); !strings.Contains(p, "joe_url") {
			t.Errorf("joe_url %q: Problem() = %q, want it named", bad, p)
		}
	}
}
