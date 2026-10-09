package platform

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

// Golden request: platform-all!884 authenticates the credential RPC with the
// org token AND the instance's own secret, each in its own header, so neither
// reaches Postgres bind-parameter logs.
func TestSupabaseHostMetricsCredentialKeepsTokenInHeaderOnly(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/rpc/supabase_host_metrics_credential" {
			t.Error("incorrect credential RPC")
		}
		if r.URL.RawQuery != "" {
			t.Error("credential RPC must not send query parameters")
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read request body: %v", err)
		}
		if string(body) != `{"instance_id":"test-instance"}` {
			t.Error("credential RPC body must contain only instance_id; body parameters can reach Postgres logs")
		}
		if got := r.Header.Values("access-token"); len(got) != 1 || got[0] != "test-org-token" {
			t.Error("access-token header must contain the org token exactly once")
		}
		if got := r.Header.Values("instance-secret"); len(got) != 1 || got[0] != "test-instance-secret" {
			t.Error("instance-secret header must contain the instance secret exactly once")
		}
		if r.Header.Get("Content-Type") != "application/json" || r.Header.Get("Accept") != "application/json" {
			t.Error("credential RPC must post and accept JSON")
		}
		for name, values := range r.Header {
			for _, value := range values {
				if value == "test-org-token" && name != http.CanonicalHeaderKey("access-token") {
					t.Errorf("org token sent in unexpected header %s", name)
				}
				if strings.Contains(value, "test-instance-secret") && name != http.CanonicalHeaderKey("instance-secret") {
					t.Errorf("instance secret sent in unexpected header %s", name)
				}
			}
		}
		io.WriteString(w, `{"status":"ok"}`)
	}))
	defer srv.Close()

	resp, err := NewClient(srv.URL, "v", time.Second).SupabaseHostMetricsCredential(
		context.Background(), Credentials{APIToken: "test-org-token", InstanceID: "test-instance"}, "test-instance-secret")
	if err != nil {
		t.Fatal(err)
	}
	if resp.Status != "ok" {
		t.Errorf("status = %q, want ok", resp.Status)
	}
}

func TestSupabaseCredentialAccessHeaders(t *testing.T) {
	for _, tc := range []struct {
		name, id, secret string
		https, want      bool
	}{
		{"both", "test-id", "test-secret", true, true},
		{"padded", " test-id ", " test-secret ", true, true},
		{"missing id", "", "test-secret", true, false},
		{"missing secret", "test-id", "", true, false},
		{"blank id", " \t ", "test-secret", true, false},
		{"blank secret", "test-id", " \t ", true, false},
		{"http", "test-id", "test-secret", false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("CF_ACCESS_CLIENT_ID", tc.id)
			t.Setenv("CF_ACCESS_CLIENT_SECRET", tc.secret)
			handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				id, secret := "", ""
				if tc.want {
					id, secret = "test-id", "test-secret"
				}
				if r.Header.Get("CF-Access-Client-Id") != id || r.Header.Get("CF-Access-Client-Secret") != secret {
					t.Error("incorrect Cloudflare Access headers")
				}
				io.WriteString(w, `{"status":"ok"}`)
			})
			var srv *httptest.Server
			if tc.https {
				srv = httptest.NewTLSServer(handler)
			} else {
				srv = httptest.NewServer(handler)
			}
			defer srv.Close()
			client := NewClient(srv.URL, "v", time.Second)
			client.httpClient.Transport = srv.Client().Transport
			_, err := client.SupabaseHostMetricsCredential(context.Background(), Credentials{}, "test-instance-secret")
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestSupabaseCredentialAccessRedirectIsNotFollowed(t *testing.T) {
	t.Setenv("CF_ACCESS_CLIENT_ID", "test-id")
	t.Setenv("CF_ACCESS_CLIENT_SECRET", "test-secret")
	for _, code := range []int{301, 302, 307, 308} {
		t.Run(http.StatusText(code), func(t *testing.T) {
			other := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				t.Error("redirect reached another origin")
				if r.Header.Get("CF-Access-Client-Id") != "" || r.Header.Get("CF-Access-Client-Secret") != "" {
					t.Error("Access headers reached another origin")
				}
				io.WriteString(w, `{"status":"ok"}`)
			}))
			defer other.Close()
			srv := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("CF-Access-Client-Id") != "test-id" || r.Header.Get("CF-Access-Client-Secret") != "test-secret" {
					t.Error("configured HTTPS origin did not receive Access headers")
				}
				http.Redirect(w, r, other.URL, code)
			}))
			defer srv.Close()
			client := NewClient(srv.URL, "v", time.Second)
			client.httpClient.Transport = srv.Client().Transport
			if _, err := client.SupabaseHostMetricsCredential(context.Background(), Credentials{}, "test-instance-secret"); err == nil {
				t.Error("redirect must fail the credential request")
			}
		})
	}
}

func TestInstanceJobsComposePassesAccessVariables(t *testing.T) {
	raw, err := os.ReadFile("../../../docker-compose.yml")
	if err != nil {
		t.Fatal(err)
	}
	_, service, ok := strings.Cut(string(raw), "\n  instance-jobs:\n")
	if !ok {
		t.Fatal("instance-jobs service missing")
	}
	service, _, _ = strings.Cut(service, "\n  rds-host-stats:\n")
	for _, name := range []string{"CF_ACCESS_CLIENT_ID", "CF_ACCESS_CLIENT_SECRET"} {
		if !strings.Contains(service, "- "+name+"=${"+name+":-}") {
			t.Errorf("instance-jobs does not pass %s", name)
		}
	}
}
