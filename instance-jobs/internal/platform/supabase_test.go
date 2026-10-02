package platform

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
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
