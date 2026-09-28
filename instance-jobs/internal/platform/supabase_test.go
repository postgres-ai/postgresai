package platform

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

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
		for name, values := range r.Header {
			if name == http.CanonicalHeaderKey("access-token") {
				continue
			}
			for _, value := range values {
				if value == "test-org-token" {
					t.Errorf("org token sent in unexpected header %s", name)
				}
			}
		}
		io.WriteString(w, `{"status":"ok"}`)
	}))
	defer srv.Close()

	resp, err := NewClient(srv.URL, "v", time.Second).SupabaseHostMetricsCredential(
		context.Background(), Credentials{APIToken: "test-org-token", InstanceID: "test-instance"})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Status != "ok" {
		t.Errorf("status = %q, want ok", resp.Status)
	}
}
