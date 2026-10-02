//go:build supabase_live

package supabase

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/config"
	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/platform"
)

func TestSupabaseLive(t *testing.T) {
	token := os.Getenv("SUPABASE_TEST_ACCESS_TOKEN")
	ref := os.Getenv("SUPABASE_TEST_PROJECT_REF")
	if token == "" || ref == "" {
		t.Skip("requires SUPABASE_TEST_ACCESS_TOKEN and SUPABASE_TEST_PROJECT_REF")
	}
	if !metricsHost.MatchString(ref + ".supabase.co") {
		t.Fatal("invalid test project reference")
	}

	var logs bytes.Buffer
	var mu sync.Mutex
	var key string
	// Check even on failure, without printing captured logs or credentials.
	defer func() {
		mu.Lock()
		defer mu.Unlock()
		if strings.Contains(logs.String(), token) || (key != "" && strings.Contains(logs.String(), key)) {
			t.Error("credential leaked into logs")
		}
	}()
	client := &http.Client{
		Timeout:       15 * time.Second,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	defer client.CloseIdleConnections()
	rpc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/rpc/supabase_host_metrics_credential" {
			t.Error("incorrect platform RPC")
			http.Error(w, "incorrect RPC", http.StatusBadRequest)
			return
		}
		var body map[string]string
		if json.NewDecoder(r.Body).Decode(&body) != nil || len(body) != 1 || body["instance_id"] != "test-instance" || r.Header.Get("access-token") != "test-org-token" {
			t.Error("incorrect platform RPC contract")
			http.Error(w, "incorrect contract", http.StatusBadRequest)
			return
		}
		fail := func(message string) {
			t.Error(message)
			http.Error(w, "credential unavailable", http.StatusBadGateway)
		}
		req, err := http.NewRequestWithContext(r.Context(), http.MethodGet,
			"https://api.supabase.com/v1/projects/"+ref+"/api-keys?reveal=true", nil)
		if err != nil {
			fail("could not construct management API request")
			return
		}
		req.Header.Set("Authorization", "Bearer "+token)
		resp, err := client.Do(req)
		if err != nil {
			fail("management API request failed")
			return
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			fail("management API returned non-200 status")
			return
		}
		var keys []struct {
			Type   string `json:"type"`
			APIKey string `json:"api_key"`
		}
		if json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&keys) != nil {
			fail("management API response could not be decoded")
			return
		}
		mu.Lock()
		defer mu.Unlock()
		for _, candidate := range keys {
			if candidate.Type == "secret" && candidate.APIKey != "" {
				key = candidate.APIKey
				break
			}
		}
		if key == "" {
			fail("management API returned no secret key")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if json.NewEncoder(w).Encode(platform.SupabaseCredential{
			Status: "ok", ProjectRef: ref,
			MetricsURL: "https://" + ref + ".supabase.co/customer/v1/privileged/metrics",
			Username:   "service_role", Password: key,
		}) != nil {
			t.Error("could not write platform RPC response")
		}
	}))
	defer rpc.Close()
	cfg := config.Config{APIToken: "test-org-token", InstanceID: "test-instance", APIBaseURL: rpc.URL, SupabaseHostMetrics: true}
	relay := New(func() (config.Config, error) { return cfg, nil },
		slog.New(slog.NewTextHandler(&logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	defer relay.client.CloseIdleConnections()
	w := httptest.NewRecorder()
	relay.Handler().ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/supabase/metrics", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("relay returned HTTP %d", w.Code)
	}
	for _, family := range []string{
		"node_cpu_seconds_total",
		"node_memory_MemAvailable_bytes",
		"node_disk_reads_completed_total",
		"node_network_receive_bytes_total",
		"node_filesystem_avail_bytes",
	} {
		found := false
		for _, line := range strings.Split(w.Body.String(), "\n") {
			if strings.HasPrefix(line, family+"{") || strings.HasPrefix(line, family+" ") {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("missing metric family %s", family)
		}
	}
}
