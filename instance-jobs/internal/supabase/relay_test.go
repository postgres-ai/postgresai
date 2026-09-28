package supabase

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
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

const projectRef = "abcdefghijklmnopqrst"
const metricsURL = "https://" + projectRef + ".supabase.co/customer/v1/privileged/metrics"
const fakeKey = "synthetic-test-password-do-not-use"

type rig struct {
	relay          *Relay
	handler        http.Handler
	logs           bytes.Buffer
	now            time.Time
	calls, scrapes int
	status, url    string
	upstreamStatus int
	contentType    string
	fixture        []byte
}

func setup(t *testing.T) *rig {
	t.Helper()
	x := &rig{now: time.Unix(1800000000, 0), status: "ok", url: metricsURL, upstreamStatus: 200, contentType: "text/plain; version=0.0.4"}
	var err error
	x.fixture, err = os.ReadFile("testdata/supabase_metrics.prom")
	if err != nil {
		t.Fatal(err)
	}
	upstream := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		x.scrapes++
		user, pass, ok := r.BasicAuth()
		if !ok || user != "service_role" || pass != fakeKey {
			t.Error("incorrect basic auth")
		}
		if r.URL.Path != "/customer/v1/privileged/metrics" {
			t.Error("incorrect metrics path")
		}
		if x.upstreamStatus != 200 {
			w.Header().Set("Location", metricsURL+"?redirect=1")
			w.WriteHeader(x.upstreamStatus)
			io.WriteString(w, fakeKey)
			return
		}
		w.Header().Set("Content-Type", x.contentType)
		w.Write(x.fixture)
	}))
	t.Cleanup(upstream.Close)
	rpc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		x.calls++
		if r.Method != "POST" || r.URL.Path != "/rpc/supabase_host_metrics_credential" {
			t.Error("incorrect RPC")
		}
		var body map[string]string
		if json.NewDecoder(r.Body).Decode(&body) != nil || body["api_token"] != "test-org-token" || body["instance_id"] != "test-instance" || len(body) != 2 {
			t.Error("incorrect RPC contract")
		}
		json.NewEncoder(w).Encode(map[string]string{"status": x.status, "project_ref": projectRef, "metrics_url": x.url, "username": "service_role", "password": fakeKey})
	}))
	t.Cleanup(rpc.Close)
	cfg := config.Config{APIToken: "test-org-token", InstanceID: "test-instance", APIBaseURL: rpc.URL, SupabaseHostMetrics: true}
	x.relay = New(func() (config.Config, error) { return cfg, nil }, slog.New(slog.NewTextHandler(&x.logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	transport := upstream.Client().Transport.(*http.Transport).Clone()
	transport.TLSClientConfig = transport.TLSClientConfig.Clone()
	transport.TLSClientConfig.ServerName = "example.com" // httptest certificate; production URL validation stays intact
	transport.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, upstream.Listener.Addr().String())
	}
	x.relay.client.Transport = transport
	x.relay.now = func() time.Time { return x.now }
	x.handler = x.relay.Handler(true)
	t.Cleanup(func() {
		if strings.Contains(x.logs.String(), fakeKey) {
			t.Error("key leaked into logs")
		}
	})
	return x
}

func (x *rig) get(path string) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	x.handler.ServeHTTP(w, httptest.NewRequest("GET", path, nil))
	return w
}

func TestPassthrough(t *testing.T) {
	x := setup(t)
	for range 2 {
		w := x.get("/supabase/metrics")
		if w.Code != 200 || !bytes.Equal(w.Body.Bytes(), x.fixture) || w.Header().Get("Content-Type") != "text/plain; version=0.0.4; charset=utf-8" {
			t.Fatal("passthrough differs")
		}
	}
	if x.calls != 1 {
		t.Fatal("credential was not cached")
	}
	if x.relay.client.Timeout != 20*time.Second {
		t.Fatal("wrong upstream timeout")
	}
}

func TestCredentialTTL(t *testing.T) {
	for _, minutes := range []int{59, 60, 61} {
		t.Run((time.Duration(minutes) * time.Minute).String(), func(t *testing.T) {
			x := setup(t)
			if x.get("/supabase/metrics").Code != 200 || x.calls != 1 {
				t.Fatal("initial credential fetch failed")
			}
			x.now = x.now.Add(time.Duration(minutes) * time.Minute)
			wantCalls := 1
			if minutes >= 60 {
				wantCalls++
			}
			for range 2 {
				if x.get("/supabase/metrics").Code != 200 || x.calls != wantCalls {
					t.Fatalf("RPCs=%d, want %d", x.calls, wantCalls)
				}
			}
		})
	}
}

func TestCredentialExpiryFetchCooldown(t *testing.T) {
	x := setup(t)
	x.get("/supabase/metrics")
	x.now = x.now.Add(61 * time.Minute)
	x.status = "no_key"
	for range 2 {
		if x.get("/supabase/metrics").Code != 503 || x.calls != 2 || x.scrapes != 1 {
			t.Fatal("expired credential retained or fetch cooldown bypassed")
		}
	}
	x.now = x.now.Add(5 * time.Minute)
	x.status = "ok"
	if x.get("/supabase/metrics").Code != 200 || x.calls != 3 {
		t.Fatal("did not recover after expiry fetch cooldown")
	}
}

func TestContentTypeDoesNotExposeCredential(t *testing.T) {
	x := setup(t)
	x.contentType = "text/plain; key=" + fakeKey
	w := x.get("/supabase/metrics")
	for name, values := range w.Header() {
		if strings.Contains(name, fakeKey) || strings.Contains(strings.Join(values, ","), fakeKey) {
			t.Fatal("key leaked into response headers")
		}
	}
	if w.Code != 200 || w.Header().Get("Content-Type") != "text/plain; version=0.0.4; charset=utf-8" {
		t.Fatal("incorrect metrics content type")
	}
}

func TestAuthRefetchOnce(t *testing.T) {
	for _, status := range []int{401, 403} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			x := setup(t)
			x.get("/supabase/metrics")
			x.now = x.now.Add(5 * time.Minute)
			x.upstreamStatus = status
			w := x.get("/supabase/metrics")
			if w.Code != 502 || x.calls != 2 || x.scrapes != 3 {
				t.Fatalf("code=%d RPCs=%d scrapes=%d", w.Code, x.calls, x.scrapes)
			}
			x.get("/supabase/metrics")
			if x.calls != 2 || x.scrapes != 3 {
				t.Fatal("rejected credential retained or rate limit bypassed")
			}
		})
	}
}

func TestAuthRateLimit(t *testing.T) {
	x := setup(t)
	x.upstreamStatus = 401
	if x.get("/supabase/metrics").Code != 503 || x.calls != 1 || x.scrapes != 1 {
		t.Fatal("initial auth failure bypassed rate limit")
	}
	for range 10 {
		x.get("/supabase/metrics")
	}
	if x.calls != 1 {
		t.Fatal("rate limit bypassed")
	}
	x.now = x.now.Add(5 * time.Minute)
	x.upstreamStatus = 200
	if x.get("/supabase/metrics").Code != 200 || x.calls != 2 {
		t.Fatal("did not recover after cooldown")
	}
}

func TestNegativeStatusAndLogThrottle(t *testing.T) {
	for _, status := range []string{"consent_needed", "not_supabase", "no_key", "upstream_error", fakeKey} {
		t.Run(status, func(t *testing.T) {
			x := setup(t)
			x.status = status
			for range 3 {
				w := x.get("/supabase/metrics")
				if w.Code != 503 || strings.Contains(w.Body.String(), fakeKey) {
					t.Fatal("unsafe negative response")
				}
			}
			if x.calls != 1 || x.scrapes != 0 {
				t.Fatal("negative cache failed")
			}
			x.now = x.now.Add(5 * time.Minute)
			x.get("/supabase/metrics")
			if x.calls != 2 || strings.Count(x.logs.String(), "level=") != 1 {
				t.Fatal("fetch/log throttling failed")
			}
			if status == "consent_needed" && !strings.Contains(x.logs.String(), "Supabase host metrics need re-authorization: open the Supabase page in the PostgresAI console and click Allow host metrics") {
				t.Fatal("missing consent guidance")
			}
			if status == "not_supabase" && !strings.Contains(x.logs.String(), "level=DEBUG") {
				t.Fatal("not_supabase must be debug")
			}
		})
	}
}

func TestBadMetricsURL(t *testing.T) {
	for _, u := range []string{"http://abcdefghijklmnopqrst.supabase.co/customer/v1/privileged/metrics", "https://example.com/customer/v1/privileged/metrics", metricsURL + "?key=" + fakeKey, metricsURL + "#fragment", "https://user:pass@abcdefghijklmnopqrst.supabase.co/customer/v1/privileged/metrics", "https://abcdefghijklmnopqrst.supabase.co:443/customer/v1/privileged/metrics", metricsURL + "/other", "https://ABCDEFGHIJKLMNOPQRST.supabase.co/customer/v1/privileged/metrics"} {
		t.Run("refused", func(t *testing.T) {
			x := setup(t)
			x.url = u
			w := x.get("/supabase/metrics")
			if w.Code != 503 || x.scrapes != 0 || strings.Contains(w.Body.String(), fakeKey) {
				t.Fatal("bad URL accepted or leaked")
			}
		})
	}
}

func TestUpstreamErrorsAndRedirect(t *testing.T) {
	for _, status := range []int{302, 429, 500} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			x := setup(t)
			x.upstreamStatus = status
			w := x.get("/supabase/metrics")
			if w.Code != 502 || x.scrapes != 1 || strings.Contains(w.Body.String(), fakeKey) {
				t.Fatal("upstream error leaked or redirect followed")
			}
		})
	}
}

func TestDisabledAndMethod(t *testing.T) {
	x := setup(t)
	for _, path := range []string{"/supabase/metrics", "/supabase/targets"} {
		w := httptest.NewRecorder()
		x.relay.Handler(false).ServeHTTP(w, httptest.NewRequest("GET", path, nil))
		if w.Code != 404 {
			t.Fatal("disabled handler exists")
		}
	}
	w := httptest.NewRecorder()
	x.handler.ServeHTTP(w, httptest.NewRequest("POST", "/supabase/metrics", nil))
	if w.Code != 405 || x.calls != 0 {
		t.Fatal("method allowed")
	}
}

func TestConcurrentNegativeFetch(t *testing.T) {
	x := setup(t)
	x.status = "no_key"
	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() { defer wg.Done(); x.get("/supabase/metrics") }()
	}
	wg.Wait()
	if x.calls != 1 {
		t.Fatal("concurrent fetches bypass limit")
	}
}

func TestPlatformErrorDoesNotExposeResponse(t *testing.T) {
	for _, code := range []int{200, 500} {
		t.Run(http.StatusText(code), func(t *testing.T) {
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(code); io.WriteString(w, fakeKey) }))
			defer s.Close()
			_, err := platform.NewClient(s.URL, "test", time.Second).SupabaseHostMetricsCredential(context.Background(), platform.Credentials{APIToken: "test", InstanceID: "id"})
			if err == nil || strings.Contains(err.Error(), fakeKey) {
				t.Fatal("unsafe platform error")
			}
		})
	}
}

func TestSingleProjectRoutes(t *testing.T) {
	x := setup(t)
	if x.get("/supabase/targets").Code != http.StatusNotFound || x.calls != 0 {
		t.Fatal("discovery route still exists")
	}
	w := x.get("/supabase/metrics?project_ref=zyxwvutsrqponmlkjihgf")
	if w.Code != http.StatusOK || !bytes.Equal(w.Body.Bytes(), x.fixture) || x.scrapes != 1 {
		t.Fatal("query parameter changed the single-project scrape")
	}
}
