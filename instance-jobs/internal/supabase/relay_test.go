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
	"strconv"
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
const instanceSecret = "synthetic-instance-secret-do-not-use"

type rig struct {
	relay          *Relay
	handler        http.Handler
	logs           bytes.Buffer
	now            time.Time
	calls, scrapes int
	status, url    string
	upstreamStatus int
	rejectOnce     bool // answer the next scrape with upstreamStatus, then 200
	truncate       bool // announce a longer body than is written
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
			if x.rejectOnce {
				x.upstreamStatus = 200
			}
			return
		}
		w.Header().Set("Content-Type", x.contentType)
		if x.truncate {
			w.Header().Set("Content-Length", strconv.Itoa(len(x.fixture)*2))
		}
		w.Write(x.fixture)
	}))
	t.Cleanup(upstream.Close)
	rpc := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		x.calls++
		if r.Method != "POST" || r.URL.Path != "/rpc/supabase_host_metrics_credential" {
			t.Error("incorrect RPC")
		}
		var body map[string]string
		if json.NewDecoder(r.Body).Decode(&body) != nil || body["instance_id"] != "test-instance" || len(body) != 1 || r.Header.Get("access-token") != "test-org-token" || r.Header.Get("instance-secret") != instanceSecret {
			t.Error("incorrect RPC contract")
		}
		json.NewEncoder(w).Encode(map[string]string{"status": x.status, "project_ref": projectRef, "metrics_url": x.url, "username": "service_role", "password": fakeKey})
	}))
	t.Cleanup(rpc.Close)
	cfg := config.Config{APIToken: "test-org-token", InstanceID: "test-instance", InstanceSecret: instanceSecret, APIBaseURL: rpc.URL, SupabaseHostMetrics: true}
	x.relay = New(func() (config.Config, error) { return cfg, nil }, slog.New(slog.NewTextHandler(&x.logs, &slog.HandlerOptions{Level: slog.LevelDebug})))
	transport := upstream.Client().Transport.(*http.Transport).Clone()
	transport.TLSClientConfig = transport.TLSClientConfig.Clone()
	transport.TLSClientConfig.ServerName = "example.com" // httptest certificate; production URL validation stays intact
	transport.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, network, upstream.Listener.Addr().String())
	}
	x.relay.client.Transport = transport
	x.relay.now = func() time.Time { return x.now }
	x.handler = x.relay.Handler()
	t.Cleanup(func() {
		if strings.Contains(x.logs.String(), fakeKey) {
			t.Error("key leaked into logs")
		}
		if strings.Contains(x.logs.String(), instanceSecret) {
			t.Error("instance secret leaked into logs")
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
	x.now = x.now.Add(time.Minute)
	if x.get("/supabase/metrics").Code != 200 || x.scrapes != 2 || x.calls != 1 {
		t.Fatalf("scrapes=%d RPCs=%d after the cache expired", x.scrapes, x.calls)
	}
}

// The listener is unauthenticated on the compose network: requests inside the
// exposition TTL must be answered from memory, not with an upstream scrape.
func TestExpositionCache(t *testing.T) {
	x := setup(t)
	for range 10 {
		w := x.get("/supabase/metrics")
		if w.Code != 200 || !bytes.Equal(w.Body.Bytes(), x.fixture) {
			t.Fatal("cached exposition differs")
		}
	}
	x.now = x.now.Add(29 * time.Second)
	x.get("/supabase/metrics")
	if x.scrapes != 1 {
		t.Fatalf("scrapes=%d inside the cache TTL, want 1", x.scrapes)
	}
	x.now = x.now.Add(time.Second)
	if x.get("/supabase/metrics").Code != 200 || x.scrapes != 2 {
		t.Fatalf("scrapes=%d after the cache TTL, want 2", x.scrapes)
	}
}

func TestConfigurationUnavailable(t *testing.T) {
	cases := map[string]func() (config.Config, error){
		"loader error": func() (config.Config, error) { return config.Config{}, os.ErrNotExist },
		"no token": func() (config.Config, error) {
			return config.Config{InstanceID: "test-instance", APIBaseURL: "https://example.com", SupabaseHostMetrics: true}, nil
		},
	}
	for name, load := range cases {
		t.Run(name, func(t *testing.T) {
			x := setup(t)
			x.relay.load = load
			for range 2 {
				w := x.get("/supabase/metrics")
				if w.Code != 503 || strings.TrimSpace(w.Body.String()) != "configuration_unavailable" {
					t.Fatalf("code=%d body=%q", w.Code, w.Body.String())
				}
			}
			if x.calls != 0 || x.scrapes != 0 {
				t.Fatal("RPC or scrape attempted without configuration")
			}
			if !strings.Contains(x.logs.String(), "status=configuration_unavailable") {
				t.Fatal("configuration problem not logged")
			}
		})
	}
}

// An instance provisioned before platform-all!884 has no secret: every RPC
// would be refused, so none is made, and the log says what to do.
func TestMissingInstanceSecret(t *testing.T) {
	x := setup(t)
	cfg, _ := x.relay.load()
	cfg.InstanceSecret = ""
	x.relay.load = func() (config.Config, error) { return cfg, nil }
	for range 2 {
		w := x.get("/supabase/metrics")
		if w.Code != 503 || strings.TrimSpace(w.Body.String()) != "instance_secret_missing" {
			t.Fatalf("code=%d body=%q", w.Code, w.Body.String())
		}
	}
	if x.calls != 0 || x.scrapes != 0 {
		t.Fatal("RPC or scrape attempted without the instance secret")
	}
	if !strings.Contains(x.logs.String(), "instance_secret") || !strings.Contains(x.logs.String(), "re-provision") {
		t.Fatalf("missing secret not explained: %s", x.logs.String())
	}
}

// A scraper that gives up must not burn the fetch cooldown: the credential RPC
// runs detached from the request context.
func TestCancelledScrapeKeepsCredential(t *testing.T) {
	x := setup(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	w := httptest.NewRecorder()
	x.handler.ServeHTTP(w, httptest.NewRequestWithContext(ctx, "GET", "/supabase/metrics", nil))
	if w.Code != 502 || x.calls != 1 || x.scrapes != 0 {
		t.Fatalf("code=%d RPCs=%d scrapes=%d", w.Code, x.calls, x.scrapes)
	}
	if x.get("/supabase/metrics").Code != 200 || x.calls != 1 || x.scrapes != 1 {
		t.Fatalf("RPCs=%d scrapes=%d: cancelled scrape consumed the cooldown", x.calls, x.scrapes)
	}
}

func TestTruncatedUpstreamIsNotRelayed(t *testing.T) {
	x := setup(t)
	x.truncate = true
	w := x.get("/supabase/metrics")
	if w.Code != 502 || bytes.Contains(w.Body.Bytes(), []byte("node_")) {
		t.Fatalf("code=%d: truncated exposition relayed", w.Code)
	}
	x.truncate = false
	if x.get("/supabase/metrics").Code != 200 || x.scrapes != 2 {
		t.Fatal("truncated exposition was cached")
	}
}

func TestUpstreamErrorLogged(t *testing.T) {
	x := setup(t)
	x.upstreamStatus = 500
	for range 3 {
		x.get("/supabase/metrics")
	}
	if strings.Count(x.logs.String(), "upstream_status=500") != 1 || strings.Contains(x.logs.String(), fakeKey) {
		t.Fatalf("upstream error log: %q", x.logs.String())
	}
}

func TestLogOnStatusChange(t *testing.T) {
	x := setup(t)
	x.status = "no_key"
	x.get("/supabase/metrics")
	x.now = x.now.Add(5 * time.Minute)
	x.status = "consent_needed"
	x.get("/supabase/metrics")
	if strings.Count(x.logs.String(), "level=") != 2 || !strings.Contains(x.logs.String(), "click Allow host metrics") {
		t.Fatalf("status change not logged: %q", x.logs.String())
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
	x.status = "upstream_error"
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
		t.Run(http.StatusText(status)+" then ok", func(t *testing.T) {
			x := setup(t)
			x.get("/supabase/metrics")
			x.now = x.now.Add(5 * time.Minute)
			x.upstreamStatus, x.rejectOnce = status, true
			w := x.get("/supabase/metrics")
			if w.Code != 200 || !bytes.Equal(w.Body.Bytes(), x.fixture) || x.calls != 2 || x.scrapes != 3 {
				t.Fatalf("code=%d RPCs=%d scrapes=%d", w.Code, x.calls, x.scrapes)
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

// A box without the feature granted is not a failed scrape: it answers an
// empty exposition, so up stays 1 and no series exist. Faults stay 503.
func TestNotApplicableIsEmptyNotDown(t *testing.T) {
	for status, want := range map[string]int{"consent_needed": 200, "not_supabase": 200, "no_key": 200, "upstream_error": 503, fakeKey: 503} {
		t.Run(status, func(t *testing.T) {
			x := setup(t)
			x.status = status
			w := x.get("/supabase/metrics")
			if w.Code != want || (want == 200 && w.Body.Len() != 0) {
				t.Fatalf("code=%d body=%q", w.Code, w.Body.String())
			}
		})
	}
}

func TestNegativeStatusAndLogThrottle(t *testing.T) {
	for _, status := range []string{"consent_needed", "not_supabase", "no_key", "upstream_error", fakeKey} {
		t.Run(status, func(t *testing.T) {
			x := setup(t)
			x.status = status
			for range 3 {
				w := x.get("/supabase/metrics")
				if w.Code == 200 && w.Body.Len() != 0 || strings.Contains(w.Body.String(), fakeKey) {
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
		t.Run(u, func(t *testing.T) {
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

func TestOnlyGetMetrics(t *testing.T) {
	x := setup(t)
	w := httptest.NewRecorder()
	x.handler.ServeHTTP(w, httptest.NewRequest("GET", "/supabase/targets", nil))
	if w.Code != 404 {
		t.Fatal("unknown path served")
	}
	w = httptest.NewRecorder()
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
			_, err := platform.NewClient(s.URL, "test", time.Second).SupabaseHostMetricsCredential(context.Background(), platform.Credentials{APIToken: "test", InstanceID: "id"}, instanceSecret)
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
