// Package supabase relays host metrics with a credential held only in memory.
package supabase

import (
	"bytes"
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"sync"
	"time"

	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/config"
	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/platform"
)

var metricsHost = regexp.MustCompile(`^[a-z]{20}\.supabase\.co$`)

const (
	// credentialTTL bounds how long a revoked connection or consent keeps
	// working: nothing re-checks the platform while a credential is cached.
	credentialTTL = time.Hour
	// fetchCooldown is consumed by every credential fetch, successful or not.
	fetchCooldown = 5 * time.Minute
	// platformTimeout keeps the credential RPC well inside the scraper's 20 s
	// budget, which the RPC shares with the upstream scrape on a TTL refresh.
	platformTimeout = 10 * time.Second
	// expositionTTL is how long the last upstream exposition is served from
	// memory. The listener is reachable from every compose-network peer, so
	// without it any peer could turn the relay into an upstream scrape loop.
	expositionTTL = 30 * time.Second
	// logInterval throttles repeated log lines for an unchanged status.
	logInterval = time.Hour
)

// Relay serializes scrapes and credential fetches to enforce a process-wide
// cooldown, including concurrent requests and negative platform responses.
type Relay struct {
	mu                  sync.Mutex
	load                func() (config.Config, error)
	logger              *slog.Logger
	client              *http.Client
	now                 func() time.Time
	credential          *platform.SupabaseCredential
	credentialFetchedAt time.Time
	status              string
	lastFetch           time.Time
	lastLog             time.Time
	lastLogStatus       string
	exposition          []byte
	expositionAt        time.Time
}

func New(load func() (config.Config, error), logger *slog.Logger) *Relay {
	return &Relay{load: load, logger: logger, now: time.Now,
		client: &http.Client{Timeout: 20 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }},
	}
}

// Handler registers no routes when the feature is disabled.
func (r *Relay) Handler(enabled bool) http.Handler {
	mux := http.NewServeMux()
	if enabled {
		mux.HandleFunc("GET /supabase/metrics", r.metrics)
	}
	return mux
}

func validCredential(c *platform.SupabaseCredential) bool {
	u, err := url.Parse(c.MetricsURL)
	return err == nil && u.Scheme == "https" && metricsHost.MatchString(u.Host) &&
		u.Path == "/customer/v1/privileged/metrics" && u.RawPath == "" &&
		u.RawQuery == "" && !u.ForceQuery && u.Fragment == "" && u.User == nil &&
		c.ProjectRef+".supabase.co" == u.Host && c.Username == "service_role" && c.Password != ""
}

// fetch must be called with mu held. Even failed calls consume the cooldown.
// The RPC runs detached from the scraper's context: a scraper that gives up
// must not burn the cooldown by cancelling a fetch that was going to succeed.
func (r *Relay) fetch(ctx context.Context) {
	if r.credential != nil && r.now().Sub(r.credentialFetchedAt) >= credentialTTL {
		r.credential = nil
	}
	if r.credential != nil || (!r.lastFetch.IsZero() && r.now().Sub(r.lastFetch) < fetchCooldown) {
		return
	}
	r.lastFetch = r.now()
	r.status = "upstream_error"
	cfg, err := r.load()
	if err != nil || cfg.Problem() != "" {
		r.status = "configuration_unavailable"
		return
	}
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), platformTimeout)
	defer cancel()
	c, err := platform.NewClient(cfg.APIBaseURL, "", platformTimeout).SupabaseHostMetricsCredential(ctx, platform.Credentials{APIToken: cfg.APIToken, InstanceID: cfg.InstanceID})
	if err != nil {
		return
	}
	switch c.Status {
	case "ok":
		if !validCredential(c) {
			r.status = "invalid_credential"
			return
		}
		r.credential = c
		r.credentialFetchedAt = r.now()
		r.status = "ok"
	case "not_supabase", "consent_needed", "no_key", "upstream_error":
		r.status = c.Status
	}
}

// logStatus must be called with mu held. It logs on every status change and
// otherwise once per logInterval, so a quiet status cannot suppress the next
// actionable one.
func (r *Relay) logStatus(status string, log func()) {
	if status == r.lastLogStatus && !r.lastLog.IsZero() && r.now().Sub(r.lastLog) < logInterval {
		return
	}
	r.lastLog, r.lastLogStatus = r.now(), status
	log()
}

func (r *Relay) unavailable(w http.ResponseWriter) {
	r.logStatus(r.status, func() {
		switch r.status {
		case "consent_needed":
			r.logger.Warn("Supabase host metrics need re-authorization: open the Supabase page in the PostgresAI console and click Allow host metrics")
		case "not_supabase":
			r.logger.Debug("Supabase host metrics: not_supabase")
		default:
			r.logger.Warn("Supabase host metrics unavailable", "status", r.status)
		}
	})
	http.Error(w, r.status, http.StatusServiceUnavailable)
}

// upstreamError reports the upstream HTTP status only; 0 is a transport error.
func (r *Relay) upstreamError(w http.ResponseWriter, code int) {
	r.logStatus("upstream_http_"+strconv.Itoa(code), func() {
		r.logger.Warn("Supabase host metrics upstream error", "upstream_status", code)
	})
	http.Error(w, "upstream_error", http.StatusBadGateway)
}

func (r *Relay) serve(w http.ResponseWriter, body []byte) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	// On a mid-stream failure, abort the HTTP response so the scraper cannot
	// accept a truncated exposition as a successful collection.
	if _, err := w.Write(body); err != nil {
		panic(http.ErrAbortHandler)
	}
}

func (r *Relay) metrics(w http.ResponseWriter, req *http.Request) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.exposition != nil && r.now().Sub(r.expositionAt) < expositionTTL {
		r.serve(w, r.exposition)
		return
	}
	r.exposition = nil
	r.fetch(req.Context())
	for attempt := 0; attempt < 2; attempt++ {
		if r.credential == nil {
			r.unavailable(w)
			return
		}
		c := r.credential
		upstream, err := http.NewRequestWithContext(req.Context(), http.MethodGet, c.MetricsURL, nil)
		if err != nil {
			r.upstreamError(w, 0)
			return
		}
		upstream.SetBasicAuth(c.Username, c.Password)
		resp, err := r.client.Do(upstream)
		if err != nil {
			r.upstreamError(w, 0)
			return
		}
		if resp.StatusCode == 401 || resp.StatusCode == 403 {
			resp.Body.Close()
			r.credential = nil
			r.status = "credential_rejected"
			if attempt == 0 {
				r.fetch(req.Context())
				continue
			}
		} else if resp.StatusCode == http.StatusOK {
			var buf bytes.Buffer
			_, err = io.Copy(&buf, resp.Body)
			resp.Body.Close()
			if err != nil {
				// A truncated upstream body must not be cached or relayed.
				r.upstreamError(w, 0)
				return
			}
			r.exposition, r.expositionAt = buf.Bytes(), r.now()
			r.serve(w, r.exposition)
			return
		} else {
			resp.Body.Close()
		}
		r.upstreamError(w, resp.StatusCode)
		return
	}
}
