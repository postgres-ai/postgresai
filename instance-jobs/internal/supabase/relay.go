// Package supabase relays host metrics with a credential held only in memory.
package supabase

import (
	"context"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"regexp"
	"sync"
	"time"

	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/config"
	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/platform"
)

var metricsHost = regexp.MustCompile(`^[a-z]{20}\.supabase\.co$`)

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
	lastFetch, lastLog  time.Time
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
func (r *Relay) fetch(ctx context.Context) {
	if r.credential != nil && r.now().Sub(r.credentialFetchedAt) >= time.Hour {
		r.credential = nil
	}
	if r.credential != nil || (!r.lastFetch.IsZero() && r.now().Sub(r.lastFetch) < 5*time.Minute) {
		return
	}
	r.lastFetch = r.now()
	r.status = "upstream_error"
	cfg, err := r.load()
	if err != nil || cfg.Problem() != "" {
		r.status = "configuration_unavailable"
		return
	}
	c, err := platform.NewClient(cfg.APIBaseURL, "", 20*time.Second).SupabaseHostMetricsCredential(ctx, platform.Credentials{APIToken: cfg.APIToken, InstanceID: cfg.InstanceID})
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

func (r *Relay) unavailable(w http.ResponseWriter) {
	if r.lastLog.IsZero() || r.now().Sub(r.lastLog) >= time.Hour {
		r.lastLog = r.now()
		switch r.status {
		case "consent_needed":
			r.logger.Warn("Supabase host metrics need re-authorization: open the Supabase page in the PostgresAI console and click Allow host metrics")
		case "not_supabase":
			r.logger.Debug("Supabase host metrics: not_supabase")
		default:
			r.logger.Warn("Supabase host metrics unavailable", "status", r.status)
		}
	}
	http.Error(w, r.status, http.StatusServiceUnavailable)
}

func (r *Relay) metrics(w http.ResponseWriter, req *http.Request) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.fetch(req.Context())
	for attempt := 0; attempt < 2; attempt++ {
		if r.credential == nil {
			r.unavailable(w)
			return
		}
		c := r.credential
		upstream, err := http.NewRequestWithContext(req.Context(), http.MethodGet, c.MetricsURL, nil)
		if err != nil {
			http.Error(w, "upstream_error", http.StatusBadGateway)
			return
		}
		upstream.SetBasicAuth(c.Username, c.Password)
		resp, err := r.client.Do(upstream)
		if err != nil {
			http.Error(w, "upstream_error", http.StatusBadGateway)
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
			w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
			// On a mid-stream failure, abort the HTTP response so the scraper cannot
			// accept a truncated exposition as a successful collection.
			_, err = io.Copy(w, resp.Body)
			resp.Body.Close()
			if err != nil {
				panic(http.ErrAbortHandler)
			}
			return
		} else {
			resp.Body.Close()
		}
		http.Error(w, "upstream_error", http.StatusBadGateway)
		return
	}
}
