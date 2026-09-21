// Package runner is the loop: ask the platform for work, run one job at a time
// against the local metric store, post the answer back, sleep.
//
// Every connection it makes is outbound. The platform never dials in.
package runner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math/rand"
	"net/url"
	"runtime/debug"
	"strings"
	"time"

	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/collect"
	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/config"
	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/platform"
)

const (
	// The poll interval the platform hands back is clamped to this range: a
	// blanked or fat-fingered platform setting must not turn the fleet into a
	// poll storm, nor park an instance for a day.
	minPollInterval = 1 * time.Second
	maxPollInterval = 1 * time.Hour
	// Used when the platform did not say, or could not be reached.
	defaultPollInterval = 10 * time.Minute

	// jitterFraction spreads a fleet that was provisioned together.
	jitterFraction = 0.2

	// jobBudget is the per-job ceiling. It bounds the healthcheck deadline too,
	// and must stay well under the platform's stuck sweep (1 hour) so a job
	// still running is never failed underneath us.
	jobBudget = 15 * time.Minute

	// storeAttempts is how often one job re-runs against the metric store
	// before it is answered as failed. Network trouble, a 5xx and a 401 from
	// the store are all momentary on a container that shares our compose
	// network.
	storeAttempts = 3
	storeBackoff  = 5 * time.Second

	// storeRequestTimeout bounds ONE request to the store. It has to be well
	// under jobBudget, or a store that accepts the connection and then stops
	// answering -- VictoriaMetrics under memory pressure -- burns the whole
	// budget on attempt one and the retries above never run.
	storeRequestTimeout = 90 * time.Second

	// submitAttempts is how often an answer is re-posted. A job is 'running'
	// platform-side until it is answered or the hourly sweep closes it, so a
	// single reset on the way back would otherwise throw away the whole
	// collection and wedge that period for an hour.
	submitAttempts = 3
	submitBackoff  = 3 * time.Second

	// A rejected credential or a missing RPC is not fixed by polling harder.
	longBackoff = 10 * time.Minute

	// unhealthyAfter consecutive failed polls flips the healthcheck.
	unhealthyAfter = 3

	// idleLogInterval keeps an un-provisioned instance to one log line an hour.
	idleLogInterval = 1 * time.Hour

	// platformTimeout bounds one poll or submit.
	platformTimeout = 30 * time.Second

	// errorMaxBytes is the platform's cap on the submitted error text.
	errorMaxBytes = 512
)

// Runner owns the loop's state.
type Runner struct {
	platformClient func(baseURL string) *platform.Client
	storeClient    func(cfg config.Config) *collect.Client
	healthPath     string
	now            func() time.Time
	// elapsed measures a job's duration. Separate from now() because now() is
	// wall-clock (`time.Now().UTC()`, and .UTC() strips Go's monotonic
	// reading), so subtracting two of its readings across a backward NTP step
	// yields a negative duration_ms. Injectable so tests can produce a
	// deterministic one.
	elapsed func(time.Time) time.Duration
	// sleep is overridden in tests; it returns false when the context ended.
	sleep func(ctx context.Context, d time.Duration) bool
	// budget is the per-job ceiling. A field rather than the constant directly
	// so a test can shorten it and see what a job that overruns is answered as.
	budget time.Duration

	consecutiveFailures    int
	consecutiveJobFailures int
	lastIdleLog            time.Time
}

// New builds a Runner with the production dependencies.
func New(healthPath, clientVersion string) *Runner {
	return &Runner{
		platformClient: func(baseURL string) *platform.Client {
			return platform.NewClient(baseURL, clientVersion, platformTimeout)
		},
		storeClient: func(cfg config.Config) *collect.Client {
			c := collect.NewClient(cfg.StoreURL, cfg.StoreUsername, cfg.StorePassword, storeRequestTimeout)
			c.OnEnrichmentError = func(err error) {
				log.Printf("query-text enrichment dropped (non-fatal): %v", sanitize(err))
			}
			return c
		},
		healthPath: healthPath,
		now:        func() time.Time { return time.Now().UTC() },
		elapsed:    time.Since,
		sleep:      sleepCtx,
		budget:     jobBudget,
	}
}

// Run loops until the context ends.
func (r *Runner) Run(ctx context.Context) error {
	// Seed the health file immediately, so a healthcheck that fires before the
	// first tick finishes does not read a missing file as a dead process.
	r.setHealth(true, "", defaultPollInterval)
	for {
		wait := r.tick(ctx)
		if !r.sleep(ctx, wait) {
			return ctx.Err()
		}
	}
}

// tick runs one poll-work-submit cycle and returns how long to sleep.
func (r *Runner) tick(ctx context.Context) time.Duration {
	cfg, err := config.Load()
	if err != nil {
		// Not a transient poll failure: the file is there and cannot be read,
		// which on this service usually means it is 0600 and owned by someone
		// else (see INSTANCE_JOBS_USER). Waiting three polls to say so would
		// leave `mon health` green for twenty minutes on a box that will never
		// collect anything.
		return r.jittered(r.idle(cfg, "config unreadable; check its owner and mode", err))
	}
	if problem := cfg.Problem(); problem != "" {
		return r.jittered(r.idle(cfg, problem, nil))
	}

	client := r.platformClient(cfg.APIBaseURL)
	creds := platform.Credentials{APIToken: cfg.APIToken, InstanceID: cfg.InstanceID}

	pollCtx, cancel := context.WithTimeout(ctx, platformTimeout)
	resp, err := client.Poll(pollCtx, creds)
	cancel()
	if err != nil {
		return r.jittered(r.pollError(err))
	}

	r.consecutiveFailures = 0
	next := clampInterval(time.Duration(resp.NextPollMS) * time.Millisecond)

	// One job at a time, each answered before the next is started. The platform
	// stamps started_at at pickup and sweeps anything still running after an
	// hour, so running a claimed batch concurrently -- or slowly -- is how real
	// answers get refused. The "one job per poll" pacing (issue #366) is the
	// platform's own (c_claim_limit := 1); this side just never runs a claimed
	// batch in parallel, and re-polls only on the next tick after the sleep.
	if len(resp.Jobs) == 0 {
		// Nothing to run means nothing is failing: a fleet being drained (the
		// flag turned off) hands out no work, and a box must not stay red on a
		// run of failures it can no longer retry.
		r.consecutiveJobFailures = 0
	}

	for _, job := range resp.Jobs {
		// Stamped before AND after each job: the deadline in the file allows one
		// job budget, so a tick that runs several would otherwise look wedged
		// while it is doing exactly what it should. It carries the CURRENT
		// verdict -- re-stamping healthy here would make a box already judged
		// dead report green for the whole duration of every later job.
		r.stampHealth(next)
		r.runJob(ctx, client, creds, cfg, job)
		if ctx.Err() != nil {
			break
		}
	}

	r.stampHealth(next)
	return r.jittered(next)
}

// idle handles an instance that cannot poll: no credential, no instance id, or
// a platform URL that would put the token on the wire in clear. It reports
// UNHEALTHY -- the container was started on purpose and is doing nothing, which
// is exactly the state that must not look green in `mon health` -- but logs at
// most once an hour, and never crash-loops.
func (r *Runner) idle(cfg config.Config, problem string, detail error) time.Duration {
	if now := r.now(); now.Sub(r.lastIdleLog) >= idleLogInterval {
		if detail != nil {
			log.Printf("idle: %s (config: %s): %v", problem, cfg.Path, detail)
		} else {
			log.Printf("idle: %s (config: %s)", problem, cfg.Path)
		}
		r.lastIdleLog = now
	}
	r.setHealth(false, "not configured: "+problem, defaultPollInterval)
	return defaultPollInterval
}

// pollError classifies a failed poll and returns how long to wait.
func (r *Runner) pollError(err error) time.Duration {
	switch platform.Classify(err) {
	case platform.ClassAuth:
		r.pollFailed("platform rejected the credential", err)
		return longBackoff
	case platform.ClassUnavailable:
		r.pollFailed(unavailableReason(err), err)
		return longBackoff
	case platform.ClassRequest:
		r.pollFailed("platform refused the poll"+codeSuffix(err), err)
		return longBackoff
	default:
		r.pollFailed("poll failed", err)
		return defaultPollInterval
	}
}

// stampHealth refreshes the health file with the verdict as it stands.
func (r *Runner) stampHealth(next time.Duration) {
	if r.consecutiveJobFailures >= unhealthyAfter {
		r.setHealth(false, fmt.Sprintf("%d jobs in a row failed", r.consecutiveJobFailures), next)
		return
	}
	r.setHealth(true, "", next)
}

// pollFailed records a failed poll and flips health after unhealthyAfter of
// them in a row. reason is ours and goes in the health file, which an operator
// reads through `docker inspect`; detail may carry a message the platform
// wrote, so it only ever reaches the local log.
func (r *Runner) pollFailed(reason string, detail error) {
	r.consecutiveFailures++
	if detail != nil {
		log.Printf("%s: %v (consecutive failures: %d)", reason, sanitize(detail), r.consecutiveFailures)
	} else {
		log.Printf("%s (consecutive failures: %d)", reason, r.consecutiveFailures)
	}
	r.setHealth(r.consecutiveFailures < unhealthyAfter, reason, defaultPollInterval)
}

// unavailableReason separates the two things ClassUnavailable covers. A
// redirect is not "there is no channel here" -- the channel may well exist and
// something in front of the platform sent us elsewhere.
func unavailableReason(err error) string {
	var apiErr *platform.APIError
	if errors.As(err, &apiErr) && apiErr.StatusCode >= 300 && apiErr.StatusCode < 400 {
		return "platform redirected the poll, which we refuse to follow"
	}
	return "platform has no job channel at this api_base_url"
}

// codeSuffix names the platform's error code, bounded by the platform client,
// without carrying the message it wrote alongside it.
func codeSuffix(err error) string {
	var apiErr *platform.APIError
	if errors.As(err, &apiErr) && apiErr.Code != "" {
		return " (code " + apiErr.Code + ")"
	}
	return ""
}

// runJob executes one job and submits its answer.
func (r *Runner) runJob(ctx context.Context, client *platform.Client, creds platform.Credentials, cfg config.Config, job platform.Job) {
	// Monotonic, not r.now(): see the elapsed field. r.now() stays wall-clock
	// for the collection window, which is compared against RFC3339 args.
	startedAt := time.Now()
	outcome, runErr := r.execute(ctx, cfg, job)
	duration := int(r.elapsed(startedAt) / time.Millisecond)
	if duration < 0 {
		duration = 0
	}

	submission := platform.Submission{JobID: job.ID, DurationMS: duration}
	if runErr == nil && !outcome.Skipped() {
		// A non-finite sample reaches the AAS payload unfiltered (only the
		// tempfile path drops them), and json.Marshal refuses +Inf/NaN. Catch it
		// here and answer the job as failed: left to the submit call it would
		// look like a transport failure and the job would sit 'running' until
		// the platform's sweep closed it an hour later.
		if _, err := json.Marshal(outcome.Payload); err != nil {
			runErr = fmt.Errorf("%w: %v", errUnencodableResult, err)
		}
	}

	switch {
	case runErr != nil:
		submission.Outcome = platform.OutcomeError
		submission.Error, submission.FailureClass = describeFailure(runErr)
		log.Printf("job %s (%s) failed after %dms: %s", job.ID, job.Kind, duration, submission.Error)
	default:
		// One mapping, shared with the fixtures: a skip names itself as one and
		// carries no payload, and a result is the BARE payload the apply reads.
		wire := outcome.AsSubmission()
		submission.Outcome, submission.Result, submission.SkipReason =
			wire.Outcome, wire.Result, wire.SkipReason
		if outcome.Skipped() {
			log.Printf("job %s (%s) skipped (%s) in %dms",
				job.ID, job.Kind, submission.SkipReason, duration)
		} else {
			log.Printf("job %s (%s) ok in %dms", job.ID, job.Kind, duration)
		}
	}

	// The verdict comes from the SUBMIT, not from the run: the platform records
	// a refused payload on the job row rather than raising, so a box that does
	// not read the reply reports a collection as landed when it was rejected.
	if err := r.submit(ctx, client, creds, submission); err != nil {
		r.consecutiveJobFailures++
		log.Printf("job %s was not accepted: %v (consecutive job failures: %d)",
			job.ID, sanitize(err), r.consecutiveJobFailures)
		return
	}
	if runErr != nil {
		// Accepted, but the box could not do the work.
		r.consecutiveJobFailures++
		return
	}
	r.consecutiveJobFailures = 0
}

// submit posts one answer, retrying while the failure is transient. The work is
// already done and the job stays 'running' platform-side until the hourly sweep
// closes it, so giving up on the first reset would lose the whole collection.
func (r *Runner) submit(ctx context.Context, client *platform.Client, creds platform.Credentials, s platform.Submission) error {
	for attempt := 1; ; attempt++ {
		submitCtx, cancel := context.WithTimeout(ctx, platformTimeout)
		err := client.Submit(submitCtx, creds, s)
		cancel()
		if err == nil {
			return nil
		}
		// A PT404 is routine, not an outage: the job was swept, answered, or
		// was never ours. Retrying it would only collect the same answer, and
		// it says nothing about whether this box is working.
		if platform.Classify(err) == platform.ClassJobGone {
			log.Printf("job %s was no longer open when the answer arrived", s.JobID)
			return nil
		}
		// A refused payload is the platform's verdict on the answer, not a
		// transport problem: re-sending the same bytes would be refused again.
		if errors.Is(err, platform.ErrResultRejected) {
			return err
		}
		if platform.Classify(err) != platform.ClassTransient || attempt >= submitAttempts {
			log.Printf("submitting job %s failed: %v", s.JobID, sanitize(err))
			return err
		}
		log.Printf("submitting job %s failed, retrying: %v", s.JobID, sanitize(err))
		if !r.sleep(ctx, submitBackoff*time.Duration(attempt)) {
			return err
		}
	}
}

// errCollectionPanicked marks a collection that panicked. It is deliberately a
// sentinel rather than a bare error string: without it the panic fell through
// retryableStoreError's "assume transient" default and classifyFailure's
// store_unreachable default, so a deterministic poison job was retried three
// times and reported to the platform as a metric-store outage on a box whose
// store was fine.
var errCollectionPanicked = errors.New("collection panicked")

// runCollectSafely turns a panic in a collection into an ordinary error.
//
// Nothing panics today -- the parsers guard their type assertions and lengths,
// and a fuzz sweep over hostile kinds, args and store responses found none. The
// guard is here because of what a panic would cost rather than how likely it
// is: the container restarts `unless-stopped`, the platform re-queues a swept
// job, and the loop runs jobs one at a time, so a single poison job would
// crash-loop the box and starve every other job on it indefinitely. The whole
// design is "never crash, always answer"; this makes that true of the job body
// too, and the job comes back as an error the platform can see.
func runCollectSafely(ctx context.Context, store *collect.Client, job platform.Job, now time.Time) (outcome collect.Outcome, err error) {
	defer func() {
		if p := recover(); p != nil {
			// The panic VALUE can carry anything -- a store response, a label --
			// so it never reaches the log or the platform. The stack carries no
			// strings or payloads (frame lines show scalar argument words and
			// pointers; a string shows as {addr, len}), and without it a panic
			// left no trace anywhere at all.
			log.Printf("job %s (%s) panicked; recovered\n%s", job.ID, job.Kind, debug.Stack())
			outcome = collect.Outcome{}
			err = fmt.Errorf("%w (kind %q)", errCollectionPanicked, job.Kind)
		}
	}()
	return collect.Run(ctx, store, job.Kind, job.Args, now)
}

// execute runs one collection, retrying the store while the failure is one a
// retry could clear.
func (r *Runner) execute(ctx context.Context, cfg config.Config, job platform.Job) (collect.Outcome, error) {
	jobCtx, cancel := context.WithTimeout(ctx, r.budget)
	defer cancel()

	store := r.storeClient(cfg)

	var lastErr error
	for attempt := 1; attempt <= storeAttempts; attempt++ {
		outcome, err := runCollectSafely(jobCtx, store, job, r.now())
		if err == nil {
			return outcome, nil
		}
		lastErr = err
		if !retryableStoreError(err) || jobCtx.Err() != nil {
			return collect.Outcome{}, err
		}
		if attempt < storeAttempts && !r.sleep(jobCtx, storeBackoff*time.Duration(attempt)) {
			break
		}
	}
	return collect.Outcome{}, lastErr
}

// retryableStoreError reports whether re-running the job could clear the error.
func retryableStoreError(err error) bool {
	if errors.Is(err, collect.ErrInvalidArgs) ||
		errors.Is(err, collect.ErrUnknownKind) ||
		errors.Is(err, collect.ErrWindowTooLong) ||
		// A panic is deterministic: retrying it just panics again, three times,
		// and burns the job budget doing it.
		errors.Is(err, errCollectionPanicked) {
		return false
	}
	var upstream *collect.UpstreamError
	if errors.As(err, &upstream) {
		return upstream.Retryable()
	}
	// Network, DNS or timeout against a container on our own compose network.
	return true
}

// errUnencodableResult marks a payload json.Marshal refuses (a non-finite
// sample that reached the AAS metrics).
var errUnencodableResult = errors.New("result cannot be encoded")

// describeFailure turns an error into the short (error, failure_class) pair the
// submit takes. The raw error is deliberately NOT forwarded: a transport error
// carries the request URL, and that URL carries the PromQL built from the job's
// labels. The platform gets a class; the local log gets the sanitized detail.
func describeFailure(err error) (string, string) {
	text, class := classifyFailure(err)
	return truncate(text, errorMaxBytes), class
}

func classifyFailure(err error) (string, string) {
	switch {
	case errors.Is(err, collect.ErrInvalidArgs):
		return "job args could not be used", "invalid_args"
	case errors.Is(err, collect.ErrUnknownKind):
		return "this instance does not know this job kind", "unknown_kind"
	case errors.Is(err, collect.ErrWindowTooLong):
		return "collection window is too long", "window_too_long"
	case errors.Is(err, context.DeadlineExceeded):
		return "collection exceeded the local time budget", "timeout"
	case errors.Is(err, context.Canceled):
		return "collection was cancelled", "cancelled"
	case errors.Is(err, errUnencodableResult):
		return "the collected result could not be encoded", "unencodable_result"
	case errors.Is(err, errCollectionPanicked):
		return "collection failed unexpectedly", "panic"
	default:
		var upstream *collect.UpstreamError
		if errors.As(err, &upstream) {
			return fmt.Sprintf("metric store returned %d", upstream.StatusCode), "store_error"
		}
		return "metric store unreachable", "store_unreachable"
	}
}

// sanitize strips the request URL out of a transport error before it reaches a
// log line: the URL carries the PromQL, and the PromQL carries the job's
// cluster and node labels.
func sanitize(err error) error {
	var urlErr *url.Error
	if errors.As(err, &urlErr) {
		return fmt.Errorf("%s request failed: %w", urlErr.Op, urlErr.Err)
	}
	return err
}

func truncate(s string, max int) string {
	// Clean FIRST, then cap. Postgres rejects ANY invalid UTF-8 with 22021, not
	// just a split rune, so validity rather than length is the requirement, and
	// an early return on length alone handed a short invalid string back.
	//
	// Nothing reaching here today is invalid or over the cap: classifyFailure
	// returns constant ASCII (41 bytes at most), and for an UpstreamError it
	// forwards only the status code, never the message -- describeFailure's own
	// docstring above says the raw error is deliberately not forwarded. This is
	// a guard for the day someone widens classifyFailure, not for today's
	// inputs. An earlier comment here claimed it was surviving upstream bodies,
	// which no caller sends.
	//
	// Both calls are needed: the second because slicing a cleaned string can
	// still land mid-rune. An earlier version instead walked the cut point back
	// while the whole prefix was invalid, which collapsed the text to "" for a
	// leading bad byte -- and an empty `error` beside a non-null failure_class
	// is a PT400, not retried and not consuming the job.
	s = strings.ToValidUTF8(s, "")
	if len(s) <= max {
		return s
	}
	return strings.ToValidUTF8(s[:max], "")
}

// setHealth records the verdict with the deadline by which the next tick must
// have written again: the sleep, plus a whole job budget, plus slack.
func (r *Runner) setHealth(healthy bool, reason string, next time.Duration) {
	now := r.now()
	if err := writeHealth(r.healthPath, healthy, reason, now, now.Add(r.healthDeadline(next))); err != nil {
		log.Printf("could not write the health file: %v", err)
	}
}

// healthDeadline is how long the next stamp may take: the sleep, one whole job,
// the submit that follows it with every retry, and slack. Budgeting only for
// the job would call a working loop wedged whenever a submit had to retry. It
// reads r.budget rather than the constant so the deadline and the job ceiling
// cannot drift apart.
func (r *Runner) healthDeadline(next time.Duration) time.Duration {
	return next + r.budget + submitAttempts*platformTimeout + 2*time.Minute
}

// clampInterval bounds what the platform asked for to [1s, 1h].
func clampInterval(d time.Duration) time.Duration {
	if d <= 0 {
		return defaultPollInterval
	}
	if d < minPollInterval {
		return minPollInterval
	}
	if d > maxPollInterval {
		return maxPollInterval
	}
	return d
}

// jittered spreads the fleet by +/-20% and re-clamps, so jitter can never carry
// the interval outside the bound the clamp just enforced.
func (r *Runner) jittered(d time.Duration) time.Duration {
	factor := 1 + (rand.Float64()*2-1)*jitterFraction
	return clampInterval(time.Duration(float64(d) * factor))
}

// sleepCtx waits, returning false if the context ended first.
func sleepCtx(ctx context.Context, d time.Duration) bool {
	timer := time.NewTimer(d)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-timer.C:
		return true
	}
}
