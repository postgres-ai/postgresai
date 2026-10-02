// Package runner is the loop: ask the platform for work, run it -- one job at a
// time on the monitoring arm, on a bounded pool on the DBLab arm -- against the
// local metric store or the local DBLab engine, post the answer back, sleep.
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
	"net/http"
	"net/url"
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/collect"
	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/config"
	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/dblab"
	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/joe"
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

	// submitAttempts is how often an answer is re-posted, submitBackoff paces
	// them. A job is 'running' platform-side until it is answered or the hourly
	// sweep closes it, so a single reset on the way back would otherwise throw
	// away the whole collection and wedge that period for an hour.
	//
	// The budget is sized against what it actually contends with rather than
	// against a generic blip: instance_job_submit waits out its own 5s
	// lock_timeout for the monitoring_instances row, and the pull path's
	// consumer can hold that row for a whole PgQ batch -- one transaction, one
	// outbound call per event. Three attempts 3s apart gave up after ~24s,
	// inside the first event of such a batch, so the retry existed but could
	// not outlast the thing it exists for. Eight paced 10s, 20s ... 70s keep
	// trying for 5m20s: long enough for a realistic batch of a few events,
	// not a pretence of outlasting a pathological one.
	submitAttempts = 8
	submitBackoff  = 10 * time.Second

	// A rejected credential or a missing RPC is not fixed by polling harder.
	longBackoff = 10 * time.Minute

	// transientBackoffBase is the first wait after a poll that reached no verdict
	// at all, doubling per consecutive failure to transientBackoffMax. Two
	// seconds rather than minPollInterval because `connection refused` returns
	// instantly: at a one-second floor that is a poll per second, and the
	// jitter's -20% is clamped back up, so the fleet stops spreading exactly
	// when all of it fails together.
	transientBackoffBase = 2 * time.Second

	// transientBackoffMax is deliberately BELOW longBackoff: a platform that
	// ANSWERED "go away" said something a dropped connection did not, and must
	// keep backing off harder. Not maxPollInterval either -- that bounds what
	// the PLATFORM may ask for. The minute is a small fraction of the 30 minutes
	// the platform goes on dispatching work here (c_job_backed_window), so a box
	// is answering again well inside it.
	transientBackoffMax = 1 * time.Minute

	// outageUnhealthyAfter is how long polls may go on failing before the
	// container reports unhealthy. See pollHealthy.
	outageUnhealthyAfter = 5 * time.Minute

	// unhealthyAfter consecutive failed polls flips the healthcheck.
	unhealthyAfter = 3

	// idleLogInterval keeps an un-provisioned instance to one log line an hour.
	idleLogInterval = 1 * time.Hour

	// platformTimeout bounds one poll or submit.
	platformTimeout = 30 * time.Second

	// submitBudget is the worst case wall clock for ONE answer: every attempt
	// burning its whole platformTimeout, plus every backoff between them.
	// Anything asking "how long may a job take" wants this rather than the
	// attempt count, which counts the requests and not the waiting.
	//
	// It is also the other half of the platform's claim arithmetic,
	// claim_limit x per_job_ceiling < public.instance_jobs_expire_sweep's
	// stuck_after: with claim_limit 1 the ceiling is jobBudget + 2*submitBudget
	// -- the answer AND the terminal close after a refusal, which retries on the
	// same ladder -- so 32m20s against an hour.
	submitBudget = submitAttempts*platformTimeout +
		submitBackoff*(submitAttempts*(submitAttempts-1)/2)

	// shutdownSubmitBudget bounds EVERY submit sent after the context has been
	// cancelled -- the answer and the terminal close that can follow it --
	// together, as one deadline rather than one each. It sits between two
	// numbers that belong to other systems, and both earlier versions of it got
	// one of them wrong:
	//
	// ABOVE the platform's 5s lock_timeout: v1.instance_job_submit sets that and
	// waits it out for the monitoring_instances row, which the pull path's
	// consumer can hold for a whole PgQ batch. A 4s deadline aborted before the
	// server could reach either outcome, making delivery strictly worse in the
	// one contention case submitAttempts was sized against.
	//
	// BELOW docker's 10s stop grace (no stop_grace_period is set for this
	// service), after which the process is SIGKILLed. 8s PER SUBMIT satisfied
	// that for one submit and not for two: runJob can send a second, and an
	// argument that it "returns fast" is not a bound. Sharing the budget makes
	// it mechanical again.
	shutdownSubmitBudget = 9 * time.Second

	// errorMaxBytes is the platform's cap on the submitted error text.
	errorMaxBytes = 512

	// dblabGetTimeout bounds ONE read against the local engine. Short, because a
	// read is retried and each attempt has to leave room for the next.
	dblabGetTimeout = 2 * time.Minute

	// dblabConcurrency is how many jobs of ONE claimed batch the DBLab arm runs at
	// once; the monitoring arm stays strictly sequential (see tick).
	//
	// IT MUST EQUAL THE PLATFORM'S CLAIM LIMIT (app.settings.dblab_job_claim_limit,
	// which platform-all#816 seeds to 5). That equality is what stops anything
	// claimed sitting unstarted, so claimed_at is a true proxy for started_at and
	// claim_limit x per_job_ceiling < the sweep's stuck_after collapses to ONE
	// ceiling whatever the batch size. Claim more and the tail is swept to
	// 'failed' while this process is still going to run it: the caller is told a
	// POST /clone failed and the clone exists anyway -- the mirror of the write
	// rule below. runBatchConcurrently logs a batch wider than the pool, and
	// TestADBLabBatchRunsAtMostThePoolAtOnce pins the number.
	//
	// Five is the Console instance page's own call count, and deliberately small
	// -- the engine is the customer's box, and the point is to stop a cheap read
	// queueing behind an expensive write, not to parallelise load onto it. It is
	// NOT applied per method, so a batch of writes runs concurrently too;
	// bounding that belongs on the claim, the only place that can refuse work
	// without having already taken it (#391).
	dblabConcurrency = 5

	// dblabWriteAttempts is 1, and that is the whole point: a POST, a PATCH or a
	// DELETE against the engine is NOT idempotent. A `POST /clone` whose answer
	// we never saw may well have created the clone, so re-sending it would create
	// a second one -- on the customer's disk, charged to the customer -- to
	// recover from a timeout. A read is safe to repeat and is repeated
	// (dblabReadAttempts); a write is answered as failed and the platform decides.
	dblabWriteAttempts = 1
	dblabReadAttempts  = 3

	// joeGetTimeout bounds ONE read against the local Joe -- the channel lookup
	// and any GET job. Short, because a read is retried and each attempt has to
	// leave room for the next, and Joe's reads answer in milliseconds or they are
	// not going to: the channel list is a read of its own config.
	joeGetTimeout = 30 * time.Second

	// joeConcurrency is how many jobs of ONE claimed batch the Joe arm runs at
	// once, and IT MUST EQUAL THE PLATFORM'S CLAIM LIMIT
	// (app.settings.joe_job_claim_limit) for the reason dblabConcurrency gives:
	// the equality is what keeps claimed_at a true proxy for started_at, so
	// claim_limit x per_job_ceiling collapses to ONE ceiling and the sweep cannot
	// fail a tail this process is still going to run.
	//
	// DERIVED, not taken from dblabConcurrency -- landing on the same number is
	// the arithmetic, not a copy. The ceiling is unchanged at 32m20s (jobBudget +
	// 2*submitBudget; same ladders) against a 1-hour stuck_after, so with pool ==
	// claim_limit the sweep bounds NO pool size at all. What bounds it is the box,
	// and Joe is cheap: /webui/command hands the message to a goroutine and
	// returns, /webui/channels reads config, so a joe_call is two sub-second local
	// calls and the pool adds no load to the customer's database -- Joe's own
	// per-channel processor serialises the SQL work. So this is sized to DRAIN A
	// POLL WINDOW rather than to parallelise load: five covers a burst of five
	// distinct users inside one active interval, and the sixth waits that interval
	// rather than the idle one.
	joeConcurrency = 5

	// joeWriteAttempts is 1, for the reason dblabWriteAttempts is: a POST
	// /webui/command is NOT idempotent. Joe answers 200 BEFORE it has done
	// anything, so a POST whose reply we never saw may well have been accepted,
	// and re-sending it would run the command a second time on the customer's
	// clone. A read is safe to repeat and is repeated.
	joeWriteAttempts = 1
	joeReadAttempts  = 3

	// joeChannelAttempts retries the channel lookup, and it is a READ ladder even
	// when the job is a write: a failed lookup means the command was never sent,
	// so repeating it cannot duplicate anything. Do not fold this into
	// joeWriteAttempts -- failing a whole command because a preliminary config
	// read blipped is stricter than the write contract asks for.
	joeChannelAttempts = 3

	// joeSkipNoChannels is the skip_reason for a Joe that ANSWERED A CHANNEL LIST
	// and served none. A SKIP rather than a failure because there is nothing to
	// deliver to and no Joe failure to report. A 200 that is not a channel list is
	// NOT this -- see joe.ErrBadChannelList.
	//
	// v1.joe_job_submit takes free text under a 64-CHARACTER cap (length(), not
	// octet_length(), which that same rpc uses for `error`). So does
	// v1.instance_job_submit: the monitoring channel's three reasons are this
	// agent's own vocabulary, not something either rpc enforces.
	joeSkipNoChannels = "no_channels"
)

// Runner owns the loop's state.
type Runner struct {
	platformClient func(baseURL string) *platform.Client
	storeClient    func(cfg config.Config) *collect.Client
	dblabClient    func(cfg config.Config) *dblab.Client
	joeClient      func(cfg config.Config) *joe.Client
	healthPath     string
	now            func() time.Time
	// elapsed measures a job's duration. Separate from now() because now() is
	// wall-clock (`time.Now().UTC()`, and .UTC() strips Go's monotonic
	// reading), so subtracting two of its readings across a backward NTP step
	// yields a negative duration_ms. Injectable so tests can produce a
	// deterministic one.
	elapsed func(time.Time) time.Duration
	// monotonic stamps and reads failingSince, for the same reason: an outage
	// timed on r.now() stays green through a backward step as long as the step.
	monotonic func() time.Time
	// sleep is overridden in tests; it returns false when the context ended.
	sleep func(ctx context.Context, d time.Duration) bool
	// shutdownDeadline is the single wall-clock deadline shared by every submit
	// sent after cancellation. Zero until the first one. See submit. Guarded by
	// mu: a DBLab batch can be cancelled with several workers in flight and any
	// of them may be the one to set it.
	shutdownDeadline time.Time
	// budget is the per-job ceiling. A field rather than the constant directly
	// so a test can shorten it and see what a job that overruns is answered as.
	budget time.Duration

	// The poll loop's own state, written only from tick/pollError/idle and
	// therefore only ever from the single loop goroutine. Kept ABOVE mu on
	// purpose: the merge that brought #388's backoff onto this branch landed
	// hardFailures and failingSince below it, inside the block mu's comment
	// enumerates, which reads as a claim that the lock covers them.
	consecutiveFailures int
	lastIdleLog         time.Time
	// hardFailures counts the polls in the current run that the platform
	// refused rather than never answered. The count rule applies to these
	// alone: transient retries are seconds apart and would trip it in six.
	hardFailures int
	// failingSince is when the current run of failed polls started. Re-stamped
	// whenever consecutiveFailures leaves zero, so a successful poll clearing
	// the counter is all it takes to forget it.
	failingSince time.Time
	// afterSlot runs between a worker slot being taken and the re-check that
	// follows it, and is nil outside tests. The window it opens onto -- a
	// cancellation arriving while the dispatch loop is parked on a slot -- is
	// not reachable from outside the loop, so without a seam the re-check in
	// runBatchConcurrently cannot be covered at all.
	afterSlot func()
	// mu guards the two fields a DBLab batch's workers share:
	// consecutiveJobFailures and shutdownDeadline. NOT the health file -- that
	// is written outside the lock, because writeHealth is write-and-rename and
	// tick re-stamps after wg.Wait(), so a concurrent stamp can only be
	// transiently stale. Move a stamp later into a worker and that stops being
	// true. The monitoring arm is single-goroutine and the lock is free there.
	mu                     sync.Mutex
	consecutiveJobFailures int
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
		// Built with the whole job budget as its client timeout: a write gets one
		// attempt and the budget IS its bound, while a read is bounded tighter by
		// a per-attempt context in executeDBLab.
		dblabClient: func(cfg config.Config) *dblab.Client {
			return dblab.NewClient(cfg.DBLabURL, cfg.DBLabVerifyToken, jobBudget)
		},
		// Same bound and the same reason: a write gets one attempt and the budget
		// IS its bound, while a read is bounded tighter per attempt in executeJoe.
		joeClient: func(cfg config.Config) *joe.Client {
			return joe.NewClient(cfg.JoeURL, cfg.JoeVerifyToken, jobBudget)
		},
		healthPath: healthPath,
		now:        func() time.Time { return time.Now().UTC() },
		elapsed:    time.Since,
		monotonic:  time.Now,
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
	switch {
	case cfg.IsDBLab():
		// The engine's OWN token and nothing else: no org token, and no instance
		// id, because the token identifies the engine (platform-all#805).
		creds = platform.Credentials{DBLabToken: cfg.DBLabToken}
	case cfg.IsJoe():
		// Joe's own token, on the same terms (#398).
		creds = platform.Credentials{JoeToken: cfg.JoeToken}
	}

	pollCtx, cancel := context.WithTimeout(ctx, platformTimeout)
	resp, err := client.Poll(pollCtx, creds)
	cancel()
	if err != nil {
		return r.jittered(r.pollError(err))
	}

	r.consecutiveFailures = 0
	r.hardFailures = 0
	next := clampInterval(time.Duration(resp.NextPollMS) * time.Millisecond)

	// HOW MANY A POLL HANDS OUT IS THE PLATFORM'S BUSINESS, not this loop's, and
	// since platform-all#816 it is a SETTING per channel
	// (app.settings.instance_job_claim_limit, 1; dblab_job_claim_limit, 5;
	// joe_job_claim_limit, 5 -- see joeConcurrency), not a constant. Nothing here
	// had to change to DRAIN a wider one: `jobs` has been
	// an array since platform-all#805, so a box installed before #816 parses and
	// answers five. Running five SEQUENTIALLY is the tail-sweep below, though,
	// which is why !866 and #391 land together and why dblab_jobs_enabled being
	// off is what holds it until they do.
	//
	// THIS SIDE DECLINES NOTHING IT IS HANDED, deliberately: a job claimed here
	// and then refused would sit 'running' until the platform's hourly sweep
	// failed it. Only the claim can refuse work without having already taken it.
	// The pool bounds how many RUN at once, never how many are answered.
	//
	// WHAT THIS SIDE OWES IN RETURN is the arithmetic behind that limit:
	// claim_limit x per_job_ceiling must stay under the sweep's stuck_after, and
	// the ceiling is NOT jobBudget alone -- it is jobBudget + 2*submitBudget
	// (32m20s; see submitBudget), because a job can burn the answer ladder and
	// then the terminal close after a refusal. Against a 1-hour stuck_after that
	// permits a SEQUENTIAL claim of one, which is why monitoring's limit is 1: a
	// sequential loop leaves the tail of a batch claimed-but-not-started for
	// (n-1) ceilings, which the platform cannot tell from a box that died.
	//
	// THE DBLAB AND JOE ARMS RUN THEIR BATCH CONCURRENTLY (runBatchConcurrently),
	// which is what collapses the product to one ceiling WITHIN A CLAIMED BATCH
	// and lets each limit be its pool size. It does not make this loop poll while
	// a batch runs: the tick still waits the batch out, so a call enqueued after
	// the claim waits for the slowest job in it. Monitoring stays strictly one at
	// a time below.
	if len(resp.Jobs) == 0 {
		// Nothing to run means nothing is failing: a fleet being drained (the
		// flag turned off) hands out no work, and a box must not stay red on a
		// run of failures it can no longer retry.
		r.mu.Lock()
		r.consecutiveJobFailures = 0
		r.mu.Unlock()
	}

	switch {
	case cfg.IsDBLab():
		r.runBatchConcurrently(ctx, client, creds, cfg, resp.Jobs, next,
			dblabConcurrency, "app.settings.dblab_job_claim_limit")
	case cfg.IsJoe():
		r.runBatchConcurrently(ctx, client, creds, cfg, resp.Jobs, next,
			joeConcurrency, "app.settings.joe_job_claim_limit")
	default:
		for _, job := range resp.Jobs {
			// Stamped before AND after each job: the deadline in the file allows
			// one job budget, so a tick that runs several would otherwise look
			// wedged while it is doing exactly what it should. It carries the
			// CURRENT verdict -- re-stamping healthy here would make a box
			// already judged dead report green for the whole duration of every
			// later job.
			r.stampHealth(next)
			if r.runJob(ctx, client, creds, cfg, job) {
				r.recordRun(0, 1)
			} else {
				r.recordRun(1, 1)
			}
			if ctx.Err() != nil {
				break
			}
		}
	}

	r.stampHealth(next)
	return r.jittered(next)
}

// runBatchConcurrently runs a claimed DBLab or Joe batch on a bounded worker
// pool. The
// monitoring arm does not come through here: that fleet is live, its claim limit
// is 1, and a pool would be a behaviour change for no gain.
//
// EACH JOB IS STILL HANDED TO runJob UNCHANGED, which keeps every rule intact: a
// write is attempted ONCE (dblabWriteAttempts, joeWriteAttempts), a read is
// retried, each job is
// answered for its own id, and a failure is submitted rather than swallowed.
// Concurrency changes WHEN jobs run, never how many times a call is sent.
//
// jobs should never exceed pool -- see dblabConcurrency and joeConcurrency for
// why that equality is load-bearing. If it does, the surplus queues here rather
// than being declined, and the log line below is the only warning anyone gets;
// claimSetting names the platform setting that has to come down.
func (r *Runner) runBatchConcurrently(
	ctx context.Context,
	client *platform.Client,
	creds platform.Credentials,
	cfg config.Config,
	jobs []platform.Job,
	next time.Duration,
	pool int,
	claimSetting string,
) {
	if len(jobs) == 0 {
		return
	}

	if len(jobs) > pool {
		// The equality this rests on has broken, and nothing else reports it:
		// the surplus waits for a slot while the platform counts it as running,
		// and the sweep can fail it to its caller before this process starts it.
		log.Printf("the platform claimed %d jobs but this agent runs %d at a time; "+
			"the surplus can be swept as failed while it is still going to be run "+
			"(lower %s, or upgrade the agent)",
			len(jobs), pool, claimSetting)
	}

	slots := make(chan struct{}, pool)
	var wg sync.WaitGroup
	// The batch's verdict, counted once and applied once: see recordRun. Only
	// the jobs this loop actually STARTED are judged -- one it never dispatched
	// says nothing about whether the box works.
	var failed atomic.Int64
	started := 0

dispatch:
	for _, job := range jobs {
		// THE LOOP NEVER DISPATCHES AFTER OBSERVING THE CONTEXT END. Not the
		// same as "nothing is started after it ends" -- the context can die
		// between the last check and the goroutine's first instruction, and no
		// arrangement here closes that. What is closed is the loop deciding to
		// start one.
		//
		// The early-out is a cheap skip and the <-ctx.Done() arm keeps the park
		// interruptible; neither is load-bearing alone. THE RE-CHECK AFTER THE
		// SEND IS: select picks uniformly at random when both cases are ready,
		// so without it a cancellation landing while the loop is parked starts
		// one more job about half the time. afterSlot is the seam that lets a
		// test reach that interleaving, which no amount of scheduling pressure
		// can produce from outside.
		//
		// The jobs left undispatched were already CLAIMED, so they are 'running'
		// platform-side and wait out the sweep; beginning a call this process is
		// about to abandon is the worse of the two.
		if ctx.Err() != nil {
			break dispatch
		}
		select {
		case <-ctx.Done():
			break dispatch
		case slots <- struct{}{}:
		}
		if r.afterSlot != nil {
			r.afterSlot()
		}
		if ctx.Err() != nil {
			<-slots
			break dispatch
		}
		wg.Add(1)
		started++
		go func(job platform.Job) {
			defer wg.Done()
			defer func() { <-slots }()
			if !r.runWorker(ctx, client, creds, cfg, job, next) {
				failed.Add(1)
			}
		}(job)
	}

	// The tick does not return until the batch is done: the next poll must not
	// overlap this one, or the platform hands out more work while these are
	// still running and the pool's equality with the claim limit stops meaning
	// anything.
	wg.Wait()
	r.recordRun(int(failed.Load()), started)
}

// runWorker is one pool worker's whole body, panic guard first.
//
// The stamp is per worker for the same reason the sequential arm stamps per
// job: the health deadline allows one job budget, so a batch that outlives one
// reads as wedged and a working box reports dead through the container
// HEALTHCHECK -- `mon health` shows a dead channel and someone restarts it,
// mid-clone-create. (Nothing restarts it automatically: the compose service
// carries `restart: unless-stopped`, which fires on process exit, not on an
// unhealthy status.)
//
// THE GUARD COVERS BOTH, and it is here rather than only around the job because
// a panic in either ends the process. runCollectSafely already covers a
// COLLECTION's body; a dblab_call has no equivalent, and a panic on this arm
// strands up to dblabConcurrency-1 siblings already CLAIMED and now never
// answered -- including a POST /clone the engine may have carried out, which is
// the outcome dblabWriteAttempts exists to prevent (#391).
//
// The answer is best effort and safe to duplicate: a job already answered comes
// back PT404, which submit treats as routine. It does NOT catch runtime.Goexit
// -- recover() returns nil for that -- so a t.Fatalf from an injected hook on a
// worker drops the job silently; assert from the test goroutine.
func (r *Runner) runWorker(ctx context.Context, client *platform.Client, creds platform.Credentials, cfg config.Config, job platform.Job, next time.Duration) (ok bool) {
	defer func() {
		if p := recover(); p == nil {
			return
		}
		ok = false
		// The panic VALUE can carry a payload or a label, so only the stack is
		// logged -- the same rule runCollectSafely follows.
		log.Printf("job %s (%s) panicked; recovered\n%s", job.ID, job.Kind, debug.Stack())
		answer := platform.Submission{JobID: job.ID, Outcome: platform.OutcomeError}
		answer.Error, answer.FailureClass = describeFailure(errJobPanicked)
		if err := r.submit(ctx, client, creds, answer); err != nil {
			log.Printf("job %s: panicked and the failure could not be recorded: %v",
				job.ID, sanitize(err))
		}
	}()
	r.stampHealth(next)
	return r.runJob(ctx, client, creds, cfg, job)
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
	// Counted before the wait is chosen: the transient ladder is a function of
	// how many polls in a row have failed, this one included.
	r.consecutiveFailures++
	if r.consecutiveFailures == 1 {
		r.failingSince = r.monotonic()
	}
	class := platform.Classify(err)
	if class != platform.ClassTransient {
		r.hardFailures++
	}
	switch class {
	case platform.ClassAuth:
		return r.pollFailed("platform rejected the credential", err, longBackoff)
	case platform.ClassUnavailable:
		return r.pollFailed(unavailableReason(err), err, longBackoff)
	case platform.ClassRequest:
		return r.pollFailed("platform refused the poll"+codeSuffix(err), err, longBackoff)
	case platform.ClassTransient:
		// Nothing was decided platform-side. Parking defaultPollInterval here
		// threw away the pacing the platform had just asked for and left the box
		// a black hole for work it went on dispatching (#388) -- green
		// throughout, since one failure never reaches unhealthyAfter. Classify
		// routes a 503 to this class rather than ClassUnavailable precisely to
		// avoid that park; this arm is what makes that true.
		return r.pollFailed("poll failed", err, r.transientBackoff())
	default:
		// ClassJobGone: on the POLL path, instance_job_auth answering PT404 for
		// an instance id this credential does not own. The token and the id
		// disagree, so seconds-scale retries would just be a misconfigured box
		// hammering the platform.
		return r.pollFailed("poll failed", err, defaultPollInterval)
	}
}

// transientBackoff is the wait after consecutiveFailures transient polls in a
// row: transientBackoffBase, doubling, up to transientBackoffMax. Doubled in a
// loop with an early return rather than shifted by the count: a platform down
// for a day reaches four figures of consecutive failures, and `base << n` is a
// negative duration -- an immediate re-poll -- long before that.
func (r *Runner) transientBackoff() time.Duration {
	wait := transientBackoffBase
	for i := 1; i < r.consecutiveFailures; i++ {
		wait *= 2
		if wait >= transientBackoffMax {
			return transientBackoffMax
		}
	}
	return wait
}

// stampHealth refreshes the health file with the verdict as it stands.
func (r *Runner) stampHealth(next time.Duration) {
	// Read under the lock: a DBLab batch's workers stamp concurrently, and the
	// branch below must act on the value it reported.
	r.mu.Lock()
	failures := r.consecutiveJobFailures
	r.mu.Unlock()
	if failures >= unhealthyAfter {
		// "runs", not "jobs": the counter now moves once per poll's worth of
		// work, so on the DBLab arm one unit is a whole batch.
		r.setHealth(false, fmt.Sprintf("%d runs in a row failed", failures), next)
		return
	}
	r.setHealth(true, "", next)
}

// pollFailed logs a failed poll, stamps the health file and hands back the wait
// it was given, so each arm of pollError is one line. reason is ours and goes in
// the health file, which an operator reads through `docker inspect`; detail may
// carry a message the platform wrote, so it only ever reaches the local log.
func (r *Runner) pollFailed(reason string, detail error, wait time.Duration) time.Duration {
	if detail != nil {
		log.Printf("%s: %v (consecutive failures: %d)", reason, sanitize(detail), r.consecutiveFailures)
	} else {
		log.Printf("%s (consecutive failures: %d)", reason, r.consecutiveFailures)
	}
	// The wait that was actually scheduled, not defaultPollInterval: next_check_by
	// says when this file is expected to have been written again, and a
	// two-second retry promising ten minutes of slack would let a genuinely
	// wedged loop go on reading fresh for the whole ten.
	r.setHealth(r.pollHealthy(), reason, wait)
	return wait
}

// pollHealthy decides whether a run of failed polls has gone on long enough to
// take the container off the channel.
//
// Refusals are counted: longBackoff paces them, so unhealthyAfter of them is
// twenty minutes. Transient failures are paced in seconds, so a run that has
// any is measured in time instead -- outageUnhealthyAfter, inside the
// platform's 30-minute routing window. The run's total must still reach
// unhealthyAfter, so a lone failure followed by a long wait is not an outage.
func (r *Runner) pollHealthy() bool {
	if r.hardFailures >= unhealthyAfter {
		return false
	}
	if r.consecutiveFailures < unhealthyAfter {
		return true
	}
	return r.monotonic().Sub(r.failingSince) < outageUnhealthyAfter
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

// runJob executes one job, submits its answer, and reports whether the job
// landed. The caller owns what a failure MEANS for the box's health, because
// the two arms answer that differently: sequentially it is one more failure in
// a row, while a DBLab batch is judged as a whole (see runBatchConcurrently).
func (r *Runner) runJob(ctx context.Context, client *platform.Client, creds platform.Credentials, cfg config.Config, job platform.Job) (ok bool) {
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
		// A refused result leaves the job RUNNING: the refusal is correctly not
		// retried, but sending nothing else means the platform never hears an
		// answer and the job waits out the hourly sweep -- during which the
		// in-flight cap locks the user out and the CLI has long since timed
		// out. Close it as failed instead, which is true and terminal.
		if refusedByPlatform(err) && submission.Outcome != platform.OutcomeError {
			terminal := platform.Submission{JobID: job.ID, DurationMS: duration, Outcome: platform.OutcomeError}
			terminal.Error, terminal.FailureClass = describeFailure(errResultRefused)
			if ferr := r.submit(ctx, client, creds, terminal); ferr != nil {
				log.Printf("job %s: the result was refused and the failure could not be recorded: %v",
					job.ID, sanitize(ferr))
			} else {
				log.Printf("job %s: result refused, job closed as failed", job.ID)
			}
		}
		log.Printf("job %s was not accepted: %v", job.ID, sanitize(err))
		return false
	}
	if runErr != nil {
		// Accepted, but the box could not do the work.
		return false
	}
	return true
}

// recordRun folds one run's verdict into the failure count the healthcheck
// reads. failed is how many jobs did not land and total is how many ran, so the
// unit is a RUN rather than a job: sequentially that is one job, and for a
// DBLab batch it is the whole batch.
//
// A batch has to be judged whole, and the old per-job bump made that impossible
// rather than merely imprecise. Workers raced to bump and reset, so an
// identical batch of 5 with 3 failures ended on 0, 1, 2 or 3 depending purely
// on which goroutine finished last -- measured across 300 ticks as
// {0: 125, 1: 94, 2: 49, 3: 32}. 11% of them crossed unhealthyAfter and took a
// working box off the channel for three bad call arguments; 42% reported a
// clean run for a box that had failed 60% of its work. unhealthyAfter was sized
// against three bad RUNS in a row, and this is what restores that meaning.
func (r *Runner) recordRun(failed, total int) {
	if total == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if failed < total {
		// Anything landed, so the box can reach the platform and do work.
		r.consecutiveJobFailures = 0
		return
	}
	r.consecutiveJobFailures++
	log.Printf("all %d job(s) of this run failed (consecutive failed runs: %d)",
		total, r.consecutiveJobFailures)
}

// submit posts one answer, retrying while the failure is transient. The work is
// already done and the job stays 'running' platform-side until the hourly sweep
// closes it, so giving up on the first reset would lose the whole collection.
func (r *Runner) submit(ctx context.Context, client *platform.Client, creds platform.Credentials, s platform.Submission) error {
	// SIGTERM mid-job. The work is already DONE, so a cancelled context here is
	// pure loss: every attempt below fails instantly on the dead context,
	// Classify calls it transient, sleep returns false on attempt 1, and the job
	// sits `running` until the platform's hourly sweep. On the query channel
	// that is worse than loss -- instance_query_enqueue's in-flight cap is 1 and
	// its PT409 says "Nothing returns its id, so wait for it", so an ordinary
	// `docker compose up -d` during a query locks the user out for up to an hour.
	//
	// So the answer goes out on a context that survives the cancellation. ONE
	// attempt, and sized to the window it actually has: docker-compose sets no
	// stop_grace_period for this service, so Docker's default 10s applies and
	// the process is SIGKILLed after it. platformTimeout (30s) is three times
	// that, and runJob can queue a SECOND detached submit for the terminal
	// close -- so the pair has to fit inside the grace, not inside the ladder
	// it is replacing (#378).
	if ctx.Err() != nil {
		// One deadline for the whole shutdown, set on the first detached submit
		// and reused by any that follow. Under mu because the DBLab arm's
		// workers can all reach this at once on a cancelled batch, and any of
		// them may be the one to set it -- which is what makes "one deadline for
		// the whole shutdown" a claim rather than a tautology.
		// r.now() is read BEFORE the lock and the unlock is deferred into a
		// closure: r.now is injectable, and a panic under this lock would
		// deadlock runWorker's recovery -- it answers the job by calling this
		// same function, which on a cancelled batch re-enters this branch and
		// takes mu again -- turning a loud crash into a silent wedge with
		// nothing to restart it.
		candidate := r.now().Add(shutdownSubmitBudget)
		deadline := func() time.Time {
			r.mu.Lock()
			defer r.mu.Unlock()
			if r.shutdownDeadline.IsZero() {
				r.shutdownDeadline = candidate
			}
			return r.shutdownDeadline
		}()
		detached, cancel := context.WithDeadline(context.WithoutCancel(ctx), deadline)
		defer cancel()
		err := client.Submit(detached, creds, s)
		if err != nil && platform.Classify(err) != platform.ClassJobGone {
			log.Printf("job %s: shutting down, the answer could not be delivered: %v", s.JobID, sanitize(err))
			return err
		}
		return nil
	}

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

// errResultRefused marks a result the platform accepted as a request and then
// refused -- oversized, or otherwise not something it will store. It is its own
// class because the remedy differs from every other failure: the answer cannot
// be made acceptable by retrying it, so the job must be closed as failed rather
// than left for the sweep.
var errResultRefused = errors.New("platform refused the result")

// refusedByPlatform reports whether the platform has decided these bytes are
// unacceptable, so re-sending them cannot help and the job needs closing by
// hand. It has two shapes and only one of them is a 200: an apply rejection
// comes back in the reply body, while a result the rpc will not take at all --
// oversize, the case the close exists for -- raises PT400, which rolls the
// whole call back and leaves the job 'running'. Keying the close on the
// in-body sentinel alone missed the second, which is the louder half.
func refusedByPlatform(err error) bool {
	// Never sent, so the platform has decided nothing about it: the job is
	// still whatever it was and this is a bug in our own submission, not a
	// verdict to relay.
	if errors.Is(err, platform.ErrUnknownOutcome) {
		return false
	}
	return errors.Is(err, platform.ErrResultRejected) ||
		platform.Classify(err) == platform.ClassRequest
}

// errCollectionPanicked marks a collection that panicked. It is deliberately a
// sentinel rather than a bare error string: without it the panic fell through
// retryableStoreError's "assume transient" default and classifyFailure's
// store_unreachable default, so a deterministic poison job was retried three
// times and reported to the platform as a metric-store outage on a box whose
// store was fine.
var errCollectionPanicked = errors.New("collection panicked")

// errJobPanicked marks a panic anywhere else in a job -- outside the collection
// body runCollectSafely guards. It is separate because the two are answered
// from different places and a dblab_call has no "collection" to name.
var errJobPanicked = errors.New("job panicked")

// runCollectSafely turns a panic in a collection into an ordinary error.
//
// Nothing panics today -- the parsers guard their type assertions and lengths,
// and a fuzz sweep over hostile kinds, args and store responses found none. The
// guard is here because of what a panic would cost rather than how likely it
// is: the container restarts `unless-stopped`, the platform re-queues a swept
// job, and the MONITORING loop runs jobs one at a time, so a single poison job
// would crash-loop the box and starve every other job on it indefinitely. (The
// DBLab arm runs a batch on a pool and is guarded by runWorker.) The whole
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

	// A DBLab or Joe call is not a collection: its args are their own shape, its
	// target is a service on this box rather than the metric store, and its result
	// is relayed verbatim instead of applied to anything.
	switch job.Kind {
	case dblab.KindCall:
		return r.executeDBLab(jobCtx, cfg, job)
	case joe.KindCall:
		return r.executeJoe(jobCtx, cfg, job)
	}

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

// executeDBLab runs one engine call.
//
// A READ is retried while the failure is one a retry could clear; a WRITE is
// not, ever -- see dblabWriteAttempts. The distinction is made here rather than
// inside the client because it is a statement about the JOB, not about the
// transport.
func (r *Runner) executeDBLab(ctx context.Context, cfg config.Config, job platform.Job) (outcome collect.Outcome, err error) {
	// EVERY failure leaving here is marked as an engine call's, so
	// classifyFailure can answer in the caller's own terms. Its deadline and
	// cancellation arms are worded for a COLLECTION, which is the monitoring
	// fleet's vocabulary and not this one's: someone waiting on `pgai dblab
	// clone create` was told their clone "exceeded the local time budget" as a
	// collection, naming work this box was never doing. Marked at the one exit
	// rather than per return, and with two %w, so errors.Is/As through the
	// chain -- the dblab sentinels, *EngineError -- all still match and every
	// arm below keeps its own wording.
	defer func() {
		if err != nil {
			err = fmt.Errorf("%w: %w", errDBLabCall, err)
		}
	}()

	req, err := dblab.Parse(job.Args)
	if err != nil {
		return collect.Outcome{}, err
	}

	engine := r.dblabClient(cfg)
	attempts := dblabWriteAttempts
	if req.Method == http.MethodGet {
		attempts = dblabReadAttempts
	}

	var lastErr error
	for attempt := 1; attempt <= attempts; attempt++ {
		callCtx := ctx
		var cancel context.CancelFunc
		if attempts > 1 {
			// Only a retried call is bounded tighter than the job budget: giving a
			// single-attempt write the same short deadline would fail a clone that
			// was about to succeed, and there would be no second attempt to save it.
			callCtx, cancel = context.WithTimeout(ctx, dblabGetTimeout)
		}
		payload, err := engine.Do(callCtx, req)
		if cancel != nil {
			cancel()
		}
		if err == nil {
			// payload is nil when the engine answered with no body (platform-all#815). It
			// marshals to a JSON null, which PostgREST v9.0.1 binds to a SQL NULL
			// -- measured, because the distinction matters:
			// public.data_usage_collect gates on `result is not null`, so an
			// empty /status reading stays invisible to it.
			return collect.Outcome{Status: collect.OutcomeOK, Payload: payload}, nil
		}
		lastErr = err
		if !retryableEngineError(err) || ctx.Err() != nil {
			return collect.Outcome{}, err
		}
		if attempt < attempts && !r.sleep(ctx, storeBackoff*time.Duration(attempt)) {
			break
		}
	}
	return collect.Outcome{}, lastErr
}

// executeJoe runs one Joe call.
//
// TWO STEPS WITH DIFFERENT RETRY RULES, which is why this is not executeDBLab
// with another client. The channel lookup is a READ and is retried even when the
// job is a write, because a failed lookup means the command was never sent. The
// call itself follows the method: a GET is retried, a POST is attempted ONCE.
func (r *Runner) executeJoe(ctx context.Context, cfg config.Config, job platform.Job) (outcome collect.Outcome, err error) {
	// EVERY failure leaving here is marked as a Joe call's, so classifyFailure can
	// answer in the caller's own terms rather than a collection's -- and so the
	// oversize ceiling, which is one shared sentinel across both channels, is
	// still reported as Joe's. Marked at the one exit and with two %w, so
	// errors.Is/As through the chain all still match.
	defer func() {
		if err != nil {
			err = fmt.Errorf("%w: %w", errJoeCall, err)
		}
	}()

	req, err := joe.Parse(job.Args)
	if err != nil {
		return collect.Outcome{}, err
	}

	client := r.joeClient(cfg)

	var channelID string
	if req.ResolveChannel {
		channelID, err = r.resolveJoeChannel(ctx, client)
		// A Joe serving no channel is a SKIP, not a failure: nothing was
		// delivered and nothing at Joe failed, so 'error' would report a fault
		// that did not happen. Every OTHER lookup failure -- unreachable, a 5xx, a
		// refused signature -- stays an error, because those are faults.
		if errors.Is(err, joe.ErrNoChannels) {
			return collect.Outcome{Status: collect.OutcomeSkipped, SkipReason: joeSkipNoChannels}, nil
		}
		if err != nil {
			return collect.Outcome{}, err
		}
	}

	attempts := joeWriteAttempts
	if req.Method == http.MethodGet {
		attempts = joeReadAttempts
	}

	var lastErr error
	for attempt := 1; attempt <= attempts; attempt++ {
		callCtx := ctx
		var cancel context.CancelFunc
		if attempts > 1 {
			// Only a retried call is bounded tighter than the job budget: giving a
			// single-attempt write the same short deadline would fail a command that
			// was about to be accepted, with no second attempt to save it.
			callCtx, cancel = context.WithTimeout(ctx, joeGetTimeout)
		}
		payload, err := client.Do(callCtx, req, channelID)
		if cancel != nil {
			cancel()
		}
		if err == nil {
			return collect.Outcome{Status: collect.OutcomeOK, Payload: joeResult(req, channelID, payload)}, nil
		}
		lastErr = err
		if !retryableJoeError(err) || ctx.Err() != nil {
			return collect.Outcome{}, err
		}
		if attempt < attempts && !r.sleep(ctx, storeBackoff*time.Duration(attempt)) {
			break
		}
	}
	return collect.Outcome{}, lastErr
}

// joeResult is what the platform stores for one Joe call.
//
// Joe's reply, verbatim, for every action -- EXCEPT that a resolve_channel job
// whose reply is empty records the channel this box chose instead. It displaces
// nothing: Joe's command handler passes the message to a goroutine and writes no
// body, so there is otherwise nothing at all to store, and this is then the only
// record anywhere of which channel the command went to -- which is exactly what
// someone debugging "the command went nowhere" has to have. The platform stores
// it and reads nothing from it (v1.joe_job_submit has no apply).
//
// A nil payload marshals to a JSON null, which PostgREST binds to a SQL NULL, so
// on every other action a successful call still stores NULL and nothing consuming
// joe_call may gate on `result is not null`.
func joeResult(req joe.Request, channelID string, payload json.RawMessage) any {
	if !req.ResolveChannel || len(payload) > 0 {
		return payload
	}
	return map[string]string{"channel_id": channelID}
}

// resolveJoeChannel asks the local Joe which channel to address.
//
// THE FIRST advertised channel, which is exactly what v1.joe_command_run takes
// today, so the inverted route addresses the same channel as the pull path does
// for the same box.
func (r *Runner) resolveJoeChannel(ctx context.Context, client *joe.Client) (string, error) {
	var lastErr error
	for attempt := 1; attempt <= joeChannelAttempts; attempt++ {
		lookupCtx, cancel := context.WithTimeout(ctx, joeGetTimeout)
		ids, err := client.Channels(lookupCtx)
		cancel()
		if err == nil {
			return ids[0], nil
		}
		lastErr = err
		if !retryableJoeError(err) || ctx.Err() != nil {
			return "", err
		}
		if attempt < joeChannelAttempts && !r.sleep(ctx, storeBackoff*time.Duration(attempt)) {
			break
		}
	}
	return "", lastErr
}

// retryableJoeError reports whether re-running the same Joe call could clear the
// error.
func retryableJoeError(err error) bool {
	if errors.Is(err, joe.ErrInvalidArgs) || errors.Is(err, joe.ErrOversizeReply) ||
		errors.Is(err, joe.ErrNoChannels) {
		// Decided by what came back rather than by the transport, so the same call
		// produces the same unusable answer. ErrNoChannels is Joe's OWN config:
		// asking three times says the same thing.
		return false
	}
	var joeErr *joe.JoeError
	if errors.As(err, &joeErr) {
		return joeErr.Retryable()
	}
	// Network or timeout against a service on this very box.
	return true
}

// retryableEngineError reports whether re-running the same engine call could
// clear the error.
func retryableEngineError(err error) bool {
	if errors.Is(err, dblab.ErrInvalidArgs) || errors.Is(err, dblab.ErrOversizeReply) {
		// Both are decided by what came back, not by the transport: the same
		// call would produce the same unusable answer.
		return false
	}
	var engineErr *dblab.EngineError
	if errors.As(err, &engineErr) {
		return engineErr.Retryable()
	}
	// Network or timeout against a service on this very box.
	return true
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

// errJoeCall marks a failure as a Joe call's rather than a collection's. It
// carries no classification of its own, and it is load-bearing beyond wording:
// the oversize ceiling is ONE shared sentinel across both channels
// (internal/reply), so this is what tells the two apart when reporting it.
var errJoeCall = errors.New("joe call")

// errDBLabCall marks a failure as an engine call's rather than a collection's.
// It carries no classification of its own -- every dblab arm of classifyFailure
// keys on its own sentinel -- and exists only so the two arms worded for the
// monitoring channel can be answered in this one's vocabulary.
var errDBLabCall = errors.New("dblab call")

// classWithStatus appends the upstream HTTP status to a failure class, so a
// client can tell a deleted clone's 404 from a broken engine's 500 without
// regexing the English text (#402: the Console spun forever on exactly that).
// The platform caps failure_class at 64 bytes and enumerates nothing. A status
// outside the HTTP range means there was none: keep the flat class.
func classWithStatus(class string, status int) string {
	if status < 100 || status > 599 {
		return class
	}
	return fmt.Sprintf("%s_%d", class, status)
}

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
		// Forward the STORE's words when there are any. For a collection the
		// args are machine-generated and "bad args" is the whole story; for a
		// promql job the caller typed the expression, and the store's message
		// naming the parse error is the entire diagnostic. Without this the
		// user is told the platform's args are bad, which points them at the
		// one place the fault is not (#378).
		//
		// stripControls FIRST: the store echoes the submitted expression back
		// inside the message and PromQL literals take Go-style escapes, so the
		// text is attacker-influenced and is rendered on a terminal by
		// `pgai promql`. truncate() then caps it and guarantees valid UTF-8.
		if msg := collect.StripControls(collect.StoreMessage(err)); msg != "" {
			return "job args could not be used: " + msg, "invalid_args"
		}
		return "job args could not be used", "invalid_args"
	case errors.Is(err, collect.ErrUnknownKind):
		return "this instance does not know this job kind", "unknown_kind"
	case errors.Is(err, collect.ErrWindowTooLong):
		return "collection window is too long", "window_too_long"
	// Before the two arms below, which say "collection": a dblab_call is not
	// one, and the text is what its caller is shown. The CLASS is unchanged --
	// the platform stores and aggregates on that, and a timeout is a timeout
	// whichever channel produced it.
	case errors.Is(err, errDBLabCall) && errors.Is(err, context.DeadlineExceeded):
		return "the engine call exceeded the local time budget", "timeout"
	case errors.Is(err, errDBLabCall) && errors.Is(err, context.Canceled):
		return "the engine call was cancelled", "cancelled"
	case errors.Is(err, errJoeCall) && errors.Is(err, context.DeadlineExceeded):
		return "the joe call exceeded the local time budget", "timeout"
	case errors.Is(err, errJoeCall) && errors.Is(err, context.Canceled):
		return "the joe call was cancelled", "cancelled"
	case errors.Is(err, context.DeadlineExceeded):
		return "collection exceeded the local time budget", "timeout"
	case errors.Is(err, context.Canceled):
		return "collection was cancelled", "cancelled"
	case errors.Is(err, errUnencodableResult):
		return "the collected result could not be encoded", "unencodable_result"
	case errors.Is(err, errCollectionPanicked):
		return "collection failed unexpectedly", "panic"
	case errors.Is(err, errJobPanicked):
		return "the job failed unexpectedly", "panic"
	case errors.Is(err, errResultRefused):
		return "the platform refused the result", "result_rejected"
	// THE JOE ARMS COME FIRST, and the oversize one is why: the ceiling is one
	// shared sentinel across both channels (internal/reply), so the engine's arm
	// below would otherwise report a Joe reply as the engine's.
	case errors.Is(err, joe.ErrNoChannels):
		// A BACKSTOP: executeJoe turns this into a skip, so it does not reach here
		// today. Kept because the default arm's fallback is "metric store
		// unreachable", which on a Joe box names a component that is not there.
		return "joe advertises no channels", "no_channels"
	case errors.Is(err, joe.ErrBadChannelList):
		// NOT the arm above, and the distinction is the whole reason the sentinel
		// exists: a body that is not a channel list says nothing about how many
		// channels Joe serves, so reporting it as a skip would close a lost command
		// as done with the box still green.
		return "joe did not answer with a channel list", "bad_channel_list"
	case errors.Is(err, joe.ErrInvalidArgs):
		return "job args could not be used", "invalid_args"
	case errors.Is(err, errJoeCall) && errors.Is(err, joe.ErrOversizeReply):
		return "the joe reply is too large to submit", "oversize_reply"
	case errors.Is(err, joe.ErrJoeUnreachable):
		// Named before the default arm, whose fallback is "metric store
		// unreachable" -- a component a Joe box does not have.
		return "joe unreachable", "joe_unreachable"
	case errors.Is(err, dblab.ErrInvalidArgs):
		return "job args could not be used", "invalid_args"
	case errors.Is(err, dblab.ErrOversizeReply):
		return "the engine reply is too large to submit", "oversize_reply"
	case errors.Is(err, dblab.ErrEngineUnreachable):
		// Named before the default arm, whose fallback is "metric store
		// unreachable" -- a component a DBLab box does not have.
		return "dblab engine unreachable", "engine_unreachable"
	default:
		var upstream *collect.UpstreamError
		if errors.As(err, &upstream) {
			return fmt.Sprintf("metric store returned %d", upstream.StatusCode), "store_error"
		}
		var joeErr *joe.JoeError
		if errors.As(err, &joeErr) {
			// Joe usually says nothing at all -- its verifier's 403 and its command
			// handler's 400 write a status and no body -- so the status IS the
			// diagnostic, and whatever text there is joins it. Controls are stripped
			// and the length capped by truncate(): the text is rendered on a terminal
			// and reaches it from outside this process.
			class := classWithStatus("joe_error", joeErr.StatusCode)
			if msg := collect.StripControls(joeErr.Message); msg != "" {
				return fmt.Sprintf("joe returned %d: %s", joeErr.StatusCode, msg), class
			}
			return fmt.Sprintf("joe returned %d", joeErr.StatusCode), class
		}
		var engineErr *dblab.EngineError
		if errors.As(err, &engineErr) {
			// The engine is ours and its message names what was wrong with the
			// call, which is the whole diagnostic to whoever made it. Controls are
			// stripped and the length capped by truncate(): the text is rendered on
			// a terminal and reaches it from outside this process.
			class := classWithStatus("engine_error", engineErr.StatusCode)
			if msg := collect.StripControls(engineErr.Message); msg != "" {
				return fmt.Sprintf("dblab engine returned %d: %s", engineErr.StatusCode, msg), class
			}
			return fmt.Sprintf("dblab engine returned %d", engineErr.StatusCode), class
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
	// This WAS a guard "for the day someone widens classifyFailure". That day
	// arrived: the invalid_args arm now forwards the metric store's own message,
	// which is attacker-influenced, multibyte, and bounded only by
	// maxErrorBodyBytes (64 KiB). So the cap here is live rather than
	// theoretical, and so is the UTF-8 cleaning -- Postgres rejects ANY invalid
	// UTF-8 with 22021, and a naive slice at 512 lands mid-rune.
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
// the TWO submits that can follow it with every retry, and slack. Budgeting only for
// the job would call a working loop wedged whenever a submit had to retry. It
// reads r.budget rather than the constant so the deadline and the job ceiling
// cannot drift apart.
//
// The slack also covers the terminal close after a refused answer. That is a
// SECOND submit on the same retry ladder, not a bounded one: the refused ANSWER
// returns on its first attempt because a refusal is not transient, but the close
// is a different submission and a 55P03 on it burns the whole budget. So the
// true worst case is r.budget + 2*submitBudget = 32m20s, still inside the
// platform's 1h stuck_after (#378).
func (r *Runner) healthDeadline(next time.Duration) time.Duration {
	return next + r.budget + 2*submitBudget + 2*time.Minute
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
