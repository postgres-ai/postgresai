package runner

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

// One poll of v1.dblab_job_poll can hand back more than one job
// (platform-all#816), because an engine shared by many people has many calls
// outstanding at once and the old cap refused all but the first.
//
// DRAINING THE ARRAY DID NOT HAVE TO CHANGE FOR THAT. `jobs` has been a json
// array since platform-all#805, PollResponse.Jobs has always been a []Job, and
// tick() has always looped over it -- so that contract is one a pre-#816 box
// already satisfies. What IS new in #391 is the concurrency underneath it: the
// DBLab arm now runs a claimed batch on a pool (dblabConcurrency).
//
// THESE TESTS ARE NOT THE COMPATIBILITY PROOF, and an earlier revision of this
// comment claimed they were. They exercise the CURRENT source; a claim about a
// PRE-CHANGE binary cannot be made by a test that compiles against the new one.
// The proof is a binary built at feature/dblab-call-channel (33e4c36, before
// every #816 commit) pointed at a database carrying the #816 migration: it was
// handed five jobs in one poll -- one claimed_at, five ids -- ran them and
// answered all five, 238ms end to end. These tests are the regression cover for
// that behaviour, not the evidence for it.
//
// The fixtures are the rpc's real reply, captured from a database with the
// #816 migration deployed:
//
//	{"jobs": [{"id": "01a0ddb0-...", "args": {"action": "/status", "method":
//	 "GET", "purpose": "api_call"}, "kind": "dblab_call"}, ... ],
//	 "server_time": "2026-09-26T12:29:14.703881+00:00", "next_poll_ms": 5000}
//
// HOW MANY the platform hands out is a SETTING since #816
// (app.settings.dblab_job_claim_limit, seeded to 5). The fixtures below use
// three, five, six and eight: this side must handle whatever arrives, and only
// TestADBLabBatchRunsAtMostThePoolAtOnce is about the number itself.
//
// The bound on that setting is the platform's -- claim_limit x per_job_ceiling
// < the sweep's stuck_after, where the ceiling is jobBudget + 2*submitBudget.
// It binds a SEQUENTIAL agent, which is why monitoring's limit is 1; running
// the batch on a pool is what collapses the product to one ceiling and lets
// DBLab's be 5. See tick() and dblabConcurrency.

// dblabBatch builds a poll reply carrying one job per (action, method) pair, in
// the reply shape captured above. ids are j1..jN so a failure names the job.
func dblabBatch(calls ...[2]string) string {
	jobs := make([]string, 0, len(calls))
	for i, c := range calls {
		jobs = append(jobs, fmt.Sprintf(
			`{"id":"j%d","kind":"dblab_call","args":{"method":%q,"action":%q,"purpose":"api_call"}}`,
			i+1, c[1], c[0]))
	}
	return "[" + strings.Join(jobs, ",") + "]"
}

// submittedIDs is the job_id of every answer the box sent, SORTED.
//
// Sorted rather than in arrival order, because the DBLab arm runs a claimed
// batch concurrently and the order answers come back in is therefore not a
// property of this code. What every test here actually needs is the SET: each
// job answered exactly once, for its own id. An assertion on arrival order
// would now be asserting the scheduler.
func submittedIDs(h *dblabHarness) []string {
	subs := h.snapshotSubmits()
	ids := make([]string, 0, len(subs))
	for _, s := range subs {
		id, _ := s["job_id"].(string)
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// sortedCopy is for the same reason: what the engine SAW is a set, not a
// sequence, once the batch runs concurrently.
func sortedCopy(in []string) []string {
	out := append([]string(nil), in...)
	sort.Strings(out)
	return out
}

// The batch itself: three jobs from ONE poll, each run against the engine and
// each answered.
//
// THE ORDER IS NO LONGER ASSERTED, and that is a deliberate consequence rather
// than a weakened test. An earlier revision required the engine to see the jobs
// oldest-first, on the reasoning that the caller waiting on the first call has
// been waiting longest. That reasoning belonged to a SEQUENTIAL loop, where the
// only way to serve someone sooner was to serve someone else later. The DBLab
// arm now runs the batch concurrently (dblabConcurrency), so nobody is behind
// anybody: all three start at about claim time and the longest-waiting caller
// is served first in the only sense that matters, which is immediately.
//
// The platform still claims oldest-first and the `jobs` array is still ordered.
// What stopped being true is that claim order implies COMPLETION order -- and
// nothing depends on that, because every answer is bound to its own job id
// (TestEachAnswerInABatchNamesItsOwnJobAndResult is the guard).
//
// So the assertion is the SET: every job reached the engine, exactly once.
func TestABatchFromOnePollRunsEveryJobAndSubmitsEachAnswer(t *testing.T) {
	var paths []string
	h := newDBLabHarness(t, dblabBatch(
		[2]string{"/status", "GET"},
		[2]string{"/branch", "GET"},
		[2]string{"/snapshot", "GET"},
	))
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		// Under the harness lock: the DBLab arm runs a batch on a worker pool,
		// so several workers enter this closure at the same time.
		h.mu.Lock()
		paths = append(paths, r.URL.Path)
		h.mu.Unlock()
		w.Write([]byte(`{"pools":[]}`))
	}

	h.runner.tick(context.Background())

	if want := []string{"/branch", "/snapshot", "/status"}; !equalStrings(sortedCopy(paths), want) {
		t.Fatalf("the engine saw %v, want exactly one call to each of %v", paths, want)
	}
	if want := []string{"j1", "j2", "j3"}; !equalStrings(submittedIDs(h), want) {
		t.Fatalf("submitted %v, want an answer for each of %v", submittedIDs(h), want)
	}
	for i, s := range h.snapshotSubmits() {
		if s["outcome"] != "ok" {
			t.Fatalf("answer %d is %v, want ok: %v", i, s["outcome"], s)
		}
	}
	// ONE poll, not one per job. The claim is what this change moved; a loop
	// that re-polled between jobs would multiply the rpc's load by the batch
	// size and claim a second batch before the first was answered.
	if got := h.pollCount(); got != 1 {
		t.Fatalf("the tick polled %d times for one batch, want 1", got)
	}
	// Concurrency must not have turned the batch into extra engine traffic: one
	// call per job and no more, which is also the write rule holding at the
	// transport level.
	if h.engineCallCount() != 3 {
		t.Fatalf("the engine was called %d times for a batch of 3", h.engineCallCount())
	}
}

// A BATCH IS NOT A TRANSACTION. One job failing must not roll back, skip or
// strand the others: the platform stamped started_at on all three at pickup and
// sweeps anything still 'running' after an hour, so a batch that stopped at its
// first failure would leave real work to be failed by the sweep -- and, on the
// interactive channel, leave a person watching a spinner until it was.
func TestAFailedJobDoesNotStopTheRestOfTheBatch(t *testing.T) {
	h := newDBLabHarness(t, dblabBatch(
		[2]string{"/status", "GET"},
		[2]string{"/branch", "GET"},
		[2]string{"/snapshot", "GET"},
	))
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		// The MIDDLE one of the claimed array. On the sequential arm that mattered
		// directly -- a loop that gave up on a failure would still have submitted
		// the first, so failing the LAST would have passed either way. On the
		// concurrent arm it still matters, because a batch that aborted the
		// remaining goroutines on the first failure would drop j3.
		if r.URL.Path == "/branch" {
			w.WriteHeader(http.StatusNotFound)
			w.Write([]byte(`{"message":"branch not found"}`))
			return
		}
		w.Write([]byte(`{"pools":[]}`))
	}

	h.runner.tick(context.Background())

	if want := []string{"j1", "j2", "j3"}; !equalStrings(submittedIDs(h), want) {
		t.Fatalf("submitted %v after a failure mid-batch, want an answer for each of %v",
			submittedIDs(h), want)
	}
	// KEYED BY JOB ID, not by position. The claim is still oldest-first and the
	// `jobs` array is still ordered, but the batch runs concurrently, so the
	// order answers ARRIVE in is the scheduler's business. An earlier revision
	// compared the outcome slice positionally and passed on this branch purely
	// by scheduling luck -- it failed the moment it was merged with an unrelated
	// change that shifted the timing. Positional assertions over a concurrent
	// batch are flakes waiting for a slow CI runner.
	//
	// What the test actually means is per-job: j2 failed, j1 and j3 did not.
	outcomes := map[string]string{}
	for _, s := range h.snapshotSubmits() {
		id, _ := s["job_id"].(string)
		o, _ := s["outcome"].(string)
		outcomes[id] = o
	}
	want := map[string]string{"j1": "ok", "j2": "error", "j3": "ok"}
	for id, w := range want {
		if outcomes[id] != w {
			t.Fatalf("job %s was answered %q, want %q (all outcomes: %v) -- the failure must be "+
				"reported as its own job's, and only its own", id, outcomes[id], w, outcomes)
		}
	}
	// The job that is NOT the failure really ran, rather than being answered ok
	// without touching the engine.
	if h.engineCallCount() != 3 {
		t.Fatalf("the engine was called %d times, want 3 -- one per job", h.engineCallCount())
	}
}

// THE WRITE RULE IS UNCHANGED BY BATCHING, and this is the test that says so.
//
// A POST or DELETE is answered as failed after ONE attempt, because a
// `POST /clone` whose reply was lost may already have created the clone.
// Claiming three jobs at once must not become a second delivery of any of them:
// the write in the middle of this batch gets exactly one engine call while the
// reads on either side of it get their retries, and all three are answered.
func TestTheWriteRuleStillHoldsInsideABatch(t *testing.T) {
	perPath := map[string]int{}
	h := newDBLabHarness(t, dblabBatch(
		[2]string{"/status", "GET"},
		[2]string{"/clone", "POST"},
		[2]string{"/snapshot", "GET"},
	))
	// Every call fails with a 500, which IS retryable -- so a read is retried
	// and a write must not be. A failure the transport calls permanent would
	// make this pass without the rule.
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		h.mu.Lock()
		perPath[r.URL.Path]++
		h.mu.Unlock()
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte(`{"message":"engine is busy"}`))
	}

	h.runner.tick(context.Background())

	if perPath["/clone"] != 1 {
		t.Fatalf("the POST reached the engine %d times inside a batch, want exactly 1: %v",
			perPath["/clone"], perPath)
	}
	// The control: without it, a build that stopped retrying everything would
	// pass the assertion above for the wrong reason.
	//
	// 3 is dblabReadAttempts, spelled out by hand. Written as the constant this
	// control cannot notice the constant changing -- and it was: with
	// dblabReadAttempts set to 1 this whole test stayed green, at which point
	// "/clone was called once" means "nothing is retried at all" rather than
	// "the write rule holds", which is the exact vacuity the control exists to
	// prevent.
	if perPath["/status"] != 3 || perPath["/snapshot"] != 3 {
		t.Fatalf("the reads were attempted %v, want 3 each -- if reads are not being retried, "+
			"the write assertion above proves nothing", perPath)
	}
	if want := []string{"j1", "j2", "j3"}; !equalStrings(submittedIDs(h), want) {
		t.Fatalf("submitted %v, want every job answered even though every job failed",
			submittedIDs(h))
	}
	for i, s := range h.snapshotSubmits() {
		if s["outcome"] != "error" {
			t.Fatalf("answer %d is %v, want error", i, s["outcome"])
		}
	}
}

// Each answer goes out for its OWN job, CARRYING ITS OWN RESULT.
//
// The platform binds a result to the job row it names, so a loop that reused
// one id would report one job three times and leave the other two running until
// the sweep failed them. That much was true of the sequential arm too.
//
// What is new once the batch runs on a pool is that a result can cross between
// workers -- a closure over the loop variable, a shared buffer -- and every
// other test here would still pass, because each of them answers the whole
// batch with one reply. So the engine answers each path differently and every
// answer is traced back to the call that produced it.
func TestEachAnswerInABatchNamesItsOwnJobAndResult(t *testing.T) {
	h := newDBLabHarness(t, dblabBatch(
		[2]string{"/status", "GET"},
		[2]string{"/branch", "GET"},
		[2]string{"/snapshot", "GET"},
	))
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `{"seen":%q}`, r.URL.Path)
	}

	h.runner.tick(context.Background())

	// dblabBatch numbers the jobs in the order it is given them.
	wantFor := map[string]string{"j1": "/status", "j2": "/branch", "j3": "/snapshot"}
	seen := map[string]bool{}
	for _, s := range h.snapshotSubmits() {
		id, _ := s["job_id"].(string)
		if id == "" {
			t.Fatalf("an answer named no job: %v", h.snapshotSubmits())
		}
		if seen[id] {
			t.Fatalf("job %s was answered twice: %v", id, submittedIDs(h))
		}
		seen[id] = true
		result, _ := s["result"].(map[string]any)
		got, _ := result["seen"].(string)
		if got != wantFor[id] {
			t.Fatalf("job %s asked the engine for %q and was answered with the reply to "+
				"%q -- a result crossed between workers: %v", id, wantFor[id], got, s)
		}
	}
	if len(seen) != 3 {
		t.Fatalf("answered %d distinct jobs, want 3", len(seen))
	}
	// Nothing in a batch may name an instance: the platform derives the engine
	// from the credential, and PostgREST resolves an rpc by its body keys, so an
	// extra key would 404 rather than being ignored.
	for _, s := range h.snapshotSubmits() {
		for _, k := range []string{"dblab_instance_id", "instance_id"} {
			if _, ok := s[k]; ok {
				t.Fatalf("an answer carried %q: %v", k, s)
			}
		}
	}
}

// A batch bigger than the platform's own cap is still run to the end rather
// than truncated here.
//
// This side deliberately enforces NO limit of its own. A job this box claimed
// and then declined to run would sit 'running' until the platform's hourly
// sweep failed it -- strictly worse than running it, and on the interactive
// channel it would hold a slot against the in-flight cap for the whole hour.
// The cap belongs to the claim, which is the only place that can refuse work
// without having already taken it.
func TestABatchIsNotTruncatedByTheBox(t *testing.T) {
	calls := make([][2]string, 0, 6)
	for i := 0; i < 6; i++ {
		calls = append(calls, [2]string{fmt.Sprintf("/clone/c%d", i), "GET"})
	}
	h := newDBLabHarness(t, dblabBatch(calls...))

	h.runner.tick(context.Background())

	want := []string{"j1", "j2", "j3", "j4", "j5", "j6"}
	if !equalStrings(submittedIDs(h), want) {
		t.Fatalf("submitted %v for a 6-job reply, want %v -- a job claimed and "+
			"not answered sits running until the platform's sweep", submittedIDs(h), want)
	}
	// ...but it does SAY SO, once. That log line is the only warning anyone gets
	// that the pool no longer equals the platform's claim limit, and `!= 1`
	// rather than `< 1` is what also pins "per batch, not per job".
	// Snapshotted, not read live: log output is process-global, so a tick that
	// outlives an already-red test keeps writing into whichever builder `log`
	// points at by then.
	logs := h.logs.String()
	if got := strings.Count(logs, "runs 5 at a time"); got != 1 {
		t.Fatalf("the wide-batch warning was logged %d times for one 6-job batch, want 1 -- "+
			"nothing else reports that the claim limit and the pool have diverged", got)
	}
}

func equalStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// FOUR READS AND A CLONE CREATE, AND THE READS DO NOT WAIT FOR IT.
//
// This is the acceptance case for platform-all#816, and it is the one that
// replaced an earlier bar of "five concurrent callers, none refused". Five reads
// would pass while the bug survived: removing the platform's in-flight cap stops
// anyone being REFUSED, but with a sequential loop the four reads still queue
// behind the write and are answered when it finishes. Measured on a rig before
// this change: a 6s write, and the reads answered at 6.04s, 6.05s, 6.10s and
// 6.16s. Nobody refused; everybody waited.
//
// So the assertion is not "all five are answered" -- the sequential arm does
// that too. It is that THE READS ARE ANSWERED WHILE THE WRITE IS STILL RUNNING.
// The write is held open on a channel rather than a sleep, so the test proves an
// ordering rather than racing a clock: if the reads were queued behind it this
// test would deadlock on its own timeout rather than pass slowly.
//
// It fails on the sequential arm by construction, which is what makes it the
// test for this change rather than a test of the platform's.
func TestReadsDoNotWaitForAWriteInTheSameBatch(t *testing.T) {
	const reads = 4

	h := newDBLabHarness(t, dblabBatch(
		// The write FIRST, which is the hostile order: claiming is oldest-first,
		// so a sequential loop would run it before any read. With the write last
		// the reads would come back promptly even sequentially and the test
		// would pass for the wrong reason.
		[2]string{"/clone", "POST"},
		[2]string{"/status", "GET"},
		[2]string{"/snapshot", "GET"},
		[2]string{"/branch", "GET"},
		[2]string{"/instance/retrieval", "GET"},
	))

	// releaseOnce, and registered as CLEANUP rather than only called at the end.
	// If an assertion below fails, t.Fatalf unwinds this goroutine while the
	// engine handler is still blocked inside the write -- and httptest's Close,
	// which runs from t.Cleanup, WAITS for outstanding requests. The test would
	// then hang instead of failing, and a wedged CI job is far worse than a red
	// one. Reproduced by forcing the sequential arm: without this the run had to
	// be killed.
	release := make(chan struct{})
	var releaseOnce sync.Once
	releaseWrite := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(releaseWrite)

	readsDone := make(chan struct{}, reads)
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/clone" {
			<-release // held open: stands in for a clone create taking minutes
			w.Write([]byte(`{"id":"c1"}`))
			return
		}
		w.Write([]byte(`{"pools":[]}`))
		// Non-blocking: the test drains exactly `reads`, so an unconditional
		// send wedges a handler goroutine the moment a read is attempted more
		// often than that -- and engineSrv.Close then waits for it forever,
		// killing the whole package on its timeout instead of failing one test.
		// Same reasoning as the releaseWrite cleanup below.
		select {
		case readsDone <- struct{}{}:
		default:
		}
	}

	// Drained on EVERY path, not just the happy one: a t.Fatalf below unwinds
	// this goroutine's test while the batch is still running against servers
	// cleanup is about to close, which shows up as connection-refused noise in
	// whichever test runs next. Cancellable and bounded so a parked loop fails
	// this test rather than hanging the package.
	tickCtx, cancelTick := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		h.runner.tick(tickCtx)
		close(done)
	}()
	t.Cleanup(func() {
		releaseWrite()
		cancelTick()
		select {
		case <-done:
		case <-time.After(20 * time.Second):
			t.Error("the tick never returned; the batch is parked")
		}
	})

	// Every read must reach the engine while the write is still held. This is
	// the whole assertion: on a sequential arm nothing arrives here at all,
	// because the loop is blocked inside the write, and the test times out.
	for i := 0; i < reads; i++ {
		select {
		case <-readsDone:
		case <-time.After(10 * time.Second):
			t.Fatalf("only %d of %d reads reached the engine while a write was in flight -- "+
				"the reads are queued behind the write, which is the defect this change exists to fix", i, reads)
		}
	}

	// ...and they are ANSWERED while it is still held, not merely started. A
	// caller is not served until its result is submitted.
	deadline := time.Now().Add(10 * time.Second)
	for {
		if len(h.snapshotSubmits()) >= reads {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("%d of %d reads were answered while the write was in flight",
				len(h.snapshotSubmits()), reads)
		}
		time.Sleep(5 * time.Millisecond)
	}

	// The write is still outstanding at this point -- it has not been answered,
	// and nothing about the reads finishing early has abandoned it.
	for _, id := range submittedIDs(h) {
		if id == "j1" {
			t.Fatal("the write was answered before it was released, so this test proved nothing about ordering")
		}
	}

	releaseWrite()
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatal("the tick did not finish after the write was released")
	}

	// And nothing was lost: every job in the batch answered exactly once, the
	// write included.
	if want := []string{"j1", "j2", "j3", "j4", "j5"}; !equalStrings(submittedIDs(h), want) {
		t.Fatalf("submitted %v, want every job in the batch answered once", submittedIDs(h))
	}
	// The write reached the engine ONCE, concurrency notwithstanding.
	if h.engineCallCount() != reads+1 {
		t.Fatalf("the engine was called %d times for %d jobs", h.engineCallCount(), reads+1)
	}
}

// gatedBatch is the fixture the concurrency tests share: a batch of n GET jobs
// whose engine handler ANNOUNCES each arrival and then blocks until that job's
// own gate is released. Holding jobs open is the only way to observe how many
// run at once; every other test here answers instantly, so slots recycle faster
// than an assertion can see.
type gatedBatch struct {
	h          *dblabHarness
	arrivals   chan string
	releaseOne func(path string)
	releaseAll func()
	done       chan struct{}
}

// path is the action job i asks the engine for.
func gatedPath(i int) string { return fmt.Sprintf("/clone/c%d", i) }

func newGatedBatch(t *testing.T, n int) *gatedBatch {
	t.Helper()

	calls := make([][2]string, 0, n)
	gates := make(map[string]chan struct{}, n)
	for i := 0; i < n; i++ {
		calls = append(calls, [2]string{gatedPath(i), "GET"})
		gates[gatedPath(i)] = make(chan struct{})
	}
	g := &gatedBatch{
		h: newDBLabHarness(t, dblabBatch(calls...)),
		// Buffered to the whole batch, so a handler only ever blocks on its own
		// gate; the default arm below turns an unexpected extra call into a red
		// test rather than a wedged one.
		arrivals: make(chan string, n),
		done:     make(chan struct{}),
	}

	// gates is written before the first request and only read afterwards. The
	// closing is serialised here, which is also what makes a second release of
	// the same path a no-op rather than a panic.
	var mu sync.Mutex
	released := map[string]bool{}
	g.releaseOne = func(path string) {
		mu.Lock()
		defer mu.Unlock()
		gate, ok := gates[path]
		if !ok {
			t.Errorf("asked to release an unknown path %q", path)
			return
		}
		if !released[path] {
			released[path] = true
			close(gate)
		}
	}
	g.releaseAll = func() {
		for path := range gates {
			g.releaseOne(path)
		}
	}
	t.Cleanup(g.releaseAll)

	g.h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		// Both lookups guarded: an unexpected path would block this handler on a
		// NIL channel forever and take engineSrv.Close with it, and an
		// unconditional send would block once a retry pushed past the buffer.
		// Either wedges the whole package on its timeout instead of failing one
		// test, which is strictly worse than a red run.
		gate, ok := gates[r.URL.Path]
		if !ok {
			t.Errorf("the engine was called at an unexpected path %q", r.URL.Path)
			return
		}
		select {
		case g.arrivals <- r.URL.Path:
		default:
			t.Errorf("the engine was called more than %d times; %q did not fit", n, r.URL.Path)
		}
		<-gate
		w.Write([]byte(`{"pools":[]}`))
	}
	return g
}

// start runs the tick in the background. The context is cancellable and the
// drain is BOUNDED: a production defect that parks the dispatch loop would
// otherwise hang the binary on its timeout, taking every other result in the
// package with it and losing this test's own message.
func (g *gatedBatch) start(t *testing.T) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		g.h.runner.tick(ctx)
		close(g.done)
	}()
	t.Cleanup(func() {
		g.releaseAll()
		cancel()
		select {
		case <-g.done:
		case <-time.After(20 * time.Second):
			t.Error("the tick never returned; the dispatch loop is parked")
		}
	})
}

// awaitInFlight waits for n jobs to be in the engine at once and returns the
// paths, oldest first.
func (g *gatedBatch) awaitInFlight(t *testing.T, n int) []string {
	t.Helper()
	paths := make([]string, 0, n)
	for i := 0; i < n; i++ {
		select {
		case path := <-g.arrivals:
			paths = append(paths, path)
		case <-time.After(10 * time.Second):
			t.Fatalf("only %d jobs were in flight at once, want %d -- a pool smaller than "+
				"the platform's claim limit leaves the tail unstarted, and the sweep fails "+
				"it to its caller while this process is still going to run it", i, n)
		}
	}
	return paths
}

// THE POOL IS FIVE, AND A BATCH WIDER THAN IT WAITS.
//
// The equality with the platform's claim limit is what this change rests on,
// and nothing pinned it: a pool of 2, 3, 4 or 100 -- or no bound at all --
// passed this whole package, because every other test's reads return at once,
// so slots recycle faster than any assertion can notice. The test above that
// looks like the acceptance case for "five in flight" is really satisfied by
// two.
//
// So the engine holds EVERY job open and the test watches how many are in
// flight at the same moment. 5 is written as a literal on purpose: a test
// phrased in terms of dblabConcurrency cannot notice dblabConcurrency changing,
// and changing it is a change in the OTHER repository too
// (app.settings.dblab_job_claim_limit).
func TestADBLabBatchRunsAtMostThePoolAtOnce(t *testing.T) {
	const pool = 5 // dblabConcurrency, spelled out by hand.
	const batch = pool + 1

	g := newGatedBatch(t, batch)
	g.start(t)

	// FIVE reach the engine while none of them has answered. A smaller pool
	// never gets here: what is missing is the fifth, not the sixth.
	inFlight := g.awaitInFlight(t, pool)

	// ...and the SIXTH does not join them. Unbounded, or any pool at or above
	// the batch size, arrives here immediately.
	select {
	case path := <-g.arrivals:
		t.Fatalf("%d jobs were in flight at once (%s joined the %d already running), want at "+
			"most %d -- the pool is not bounding the batch, and it is the customer's engine "+
			"on the other end", pool+1, path, pool, pool)
	case <-time.After(500 * time.Millisecond):
	}

	// Freeing ONE slot is what starts it, which is the other half of "bounded":
	// the surplus waits rather than being dropped.
	g.releaseOne(inFlight[0])
	select {
	case <-g.arrivals:
	case <-time.After(10 * time.Second):
		t.Fatal("the queued job never started, though a slot had been freed")
	}

	g.releaseAll()
	select {
	case <-g.done:
	case <-time.After(20 * time.Second):
		t.Fatal("the tick did not finish after the batch was released")
	}

	want := []string{"j1", "j2", "j3", "j4", "j5", "j6"}
	if !equalStrings(submittedIDs(g.h), want) {
		t.Fatalf("submitted %v, want %v -- the queued job must still be answered",
			submittedIDs(g.h), want)
	}
}

// THE HEALTH DEADLINE SLIDES WHILE A BATCH RUNS.
//
// Each worker stamps as it starts, and that has to be true of a QUEUED one too:
// the deadline in the health file allows one job budget, so a batch that
// outlives one would read as wedged, and a working box would report dead
// through the container HEALTHCHECK -- `mon health` shows a dead channel and
// someone goes chasing it, mid-clone-create. The sequential arm's
// equivalent is covered by TestHealthIsRefreshedBetweenJobs; deleting the
// concurrent one left this package green.
func TestAQueuedJobStampsHealthWhenItStarts(t *testing.T) {
	const pool = 5 // dblabConcurrency, spelled out by hand.

	g := newGatedBatch(t, pool+1)

	// The clock only moves when this test moves it, so "was stamped again" is an
	// ordering rather than a race with the wall clock.
	var clockMu sync.Mutex
	now := time.Date(2026, 9, 26, 12, 0, 0, 0, time.UTC)
	g.h.runner.now = func() time.Time {
		clockMu.Lock()
		defer clockMu.Unlock()
		return now
	}
	advance := func(d time.Duration) {
		clockMu.Lock()
		defer clockMu.Unlock()
		now = now.Add(d)
	}

	g.start(t)
	inFlight := g.awaitInFlight(t, pool)
	first := g.h.healthStamp(t)

	// A whole job budget later, the queued job starts -- and must stamp as it
	// starts. The end-of-tick stamp cannot be what satisfies this: it is behind
	// wg.Wait() and four gates are still held when the assertion runs.
	advance(20 * time.Minute)
	g.releaseOne(inFlight[0])
	select {
	case <-g.arrivals:
	case <-time.After(10 * time.Second):
		t.Fatal("the queued job never started, though a slot had been freed")
	}

	deadline := time.Now().Add(10 * time.Second)
	for {
		got := g.h.healthStamp(t)
		if got.UpdatedAt.After(first.UpdatedAt) {
			// The deadline has to carry a whole job budget FORWARD from the new
			// stamp. Asserting it moved would be tautological -- next_check_by is
			// updated_at plus a constant within one tick -- so assert the gap.
			if gap := got.NextCheckBy.Sub(got.UpdatedAt); gap < jobBudget {
				t.Fatalf("the re-stamped deadline allows only %v, want at least one job "+
					"budget (%v)", gap, jobBudget)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the health file was last written at %v, %v after the batch started -- "+
				"a job that starts late must stamp as it starts, or a long batch reads as "+
				"wedged and `mon health` shows a dead channel on a working box",
				first.UpdatedAt, 20*time.Minute)
		}
		time.Sleep(5 * time.Millisecond)
	}

	g.releaseAll()
	select {
	case <-g.done:
	case <-time.After(20 * time.Second):
		t.Fatal("the tick did not finish after the batch was released")
	}
}

// A PANIC IN ONE WORKER MUST NOT TAKE THE BATCH WITH IT.
//
// Go kills the process for an unrecovered panic in a goroutine, and the DBLab
// arm's jobs now run on goroutines this package spawns. Without runJobSafely
// one poison job ends the run with up to dblabConcurrency-1 siblings already
// CLAIMED and never answered: they sit 'running' until the platform's hourly
// sweep, and one of them may be a POST /clone the engine carried out.
//
// Remove the guard and this test does not fail, it CRASHES THE TEST BINARY --
// which is exactly what it does on a customer's box, and why the assertion is
// worth having rather than trusting that nothing panics.
func TestAPanickingJobDoesNotTakeTheBatchWithIt(t *testing.T) {
	h := newDBLabHarness(t, dblabBatch(
		[2]string{"/status", "GET"},
		[2]string{"/branch", "GET"},
		[2]string{"/snapshot", "GET"},
	))
	// elapsed is reached once per job, after the engine call and before any
	// answer goes out -- so exactly one worker panics with nothing submitted for
	// it, and the other two run to completion beside it.
	var once sync.Once
	h.runner.elapsed = func(time.Time) time.Duration {
		boom := false
		once.Do(func() { boom = true })
		if boom {
			panic("a job panicked")
		}
		return time.Millisecond
	}

	h.runner.tick(context.Background())

	if want := []string{"j1", "j2", "j3"}; !equalStrings(submittedIDs(h), want) {
		t.Fatalf("submitted %v after a panic mid-batch, want an answer for each of %v -- "+
			"a claimed job nobody answers sits running until the sweep", submittedIDs(h), want)
	}
	panics := 0
	for _, s := range h.snapshotSubmits() {
		if s["failure_class"] == "panic" {
			panics++
		}
	}
	if panics != 1 {
		t.Fatalf("%d answers carry failure_class panic, want exactly 1: %v -- the panicking "+
			"job must be answered as failed, not merely survived", panics, h.snapshotSubmits())
	}
}

// SIGTERM MID-BATCH STARTS NOTHING MORE, AND STILL DELIVERS WHAT IT HAS.
//
// The dispatch loop has to stop handing out slots while it is PARKED waiting
// for one: a plain check before a blocking send leaves it wedged on the send
// and then starts a job into a dead context. Deleting the check entirely used
// to pass this whole package.
//
// The undispatched jobs were already claimed, so they are 'running'
// platform-side and wait out the sweep -- beginning a call this process is
// about to abandon is the worse of the two, not the better.
func TestACancelledDBLabBatchStartsNothingMore(t *testing.T) {
	const batch = 8 // wider than the pool, so the loop parks on a slot

	calls := make([][2]string, 0, batch)
	for i := 0; i < batch; i++ {
		calls = append(calls, [2]string{fmt.Sprintf("/clone/c%d", i), "GET"})
	}
	h := newDBLabHarness(t, dblabBatch(calls...))

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	var cancelOnce sync.Once
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		cancelOnce.Do(cancel)
		w.Write([]byte(`{"pools":[]}`))
	}

	h.runner.tick(ctx)

	// BOUNDED AT THE POOL, not at the batch: "fewer than 8" was satisfied by the
	// pool alone and passed for the very defect this test is named after.
	//
	// 5 is a CEILING here, not an expected value, so this cannot false-fail
	// under CI load -- a receive of slot k happens-before the send of slot k+5
	// completing, so any send past the fifth is ordered after a completed engine
	// call and therefore after cancel(). Measured: 500/500 exactly 5 at
	// GOMAXPROCS=1 under 64 spinners, and 1000 runs at GOMAXPROCS=32 never
	// exceeded 5 (they degrade downward). 5 is dblabConcurrency by hand.
	const pool = 5
	answered := submittedIDs(h)
	if len(answered) > pool {
		t.Fatalf("%d jobs were answered after the context ended, want at most the pool (%d) "+
			"-- the dispatch loop took a slot through a cancellation: %v",
			len(answered), pool, answered)
	}
	// The control. Without it this passes for a build that answers nothing at
	// all, which is the opposite failure: work already in flight when the
	// context ends is still delivered, on the detached submit path -- and that
	// is what puts several workers in submit's shutdown branch together, which
	// is the reason shutdownDeadline is under a lock.
	if len(answered) == 0 {
		t.Fatalf("no job was answered at all -- an answer already in hand must still go " +
			"out on the detached context rather than being dropped on SIGTERM")
	}
}

// A CANCELLATION LANDING WHILE THE LOOP IS PARKED ON A SLOT STARTS NOTHING.
//
// This is the one window no amount of scheduling pressure can produce from
// outside, and the test above cannot see it: there the context dies inside the
// first engine call, which is strictly before any worker can free a slot, so
// when the parked select wakes, ctx.Done() has already won. The "both cases
// ready, Go picks at random" state never arises -- instrumented, the re-check
// branch was taken 0 times in 1000 runs of the whole suite.
//
// So the loop has a seam. afterSlot fires between the slot being taken and the
// re-check; cancelling there is exactly the interleaving, and without the
// re-check the sixth job starts into a dead context.
func TestACancellationWhileParkedOnASlotStartsNothing(t *testing.T) {
	const pool = 5 // dblabConcurrency, spelled out by hand.

	g := newGatedBatch(t, pool+1)

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	// The sixth dispatch is the one that had to wait for a slot. Cancel as it
	// takes that slot, which is the moment both select cases would be ready.
	var dispatches int
	g.h.runner.afterSlot = func() {
		dispatches++
		if dispatches == pool+1 {
			cancel()
		}
	}

	done := make(chan struct{})
	go func() {
		g.h.runner.tick(ctx)
		close(done)
	}()
	t.Cleanup(func() {
		g.releaseAll()
		select {
		case <-done:
		case <-time.After(20 * time.Second):
			t.Error("the tick never returned; the dispatch loop is parked")
		}
	})

	inFlight := g.awaitInFlight(t, pool)
	// Free a slot so the loop wakes and takes it -- and cancels in afterSlot as
	// it does.
	g.releaseOne(inFlight[0])

	g.releaseAll()
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatal("the tick did not finish; the slot taken for the undispatched job was not handed back")
	}

	// THE ASSERTION IS THE ANSWER SET, not the engine. A job dispatched into a
	// dead context never reaches the engine either -- execute derives its
	// context from this one -- so "the engine was not called" is satisfied by
	// the cancellation itself and would pass without the re-check. What only
	// happens if the job was STARTED is that it gets answered, on the detached
	// submit path.
	for _, id := range submittedIDs(g.h) {
		if id == "j6" {
			t.Fatalf("j6 was answered, so it was started after the context ended: %v -- the "+
				"loop took a slot and dispatched into a dead context, which is the one "+
				"thing the re-check after the send exists to prevent", submittedIDs(g.h))
		}
	}
	// The control, and a lower bound rather than an exact count: how many of the
	// in-flight five are answered depends on where each worker was when the
	// cancellation landed -- one already past its submit check answers on the
	// detached path, one caught mid-submit does not. What must not happen is
	// nothing being answered at all, which is how "j6 is absent" would pass for
	// a build that dropped the whole batch.
	if len(submittedIDs(g.h)) == 0 {
		t.Fatal("no job was answered at all -- an answer already in hand must still go out " +
			"on the detached context rather than being dropped on SIGTERM")
	}
}

// THE HEALTH VERDICT FOR A BATCH IS DECIDED ONCE, FOR THE WHOLE BATCH.
//
// It used to be decided per job, by workers racing to bump and reset a shared
// counter, so an identical batch ended on any of four values depending on which
// goroutine finished last -- measured {0: 125, 1: 94, 2: 49, 3: 32} over 300
// ticks, with 11% crossing unhealthyAfter and taking a working box off the
// channel over three bad call arguments.
//
// The rule is the one unhealthyAfter was sized against: anything landed means
// the box works, so the run counts as good however many jobs in it failed.
func TestABatchWithOneSuccessIsNotAFailedRun(t *testing.T) {
	// ONE runner across three polls, which is what "in a row" means. A fresh
	// harness per tick makes this test VACUOUS: a run bumps the counter at most
	// once, so a per-tick runner never gets near unhealthyAfter and the health
	// assertion holds whatever the reset rule is. Measured: narrowing the reset
	// from "anything landed" to "everything landed" -- which is the rule this
	// test is named after -- left the per-tick version green.
	// unhealthyAfter, spelled out by hand, and it is the LOOP that matters:
	// the "everything landed" mutant crosses the threshold on tick 3 and on no
	// earlier one, so a shorter loop sees nothing.
	const ticks = 3

	// The batch size is this fixture's own, so it is taken from the fixture --
	// unlike unhealthyAfter and dblabConcurrency, which are production constants
	// a test must not be able to follow silently.
	calls := [][2]string{
		{"/status", "GET"},
		{"/branch", "GET"},
		{"/snapshot", "GET"},
		{"/instance/retrieval", "GET"},
		{"/clone", "POST"},
	}
	h := newDBLabHarness(t, dblabBatch(calls...))
	// Four of the five fail permanently; one lands. A 404 is not retryable,
	// so this is deterministic rather than a race with the retry ladder.
	h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/status" {
			w.Write([]byte(`{"pools":[]}`))
			return
		}
		w.WriteHeader(http.StatusNotFound)
		w.Write([]byte(`{"message":"nope"}`))
	}

	for i := 1; i <= ticks; i++ {
		// The harness hands the batch out on the FIRST poll only, so the counter
		// is what has to survive between ticks, not the work.
		h.mu.Lock()
		h.polls = 0
		h.mu.Unlock()

		h.runner.tick(context.Background())

		if got, want := len(submittedIDs(h)), len(calls)*i; got != want {
			t.Fatalf("submitted %d answers after %d batches, want %d", got, i, want)
		}
		if err := CheckHealth(h.runner.healthPath, h.runner.now()); err != nil {
			t.Fatalf("the box reported unhealthy after batch %d of %d, in EACH of which a job "+
				"landed: %v -- a run with any success is not a failed run, and the verdict "+
				"must not depend on which worker finished last", i, ticks, err)
		}
	}
}

// ...AND A BATCH IN WHICH NOTHING LANDS COUNTS ONCE, NOT ONCE PER JOB.
//
// The control for the test above: without it, "never unhealthy" would pass for
// a build that had simply stopped counting. unhealthyAfter is 3, so two
// all-failed batches must NOT flip it and the third must.
func TestAllFailedBatchesCountOncePerRun(t *testing.T) {
	newFailingBatch := func(t *testing.T) *dblabHarness {
		h := newDBLabHarness(t, dblabBatch(
			[2]string{"/status", "GET"},
			[2]string{"/branch", "GET"},
			[2]string{"/snapshot", "GET"},
		))
		h.engine = func(h *dblabHarness, w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotFound)
			w.Write([]byte(`{"message":"nope"}`))
		}
		return h
	}

	// One runner across three polls, which is what "in a row" means.
	h := newFailingBatch(t)
	for i := 1; i <= 2; i++ {
		h.mu.Lock()
		h.polls = 0
		h.mu.Unlock()
		h.runner.tick(context.Background())
		if err := CheckHealth(h.runner.healthPath, h.runner.now()); err != nil {
			t.Fatalf("health flipped after %d all-failed batch(es), want 3: %v -- a batch "+
				"of 3 must count as ONE failed run, not three", i, err)
		}
	}
	h.mu.Lock()
	h.polls = 0
	h.mu.Unlock()
	h.runner.tick(context.Background())
	if err := CheckHealth(h.runner.healthPath, h.runner.now()); err == nil {
		t.Fatal("three all-failed batches in a row and the box still reports healthy")
	} else if !strings.Contains(err.Error(), "runs in a row failed") {
		t.Fatalf("the health reason does not name the cause: %v", err)
	}
}

// A PANIC IN THE HEALTH STAMP IS A PANIC IN THE WORKER.
//
// The stamp sits INSIDE runWorker's recover rather than above it, which is a
// one-line choice with the same consequence as the job body: unrecovered, it
// ends the process with up to dblabConcurrency-1 siblings already claimed and
// never answered. r.now is the injection point it reaches.
func TestAPanicInTheHealthStampDoesNotTakeTheBatchWithIt(t *testing.T) {
	h := newDBLabHarness(t, dblabBatch(
		[2]string{"/status", "GET"},
		[2]string{"/branch", "GET"},
		[2]string{"/snapshot", "GET"},
	))
	var once sync.Once
	realNow := h.runner.now
	h.runner.now = func() time.Time {
		boom := false
		once.Do(func() { boom = true })
		if boom {
			panic("the clock panicked")
		}
		return realNow()
	}

	h.runner.tick(context.Background())

	if want := []string{"j1", "j2", "j3"}; !equalStrings(submittedIDs(h), want) {
		t.Fatalf("submitted %v after a panic in the health stamp, want an answer for each "+
			"of %v -- a claimed job nobody answers sits running until the sweep",
			submittedIDs(h), want)
	}
}
