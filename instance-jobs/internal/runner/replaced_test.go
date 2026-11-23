package runner

import (
	"context"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// #523: a single-file bind mount keeps the old .pgwatch-config after the host replaces it
// (sed -i, Ansible, mv); by path the container then sees link count 0. Run stops between
// batches with ErrConfigReplaced (main exits 75) so Docker restarts it with the new file.
func TestARunStopsBetweenBatchesWhenTheConfigWasReplaced(t *testing.T) {
	h := newHarness(t, true, func(_ *harness, w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"server_time":"2026-09-16T00:00:00Z","jobs":[],"next_poll_ms":5000}`))
	})
	counts := []uint64{1, 1, 0, 0}
	calls := 0
	h.runner.linkCount = func(string) (uint64, bool) {
		n := counts[min(calls, len(counts)-1)]
		calls++
		return n, true
	}
	sleeps := 0
	h.runner.sleep = func(context.Context, time.Duration) bool { sleeps++; return sleeps < 10 }

	err := h.runner.Run(context.Background())

	if !errors.Is(err, ErrConfigReplaced) {
		t.Fatalf("Run = %v, want ErrConfigReplaced", err)
	}
	if h.polls != 2 || sleeps != 2 {
		t.Fatalf("polls = %d, sleeps = %d; want 2 and 2: the tick that sees the replaced file must not poll", h.polls, sleeps)
	}
	if !strings.Contains(h.logs.String(), "was replaced or removed on the host") ||
		!strings.Contains(h.logs.String(), "docker compose up -d") {
		t.Fatalf("log does not say what happened and how to recover:\n%s", h.logs.String())
	}
	if ReplacedExitCode != 75 {
		t.Fatalf("ReplacedExitCode = %d, want 75 (as pgai-mon and the reporter)", ReplacedExitCode)
	}
}

// File sharing that always reports link count 0 must not restart the container forever.
func TestLinkCountZeroFromTheStartIsNotAReplacement(t *testing.T) {
	r := &Runner{linkCount: func(string) (uint64, bool) { return 0, true }}
	for i := 0; i < 3; i++ {
		if r.configReplaced("/app/.pgwatch-config") {
			t.Fatalf("tick %d: a file at link count 0 from the start was reported as replaced", i)
		}
	}
}

func TestAConfigNotThereYetIsCheckedOnceItAppears(t *testing.T) {
	seq := []struct {
		n  uint64
		ok bool
	}{{0, false}, {1, true}, {0, true}, {0, true}}
	i := 0
	r := &Runner{linkCount: func(string) (uint64, bool) { s := seq[i]; i++; return s.n, s.ok }}
	got := []bool{r.configReplaced("p"), r.configReplaced("p"), r.configReplaced("p"), r.configReplaced("p")}
	if want := []bool{false, false, true, false}; !equalBools(got, want) {
		t.Fatalf("configReplaced = %v, want %v (reported once, after it was seen linked)", got, want)
	}
}

// On a plain file system a file replaced by rename keeps link count 1 by path, so a
// standalone instance-jobs never restarts on an ordinary edit.
func TestStatLinkCountFollowsThePath(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, ".pgwatch-config")
	if err := os.WriteFile(f, []byte("api_key=a\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	r := &Runner{linkCount: statLinkCount}
	if r.configReplaced(f) {
		t.Fatal("first look reported a replacement")
	}
	tmp := f + ".tmp"
	if err := os.WriteFile(tmp, []byte("api_key=b\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(tmp, f); err != nil {
		t.Fatal(err)
	}
	if r.configReplaced(f) {
		t.Fatal("a rename on a plain file system was reported as a replaced mount")
	}
	if n, ok := statLinkCount(filepath.Join(dir, "missing")); ok || n != 0 {
		t.Fatalf("statLinkCount(missing) = %d, %v", n, ok)
	}
}

func equalBools(a, b []bool) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
