package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The subcommands are a contract with instance-jobs/Dockerfile's HEALTHCHECK and
// with the operator, and `healthcheck` exiting 0 on an unwell loop would make
// the container permanently green -- silently voiding the whole service.
//
// They are exercised by re-running the test binary as a child, which is the
// only way to observe os.Exit.
func TestSubcommands(t *testing.T) {
	if os.Getenv("INSTANCE_JOBS_SUBPROCESS") == "1" {
		// The child is the test binary, so its own -test.* flags are in os.Args;
		// hand main() the argv the case is really about.
		os.Args = append([]string{"instance-jobs"}, strings.Fields(os.Getenv("INSTANCE_JOBS_ARGS"))...)
		main()
		return
	}

	dir := t.TempDir()
	healthy := filepath.Join(dir, "healthy")
	writeState(t, healthy, true, time.Now().Add(time.Hour))
	unhealthy := filepath.Join(dir, "unhealthy")
	writeState(t, unhealthy, false, time.Now().Add(time.Hour))
	stale := filepath.Join(dir, "stale")
	writeState(t, stale, true, time.Now().Add(-time.Hour))
	corrupt := filepath.Join(dir, "corrupt")
	if err := os.WriteFile(corrupt, []byte("not json"), 0o600); err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name       string
		args       []string
		healthFile string
		wantExit   int
		wantStdout string
		wantStderr string
	}{
		{"healthcheck on a fresh verdict", []string{"healthcheck"}, healthy, 0, "", ""},
		{"healthcheck on an unwell loop", []string{"healthcheck"}, unhealthy, 1, "", "unhealthy"},
		{"healthcheck past the deadline", []string{"healthcheck"}, stale, 1, "", "no health update"},
		{"healthcheck with no file", []string{"healthcheck"}, filepath.Join(dir, "absent"), 1, "", "unreadable"},
		// A garbage file must not read as green: that is the same silent void
		// as a healthcheck that always exits 0.
		{"healthcheck on a corrupt file", []string{"healthcheck"}, corrupt, 1, "", "unparseable"},
		{"version", []string{"version"}, healthy, 0, "dev", ""},
		{"an unknown subcommand", []string{"nonsense"}, healthy, 2, "", "unknown subcommand"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// With a deadline: a child that falls through to the production
			// loop (which a missing os.Exit would cause) would otherwise hang
			// the whole run until Go's global timeout.
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=TestSubcommands")
			cmd.Env = append(os.Environ(),
				"INSTANCE_JOBS_SUBPROCESS=1",
				"INSTANCE_JOBS_ARGS="+strings.Join(tc.args, " "),
				"INSTANCE_JOBS_HEALTH_FILE="+tc.healthFile)
			var stdout, stderr strings.Builder
			cmd.Stdout = &stdout
			cmd.Stderr = &stderr
			err := cmd.Run()

			exit := 0
			if ee, ok := err.(*exec.ExitError); ok {
				exit = ee.ExitCode()
			} else if err != nil {
				t.Fatal(err)
			}
			if exit != tc.wantExit {
				t.Fatalf("exit = %d, want %d (stdout %q, stderr %q)",
					exit, tc.wantExit, stdout.String(), stderr.String())
			}
			if tc.wantStdout != "" && !strings.Contains(stdout.String(), tc.wantStdout) {
				t.Fatalf("stdout = %q, want it to contain %q", stdout.String(), tc.wantStdout)
			}
			if tc.wantStderr != "" && !strings.Contains(stderr.String(), tc.wantStderr) {
				t.Fatalf("stderr = %q, want it to contain %q", stderr.String(), tc.wantStderr)
			}
		})
	}
}

// The Dockerfile invokes the subcommand by name; a rename here and not there
// leaves the container with a healthcheck that always fails.
func TestTheDockerfileInvokesTheSubcommandThisBinaryHas(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("..", "..", "Dockerfile"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"healthcheck"`) {
		t.Fatal("the Dockerfile's HEALTHCHECK does not invoke `healthcheck`")
	}
}

func writeState(t *testing.T, path string, healthy bool, nextCheckBy time.Time) {
	t.Helper()
	state := map[string]any{
		"healthy":       healthy,
		"reason":        "because",
		"next_check_by": nextCheckBy.UTC().Format(time.RFC3339Nano),
		"updated_at":    time.Now().UTC().Format(time.RFC3339Nano),
	}
	encoded, err := json.Marshal(state)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, encoded, 0o600); err != nil {
		t.Fatal(err)
	}
}
