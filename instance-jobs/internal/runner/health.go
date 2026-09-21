package runner

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// DefaultHealthPath is where the loop records its health. A file rather than a
// port because the service is defined with no ports at all, and under /tmp
// because the service runs read_only with only a tmpfs there -- anywhere else
// and every write fails.
const DefaultHealthPath = "/tmp/instance-jobs-health"

// healthState is what the `healthcheck` subcommand reads back.
type healthState struct {
	Healthy bool   `json:"healthy"`
	Reason  string `json:"reason,omitempty"`
	// NextCheckBy is when the loop expects to have written this file again.
	// Past it the process is stuck or dead, whatever the last verdict said, so
	// a wedged loop cannot keep reporting green.
	NextCheckBy time.Time `json:"next_check_by"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// writeHealth records the current verdict. It is best-effort: failing to write
// the file must not take down a loop that is otherwise working.
func writeHealth(path string, healthy bool, reason string, now, nextCheckBy time.Time) error {
	state := healthState{
		Healthy:     healthy,
		Reason:      reason,
		NextCheckBy: nextCheckBy.UTC(),
		UpdatedAt:   now.UTC(),
	}
	encoded, err := json.Marshal(state)
	if err != nil {
		return err
	}
	// Write-and-rename, so the healthcheck never reads a half-written file.
	tmp, err := os.CreateTemp(filepath.Dir(path), ".health-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(encoded); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), path)
}

// CheckHealth reads the health file and reports whether the loop is well. It is
// what the container's HEALTHCHECK runs, so `mon health` sees a dead channel
// instead of a green container that has been idling since it started.
func CheckHealth(path string, now time.Time) error {
	raw, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("health file unreadable: %w", err)
	}
	var state healthState
	if err := json.Unmarshal(raw, &state); err != nil {
		return fmt.Errorf("health file unparseable: %w", err)
	}
	if now.After(state.NextCheckBy) {
		return fmt.Errorf("no health update since %s (expected by %s)",
			state.UpdatedAt.Format(time.RFC3339), state.NextCheckBy.Format(time.RFC3339))
	}
	if !state.Healthy {
		return fmt.Errorf("unhealthy: %s", state.Reason)
	}
	return nil
}
