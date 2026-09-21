package runner

import (
	"os"
	"path/filepath"
	"testing"
)

// The GC sizes the heap against the machine, not the cgroup, so without this a
// 256 MiB container grows past its ceiling and is OOM-killed instead of
// collecting -- which then trips unhealthyAfter and reports the box as faulty.
func TestTheCgroupLimitIsReadAndHeadroomLeft(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "memory.max")
	if err := os.WriteFile(path, []byte("268435456\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	prev := cgroupMemoryLimitPaths
	cgroupMemoryLimitPaths = []string{path}
	defer func() { cgroupMemoryLimitPaths = prev }()
	t.Setenv("GOMEMLIMIT", "")

	got := cgroupMemoryLimit()
	if got != 268435456 {
		t.Fatalf("cgroupMemoryLimit = %d, want 268435456", got)
	}

	applied := ApplyMemoryLimit()
	if applied >= got {
		t.Fatalf("applied %d with no headroom under the %d ceiling: a limit AT the ceiling "+
			"just moves the OOM kill to the moment it is reached", applied, got)
	}
	if applied < got/2 {
		t.Fatalf("applied %d, far under the %d ceiling: that is a different limit, not headroom",
			applied, got)
	}
}

// "max", an unparseable value, or a sentinel so large it means unlimited are
// all "no limit here", not a limit of zero.
func TestAnAbsentOrUnlimitedCgroupLeavesTheRuntimeAlone(t *testing.T) {
	dir := t.TempDir()
	for name, content := range map[string]string{
		"unlimited v2": "max\n",
		"garbage":      "not-a-number\n",
		"v1 sentinel":  "9223372036854771712\n",
		"zero":         "0\n",
		"empty":        "",
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(dir, "memory.max."+name)
			if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
				t.Fatal(err)
			}
			prev := cgroupMemoryLimitPaths
			cgroupMemoryLimitPaths = []string{path}
			defer func() { cgroupMemoryLimitPaths = prev }()

			if got := cgroupMemoryLimit(); got != 0 {
				t.Fatalf("read %q as a limit of %d", content, got)
			}
		})
	}
}

// An operator who set GOMEMLIMIT means it; the runtime has already applied it.
func TestAnExplicitGomemlimitIsNotOverridden(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "memory.max")
	if err := os.WriteFile(path, []byte("268435456\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	prev := cgroupMemoryLimitPaths
	cgroupMemoryLimitPaths = []string{path}
	defer func() { cgroupMemoryLimitPaths = prev }()

	t.Setenv("GOMEMLIMIT", "100MiB")
	if applied := ApplyMemoryLimit(); applied != 0 {
		t.Fatalf("overrode an explicit GOMEMLIMIT with %d", applied)
	}
}
