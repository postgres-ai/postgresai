package runner

import (
	"os"
	"runtime/debug"
	"strconv"
	"strings"
)

// cgroup memory limit files, v2 first.
var cgroupMemoryLimitPaths = []string{
	"/sys/fs/cgroup/memory.max",                   // v2
	"/sys/fs/cgroup/memory/memory.limit_in_bytes", // v1
}

// ApplyMemoryLimit points the Go GC at the container's actual memory ceiling.
//
// The GC sizes the heap against the MACHINE's memory and cannot see a cgroup
// limit, so in a 256 MiB container it will happily grow past it and be
// OOM-killed rather than collect. That failure is not slowness: the container
// dies, restarts, and three of them trip unhealthyAfter, so the box reports as
// faulty for what is really one oversized response.
//
// Read from the cgroup rather than configured, deliberately: a second copy of
// the compose mem_limit is one more number to drift. Returns the limit applied,
// or 0 when it left the runtime alone.
func ApplyMemoryLimit() int64 {
	// An explicit GOMEMLIMIT wins: the runtime has already applied it, and an
	// operator who set one means it.
	if strings.TrimSpace(os.Getenv("GOMEMLIMIT")) != "" {
		return 0
	}

	limit := cgroupMemoryLimit()
	if limit <= 0 {
		return 0
	}
	// Headroom for the parts of the process the GC does not manage -- stacks,
	// the runtime itself, and whatever a cgo-free binary still maps. A limit
	// set AT the ceiling just moves the OOM kill to the moment it is reached.
	target := limit / 10 * 9
	debug.SetMemoryLimit(target)
	return target
}

func cgroupMemoryLimit() int64 {
	for _, p := range cgroupMemoryLimitPaths {
		raw, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		v := strings.TrimSpace(string(raw))
		// cgroup v2 writes "max" when unlimited; v1 writes a number so large it
		// means the same thing. Neither is a limit worth honouring.
		if v == "" || v == "max" {
			continue
		}
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil || n <= 0 || n >= 1<<62 {
			continue
		}
		return n
	}
	return 0
}
