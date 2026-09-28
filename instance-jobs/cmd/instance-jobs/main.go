// Command instance-jobs runs collection jobs on a monitoring instance and posts
// the results to the platform, with an optional internal Supabase metrics relay.
//
// Subcommands:
//
//	(none)        run the loop
//	healthcheck   exit non-zero when the loop is unwell (the container's HEALTHCHECK)
//	version       print the build version
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/config"
	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/runner"
	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/supabase"
)

// version is stamped at build time with -ldflags "-X main.version=$PGAI_TAG".
var version = "dev"

// clientVersionMax is the platform's cap on client_version.
const clientVersionMax = 64

func main() {
	log.SetFlags(log.LstdFlags | log.LUTC)
	log.SetPrefix("instance-jobs: ")

	// Before anything allocates: the GC cannot see the container's cgroup limit
	// on its own, and decoding a large store response into label maps is many
	// times the wire size in heap.
	if applied := runner.ApplyMemoryLimit(); applied > 0 {
		log.Printf("memory limit set to %d bytes from the cgroup", applied)
	}

	healthPath := runner.DefaultHealthPath
	if v := strings.TrimSpace(os.Getenv("INSTANCE_JOBS_HEALTH_FILE")); v != "" {
		healthPath = v
	}

	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "healthcheck":
			if err := runner.CheckHealth(healthPath, time.Now().UTC()); err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			return
		case "version":
			fmt.Println(version)
			return
		default:
			fmt.Fprintf(os.Stderr, "unknown subcommand %q (want: healthcheck, version)\n", os.Args[1])
			os.Exit(2)
		}
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	// Load returns the environment settings even if the credential file cannot
	// yet be read. The relay re-reads platform credentials when it needs them.
	cfg, _ := config.Load()
	if cfg.SupabaseHostMetrics {
		listener, err := net.Listen("tcp", cfg.SupabaseMetricsListen)
		if err != nil {
			log.Fatal("Supabase metrics listener failed")
		}
		server := &http.Server{Handler: supabase.New(config.Load, slog.Default()).Handler(true), ReadHeaderTimeout: 5 * time.Second, IdleTimeout: 60 * time.Second, WriteTimeout: 60 * time.Second}
		defer server.Close()
		go func() {
			if err := server.Serve(listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
				log.Print("Supabase metrics listener stopped")
				stop()
			}
		}()
	}

	log.Printf("starting (version %s)", version)
	err := runner.New(healthPath, clientVersion()).Run(ctx)
	if err != nil && !errors.Is(err, context.Canceled) {
		log.Fatal(err)
	}
	log.Print("stopped")
}

func clientVersion() string {
	v := "instance-jobs/" + version
	if len(v) > clientVersionMax {
		v = v[:clientVersionMax]
	}
	return v
}
