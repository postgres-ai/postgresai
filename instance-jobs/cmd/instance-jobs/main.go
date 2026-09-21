// Command instance-jobs runs collection jobs on a monitoring instance and posts
// the results to the platform. Every connection it makes is outbound.
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
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"gitlab.com/postgres-ai/postgresai/instance-jobs/internal/runner"
)

// version is stamped at build time with -ldflags "-X main.version=$PGAI_TAG".
var version = "dev"

// clientVersionMax is the platform's cap on client_version.
const clientVersionMax = 64

func main() {
	log.SetFlags(log.LstdFlags | log.LUTC)
	log.SetPrefix("instance-jobs: ")

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
