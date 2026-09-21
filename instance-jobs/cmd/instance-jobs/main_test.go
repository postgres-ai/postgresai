package main

import (
	"strings"
	"testing"
)

// The platform caps client_version at 64 characters and answers PT400 over it,
// which would fail every poll and every submit.
func TestClientVersionFitsThePlatformCap(t *testing.T) {
	original := version
	defer func() { version = original }()

	version = strings.Repeat("v", 200)
	got := clientVersion()
	if len(got) > 64 {
		t.Fatalf("client_version is %d characters, over the platform's 64", len(got))
	}

	version = "0.17.0"
	if got := clientVersion(); got != "instance-jobs/0.17.0" {
		t.Fatalf("clientVersion = %q, want the name and the build version", got)
	}
}
