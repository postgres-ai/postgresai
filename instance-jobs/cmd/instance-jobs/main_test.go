package main

import (
	"io"
	"net"
	"net/http"
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

// The relay is optional: an unusable listen address must be reported, not fatal.
func TestServeRelay(t *testing.T) {
	busy, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer busy.Close()
	if _, err := serveRelay(busy.Addr().String(), http.NotFoundHandler()); err == nil {
		t.Fatal("serveRelay bound an address that is in use")
	}
	if _, err := serveRelay("not an address", http.NotFoundHandler()); err == nil {
		t.Fatal("serveRelay accepted an invalid address")
	}

	free, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := free.Addr().String()
	free.Close()
	closeRelay, err := serveRelay(addr, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, "relay") }))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Get("http://" + addr + "/")
	if err != nil {
		t.Fatal(err)
	}
	body, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if string(body) != "relay" {
		t.Fatalf("body = %q", body)
	}
	closeRelay()
	if _, err := http.Get("http://" + addr + "/"); err == nil {
		t.Fatal("relay still serving after close")
	}
}
