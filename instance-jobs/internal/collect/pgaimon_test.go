package collect

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// startPgaiMon loads a tiny store and serves it with the pgai-mon binary named
// by PGAI_MON_BIN. Every other test in this package answers the SQL from a
// double keyed by the statement's name, so only this one shows that the store
// parses and runs what the builders render. Skipped without a binary; CI must
// provide one.
func startPgaiMon(t *testing.T, samples, texts []string) *Client {
	t.Helper()
	bin := os.Getenv("PGAI_MON_BIN")
	if bin == "" {
		if os.Getenv("CI") != "" {
			t.Fatal("PGAI_MON_BIN is not set: CI runs the builders' SQL on a real pgai-mon")
		}
		t.Skip("PGAI_MON_BIN is not set")
	}
	dir := t.TempDir()
	write := func(name string, lines []string) string {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	store := filepath.Join(dir, "store")
	load := exec.Command(bin, "load", "--store", store, "--input", write("samples.jsonl", samples),
		"--query-texts", write("texts.jsonl", texts))
	if out, err := load.CombinedOutput(); err != nil {
		t.Fatalf("pgai-mon load: %v\n%s", err, out)
	}
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := l.Addr().String()
	l.Close()
	mon, err := filepath.Abs("../../../mon")
	if err != nil {
		t.Fatal(err)
	}
	// loopback only, so no token; the query limits are the defaults, as in the compose stack
	serve := exec.Command(bin, "serve", "--store", store, "--listen", addr, "--no-compact", "--no-auth",
		"--metrics-yml", filepath.Join(mon, "metrics.yml"), "--macros-sql", filepath.Join(mon, "sql", "macros.sql"))
	serve.Dir = filepath.Dir(mon)
	if err := serve.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		serve.Process.Kill()
		serve.Wait()
	})
	for deadline := time.Now().Add(60 * time.Second); ; time.Sleep(100 * time.Millisecond) {
		if resp, err := http.Get("http://" + addr + "/health"); err == nil {
			resp.Body.Close()
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("pgai-mon serve did not become ready")
		}
	}
	return NewClient("http://"+addr, "", 30*time.Second)
}

func TestBuildersRunOnPgaiMon(t *testing.T) {
	const base = int64(1794236400) // a whole minute; samples every minute for 10 minutes
	series := func(labels string, value func(i int64) float64) string {
		var ts, vs []string
		for i := int64(0); i <= 10; i++ {
			ts = append(ts, fmt.Sprint((base+i*60)*1000))
			vs = append(vs, fmt.Sprint(value(i)))
		}
		return fmt.Sprintf(`{"metric":{%s},"timestamps":[%s],"values":[%s]}`, labels, strings.Join(ts, ","), strings.Join(vs, ","))
	}
	wait := func(cluster string, v float64) string {
		return series(`"__name__":"pgwatch_wait_events_total","cluster":"`+cluster+`","node_name":"n1","datname":"d",`+
			`"wait_event_type":"IO","wait_event":"DataFileRead","query_id":"7"`, func(int64) float64 { return v })
	}
	text := func(cluster, query string) string {
		return fmt.Sprintf(`{"queryid":7,"dbname":"d","query":%q,"first_seen_ms":%d,"last_seen_ms":%d,"source":"s","cluster":%q,"node_name":"n1"}`,
			query, (base+60)*1000, (base+300)*1000, cluster)
	}
	c := startPgaiMon(t,
		[]string{wait("c1", 2), wait("c'2", 5),
			series(`"__name__":"pgwatch_db_stats_temp_bytes","cluster":"c1","node_name":"n1","datname":"d"`,
				func(i int64) float64 { return float64(i * 60 * 1024 * 1024) })},
		[]string{text("c1", "select  1"), text("c'2", "select 2")})

	ctx := context.Background()
	f := SeriesFilter{Cluster: "c1", NodeName: "n1"}
	start, end := time.Unix(base, 0), time.Unix(base+600, 0)

	types, err := c.QueryPerType(ctx, f, start, end)
	if err != nil || len(types) != 1 || types[0].Type != "IO" || len(types[0].Samples) != 11 || types[0].Samples[0] != 2 {
		t.Fatalf("per type = %+v, %v", types, err)
	}
	events, err := c.QueryPerEvent(ctx, f, start, end)
	if err != nil || len(events) != 1 || events[0].Event != "DataFileRead" || len(events[0].Samples) != 11 {
		t.Fatalf("per event = %+v, %v", events, err)
	}
	// "total" and "io" hold the statement; the four other classes are empty
	ids, err := c.QueryPerQueryID(ctx, f, start, end)
	if err != nil || len(ids) != 2 || ids[0].QueryID != "7" || ids[0].Datname != "d" || len(ids[0].Samples) != 11 {
		t.Fatalf("per query id = %+v, %v", ids, err)
	}
	total, err := c.QueryTotalSeries(ctx, f, start, end)
	if err != nil || len(total) != 11 || total[10] != 2 {
		t.Fatalf("total = %v, %v", total, err)
	}
	// 1 MiB/s; a rate needs two samples, so the first steps have none
	rate, err := c.QueryTempfileWriteRate(ctx, f, start, end)
	if err != nil || len(rate) == 0 || rate[len(rate)-1] != 1 {
		t.Fatalf("tempfile rate = %v, %v", rate, err)
	}
	for _, kind := range []string{KindAAS, KindTempfile} {
		if o := c.queryObservation(ctx, kind, f, start, end); o == nil || o.ObservedAt != end.UTC().Format(time.RFC3339) {
			t.Fatalf("%s observation = %+v", kind, o)
		}
	}

	// The same queryid and database exist on two clusters: each sees its own text.
	for cluster, want := range map[string]string{"c1": "select  1", "c'2": "select 2"} {
		texts, err := c.QueryInfoTexts(ctx, SeriesFilter{Cluster: cluster, NodeName: "n1"}, []string{"7"}, start, end)
		if collapsed := strings.Join(strings.Fields(want), " "); err != nil || len(texts) != 1 || texts["7"] != collapsed {
			t.Fatalf("%s: texts = %v, %v", cluster, texts, err)
		}
		args, _ := json.Marshal(QueryTextArgs{Cluster: cluster, Node: "n1", Database: "d", QueryID: "7",
			Start: float64(base), End: float64(base + 600)})
		out, err := QueryText(ctx, c, args, end)
		if err != nil {
			t.Fatalf("%s: query text: %v", cluster, err)
		}
		if res := out.Payload.(QueryTextResult); !res.Available || *res.Query != want || *res.ObservedAt != float64(base+300) {
			t.Fatalf("%s: query text = %+v", cluster, res)
		}
	}
	args, _ := json.Marshal(QueryTextArgs{Cluster: "c1", Node: "n2", Database: "d", QueryID: "7", Start: float64(base), End: float64(base + 600)})
	if out, err := QueryText(ctx, c, args, end); err != nil || out.Payload.(QueryTextResult).Available {
		t.Fatalf("another node's text: %+v, %v", out.Payload, err)
	}
}
