package supabase

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestTargetsUseInstanceLabels(t *testing.T) {
	x := setup(t)
	p := filepath.Join(t.TempDir(), "instances.yml")
	// Two databases on one node must not double the host metrics.
	data := `- name: first
  conn_str: postgresql://monitor@db.abcdefghijklmnopqrst.supabase.co/postgres
  is_enabled: true
  custom_tags: {cluster: 'my cluster', node_name: 'node-one'}
- name: duplicate
  conn_str: postgresql://monitor.abcdefghijklmnopqrst@aws-0-us-east-1.pooler.supabase.com/other
  is_enabled: true
  custom_tags: {cluster: 'my cluster', node_name: 'node-one'}
- name: disabled
  conn_str: postgresql://monitor@db.abcdefghijklmnopqrst.supabase.co/postgres
  is_enabled: false
  custom_tags: {cluster: wrong, node_name: wrong}
- name: other-provider
  conn_str: postgresql://monitor@example.com/postgres
  is_enabled: true
  custom_tags: {cluster: wrong, node_name: wrong}
`
	if err := os.WriteFile(p, []byte(data), 0600); err != nil {
		t.Fatal(err)
	}
	x.relay.instancesPath = p
	w := x.get("/supabase/targets")
	var groups []struct {
		Targets []string          `json:"targets"`
		Labels  map[string]string `json:"labels"`
	}
	if w.Code != 200 || json.Unmarshal(w.Body.Bytes(), &groups) != nil || len(groups) != 1 {
		t.Fatal("incorrect discovery groups")
	}
	g := groups[0]
	if len(g.Targets) != 1 || g.Targets[0] != "instance-jobs:9188" || g.Labels["cluster"] != "my cluster" || g.Labels["node_name"] != "node-one" || g.Labels["__param_project_ref"] != projectRef {
		t.Fatal("incorrect labels")
	}
	if x.calls != 0 {
		t.Fatal("discovery fetched credentials")
	}
	if x.get("/supabase/metrics?project_ref=zyxwvutsrqponmlkjihgf").Code != 503 || x.scrapes != 0 {
		t.Fatal("wrong project mislabeled")
	}
}
