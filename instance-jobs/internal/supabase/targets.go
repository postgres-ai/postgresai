package supabase

import (
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

type targetGroup struct {
	Targets []string          `json:"targets"`
	Labels  map[string]string `json:"labels"`
}

// targets copies pgwatch's custom tags from enabled instances without exposing
// connection strings. Discovery never asks the platform for a credential.
func (r *Relay) targets(w http.ResponseWriter, req *http.Request) {
	raw, err := os.ReadFile(r.instancesPath)
	if err != nil {
		http.Error(w, "instances_unavailable", 503)
		return
	}
	var instances []struct {
		ConnStr string            `yaml:"conn_str"`
		Enabled bool              `yaml:"is_enabled"`
		Tags    map[string]string `yaml:"custom_tags"`
	}
	if yaml.Unmarshal(raw, &instances) != nil {
		http.Error(w, "instances_invalid", 503)
		return
	}
	groups := []targetGroup{}
	seen := map[[3]string]bool{}
	for _, instance := range instances {
		if !instance.Enabled {
			continue
		}
		ref := projectReference(instance.ConnStr)
		cluster, node := instance.Tags["cluster"], instance.Tags["node_name"]
		key := [3]string{ref, cluster, node}
		if ref == "" || cluster == "" || node == "" || seen[key] {
			continue
		}
		seen[key] = true
		groups = append(groups, targetGroup{Targets: []string{r.target}, Labels: map[string]string{
			"cluster": cluster, "node_name": node, "__param_project_ref": ref,
		}})
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(groups)
}

func projectReference(conn string) string {
	u, err := url.Parse(conn)
	if err != nil {
		return ""
	}
	host := u.Hostname()
	direct := strings.TrimPrefix(host, "db.")
	if metricsHost.MatchString(direct) {
		return strings.TrimSuffix(direct, ".supabase.co")
	}
	var ref string
	if strings.HasSuffix(host, ".pooler.supabase.com") {
		if u.User != nil {
			_, ref, _ = strings.Cut(u.User.Username(), ".")
		}
		if ref == "" {
			ref = strings.TrimSuffix(host, ".pooler.supabase.com")
		}
	}
	if metricsHost.MatchString(ref + ".supabase.co") {
		return ref
	}
	return ""
}
