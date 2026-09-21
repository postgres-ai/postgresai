package collect

import "strconv"

// formatFloat1 formats a float with exactly one decimal (89 -> "89.0").
func formatFloat1(v float64) string {
	return strconv.FormatFloat(v, 'f', 1, 64)
}

// TopEvents holds the top-3 pre-formatted event strings per wait-event type.
type TopEvents struct {
	IO     []string `json:"io"`
	IPC    []string `json:"ipc"`
	Lock   []string `json:"lock"`
	LWLock []string `json:"lwlock"`
}

// NodeData is the per-node payload body. ClusterName is echoed back because
// v1.get_health_matrix reads data->>'cluster_name' and node_name alone is
// ambiguous across clusters on a multi-cluster fleet.
type NodeData struct {
	ClusterName      string    `json:"cluster_name"`
	VCPUs            int       `json:"vcpus"`
	AASAvg           Metrics   `json:"aas_avg"`
	AASWorst1m       Metrics   `json:"aas_worst1m"`
	AASP99           Metrics   `json:"aas_p99"`
	AASP999          Metrics   `json:"aas_p999"`
	TopEventsWorst1m TopEvents `json:"top_events_worst1m"`
	TopEventsP99     TopEvents `json:"top_events_p99"`
	TopEventsP999    TopEvents `json:"top_events_p999"`

	// class -> entries. The reader passes the per-class arrays through with ->
	// unchanged, so queryid must stay a JSON string here.
	TopQueryIDsWorst1m map[string][]QueryIDEntry `json:"top_queryids_worst1m"`
	TopQueryIDsP99     map[string][]QueryIDEntry `json:"top_queryids_p99"`
	TopQueryIDsP999    map[string][]QueryIDEntry `json:"top_queryids_p999"`
}

type nodeEntry struct {
	Data NodeData `json:"data"`
}

// Payload is the STEP-5 checkup report object stored verbatim into
// checkup_report_jsons.data.
type Payload struct {
	CheckID string               `json:"checkId"`
	Results map[string]nodeEntry `json:"results"`
}

// toTopEvents converts the key->list map into the typed struct, defaulting a
// missing list to an empty slice so the JSON emits [] and never null.
func toTopEvents(m map[string][]string) TopEvents {
	get := func(k string) []string {
		if v, ok := m[k]; ok && v != nil {
			return v
		}
		return []string{}
	}
	return TopEvents{IO: get("io"), IPC: get("ipc"), Lock: get("lock"), LWLock: get("lwlock")}
}

// toTopQueryIDs returns a class->entries map carrying all six classes, an
// absent one as [] rather than omitted or null.
func toTopQueryIDs(m map[string][]QueryIDEntry) map[string][]QueryIDEntry {
	out := make(map[string][]QueryIDEntry, len(QueryIDClassKeys))
	for _, k := range QueryIDClassKeys {
		if v, ok := m[k]; ok && v != nil {
			out[k] = v
		} else {
			out[k] = []QueryIDEntry{}
		}
	}
	return out
}

// BuildPayload assembles the AAS payload for a single node.
func BuildPayload(nodeName, clusterName string, vcpus int, r Result) Payload {
	return Payload{
		CheckID: "AAS",
		Results: map[string]nodeEntry{
			nodeName: {Data: NodeData{
				ClusterName:        clusterName,
				VCPUs:              vcpus,
				AASAvg:             r.Avg,
				AASWorst1m:         r.Worst1m,
				AASP99:             r.P99,
				AASP999:            r.P999,
				TopEventsWorst1m:   toTopEvents(r.TopWorst1m),
				TopEventsP99:       toTopEvents(r.TopP99),
				TopEventsP999:      toTopEvents(r.TopP999),
				TopQueryIDsWorst1m: toTopQueryIDs(r.TopQueryIDsWorst1m),
				TopQueryIDsP99:     toTopQueryIDs(r.TopQueryIDsP99),
				TopQueryIDsP999:    toTopQueryIDs(r.TopQueryIDsP999),
			}},
		},
	}
}
