package collect

import (
	"math"
	"sort"
)

// TempfileResult is the period aggregate consumed by TEMPFILE/TEMPFILEp.
type TempfileResult struct {
	AvgMiBPS   float64
	WorstMiBPS float64
	P99MiBPS   float64
}

// AggregateTempfile computes the observed-time average, peak and p99.
//
// It is DELIBERATELY the opposite convention to the AAS path: rate(...[5m]) is
// a dense series, not a sparse one, so a missing monitoring sample means "not
// observed", not "zero disk writes". Dividing by expectedSlots here, or
// zero-padding the quantile, would report a gap in monitoring as idle disk.
// Non-finite and negative values are dropped for the same reason -- a counter
// reset artefact is not an observation. (The AAS path keeps +Inf/NaN, where
// they are real samples of a real metric.)
func AggregateTempfile(samples []float64) TempfileResult {
	if len(samples) == 0 {
		return TempfileResult{}
	}
	clean := make([]float64, 0, len(samples))
	var sum, peak float64
	for _, v := range samples {
		// v == 0 is kept: the series is dense, so a zero is "disk idle", a real
		// observation. Dropping zeros would shrink the denominator below and
		// overstate the average.
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			continue
		}
		clean = append(clean, v)
		sum += v
		if v > peak {
			peak = v
		}
	}
	if len(clean) == 0 {
		return TempfileResult{}
	}
	sort.Float64s(clean)
	h := 0.99 * float64(len(clean)-1)
	lo, hi := int(math.Floor(h)), int(math.Ceil(h))
	p99 := clean[lo]
	if lo != hi {
		p99 += (h - float64(lo)) * (clean[hi] - clean[lo])
	}
	return TempfileResult{
		AvgMiBPS:   round1(sum / float64(len(clean))),
		WorstMiBPS: round1(peak),
		P99MiBPS:   round1(p99),
	}
}

// tempfileNodeData carries MiB/s although the JSON keys say "_mbps": the key
// names are the platform's consumer contract (health_matrix_evaluate.sql reads
// temp_write_*_mbps and labels the values MiB/s). Do not rename them without
// migrating that evaluator and the stored payloads.
type tempfileNodeData struct {
	ClusterName         string  `json:"cluster_name"`
	TempWriteAvgMiBPS   float64 `json:"temp_write_avg_mbps"`
	TempWriteWorstMiBPS float64 `json:"temp_write_worst_mbps"`
	TempWriteP99MiBPS   float64 `json:"temp_write_p99_mbps"`
}

type tempfileNodeEntry struct {
	Data tempfileNodeData `json:"data"`
}

// TempfilePayload is stored as TEMPFILE.json; the platform derives the paired
// TEMPFILEp health-matrix row from the same object.
type TempfilePayload struct {
	CheckID string                       `json:"checkId"`
	Results map[string]tempfileNodeEntry `json:"results"`
}

// BuildTempfilePayload wraps a TempfileResult into the TEMPFILE payload.
func BuildTempfilePayload(nodeName, clusterName string, r TempfileResult) TempfilePayload {
	return TempfilePayload{
		CheckID: "TEMPFILE",
		Results: map[string]tempfileNodeEntry{
			nodeName: {Data: tempfileNodeData{
				ClusterName:         clusterName,
				TempWriteAvgMiBPS:   r.AvgMiBPS,
				TempWriteWorstMiBPS: r.WorstMiBPS,
				TempWriteP99MiBPS:   r.P99MiBPS,
			}},
		},
	}
}
