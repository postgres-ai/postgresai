package collect

import (
	"context"
	"fmt"
	"math"
	"time"
)

// Observation describes actual raw metric sample time. Range evaluation times
// and period boundaries cannot establish that the collector observed a source.
type Observation struct {
	Source      string `json:"source"`
	ObservedAt  string `json:"observed_at"`
	SampleCount int    `json:"sample_count"`
}

func (c *Client) queryObservation(ctx context.Context, kind string, f SeriesFilter, start, end time.Time) *Observation {
	metric, matcher := "pgwatch_wait_events_total", f.labelMatcher()
	if kind == KindTempfile {
		metric, matcher = "pgwatch_db_stats_temp_bytes", f.baseMatcher()
	}
	matrix, err := c.queryRange(ctx, fmt.Sprintf("max(timestamp(%s{%s}))", metric, matcher), start, end)
	if err != nil {
		return nil
	}
	var latest float64
	count := 0
	for _, s := range matrix {
		for _, v := range s.values {
			if math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 || v < float64(start.Unix()) || v > float64(end.Unix()) {
				continue
			}
			count++
			if v > latest {
				latest = v
			}
		}
	}
	if count == 0 {
		return nil
	}
	return &Observation{Source: "prometheus_sample_timestamp", ObservedAt: time.Unix(int64(latest), 0).UTC().Format(time.RFC3339), SampleCount: count}
}
