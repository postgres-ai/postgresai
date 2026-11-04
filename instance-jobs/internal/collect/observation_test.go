package collect

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestSourceObservationUsesMetricValueNotEvaluationTimestamp(t *testing.T) {
	var query string
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		query = r.URL.Query().Get("query")
		w.Write([]byte(`{"status":"success","data":{"resultType":"matrix","result":[{"metric":{},"values":[[2000,"1000"],[2060,"1060"],[2120,"NaN"],[2180,"9999"]]}]}}`))
	}))
	defer s.Close()
	o := NewClient(s.URL, "", "", time.Second).queryObservation(context.Background(), KindAAS, SeriesFilter{Cluster: "c", NodeName: "n"}, time.Unix(900, 0), time.Unix(2200, 0))
	if o == nil || o.ObservedAt != "1970-01-01T00:17:40Z" || o.SampleCount != 2 {
		t.Fatalf("wrong source observation: %+v", o)
	}
	if !strings.Contains(query, "timestamp(pgwatch_wait_events_total{") {
		t.Fatalf("not source timestamp: %s", query)
	}
}
func TestSourceObservationAbsentOnEmptyOrFailure(t *testing.T) {
	for _, body := range []string{`{"status":"success","data":{"resultType":"matrix","result":[]}}`, `bad`} {
		s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(body)) }))
		o := NewClient(s.URL, "", "", time.Second).queryObservation(context.Background(), KindTempfile, SeriesFilter{}, time.Unix(900, 0), time.Unix(2200, 0))
		s.Close()
		if o != nil {
			t.Fatalf("invented observation: %+v", o)
		}
	}
}
