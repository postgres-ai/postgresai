package config

import "testing"

func TestSupabaseSettings(t *testing.T) {
	t.Setenv("INSTANCE_JOBS_CONFIG_PATH", t.TempDir()+"/missing")
	t.Setenv("PGAI_SUPABASE_HOST_METRICS", "")
	t.Setenv("PGAI_SUPABASE_METRICS_LISTEN", "")
	c, err := Load()
	if err != nil || c.SupabaseHostMetrics || c.SupabaseMetricsListen != ":9188" {
		t.Fatal("incorrect defaults")
	}
	t.Setenv("PGAI_SUPABASE_HOST_METRICS", "true")
	t.Setenv("PGAI_SUPABASE_METRICS_LISTEN", "127.0.0.1:9189")
	c, err = Load()
	if err != nil || !c.SupabaseHostMetrics || c.SupabaseMetricsListen != "127.0.0.1:9189" {
		t.Fatal("settings not loaded")
	}
}
