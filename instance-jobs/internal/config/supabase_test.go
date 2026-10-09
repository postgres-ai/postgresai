package config

import (
	"os"
	"testing"
)

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

// platform-all!884: provisioning writes the instance's own secret into
// .pgwatch-config. It is read from the file only; an environment variable is
// visible to anyone who can inspect the container.
func TestInstanceSecretFromFileOnly(t *testing.T) {
	path := t.TempDir() + "/.pgwatch-config"
	t.Setenv("INSTANCE_JOBS_CONFIG_PATH", path)
	t.Setenv("PGAI_INSTANCE_SECRET", "from-env")
	if err := os.WriteFile(path, []byte("api_key=k\ninstance_id=i\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err := Load()
	if err != nil || c.InstanceSecret != "" {
		t.Fatal("instance secret must not come from the environment")
	}
	if err := os.WriteFile(path, []byte("api_key=k\ninstance_id=i\ninstance_secret= s3cr=t \n"), 0o600); err != nil {
		t.Fatal(err)
	}
	c, err = Load()
	if err != nil || c.InstanceSecret != "s3cr=t" {
		t.Fatal("instance secret not read from the file")
	}
}
