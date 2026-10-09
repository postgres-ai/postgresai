package platform

import (
	"context"
	"errors"
)

// SupabaseCredential is ephemeral: never log or persist this response.
type SupabaseCredential struct {
	Status     string `json:"status"`
	ProjectRef string `json:"project_ref"`
	MetricsURL string `json:"metrics_url"`
	Username   string `json:"username"`
	Password   string `json:"password"`
}

// SupabaseHostMetricsCredential authenticates through the access-token header,
// like Poll and Submit, plus the instance's own secret in the instance-secret
// header (platform-all!884), so neither reaches Postgres bind-parameter logs.
func (c *Client) SupabaseHostMetricsCredential(ctx context.Context, creds Credentials, instanceSecret string) (*SupabaseCredential, error) {
	var out SupabaseCredential
	err := c.callWithHeaders(ctx, "supabase_host_metrics_credential", creds, map[string]string{"instance-secret": instanceSecret}, map[string]any{
		"instance_id": creds.InstanceID,
	}, &out)
	if err != nil || out.Status == "" {
		// Neither response messages nor transport errors may escape this boundary:
		// the platform could have echoed a secret into either.
		return nil, errors.New("Supabase credential request failed")
	}
	return &out, nil
}
