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
// like Poll and Submit, so the org token cannot reach Postgres bind-parameter logs.
func (c *Client) SupabaseHostMetricsCredential(ctx context.Context, creds Credentials) (*SupabaseCredential, error) {
	var out SupabaseCredential
	err := c.call(ctx, "supabase_host_metrics_credential", creds, map[string]any{
		"instance_id": creds.InstanceID,
	}, &out)
	if err != nil || out.Status == "" {
		// Neither response messages nor transport errors may escape this boundary:
		// the platform could have echoed a secret into either.
		return nil, errors.New("Supabase credential request failed")
	}
	return &out, nil
}
