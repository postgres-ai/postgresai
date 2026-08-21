import { describe, test, expect } from "bun:test";
import { buildLocalInstallEnv } from "../bin/postgres-ai";

/**
 * `mon local-install` used to rebuild .env from a fixed allowlist, silently
 * dropping operator-set keys. The worst case: VM_RETENTION_PERIOD and
 * QUERYID_RETENTION_HOURS reverting to the short compose defaults, which
 * deletes the customer's metrics weeks after the re-install.
 */
describe("buildLocalInstallEnv", () => {
  const parse = (content: string): Record<string, string> => {
    const out: Record<string, string> = {};
    for (const line of content.split("\n")) {
      const m = line.match(/^([A-Za-z_][A-Za-z0-9_]*)=(.*)$/);
      if (m) out[m[1]] = m[2];
    }
    return out;
  };

  test("preserves retention keys across a rewrite", () => {
    const existing = [
      "PGAI_TAG=0.14.0",
      "VM_RETENTION_PERIOD=13months",
      "QUERYID_RETENTION_HOURS=9600",
      "",
    ].join("\n");

    const { content, preservedKeys } = buildLocalInstallEnv(existing, "0.15.0");
    const env = parse(content);

    expect(env.VM_RETENTION_PERIOD).toBe("13months");
    expect(env.QUERYID_RETENTION_HOURS).toBe("9600");
    expect(preservedKeys).toEqual(["VM_RETENTION_PERIOD", "QUERYID_RETENTION_HOURS"]);
  });

  test("preserves the keys ansible writes (Grafana, bind hosts, OAuth)", () => {
    const ansibleKeys: Record<string, string> = {
      GF_SECURITY_ADMIN_USER: "monitor",
      BIND_HOST: "127.0.0.1:",
      GRAFANA_BIND_HOST: "127.0.0.1:",
      GF_SERVER_ROOT_URL: "https://mon.example.com",
      GRAFANA_OAUTH_ENABLED: "true",
      GRAFANA_OAUTH_NAME: "PostgresAI",
      GRAFANA_OAUTH_ALLOW_SIGN_UP: "true",
      GRAFANA_OAUTH_CLIENT_ID: "cid",
      GRAFANA_OAUTH_CLIENT_SECRET: "secret",
      GRAFANA_OAUTH_SCOPES: "openid email profile",
      GRAFANA_OAUTH_AUTH_URL: "https://console.example.com/auth",
      GRAFANA_OAUTH_TOKEN_URL: "https://console.example.com/rpc/grafana_oauth_token",
      GRAFANA_OAUTH_API_URL: "https://console.example.com/rpc/grafana_oauth_userinfo",
      GRAFANA_DISABLE_LOGIN_FORM: "false",
    };
    const existing =
      "PGAI_TAG=0.14.0\n" +
      Object.entries(ansibleKeys).map(([k, v]) => `${k}=${v}`).join("\n") +
      "\n";

    const env = parse(buildLocalInstallEnv(existing, "0.15.0").content);
    for (const [k, v] of Object.entries(ansibleKeys)) {
      expect(env[k]).toBe(v);
    }
  });

  test("PGAI_TAG is always replaced with the CLI version", () => {
    const { content } = buildLocalInstallEnv("PGAI_TAG=0.14.0\n", "0.15.0");
    expect(parse(content).PGAI_TAG).toBe("0.15.0");
    // Exactly one PGAI_TAG line - the old one must not linger.
    expect(content.split("\n").filter((l) => l.startsWith("PGAI_TAG=")).length).toBe(1);
  });

  test("managed keys are carried over, not duplicated or regenerated", () => {
    const existing = [
      "PGAI_TAG=0.14.0",
      "PGAI_REGISTRY=registry.example.com/pgai",
      "GF_SECURITY_ADMIN_PASSWORD=grafana-pw",
      "REPLICATOR_PASSWORD=replicator-pw",
      "VM_AUTH_USERNAME=vmauth",
      'VM_AUTH_PASSWORD="quoted-pw"',
      "",
    ].join("\n");

    const { content, preservedKeys } = buildLocalInstallEnv(existing, "0.15.0");
    const env = parse(content);

    expect(env.PGAI_REGISTRY).toBe("registry.example.com/pgai");
    expect(env.GF_SECURITY_ADMIN_PASSWORD).toBe("grafana-pw");
    expect(env.REPLICATOR_PASSWORD).toBe("replicator-pw");
    expect(env.VM_AUTH_USERNAME).toBe("vmauth");
    expect(env.VM_AUTH_PASSWORD).toBe("quoted-pw");
    expect(preservedKeys).toEqual([]);
    for (const key of Object.keys(env)) {
      expect(content.split("\n").filter((l) => l.startsWith(`${key}=`)).length).toBe(1);
    }
  });

  test("generates secrets on a green-field install", () => {
    const env = parse(buildLocalInstallEnv("", "0.15.0").content);
    expect(env.PGAI_TAG).toBe("0.15.0");
    expect(env.REPLICATOR_PASSWORD.length).toBeGreaterThan(0);
    expect(env.VM_AUTH_USERNAME).toBe("vmauth");
    expect(env.VM_AUTH_PASSWORD.length).toBeGreaterThan(0);
    expect(env.PGAI_REGISTRY).toBeUndefined();
  });

  test("keeps comments and the order of unmanaged lines", () => {
    const existing = [
      "# operator notes",
      "PGAI_TAG=0.14.0",
      "VM_RETENTION_PERIOD=13months",
      "# retention above set by ansible",
      "CUSTOM_KEY=custom value",
      "",
    ].join("\n");

    const { content } = buildLocalInstallEnv(existing, "0.15.0");
    expect(content).toContain("# operator notes");
    expect(content).toContain("# retention above set by ansible");
    expect(content).toContain("CUSTOM_KEY=custom value");
    expect(content.indexOf("VM_RETENTION_PERIOD")).toBeLessThan(content.indexOf("CUSTOM_KEY"));
    expect(content.endsWith("\n")).toBe(true);
  });
});
