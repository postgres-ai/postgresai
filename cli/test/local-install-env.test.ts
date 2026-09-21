import { describe, test, expect } from "bun:test";
import fs from "fs";
import os from "os";
import path from "path";
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
      "VM_DELETE_AUTH_KEY=delete-key",
      "VM_SNAPSHOT_AUTH_KEY=snapshot-key",
      'VM_FORCE_MERGE_AUTH_KEY="force-merge-key"',
      "VM_PPROF_AUTH_KEY=pprof-key",
      "",
    ].join("\n");

    const { content, preservedKeys } = buildLocalInstallEnv(existing, "0.15.0");
    const env = parse(content);

    expect(env.PGAI_REGISTRY).toBe("registry.example.com/pgai");
    expect(env.GF_SECURITY_ADMIN_PASSWORD).toBe("grafana-pw");
    expect(env.REPLICATOR_PASSWORD).toBe("replicator-pw");
    expect(env.VM_AUTH_USERNAME).toBe("vmauth");
    expect(env.VM_AUTH_PASSWORD).toBe("quoted-pw");
    // Re-installing must not rotate the admin keys out from under an operator
    // who noted one down (#359).
    expect(env.VM_DELETE_AUTH_KEY).toBe("delete-key");
    expect(env.VM_SNAPSHOT_AUTH_KEY).toBe("snapshot-key");
    expect(env.VM_FORCE_MERGE_AUTH_KEY).toBe("force-merge-key");
    expect(env.VM_PPROF_AUTH_KEY).toBe("pprof-key");
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

  /**
   * postgresai#359: with no admin keys on the sink-prometheus command line,
   * VictoriaMetrics demanded none, and Grafana's datasource proxy forwards
   * every GET, so a Viewer token could delete the whole store. A green-field
   * install must mint the keys, and must not reuse a credential that anything
   * querying the store already holds.
   */
  test("mints distinct VictoriaMetrics admin keys, never shared with the query credentials (#359)", () => {
    // Seed the Grafana password: buildLocalInstallEnv only emits it when it is
    // already present, and comparing against an undefined value would pass no
    // matter what the code did.
    const env = parse(
      buildLocalInstallEnv("GF_SECURITY_ADMIN_PASSWORD=grafana-pw\n", "0.17.0").content,
    );
    const adminKeys = [
      env.VM_DELETE_AUTH_KEY,
      env.VM_SNAPSHOT_AUTH_KEY,
      env.VM_FORCE_MERGE_AUTH_KEY,
      env.VM_PPROF_AUTH_KEY,
    ];

    for (const key of adminKeys) {
      expect(key).toMatch(/^[a-f0-9]{64}$/);
    }
    expect(new Set(adminKeys).size).toBe(4);
    for (const shared of [env.VM_AUTH_PASSWORD, env.REPLICATOR_PASSWORD, env.GF_SECURITY_ADMIN_PASSWORD]) {
      expect(shared).toBeDefined();
      expect(adminKeys).not.toContain(shared);
    }
  });

  test("re-mints admin keys that exist but are blank or quoted-empty (#359)", () => {
    // The `cp .env.example .env` shape, plus the quoted form people copy from
    // docs, plus `export `. A blank key is the vulnerability, not a setting.
    const existing = [
      "PGAI_TAG=0.16.0",
      "VM_DELETE_AUTH_KEY=",
      'VM_SNAPSHOT_AUTH_KEY=""',
      "VM_FORCE_MERGE_AUTH_KEY=''",
      "export VM_PPROF_AUTH_KEY=",
      "",
    ].join("\n");

    const { content } = buildLocalInstallEnv(existing, "0.17.0");
    const env = parse(content);

    for (const key of [
      "VM_DELETE_AUTH_KEY",
      "VM_SNAPSHOT_AUTH_KEY",
      "VM_FORCE_MERGE_AUTH_KEY",
      "VM_PPROF_AUTH_KEY",
    ]) {
      expect(env[key]).toMatch(/^[a-f0-9]{64}$/);
      // The stale blank line must not be carried through: compose reads the
      // LAST assignment, so a preserved `export KEY=` would win.
      expect(
        content.split("\n").filter((l) => new RegExp(`^\\s*(?:export\\s+)?${key}=`).test(l)).length,
      ).toBe(1);
    }
  });

  test("an operator-set key written as `export KEY=value` is preserved, not re-minted", () => {
    const { content } = buildLocalInstallEnv(
      "PGAI_TAG=0.16.0\nexport VM_DELETE_AUTH_KEY=operator-key\n",
      "0.17.0",
    );
    expect(parse(content).VM_DELETE_AUTH_KEY).toBe("operator-key");
    expect(content.split("\n").filter((l) => l.includes("VM_DELETE_AUTH_KEY=")).length).toBe(1);
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

describe("the monitoring instance id in .env", () => {
  // It reaches the CLI as --instance-id / PGAI_INSTANCE_ID and was written
  // nowhere on the box, so a compose service could not see it. postgresai#366.
  test("an id supplied by this run is written", () => {
    const { content } = buildLocalInstallEnv("", "0.17.0", "11111111-1111-1111-1111-111111111111");
    expect(content).toContain("PGAI_INSTANCE_ID=11111111-1111-1111-1111-111111111111");
  });

  test("an id already in .env survives a re-install that supplies none", () => {
    const { content } = buildLocalInstallEnv("PGAI_INSTANCE_ID=kept\n", "0.17.0");
    expect(content).toContain("PGAI_INSTANCE_ID=kept");
    // Written once, by the managed block, not also preserved verbatim.
    expect(content.match(/PGAI_INSTANCE_ID=/g)).toHaveLength(1);
  });

  test("a supplied id replaces the stored one", () => {
    const { content } = buildLocalInstallEnv("PGAI_INSTANCE_ID=old\n", "0.17.0", "new");
    expect(content).toContain("PGAI_INSTANCE_ID=new");
    expect(content).not.toContain("PGAI_INSTANCE_ID=old");
  });

  test("nothing is written when there is no id at all", () => {
    const { content } = buildLocalInstallEnv("", "0.17.0");
    expect(content).not.toContain("PGAI_INSTANCE_ID");
  });
});

describe("who the instance-jobs container runs as", () => {
  // The point of the fix: the uid comes from the FILE, not from whoever ran the
  // command. Those differ under sudo, which the Terraform box's own documented
  // commands use.
  test("the owner of the credential file wins over the calling process", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pgai-owner-"));
    const file = path.join(dir, ".pgwatch-config");
    fs.writeFileSync(file, "api_key=k\n", { mode: 0o600 });

    // Re-own the file to a supplementary group so its gid differs from the
    // process's. Skip where the runner has none (a CI container as root).
    const groups = (process.getgroups?.() ?? []).filter(g => g !== process.getgid!());
    if (groups.length === 0) {
      return; // nothing to discriminate against on this host
    }
    fs.chownSync(file, process.getuid!(), groups[0]);
    const owner = fs.statSync(file);

    const { content } = buildLocalInstallEnv("", "0.17.0", null, file);
    expect(content).toContain(`INSTANCE_JOBS_USER=${owner.uid}:${owner.gid}`);
    expect(content).not.toContain(`INSTANCE_JOBS_USER=${process.getuid!()}:${process.getgid!()}`);
  });

  test("the fallback writes uid before gid, not the other way round", () => {
    // Compose consumes this as `user: "uid:gid"`. A reversed pair runs the
    // container as the wrong user and it cannot read the 0600 credential --
    // and it is invisible on the usual box where uid equals gid, so stub them
    // to distinct values.
    const realUid = process.getuid;
    const realGid = process.getgid;
    (process as unknown as { getuid: () => number }).getuid = () => 4242;
    (process as unknown as { getgid: () => number }).getgid = () => 7;
    try {
      const { content } = buildLocalInstallEnv("", "0.17.0", null, "/nonexistent/.pgwatch-config");
      expect(content).toContain("INSTANCE_JOBS_USER=4242:7");
    } finally {
      (process as unknown as { getuid?: () => number }).getuid = realUid;
      (process as unknown as { getgid?: () => number }).getgid = realGid;
    }
  });

  test("a missing credential file falls back to the calling process", () => {
    const { content } = buildLocalInstallEnv("", "0.17.0", null, "/nonexistent/.pgwatch-config");
    expect(content).toContain(`INSTANCE_JOBS_USER=${process.getuid!()}:${process.getgid!()}`);
  });

  test("a stray DIRECTORY at the credential path writes no key at all", () => {
    // Docker creates a bind-mount target as a root-owned directory when the
    // file is missing. Copying its owner wrote INSTANCE_JOBS_USER=0:0, and
    // falling back to the process is no better -- under sudo, the documented
    // install path, that is 0:0 too. Emitting NO key is what makes compose use
    // the image's unprivileged user and fail loudly on the 0600 credential,
    // which is what docker-compose.yml and the README promise.
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "strayd-"));
    try {
      const asDirectory = path.join(dir, ".pgwatch-config");
      fs.mkdirSync(asDirectory);
      const { content } = buildLocalInstallEnv("", "0.17.0", null, asDirectory);
      expect(content).not.toContain("INSTANCE_JOBS_USER=");
    } finally {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  // It bind-mounts .pgwatch-config, which this command keeps at 0600, so the
  // container has to be that file's owner or it cannot read the credential.
  test("the install writes the current uid:gid", () => {
    const { content } = buildLocalInstallEnv("", "0.17.0");
    expect(content).toContain(`INSTANCE_JOBS_USER=${process.getuid!()}:${process.getgid!()}`);
  });

  test("a re-install refreshes it rather than keeping a stale copy", () => {
    const { content } = buildLocalInstallEnv("INSTANCE_JOBS_USER=999:999\n", "0.17.0");
    expect(content).not.toContain("INSTANCE_JOBS_USER=999:999");
    expect(content.match(/INSTANCE_JOBS_USER=/g)).toHaveLength(1);
  });
});
