import { describe, test, expect, beforeEach, afterEach, spyOn } from "bun:test";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

/**
 * The config file holds the API key, and under #327 that can be a GLOBAL token
 * -- the widest credential the platform issues. `writeFileSync`'s `mode` option
 * applies only when the file is CREATED, so a config that already exists with
 * looser permissions (an older CLI, config management, a restored backup, a
 * shared CI image) keeps them while receiving that credential.
 */

let home: string;
let saved: string | undefined;

const configPath = () => path.join(home, "postgresai", "config.json");

// getConfigDir() reads process.env on every call, so re-importing is enough for
// isolation -- there is no module-level cache of the path to reset.
const freshConfigModule = async () => await import("../lib/config");

beforeEach(() => {
  saved = process.env.XDG_CONFIG_HOME;
  home = fs.mkdtempSync(path.join(os.tmpdir(), "pgai-cfgperm-"));
  process.env.XDG_CONFIG_HOME = home;
  fs.mkdirSync(path.dirname(configPath()), { recursive: true });
});

afterEach(() => {
  if (saved === undefined) delete process.env.XDG_CONFIG_HOME;
  else process.env.XDG_CONFIG_HOME = saved;
  fs.rmSync(home, { recursive: true, force: true });
});

const modeOf = (p: string) => (fs.statSync(p).mode & 0o777).toString(8);

describe("config file permissions", () => {
  test("a newly created config is owner-only", async () => {
    const config = await freshConfigModule();
    config.writeConfig({ apiKey: "pai_global_" + "a".repeat(43) });
    expect(modeOf(configPath())).toBe("600");
  });

  test("writeConfig TIGHTENS an existing world-readable config", async () => {
    // The regression: writeFileSync's `mode` is ignored when the file exists.
    fs.writeFileSync(configPath(), "{}\n", { mode: 0o644 });
    fs.chmodSync(configPath(), 0o644);
    expect(modeOf(configPath())).toBe("644");

    const config = await freshConfigModule();
    config.writeConfig({ apiKey: "pai_global_" + "a".repeat(43) });

    expect(modeOf(configPath())).toBe("600");
  });

  test("deleteConfigKeys tightens it too", async () => {
    fs.writeFileSync(configPath(), JSON.stringify({ apiKey: "k", orgId: 42 }) + "\n");
    fs.chmodSync(configPath(), 0o644);

    const config = await freshConfigModule();
    config.deleteConfigKeys(["orgId"]);

    expect(modeOf(configPath())).toBe("600");
  });
});

describe("tightening must never break saving the credential", () => {
  // A config owned by another UID but group-writable (a shared CI image, a
  // container running as a different user): the WRITE succeeds and the chmod
  // raises EPERM. Tightening permissions is best-effort hardening -- it must
  // never turn a working `pgai auth login` into a crash.
  //
  // chmodSync is spied rather than assigned to: `import * as fs` yields a
  // readonly namespace object, so a plain assignment throws before the code
  // under test is even reached.
  const failChmod = () =>
    spyOn(fs, "chmodSync").mockImplementation(() => {
      const err = new Error("EPERM: operation not permitted, chmod") as NodeJS.ErrnoException;
      err.code = "EPERM";
      throw err;
    });

  test("writeConfig survives a chmod it is not allowed to perform", async () => {
    fs.writeFileSync(configPath(), "{}\n");
    const spy = failChmod();
    try {
      const config = await freshConfigModule();
      expect(() =>
        config.writeConfig({ apiKey: "pai_global_" + "a".repeat(43) }),
      ).not.toThrow();
      // ...and the credential still landed.
      expect(JSON.parse(fs.readFileSync(configPath(), "utf8")).apiKey).toContain(
        "pai_global_",
      );
    } finally {
      spy.mockRestore();
    }
  });

  test("deleteConfigKeys survives it too", async () => {
    fs.writeFileSync(configPath(), JSON.stringify({ apiKey: "k", orgId: 42 }) + "\n");
    const spy = failChmod();
    try {
      const config = await freshConfigModule();
      expect(() => config.deleteConfigKeys(["orgId"])).not.toThrow();
      expect(JSON.parse(fs.readFileSync(configPath(), "utf8")).orgId).toBeUndefined();
    } finally {
      spy.mockRestore();
    }
  });
});

/**
 * config.json is not the only file that receives the API key. `.pgwatch-config`
 * is a second credential store -- readConfig() parses `api_key=` out of it -- and
 * `mon local-install` writes the token to both. It lives in the project
 * directory, not under ~/.config, so it is likelier to be group-readable.
 *
 * In scope here because 03e3e71 is what makes `mon local-install` work under a
 * global token, i.e. this MR is what routes the widest credential onto this
 * writer.
 */
describe(".pgwatch-config credential file permissions", () => {
  test("updatePgwatchConfig tightens an existing loose file", async () => {
    const p = path.join(home, ".pgwatch-config");
    // generate-grafana-password creates it with no mode at all -> umask default.
    fs.writeFileSync(p, "grafana_password=x\n", "utf8");
    fs.chmodSync(p, 0o644);

    const { updatePgwatchConfig } = await import("../bin/postgres-ai");
    updatePgwatchConfig(p, { api_key: "pai_global_" + "a".repeat(43) });

    expect(fs.readFileSync(p, "utf8")).toContain("pai_global_");
    expect(modeOf(p)).toBe("600");
  });

  test("a newly created .pgwatch-config is owner-only", async () => {
    const p = path.join(home, ".pgwatch-config-new");
    const { updatePgwatchConfig } = await import("../bin/postgres-ai");
    updatePgwatchConfig(p, { api_key: "pai_global_" + "a".repeat(43) });
    expect(modeOf(p)).toBe("600");
  });
});
