import { describe, test, expect, afterEach } from "bun:test";
import fs from "fs";
import os from "os";
import path from "path";
import {
  buildLocalInstallEnv,
  composeProfilesValue,
  instanceJobsProfileEnabled,
  instanceJobsRemovalOutcome,
  instanceJobsStartSkipReason,
  monitoringHealthServices,
  judgeServiceHealth,
  readComposeProfiles,
  resolveInstanceJobsRequest,
} from "../bin/postgres-ai";

/**
 * postgresai#381: the instance-jobs container is gated by a compose profile and
 * nothing turned that profile on, so an operator enabled it by hand -- and then
 * `mon stop` deleted the container, `mon start` did not bring it back,
 * `mon update` left it on its old image, and `mon health` called the hole
 * `- not enabled`.
 *
 * The fix is a value in `.env` rather than a `--profile` argument on every
 * command: compose reads COMPOSE_PROFILES from `.env` itself, so a plain
 * `up -d` / `pull` / `down` covers the service once the key is there. These
 * pin the three decisions that make that work.
 */
describe("composeProfilesValue", () => {
  test("a run with no opinion leaves the value alone", () => {
    // The ordinary `mon local-install` / ansible re-deploy: neither the flag
    // nor PGAI_INSTANCE_JOBS was given, so the channel must not turn itself on
    // OR off. An upgrade that enabled it would break the "two gates, both
    // closed by default" promise; one that disabled it would stop a live
    // customer's collection.
    expect(composeProfilesValue("instance-jobs")).toBe("instance-jobs");
    expect(composeProfilesValue(null)).toBeNull();
    expect(composeProfilesValue("")).toBeNull();
  });

  test("enabling adds the profile, and is idempotent", () => {
    expect(composeProfilesValue(null, true)).toBe("instance-jobs");
    expect(composeProfilesValue("instance-jobs", true)).toBe("instance-jobs");
  });

  test("disabling removes it, and is idempotent", () => {
    expect(composeProfilesValue("instance-jobs", false)).toBeNull();
    expect(composeProfilesValue(null, false)).toBeNull();
  });

  test("other profiles survive both directions", () => {
    // COMPOSE_PROFILES is compose's own variable, not ours. This stack defines
    // only `instance-jobs` today, but clobbering an operator's list would
    // silently switch off whatever else they had asked for.
    expect(composeProfilesValue("debug", true)).toBe("debug,instance-jobs");
    expect(composeProfilesValue("debug,instance-jobs,extra", false)).toBe("debug,extra");
    expect(composeProfilesValue("debug,extra", false)).toBe("debug,extra");
  });

  test("whitespace and duplicates are normalised", () => {
    expect(composeProfilesValue(" debug , instance-jobs ")).toBe("debug,instance-jobs");
    expect(composeProfilesValue("debug,debug", true)).toBe("debug,instance-jobs");
    expect(composeProfilesValue(" , ,", true)).toBe("instance-jobs");
  });
});

describe("instanceJobsProfileEnabled", () => {
  test("recognises the profile among others, however spaced", () => {
    expect(instanceJobsProfileEnabled("instance-jobs")).toBe(true);
    expect(instanceJobsProfileEnabled("debug, instance-jobs")).toBe(true);
    expect(instanceJobsProfileEnabled("instance-jobs,debug")).toBe(true);
  });

  test("is not fooled by a substring", () => {
    // A `startsWith`/`includes` on the raw string would read these as enabled
    // and make `mon health` fail a box that never asked for the channel.
    expect(instanceJobsProfileEnabled("instance-jobs-staging")).toBe(false);
    expect(instanceJobsProfileEnabled("my-instance-jobs")).toBe(false);
  });

  test("nothing set is not enabled", () => {
    expect(instanceJobsProfileEnabled(null)).toBe(false);
    expect(instanceJobsProfileEnabled("")).toBe(false);
    expect(instanceJobsProfileEnabled(undefined)).toBe(false);
  });
});

describe("resolveInstanceJobsRequest", () => {
  test("the flag wins over the environment", () => {
    expect(resolveInstanceJobsRequest(true, "false")).toEqual({ enabled: true });
    expect(resolveInstanceJobsRequest(false, "true")).toEqual({ enabled: false });
  });

  test("the environment is read when no flag was given", () => {
    // Ansible sets PGAI_INSTANCE_JOBS rather than passing a flag: a CLI older
    // than the flag rejects an unknown option and fails the whole install,
    // but ignores an unknown env var. Same reason as PGAI_INSTANCE_ID.
    for (const on of ["true", "TRUE", "1", "yes", "on", " true "]) {
      expect(resolveInstanceJobsRequest(undefined, on)).toEqual({ enabled: true });
    }
    for (const off of ["false", "0", "no", "off", "OFF"]) {
      expect(resolveInstanceJobsRequest(undefined, off)).toEqual({ enabled: false });
    }
  });

  test("absent or empty is no opinion, not 'off'", () => {
    // The ansible role renders '' when the variable is unset, and compose's own
    // ${VAR:-default} treats '' as unset, so this is the shape every other
    // pass-through in that role already has.
    expect(resolveInstanceJobsRequest(undefined, undefined)).toEqual({});
    expect(resolveInstanceJobsRequest(undefined, "")).toEqual({});
    expect(resolveInstanceJobsRequest(undefined, "   ")).toEqual({});
  });

  test("a value that is not a boolean is an error, not a silent 'off'", () => {
    // Reading `ture` as off would leave the channel dead on a machine somebody
    // enabled on purpose -- the exact failure this mechanism exists to prevent.
    const bad = resolveInstanceJobsRequest(undefined, "ture");
    expect(bad.enabled).toBeUndefined();
    expect(bad.error).toContain("ture");
  });
});

describe("buildLocalInstallEnv writes COMPOSE_PROFILES", () => {
  const read = (content: string): string | undefined =>
    content.split("\n").find((l) => l.startsWith("COMPOSE_PROFILES="))?.slice("COMPOSE_PROFILES=".length);

  test("enabling writes the key", () => {
    const { content } = buildLocalInstallEnv("", "0.17.0", null, null, true);
    expect(read(content)).toBe("instance-jobs");
  });

  test("a re-install with no opinion carries the value over exactly once", () => {
    // The whole approach rests on this: `mon update` and a plain re-install
    // must not turn the channel off, and a second COMPOSE_PROFILES line would
    // be read by compose instead of the first.
    const { content } = buildLocalInstallEnv("COMPOSE_PROFILES=instance-jobs\n", "0.17.0");
    expect(read(content)).toBe("instance-jobs");
    expect(content.match(/^COMPOSE_PROFILES=/gm)).toHaveLength(1);
  });

  test("disabling drops the key entirely", () => {
    const { content } = buildLocalInstallEnv("COMPOSE_PROFILES=instance-jobs\n", "0.17.0", null, null, false);
    expect(content).not.toContain("COMPOSE_PROFILES=");
  });

  test("the key is managed, so it is never also re-appended verbatim", () => {
    const { content, preservedKeys } = buildLocalInstallEnv(
      "COMPOSE_PROFILES=debug\nVM_RETENTION_PERIOD=13months\n",
      "0.17.0",
      null,
      null,
      true,
    );
    expect(read(content)).toBe("debug,instance-jobs");
    expect(content.match(/^COMPOSE_PROFILES=/gm)).toHaveLength(1);
    // Still reported as operator-owned nowhere: it is in the managed block now.
    expect(preservedKeys).toEqual(["VM_RETENTION_PERIOD"]);
    expect(content).toContain("VM_RETENTION_PERIOD=13months");
  });

  test("a quoted or exported existing value is read, not re-emitted with its quotes", () => {
    // Compose strips matching quotes and reads the LAST assignment; a naive
    // read would produce COMPOSE_PROFILES="debug",instance-jobs.
    const { content } = buildLocalInstallEnv('export COMPOSE_PROFILES="debug"\n', "0.17.0", null, null, true);
    expect(read(content)).toBe("debug,instance-jobs");
    expect(content.match(/COMPOSE_PROFILES=/g)).toHaveLength(1);
  });

  test("an empty existing value counts as absent", () => {
    const { content } = buildLocalInstallEnv("COMPOSE_PROFILES=\n", "0.17.0");
    expect(content).not.toContain("COMPOSE_PROFILES=");
  });
});

describe("mon health follows the profile", () => {
  const entry = (enabled: boolean) =>
    monitoringHealthServices(enabled).find((s) => s.container === "instance-jobs")!;

  test("profile off: an absent container is skipped, as before", () => {
    const s = entry(false);
    expect(s.optional).toBe(true);
    expect(judgeServiceHealth(s, false, "", "")).toMatchObject({ ok: true, label: "not enabled" });
  });

  test("profile on: an absent container is a fault, and says why", () => {
    // `mon stop` deletes this container; before the profile was wired up,
    // nothing brought it back and the health line still read `- not enabled`.
    const s = entry(true);
    expect(s.optional).toBe(false);
    const verdict = judgeServiceHealth(s, false, "", "");
    expect(verdict.ok).toBe(false);
    expect(verdict.mark).toBe("✗");
    expect(verdict.label).toBe("enabled but not running");
  });

  test("the default is still off, and the other six are untouched either way", () => {
    expect(monitoringHealthServices().find((s) => s.container === "instance-jobs")!.optional).toBe(true);
    for (const enabled of [true, false]) {
      const others = monitoringHealthServices(enabled).filter((s) => s.container !== "instance-jobs");
      expect(others.map((s) => s.container)).toEqual([
        "grafana-with-datasources",
        "sink-prometheus",
        "pgwatch-postgres",
        "pgwatch-prometheus",
        "target-db",
        "sink-postgres",
      ]);
      for (const s of others) {
        expect(s.optional).toBeUndefined();
        expect(s.absentLabel).toBeUndefined();
      }
    }
  });

  test("a running container is judged the same way whichever side the gate is on", () => {
    for (const enabled of [true, false]) {
      expect(judgeServiceHealth(entry(enabled), true, "running", "healthy")).toMatchObject({ ok: true });
      expect(judgeServiceHealth(entry(enabled), true, "running", "unhealthy")).toMatchObject({ ok: false });
    }
  });
});

describe("the scoped up is skipped unless the stack is running", () => {
  // A scoped `up` against a STOPPED project does not fail: compose creates the
  // network and volumes and starts instance-jobs alone. `mon stop && mon
  // update` would then leave a box the operator believes is stopped with one
  // container polling -- the silently-polling state this mechanism exists to
  // make visible.
  test("a running stack goes ahead", () => {
    expect(instanceJobsStartSkipReason(true)).toBeNull();
  });

  test("a stopped stack is skipped, and says the container comes up with it", () => {
    const reason = instanceJobsStartSkipReason(false);
    expect(reason).toContain("not running");
    expect(reason).toContain("mon start");
  });

  test("UNKNOWN declines too -- the fail-safe direction", () => {
    // The load-bearing case. An unstarted container is a visible fault in
    // `mon health`; a wrongly-started one is invisible. Flipping this to
    // "start when unsure" is the regression to catch.
    const reason = instanceJobsStartSkipReason(null);
    expect(reason).not.toBeNull();
    expect(reason).toContain("could not tell");
  });
});

describe("readComposeProfiles never answers 'off' by accident", () => {
  const dirs: string[] = [];
  const mkdir = (): string => {
    const d = fs.mkdtempSync(path.join(os.tmpdir(), "pgai-profiles-"));
    dirs.push(d);
    return d;
  };
  const withoutProcessEnv = <T>(fn: () => T): T => {
    const saved = process.env.COMPOSE_PROFILES;
    delete process.env.COMPOSE_PROFILES;
    try {
      return fn();
    } finally {
      if (saved !== undefined) process.env.COMPOSE_PROFILES = saved;
    }
  };

  afterEach(() => {
    for (const d of dirs.splice(0)) fs.rmSync(d, { recursive: true, force: true });
  });

  test("reads the key, last assignment winning as compose does", () => {
    const d = mkdir();
    fs.writeFileSync(path.join(d, ".env"), "COMPOSE_PROFILES=first\nexport COMPOSE_PROFILES=\"debug,instance-jobs\"\n");
    expect(withoutProcessEnv(() => readComposeProfiles(d))).toBe("debug,instance-jobs");
  });

  test("no .env at all is a real answer, and stays quiet", () => {
    const d = mkdir();
    const errs: unknown[][] = [];
    const real = console.error;
    console.error = (...a: unknown[]) => { errs.push(a); };
    try {
      expect(withoutProcessEnv(() => readComposeProfiles(d))).toBeNull();
    } finally {
      console.error = real;
    }
    expect(errs).toEqual([]);
  });

  test("an UNREADABLE .env says so instead of silently reading as 'off'", () => {
    // Silent-green is the one direction a health check must not fail in: if
    // the profile is in fact on, answering null makes `mon health` print a
    // green `- not enabled` for a container it never looked for. A directory
    // at the path gives a non-ENOENT error for any user, including root --
    // and Docker really does create bind-mount targets as directories.
    const d = mkdir();
    fs.mkdirSync(path.join(d, ".env"));
    const errs: string[] = [];
    const real = console.error;
    console.error = (...a: unknown[]) => { errs.push(a.map(String).join(" ")); };
    try {
      expect(withoutProcessEnv(() => readComposeProfiles(d))).toBeNull();
    } finally {
      console.error = real;
    }
    expect(errs.join("\n")).toContain("Could not read");
    expect(errs.join("\n")).toContain("not enabled");
  });

  test("the process environment wins over the file, as compose reads it", () => {
    const d = mkdir();
    fs.writeFileSync(path.join(d, ".env"), "COMPOSE_PROFILES=from-file\n");
    const saved = process.env.COMPOSE_PROFILES;
    process.env.COMPOSE_PROFILES = "from-env";
    try {
      expect(readComposeProfiles(d)).toBe("from-env");
    } finally {
      if (saved === undefined) delete process.env.COMPOSE_PROFILES;
      else process.env.COMPOSE_PROFILES = saved;
    }
  });
});

describe("disabling says whether it just ended a live collection channel", () => {
  /**
   * The removal is the one destructive thing this command does, and both of its
   * outcomes used to print the identical `✓ instance-jobs is disabled and not
   * running`. `docker rm --force` exits 0 either way, measured at the wire:
   *   <existing> -> stdout=[<name>] stderr=[]                  rc=0
   *   <missing>  -> stdout=[]       stderr=[No such container] rc=0
   * so only the state observed BEFORE the removal separates them.
   */
  test("a RUNNING container that is destroyed is a warning, not a tick", () => {
    const v = instanceJobsRemovalOutcome("running", "absent", 0);
    expect(v.ok).toBe(true);
    expect(v.message.startsWith("⚠")).toBe(true);
    expect(v.message).toContain("RUNNING");
    // The consequence has to be in the sentence: an operator scanning output
    // needs to see that collection stopped, not just that a container went.
    expect(v.message).toContain("OFF");
  });

  test("a stopped container that is removed is an ordinary tick", () => {
    const v = instanceJobsRemovalOutcome("stopped", "absent", 0);
    expect(v.ok).toBe(true);
    expect(v.message.startsWith("✓")).toBe(true);
    expect(v.message).toContain("not running");
  });

  test("nothing there says nothing was there", () => {
    const v = instanceJobsRemovalOutcome("absent", "absent", 0);
    expect(v.ok).toBe(true);
    expect(v.message).toContain("no container was present");
    // Must not claim a removal it did not perform.
    expect(v.message).not.toContain("removed");
  });

  test("the three outcomes are distinguishable from each other", () => {
    // The whole defect was that two of them were byte-identical.
    const msgs = [
      instanceJobsRemovalOutcome("running", "absent", 0).message,
      instanceJobsRemovalOutcome("stopped", "absent", 0).message,
      instanceJobsRemovalOutcome("absent", "absent", 0).message,
    ];
    expect(new Set(msgs).size).toBe(3);
  });

  test("a container still present afterwards fails, even at exit code 0", () => {
    for (const after of ["running", "stopped"] as const) {
      const v = instanceJobsRemovalOutcome("running", after, 0);
      expect(v.ok).toBe(false);
      expect(v.message).toContain("still polling");
    }
  });

  test("a non-zero removal fails", () => {
    const v = instanceJobsRemovalOutcome("running", "absent", 1);
    expect(v.ok).toBe(false);
  });

  test("'could not tell' is never reported as success, on either side", () => {
    // `docker inspect` exits 1 for a missing container AND for a dead daemon.
    // Reading the second as the first is how a ✓ gets printed over a container
    // that is still polling.
    expect(instanceJobsRemovalOutcome("unknown", "unknown", 0).ok).toBe(false);
    expect(instanceJobsRemovalOutcome("running", "unknown", 0).ok).toBe(false);
  });
});
