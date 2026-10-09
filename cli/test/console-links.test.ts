import { describe, expect, test } from "bun:test";
import { connectStatus, grafanaOAuthUrl, healthMatrixUrl, platformDeps, type Database } from "../lib/connect";

// After mon deploy, the first link is the project's Health Matrix in the console;
// the second is Grafana, through its PostgresAI sign-in (generic OAuth).

describe("healthMatrixUrl", () => {
  test("console origin, org alias and project alias, each path segment encoded", () => {
    expect(healthMatrixUrl("https://console.example", "acme", "prod")).toBe("https://console.example/acme/projects/prod/health");
    expect(healthMatrixUrl("https://console.example/", "acme co/x", "my proj#1?")).toBe("https://console.example/acme%20co%2Fx/projects/my%20proj%231%3F/health");
  });
});

describe("grafanaOAuthUrl", () => {
  test("the box's Grafana, signed in with PostgresAI, landing on its home", () => {
    expect(grafanaOAuthUrl("https://abc.pgai.watch")).toBe("https://abc.pgai.watch/login/generic_oauth?redirectTo=%2F");
    expect(grafanaOAuthUrl("https://abc.pgai.watch/")).toBe("https://abc.pgai.watch/login/generic_oauth?redirectTo=%2F");
  });
  test("a path in the address is where it lands", () => {
    expect(grafanaOAuthUrl("https://abc.pgai.watch/d/node?orgId=1&var-db=a b")).toBe(
      `https://abc.pgai.watch/login/generic_oauth?redirectTo=${encodeURIComponent("/d/node?orgId=1&var-db=a%20b")}`,
    );
  });
  test("already a sign-in address, none, or not a URL: as it is", () => {
    const done = "https://abc.pgai.watch/login/generic_oauth?redirectTo=%2F";
    expect(grafanaOAuthUrl(done)).toBe(done);
    expect(grafanaOAuthUrl(null)).toBeNull();
    expect(grafanaOAuthUrl("not a url")).toBe("not a url");
  });
});

const ROW: Database = { id: "i-1", name: "db.example.com/app", provider: "self-managed", status: "active", dashboard_url: "https://abc.pgai.watch", host_metrics: false };

describe("connectStatus", () => {
  test("connected: the Health Matrix is the next step, before Grafana", () => {
    const r = connectStatus({ ...ROW, health_url: "https://console.example/acme/projects/p/health" });
    expect(r.next).toBe("Open https://console.example/acme/projects/p/health");
    expect(Object.keys(r).indexOf("health_url")).toBeLessThan(Object.keys(r).indexOf("dashboard_url"));
  });
  test("without a Health Matrix link: Grafana", () => {
    expect(connectStatus(ROW).next).toBe("Open https://abc.pgai.watch");
    expect("health_url" in connectStatus(ROW)).toBe(false);
  });
});

async function withPlatform(routes: Record<string, unknown>, fn: (deps: ReturnType<typeof platformDeps>, calls: string[]) => Promise<void>) {
  const calls: string[] = [];
  const server = Bun.serve({
    hostname: "127.0.0.1", port: 0,
    fetch(req) {
      const fn = new URL(req.url).pathname.replace(/^.*\/rpc\//, "");
      calls.push(fn);
      return fn in routes ? Response.json(routes[fn]) : new Response("not found", { status: 404 });
    },
  });
  try {
    await fn(platformDeps({ apiKey: "k", apiBaseUrl: `http://127.0.0.1:${server.port}`, uiBaseUrl: "https://console.example" }), calls);
  } finally {
    server.stop(true);
  }
}

describe("platformDeps.links", () => {
  const rows = [
    ROW,
    // Still starting: not yet in its project's active instances; found by the project's name.
    { ...ROW, id: "i-2", name: "Other DB", status: "launch_requested", dashboard_url: null },
  ];
  const projects = [
    { project_id: 1, alias: "db-example-com-app", name: "db.example.com/app", monitoring_instance_ids: ["i-1"] },
    { project_id: 2, alias: "other-db", name: "Other DB", monitoring_instance_ids: [] },
  ];

  test("each row gets its Health Matrix link and the Grafana sign-in address, in that order", async () => {
    await withPlatform({ projects_list: projects, orgs_list: [{ org_id: 7, alias: "acme co", name: "Acme", is_active: true }] }, async (deps) => {
      const [a, b] = await Promise.all(rows.map(deps.links));
      expect(a).toMatchObject({ health_url: "https://console.example/acme%20co/projects/db-example-com-app/health", dashboard_url: "https://abc.pgai.watch/login/generic_oauth?redirectTo=%2F" });
      expect(Object.keys(a!).indexOf("health_url")).toBe(Object.keys(a!).indexOf("dashboard_url") - 1);
      expect(b).toMatchObject({ health_url: "https://console.example/acme%20co/projects/other-db/health", dashboard_url: null });
    });
  });

  test("the org and the projects are looked up once, not on each poll", async () => {
    await withPlatform({ projects_list: projects, orgs_list: [{ org_id: 7, alias: "acme", name: "Acme", is_active: true }] }, async (deps, calls) => {
      for (const row of rows) await deps.links(row);
      for (const row of rows) await deps.links(row);
      expect(calls.filter((c) => c === "orgs_list")).toHaveLength(1);
      expect(calls.filter((c) => c === "projects_list")).toHaveLength(1);
    });
  });

  test("a project not found yet is looked up again on the next poll", async () => {
    await withPlatform({ projects_list: projects, orgs_list: [{ org_id: 7, alias: "acme", name: "Acme", is_active: true }] }, async (deps, calls) => {
      const row = { ...ROW, id: "i-9", name: "new" };
      expect("health_url" in (await deps.links(row))).toBe(false);
      await deps.links(row);
      expect(calls.filter((c) => c === "projects_list")).toHaveLength(2);
    });
  });

  test("when the console links cannot be looked up: Grafana only, no error", async () => {
    await withPlatform({}, async (deps) => {
      const a = await deps.links(ROW);
      expect("health_url" in a!).toBe(false);
      expect(a!.dashboard_url).toBe("https://abc.pgai.watch/login/generic_oauth?redirectTo=%2F");
    });
  });
});
