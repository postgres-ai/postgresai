import { describe, expect, test } from "bun:test";
import { cleanOrphanVolumes, composeProjectName, DEFAULT_GRAFANA_PORT, DEFAULT_PGAI_MON_PORT, grafanaBindNotice, hostPort, parseProjectPs, projectPsArgs } from "../lib/stack";

// #510: the stack's containers have no fixed names, so the CLI finds them by
// their compose project and service labels, and its ports come from .env.
describe("containers by compose project", () => {
  test("lists one project's containers by label, running only unless asked", () => {
    expect(projectPsArgs("pgai-v4", false)).toEqual([
      "ps", "--filter", "label=com.docker.compose.project=pgai-v4",
      "--format", '{{.Label "com.docker.compose.service"}}\t{{.ID}}',
    ]);
    expect(projectPsArgs("pgai-v4", true)).toContain("-a");
    expect(projectPsArgs("pgai-v4", true).join(" ")).not.toContain("name=");
  });

  test("the project name is compose's: from the directory, normalized as compose does", () => {
    // `docker compose config` in a directory named "_-My.Proj" says "name: myproj".
    expect(composeProjectName("_-My.Proj")).toBe("myproj");
    expect(composeProjectName("monitoring", " pgai-v4 ")).toBe("pgai-v4");
  });

  test("maps each service to its first container", () => {
    const m = parseProjectPs("grafana\tabc\npgai-mon\tdef\n\ngrafana\tzzz\n");
    expect([...m]).toEqual([["grafana", "abc"], ["pgai-mon", "def"]]);
    expect(parseProjectPs("")).toEqual(new Map());
  });
});

describe("host ports from .env", () => {
  test("a port number is taken; unset, blank or junk falls back", () => {
    expect(hostPort("3001", DEFAULT_GRAFANA_PORT)).toBe(3001);
    expect(hostPort('"18765"', DEFAULT_PGAI_MON_PORT)).toBe(18765);
    for (const v of [undefined, null, "", " ", "abc", "0", "70000", "127.0.0.1:3001"]) {
      expect(hostPort(v, DEFAULT_GRAFANA_PORT)).toBe(3000);
    }
  });

  test("the Grafana bind notice names the port in use", () => {
    expect(grafanaBindNotice(undefined)).toContain("127.0.0.1:3000");
    expect(grafanaBindNotice(undefined, 3001)).toContain("ssh -L 3001:localhost:3001");
  });
});

describe("mon clean's leftover volumes", () => {
  const candidates = ["monitoring_grafana_data", "monitoring_victoria_metrics_data", "postgres_ai_configs_grafana_data"];
  const existing = [...candidates, "pgai-v4_grafana_data"];

  test("another project keeps the default install's monitoring_* volumes", () => {
    expect(cleanOrphanVolumes("pgai-v4", existing, candidates)).toEqual({
      remove: ["postgres_ai_configs_grafana_data"],
      kept: ["monitoring_grafana_data", "monitoring_victoria_metrics_data"],
    });
  });

  test("the monitoring project removes them as before; absent volumes are not listed", () => {
    expect(cleanOrphanVolumes("monitoring", existing, candidates)).toEqual({ remove: candidates, kept: [] });
    expect(cleanOrphanVolumes("pgai-v4", [], candidates)).toEqual({ remove: [], kept: [] });
  });
});
