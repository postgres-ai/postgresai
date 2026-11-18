import { describe, expect, test } from "bun:test";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

const repo = path.resolve(import.meta.dir, "../..");

// Run the real host callers with local executables; no requests or Docker calls.
describe("host callers use this install's ports", () => {
  const cases: { name: string; file?: string; port?: string; override?: string; expected: string; prefix?: string }[] = [
    { name: "defaults", file: undefined, port: undefined, override: undefined, expected: "default" },
    { name: ".env", file: "13001", port: undefined, override: undefined, expected: "13001" },
    { name: "quoted .env", file: '"13002"', port: undefined, override: undefined, expected: "13002" },
    { name: "inline comment", file: "13004 # this install", expected: "13004" },
    { name: "export and whitespace", prefix: "  export ", file: "'13005' # this install", expected: "13005" },
    { name: "empty .env", file: "", expected: "default" },
    { name: "environment wins", file: "13001", port: "13003", override: undefined, expected: "13003" },
    { name: "empty environment uses default", file: "13001", port: "", override: undefined, expected: "default" },
    { name: "explicit URL wins", file: "13001", port: "13003", override: "http://chosen:9000", expected: "override" },
  ];

  for (const caller of ["reporter", "grafana"] as const) {
    for (const c of cases) {
      test(`${caller}: ${c.name}`, () => {
        const dir = fs.mkdtempSync(path.join(os.tmpdir(), "pgai-host-ports-test-"));
        try {
          const portKey = caller === "reporter" ? "PGAI_MON_PORT" : "GRAFANA_PORT";
          const urlKey = caller === "reporter" ? "PGAI_MON_URL" : "GRAFANA_URL";
          const bin = path.join(dir, "bin");
          fs.mkdirSync(bin);
          if (c.file !== undefined) fs.writeFileSync(path.join(dir, ".env"), `${c.prefix ?? ""}${portKey}=${c.file}\n`);
          const env: Record<string, string> = { PATH: `${bin}:${process.env.PATH}`, HOME: dir };
          if (c.port !== undefined) env[portKey] = c.port;
          if (c.override !== undefined) env[urlKey] = c.override;
          let script: string;
          if (caller === "reporter") {
            fs.mkdirSync(path.join(dir, "scripts"));
            script = path.join(dir, "scripts/run_reporter_local.sh");
            fs.copyFileSync(path.join(repo, "scripts/run_reporter_local.sh"), script);
            fs.writeFileSync(path.join(bin, "python"), '#!/bin/sh\nwhile [ "$#" -gt 0 ]; do if [ "$1" = --mon-url ]; then printf "URL=%s\\n" "$2"; exit 0; fi; shift; done\nexit 1\n', { mode: 0o700 });
          } else {
            script = path.join(dir, "grafana.sh");
            const source = fs.readFileSync(path.join(repo, "tests/e2e.cli.sh"), "utf8");
            // Load its functions without starting the E2E scenario.
            fs.writeFileSync(script, source.slice(0, source.indexOf("# Ensure instances.yml")) + '\ngrafana_auth_header() { :; }\ngrafana_datasources_json\n');
            fs.writeFileSync(path.join(bin, "curl"), '#!/bin/sh\nfor arg do case "$arg" in http*) printf "URL=%s\\n" "$arg" ;; esac; done\n', { mode: 0o700 });
            env.DOCKER_HOST = "tcp://docker-host:2375";
          }
          const r = Bun.spawnSync(["bash", script], { cwd: dir, env });
          expect(r.exitCode).toBe(0);
          const port = c.expected === "default" ? (caller === "reporter" ? "8765" : "3000") : c.expected;
          const base = c.expected === "override" ? c.override : `http://${caller === "reporter" ? "127.0.0.1" : "docker-host"}:${port}`;
          expect(r.stdout.toString()).toContain(`URL=${base}${caller === "grafana" ? "/api/datasources" : ""}\n`);
        } finally {
          fs.rmSync(dir, { recursive: true, force: true });
        }
      });
    }
  }
});
