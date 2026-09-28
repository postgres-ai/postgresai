import { afterEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import * as instances from "../lib/instances";

let dir: string | undefined;
afterEach(() => { if (dir) rmSync(dir, { recursive: true, force: true }); });

test.each(["postgres", "postgresql"])("local-install shared target builder preserves %s probe channel binding and collector TLS", (scheme) => {
  dir = mkdtempSync(`${tmpdir()}/local-install-collector-`);
  const connStr = `${scheme}://monitor:password@test.pg.clickhouse.cloud:5432/postgres?sslmode=require&channel_binding=require`;
  const { instance, probeConfig } = instances.buildLocalInstallTarget("test", connStr);
  instances.addInstanceToFile(`${dir}/instances.yml`, instance);
  expect(instances.loadInstances(`${dir}/instances.yml`)[0].conn_str).toBe(
    `${scheme}://monitor:password@test.pg.clickhouse.cloud:5432/postgres?sslmode=require`,
  );
  expect(probeConfig).toEqual(instances.buildClientConfig(connStr, { connectionTimeoutMillis: 10000 }));
  expect(probeConfig.enableChannelBinding).toBe(true);
  expect(probeConfig.ssl).toEqual({ rejectUnauthorized: false });
});
