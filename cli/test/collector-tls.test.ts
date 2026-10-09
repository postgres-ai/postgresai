import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import dns from "node:dns";
import { resolve } from "node:path";
import { verifyCollectorTls } from "../lib/instances";

const fixtures = resolve(import.meta.dir, "fixtures/tls");
let lookup: ReturnType<typeof spyOn>;

beforeEach(() => {
  // Keep localhost resolution offline, including runtimes that request all addresses.
  lookup = spyOn(dns, "lookup").mockImplementation(((host: string, options: dns.LookupOptions, callback: Function) => {
    if (host !== "localhost") throw new Error(`Unexpected DNS lookup: ${host}`);
    if (options.all) callback(null, [{ address: "127.0.0.1", family: 4 }]);
    else callback(null, "127.0.0.1", 4);
  }) as unknown as typeof dns.lookup);
});
afterEach(() => lookup.mockRestore());

async function probe(cert: string, useCa: boolean, answer = "auth"): Promise<string[]> {
  const server = Bun.spawn(["node", "-e", String.raw`
    const assert = require("node:assert/strict");
    const { readFileSync } = require("node:fs");
    const { createServer } = require("node:net");
    const { createSecureContext, TLSSocket } = require("node:tls");
    const [fixtures, cert, answer] = process.argv.slice(1);
    const secureContext = createSecureContext({
      cert: readFileSync(fixtures + "/" + cert + ".pem"),
      key: readFileSync(fixtures + "/" + cert + "-key.pem"),
    });
    const sockets = new Set();
    const seen = [];
    const server = createServer(socket => {
      sockets.add(socket);
      socket.on("error", () => {});
      socket.on("close", () => sockets.delete(socket));
      let request = Buffer.alloc(0);
      const sslRequest = chunk => {
        request = Buffer.concat([request, chunk]);
        if (request.length < 8) return;
        assert.deepEqual(request, Buffer.from([0, 0, 0, 8, 4, 210, 22, 47]));
        seen.push("ssl");
        socket.removeListener("data", sslRequest);
        socket.write("S");
        const tls = new TLSSocket(socket, { isServer: true, secureContext });
        sockets.add(tls);
        tls.on("error", () => {});
        tls.on("close", () => sockets.delete(tls));
        let startup = Buffer.alloc(0);
        const startupMessage = chunk => {
          startup = Buffer.concat([startup, chunk]);
          if (startup.length < 4 || startup.length < startup.readInt32BE(0)) return;
          assert.equal(startup.readInt32BE(4), 196608);
          seen.push("startup", answer);
          tls.removeListener("data", startupMessage);
          if (answer === "close") { tls.end(); return; }
          const body = Buffer.from("SFATAL\0C28P01\0Mauthentication failed\0\0");
          const header = Buffer.alloc(5);
          header[0] = 69;
          header.writeInt32BE(body.length + 4, 1);
          tls.end(Buffer.concat([header, body]));
        };
        tls.on("data", startupMessage);
      };
      socket.on("data", sslRequest);
    });
    server.listen(0, "127.0.0.1", () => console.log(server.address().port));
    process.stdin.once("data", () => {
      for (const socket of sockets) socket.destroy();
      server.close(() => { console.log(JSON.stringify(seen)); process.exit(0); });
    });
  `, fixtures, cert, answer], { stdin: "pipe", stdout: "pipe", stderr: "pipe" });
  const stderr = new Response(server.stderr).text();
  const reader = server.stdout.getReader();
  try {
    const { value, done } = await reader.read();
    if (done) throw new Error(await stderr);
    const port = Number(new TextDecoder().decode(value).trim());
    const url = new URL(`postgres://fixture:fixture@localhost:${port}/postgres?sslmode=verify-full`);
    if (useCa) url.searchParams.set("sslrootcert", `${fixtures}/ca.pem`);
    await verifyCollectorTls(url.toString());
    server.stdin.write("stop\n");
    let output = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      output += new TextDecoder().decode(value);
    }
    const seen = JSON.parse(output.trim());
    expect(await server.exited).toBe(0);
    expect(await stderr).toBe("");
    return seen;
  } finally {
    reader.releaseLock();
    server.kill();
    await server.exited;
  }
}

test.each(["auth", "close"])("collector TLS accepts a trusted localhost certificate after %s", async (answer) => {
  await expect(probe("localhost", true, answer)).resolves.toEqual(["ssl", "startup", answer]);
  expect(lookup).toHaveBeenCalled();
});

test("collector TLS refuses a trusted certificate for another host", async () => {
  await expect(probe("other-name", true)).rejects.toMatchObject({ code: "ERR_TLS_CERT_ALTNAME_INVALID" });
  expect(lookup).toHaveBeenCalled();
});

test("collector TLS refuses the fixture CA when only system roots are used", async () => {
  const error = await probe("localhost", false).then(() => null, (error) => error);
  expect(error).toBeInstanceOf(Error);
  expect(["UNABLE_TO_GET_ISSUER_CERT_LOCALLY", "UNABLE_TO_VERIFY_LEAF_SIGNATURE", "SELF_SIGNED_CERT_IN_CHAIN"]).toContain(error.code);
  expect(lookup).toHaveBeenCalled();
});
