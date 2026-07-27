/**
 * Regression test: RPC responses must be decoded as UTF-8 across chunk
 * boundaries.
 *
 * postRpc() used to accumulate the HTTP response as `data += chunk` on the
 * raw stream without res.setEncoding("utf8"). Each Buffer chunk was decoded
 * independently, so a multibyte UTF-8 character whose bytes were split
 * across two TCP chunks was corrupted into U+FFFD replacement characters
 * (mojibake). This affects any RPC response containing non-ASCII text:
 * query texts, object names, org names with accents/CJK/emoji, etc.
 *
 * The test drives convertCheckupReportJsonToMarkdown (the thinnest exported
 * wrapper over postRpc) against a local http.Server that deliberately splits
 * the JSON body inside a multibyte sequence, flushing each half in a
 * separate socket write so the client receives it as two chunks.
 */
import { describe, test, expect, afterAll } from "bun:test";
import * as http from "http";
import type { AddressInfo } from "net";
import { convertCheckupReportJsonToMarkdown } from "../lib/checkup-api";

// Non-ASCII payload: accented Latin (2-byte), CJK (3-byte), emoji (4-byte).
const ORIGINAL = "Résumé — 日本語テスト 🐘 café";

const servers: http.Server[] = [];

afterAll(() => {
  for (const s of servers) s.close();
});

function startSplittingServer(splitOffset: (body: Buffer) => number): Promise<number> {
  const server = http.createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      const body = Buffer.from(JSON.stringify({ result: ORIGINAL }), "utf8");
      const at = splitOffset(body);
      res.writeHead(200, {
        "Content-Type": "application/json",
        "Content-Length": String(body.length),
      });
      // First half ends INSIDE a multibyte UTF-8 sequence.
      res.write(body.subarray(0, at));
      // Flush the first half as its own TCP segment, then send the rest a
      // moment later so the client necessarily sees two separate chunks.
      setTimeout(() => {
        res.end(body.subarray(at));
      }, 30);
    });
  });
  servers.push(server);
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      resolve((server.address() as AddressInfo).port);
    });
  });
}

/** Byte offset that lands strictly inside the first multibyte character. */
function offsetInsideFirstMultibyte(body: Buffer): number {
  for (let i = 0; i < body.length; i++) {
    // UTF-8 lead byte of a multibyte sequence: 0b11xxxxxx
    if ((body[i] & 0b1100_0000) === 0b1100_0000) return i + 1;
  }
  throw new Error("no multibyte character found in body");
}

describe("postRpc UTF-8 decoding across chunk boundaries", () => {
  test("multibyte character split across two response chunks is not corrupted", async () => {
    const port = await startSplittingServer(offsetInsideFirstMultibyte);
    const resp = await convertCheckupReportJsonToMarkdown({
      apiKey: "test-key",
      apiBaseUrl: `http://127.0.0.1:${port}`,
      checkId: "H002",
      jsonPayload: { dummy: true },
    });
    expect(resp).toBe(ORIGINAL);
  });

  test("split inside a 4-byte emoji sequence is not corrupted", async () => {
    const port = await startSplittingServer((body) => {
      const idx = body.indexOf(Buffer.from("🐘", "utf8"));
      if (idx < 0) throw new Error("emoji not found in body");
      return idx + 2; // middle of the 4-byte sequence
    });
    const resp = await convertCheckupReportJsonToMarkdown({
      apiKey: "test-key",
      apiBaseUrl: `http://127.0.0.1:${port}`,
      checkId: "H002",
      jsonPayload: { dummy: true },
    });
    expect(resp).toBe(ORIGINAL);
  });
});
