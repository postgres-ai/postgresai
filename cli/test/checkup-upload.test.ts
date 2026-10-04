import { afterEach, expect, test } from "bun:test";
import { createCheckupReport, uploadCheckupReportJson } from "../lib/checkup-api";
import { saveCheckupReport } from "../lib/connect";
import { callRpc } from "../lib/joe";
import { ORG_ID_HEADER, setActiveOrgScope } from "../lib/org-scope";

type Call = { fn: string; body: Record<string, unknown>; headers: Headers };
const token = "tok";
const reports = { H001: { checkId: "H001", value: 1 } };
afterEach(() => setActiveOrgScope(undefined));

async function withBackend(run: (url: string, calls: Call[]) => Promise<void>, answer?: (call: Call) => unknown) {
  const calls: Call[] = [];
  const server = Bun.serve({ hostname: "127.0.0.1", port: 0, async fetch(req) {
    const call = { fn: new URL(req.url).pathname.split("/").at(-1)!, body: await req.json() as Record<string, unknown>, headers: req.headers };
    calls.push(call);
    return Response.json(answer ? answer(call) : call.fn === "checkup_report_create" ? { report_id: "9" } : { report_chunck_id: "4" });
  } });
  try { await run(server.url.origin, calls); } finally { server.stop(true); }
}

const scopedRpc = (apiBaseUrl: string) => <T>(fn: string, body: Record<string, unknown>) =>
  callRpc<T>({ apiKey: token, apiBaseUrl, fn, body, operation: fn, orgScope: { id: 6333, source: "--org-id" } });

test("CLI and connect send the same report wire bodies through their own org-scoped transports", async () => {
  setActiveOrgScope({ id: 5225, source: "--org-id" });
  await withBackend(async (url, calls) => {
    const auth = { apiKey: token, apiBaseUrl: url };
    const { reportId } = await createCheckupReport({ ...auth, project: "db/app" });
    expect(await uploadCheckupReportJson({ ...auth, reportId, filename: "H001.json", checkId: "H001", jsonText: JSON.stringify(reports.H001, null, 2) })).toEqual({ reportChunkId: 4 });
    expect(await saveCheckupReport(scopedRpc(url), token, "db/app", reports)).toBe(reportId);
    expect(calls[0].body).toEqual(calls[2].body);
    expect(calls[1].body).toEqual(calls[3].body);
    expect(calls.map(c => c.headers.get(ORG_ID_HEADER))).toEqual(["5225", "5225", "6333", "6333", "6333"]);
    expect(calls[4].body).toEqual({ access_token: token, report_id: 9, status: "completed" });
  });
});

test("upload accepts either backend chunk spelling, including a null legacy field", async () => {
  for (const reply of [{ report_chunck_id: 2 }, { report_chunk_id: 3 }, { report_chunck_id: null, report_chunk_id: 3 }]) {
    await withBackend(async (apiBaseUrl) => {
      const result = await uploadCheckupReportJson({ apiKey: token, apiBaseUrl, reportId: 9, filename: "H001.json", checkId: "H001", jsonText: "{}" });
      expect(result.reportChunkId).toBe(reply.report_chunck_id ?? reply.report_chunk_id!);
    }, () => reply);
  }
});

test("creation preserves optional status and rejects a missing or non-finite ID", async () => {
  await withBackend(async (apiBaseUrl, calls) => {
    await createCheckupReport({ apiKey: token, apiBaseUrl, project: "p", status: "pending" });
    expect(calls[0].body.status).toBe("pending");
  });
  for (const reply of [{}, { report_id: "Infinity" }, { report_id: 0 }]) {
    await withBackend(async (apiBaseUrl) => {
      await expect(createCheckupReport({ apiKey: token, apiBaseUrl, project: "p" })).rejects.toThrow("checkup_report_create");
      await expect(saveCheckupReport(scopedRpc(apiBaseUrl), token, "p", {})).rejects.toThrow("checkup_report_create");
    }, () => reply);
  }
});

test("a rejected chunk retains the backend message and marks the connect report failed", async () => {
  await withBackend(async (apiBaseUrl, calls) => {
    await expect(saveCheckupReport(scopedRpc(apiBaseUrl), token, "p", reports)).rejects.toThrow("Upload rejected");
    expect(calls.at(-1)!.body.status).toBe("failed");
  }, call => call.fn === "checkup_report_create" ? { report_id: 9 } : { message: "Upload rejected" });
});
