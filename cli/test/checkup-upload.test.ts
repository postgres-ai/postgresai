import { afterEach, expect, test } from "bun:test";
import { createCheckupReport, uploadCheckupReportJson } from "../lib/checkup-api";
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

test("the CLI sends the report wire bodies with its org scope", async () => {
  setActiveOrgScope({ id: 5225, source: "--org-id" });
  await withBackend(async (url, calls) => {
    const auth = { apiKey: token, apiBaseUrl: url };
    const { reportId } = await createCheckupReport({ ...auth, project: "db/app" });
    expect(await uploadCheckupReportJson({ ...auth, reportId, filename: "H001.json", checkId: "H001", jsonText: JSON.stringify(reports.H001, null, 2) })).toEqual({ reportChunkId: 4 });
    expect(calls[0].body).toEqual({ access_token: token, project: "db/app" });
    expect(calls[1].body).toEqual({ access_token: token, checkup_report_id: 9, filename: "H001.json", check_id: "H001", data: JSON.stringify(reports.H001, null, 2), type: "json", generate_issue: true });
    expect(calls.map(c => c.headers.get(ORG_ID_HEADER))).toEqual(["5225", "5225"]);
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
    }, () => reply);
  }
});
