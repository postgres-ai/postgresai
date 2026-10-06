/** The report wire contract; callers supply their own transport and org scope. */
type Rpc = (fn: string, body: Record<string, unknown>) => Promise<unknown>;

function responseId(fn: string, response: unknown, ...keys: string[]): number {
  const answer = (response ?? {}) as Record<string, unknown>;
  const id = Number(keys.map((key) => answer[key]).find((value) => value != null));
  if (!Number.isFinite(id) || id <= 0) {
    throw new Error(typeof answer.message === "string" ? answer.message : `Unexpected ${fn} response: ${JSON.stringify(response)}`);
  }
  return id;
}

export async function createCheckupReport(rpc: Rpc, params: {
  apiKey: string; project: string; status?: string;
}): Promise<{ reportId: number }> {
  const fn = "checkup_report_create";
  const response = await rpc(fn, {
    access_token: params.apiKey, project: params.project,
    ...(params.status ? { status: params.status } : {}),
  });
  return { reportId: responseId(fn, response, "report_id") };
}

export async function uploadCheckupReportJson(rpc: Rpc, params: {
  apiKey: string; reportId: number; filename: string; checkId: string; jsonText: string;
}): Promise<{ reportChunkId: number }> {
  const fn = "checkup_report_file_post";
  const response = await rpc(fn, {
    access_token: params.apiKey, checkup_report_id: params.reportId,
    filename: params.filename, check_id: params.checkId, data: params.jsonText,
    type: "json", generate_issue: true,
  });
  // The platform's original spelling has a typo; both remain supported.
  return { reportChunkId: responseId(fn, response, "report_chunck_id", "report_chunk_id") };
}
