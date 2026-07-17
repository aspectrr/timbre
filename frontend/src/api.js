// API base: set VITE_API_URL at build time (Render static), or empty for dev proxy.
const BASE = import.meta.env.VITE_API_URL || "";

// Throw on !ok so callers can branch on e.status (404 = job truly gone).
// A 200 with unparseable body (e.g. a Fly 503 HTML page during cold start)
// also throws — the poll loop treats that as transient and keeps the job.
async function _json(r) {
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(data.error || "request failed"), { status: r.status });
  return data;
}

export async function createJob(author, synthModel, files) {
  const fd = new FormData();
  fd.append("author", author);
  fd.append("synth_model", synthModel);
  for (const f of files) fd.append("files", f);
  return _json(await fetch(`${BASE}/api/jobs`, { method: "POST", body: fd }));
}

export async function getStatus(jobId) {
  return _json(await fetch(`${BASE}/api/jobs/${jobId}`));
}

export async function resumeJob(jobId) {
  const r = await fetch(`${BASE}/api/jobs/${jobId}/resume`, { method: "POST" });
  return r.json();
}

export function downloadUrl(jobId, which) {
  return `${BASE}/api/jobs/${jobId}/download/${which}`;
}
