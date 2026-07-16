// API base: set VITE_API_URL at build time (Render static), or empty for dev proxy.
const BASE = import.meta.env.VITE_API_URL || "";

export async function createJob(author, synthModel, files) {
  const fd = new FormData();
  fd.append("author", author);
  fd.append("synth_model", synthModel);
  for (const f of files) fd.append("files", f);
  const r = await fetch(`${BASE}/api/jobs`, { method: "POST", body: fd });
  return r.json();
}

export async function getStatus(jobId) {
  const r = await fetch(`${BASE}/api/jobs/${jobId}`);
  return r.json();
}

export async function resumeJob(jobId) {
  const r = await fetch(`${BASE}/api/jobs/${jobId}/resume`, { method: "POST" });
  return r.json();
}

export function downloadUrl(jobId, which) {
  return `${BASE}/api/jobs/${jobId}/download/${which}`;
}
