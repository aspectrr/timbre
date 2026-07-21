// API base: set VITE_API_URL at build time (Render static), or empty for dev proxy.
export const apiBase = () => import.meta.env.VITE_API_URL || "";
const BASE = apiBase();

const API_KEY = "styleclone:api_key";

// ── API key (auto-managed in localStorage; plaintext never re-fetched) ─────
export function loadKey() { return localStorage.getItem(API_KEY); }
function saveKey(k) { localStorage.setItem(API_KEY, k); }
export function clearKey() { localStorage.removeItem(API_KEY); }

// One-time mint. Persists the plaintext (the only time it's available); the
// server stores only its hash and never returns it again.
export async function mintKey(label = "web") {
  const r = await fetch(`${BASE}/api/keys`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ label }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(data.error || "key mint failed"), { status: r.status });
  saveKey(data.key);
  return data.key;
}

function _authHeaders(extra = {}) {
  const k = loadKey();
  return k ? { Authorization: `Bearer ${k}`, ...extra } : { ...extra };
}

// fetch wrapper: injects the Bearer header; on 401 (key gone/invalid) re-mints
// once and retries, so callers never see an auth failure for a saved key.
async function _fetch(path, opts = {}) {
  let res = await fetch(`${BASE}${path}`, { ...opts, headers: _authHeaders(opts.headers) });
  if (res.status === 401) {
    await mintKey();
    res = await fetch(`${BASE}${path}`, { ...opts, headers: _authHeaders(opts.headers) });
  }
  return res;
}

// Throw on !ok so callers can branch on e.status (404 = job truly gone).
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
  return _json(await _fetch("/api/jobs", { method: "POST", body: fd }));
}

export async function getStatus(jobId) {
  return _json(await _fetch(`/api/jobs/${jobId}`));
}

export async function listJobs() {
  return _json(await _fetch("/api/jobs"));
}

export async function resumeJob(jobId) {
  const r = await _fetch(`/api/jobs/${jobId}/resume`, { method: "POST" });
  return r.json();
}

// Downloads need the Bearer header, so a plain <a href> won't work — fetch the
// blob with auth and trigger a download via an object URL.
export async function downloadFile(jobId, which) {
  const r = await _fetch(`/api/jobs/${jobId}/download/${which}`);
  if (!r.ok) throw Object.assign(new Error("download failed"), { status: r.status });
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = which;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
