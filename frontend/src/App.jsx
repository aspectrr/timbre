import { createSignal, onCleanup, onMount, Show } from "solid-js";
import Guide from "./Guide.jsx";
import { createJob, getStatus, resumeJob, downloadFile, loadKey, mintKey, listJobs } from "./api.js";

// ponytail: one-slot localStorage — fine until a user needs a job history list.
const JOB_KEY = "styleclone:job_id";
const LOSS_KEY = "styleclone:loss";   // [step,loss] points, persisted across refresh
const EVAL_KEY = "styleclone:eval";   // [step,eval_loss] points

// append [step,val] to a persisted signal, deduping consecutive same-step
// points. Writes localStorage so a refresh rebuilds the full curve.
function _push(key, setter, step, val) {
  setter((p) => p.length && p[p.length - 1][0] === step ? p
    : (() => { const n = [...p, [step, val]];
               localStorage.setItem(key, JSON.stringify(n)); return n; })());
}

const STAGES = ["ingesting", "curating", "synthesizing", "training", "exporting", "done"];
const LABELS = {
  ingesting: "Ingest", curating: "Curate", synthesizing: "Synthesize",
  training: "Train", exporting: "Export", done: "Done",
  queued: "Queued", error: "Error",
};

const ACCEPTED = [".mbox", ".eml", ".txt", ".md", ".docx", ".pdf"];
const isAccepted = (name) =>
  ACCEPTED.some((ext) => name.toLowerCase().endsWith(ext));

// Resolve a file's effective name from its relative path. The macOS Mail
// bundle stores its data in a bare file named `mbox` (no extension); the
// backend routes by extension, so rename it to the enclosing `<bundle>.mbox`.
// Finds the nearest ancestor dir ending in .mbox so nesting works too.
const _resolveName = (name, relPath) => {
  if (name.toLowerCase() === "mbox" && relPath) {
    const segs = relPath.split("/").filter(Boolean);
    const bundle = segs.slice(0, -1).reverse()
      .find((s) => s.toLowerCase().endsWith(".mbox"));
    if (bundle) return bundle;
  }
  return name;
};
// map a raw file + its relative path to a (possibly renamed) accepted File
const _resolveFile = (file, relPath) => {
  const name = _resolveName(file.name, relPath);
  if (!isAccepted(name)) return null;
  return name === file.name ? file : new File([file], name, { type: file.type });
};

// ── drag-and-drop folder/bundle traversal ────────────────────────────────
// A macOS Mail export is a `.mbox` *bundle directory* (Finder shows one icon).
// The file picker can't select a folder, but a DROP can read into it via the
// FileSystem Entry API. We walk any dropped folder and pull out accepted
// files (renaming the inner `mbox`). ponytail: webkitGetAsEntry only —
// covers Chrome/Edge/Safari/Firefox; the folder picker handles the rest.
const _readEntries = (reader) => new Promise((resolve) => {
  const out = []; // readEntries returns in batches — loop until empty
  const step = () => reader.readEntries((batch) =>
    batch.length ? (out.push(...batch), step()) : resolve(out));
  step();
});
const _entryFile = (entry) => new Promise((res) => entry.file(res));

async function _walk(entry, dirPath = "") {
  if (entry.isFile) {
    const file = await _entryFile(entry);
    const rel = dirPath ? `${dirPath}/${file.name}` : file.name;
    const got = _resolveFile(file, rel);
    return got ? [got] : [];
  }
  if (entry.isDirectory) {
    const here = dirPath ? `${dirPath}/${entry.name}` : entry.name;
    const sub = await _readEntries(entry.createReader());
    const out = [];
    for (const e of sub) out.push(...(await _walk(e, here)));
    return out;
  }
  return [];
}

export default function App() {
  const [author, setAuthor] = createSignal("");
  const [synth, setSynth] = createSignal("anthropic/claude-opus-4.8");
  const [files, setFiles] = createSignal([]);
  const [dragging, setDragging] = createSignal(false);

  const savedId = localStorage.getItem(JOB_KEY);
  const [tab, setTab] = createSignal(savedId ? "train" : "guide");
  const [jobId, setJobId] = createSignal(savedId);
  const [status, setStatus] = createSignal(null);
  // conn: "ok" | "reconnecting" | "gone" — survives transient blips without
  // wiping the saved job. The old code cleared localStorage on any non-stage
  // response, so one Fly cold-start 503 = Guide tab.
  const [conn, setConn] = createSignal("ok");
  const [submitErr, setSubmitErr] = createSignal("");
  const [pickHint, setPickHint] = createSignal("");
  const [busy, setBusy] = createSignal(false);
  // loss history for the live chart — persisted to localStorage so a refresh
  // rebuilds the full curve, not just points after the reload.
  const [lossPts, setLossPts] = createSignal(
    JSON.parse(localStorage.getItem(LOSS_KEY) || "[]"));
  const [evalPts, setEvalPts] = createSignal(
    JSON.parse(localStorage.getItem(EVAL_KEY) || "[]"));
  let pollTimer;
  let runTimer;

  // One-time API key reveal: minted on first visit, persisted to localStorage,
  // shown once here. _fetch re-mints transparently if a saved key ever 401s.
  const [revealedKey, setRevealedKey] = createSignal("");

  // An agent (sharing this browser's API key) may start a job out-of-band.
  // If we have no tracked job, look for an active one under this key and adopt
  // it so the user sees their agent's run here. watchForRuns keeps looking
  // while idle, so a run started moments later pops in live.
  const adoptActiveRun = async () => {
    try {
      const jobs = await listJobs();
      const active = (jobs || []).find(
        (j) => j.stage && j.stage !== "done" && j.stage !== "error");
      if (active) {
        localStorage.setItem(JOB_KEY, active.job_id);
        setJobId(active.job_id);
        poll(active.job_id);
        return true;
      }
    } catch { /* transient — keep watching */ }
    return false;
  };

  const watchForRuns = () => {
    clearTimeout(runTimer);
    runTimer = setTimeout(async () => {
      if (jobId()) return;              // adopted/submitted — stop watching
      if (!await adoptActiveRun()) watchForRuns();
    }, 5000);
  };

  onMount(async () => {
    if (!loadKey()) {
      try { setRevealedKey(await mintKey()); }
      catch { /* network down — the first request will mint lazily on 401 */ }
    }
    if (jobId()) {
      poll(jobId());
    } else if (await adoptActiveRun()) {
      // adopted an agent-started run; poll already running
    } else {
      watchForRuns();
    }
  });

  // append — a second pick/drop adds to the list, doesn't wipe the first.
  // reset the input value after so onChange always fires (without this, picking
  // a file with the same name as a prior pick is a no-op — value unchanged).
  // macOS Mail exports .mbox as a *bundle directory* (Finder shows one icon,
  // but it's a folder). The file picker can't select a directory, so it fires
  // Two click paths: a FILE picker (flat .mbox/.eml/.txt/.md — e.g. a Google
  // Takeout export) and a FOLDER picker (macOS Mail .mbox bundles). A native
  // dialog can't mix files and folders, so the standard fix is two pickers.
  // Drag-and-drop (onDrop) accepts both in one motion.
  const onPickFiles = (e) => {
    const picked = Array.from(e.target.files)
      .filter((f) => isAccepted(f.name));
    setPickHint(picked.length ? "" : "Choose .mbox, .eml, .txt, .md, .docx, or .pdf files.");
    if (picked.length) setFiles((p) => [...p, ...picked]);
    e.target.value = "";
  };
  const onPickFolder = (e) => {
    const picked = Array.from(e.target.files);
    const resolved = picked
      .map((f) => _resolveFile(f, f.webkitRelativePath || f.name))
      .filter(Boolean);
    setPickHint(resolved.length ? ""
      : "No writing files (.mbox, .eml, .txt, .md, .docx, .pdf) found in that folder.");
    if (resolved.length) setFiles((p) => [...p, ...resolved]);
    e.target.value = "";
  };
  const onDrop = (e) => {
    e.preventDefault(); setDragging(false);
    // Capture entries SYNCHRONOUSLY — the DataTransferItemList is invalidated
    // once the handler awaits. Then walk async.
    const entries = [...(e.dataTransfer?.items ?? [])]
      .filter((it) => it.kind === "file")
      .map((it) => it.webkitGetAsEntry?.())
      .filter(Boolean);
    if (entries.length) {
      (async () => {
        const files = (await Promise.all(entries.map(_walk))).flat();
        if (files.length) setFiles((p) => [...p, ...files]);
      })();
    } else if (e.dataTransfer?.files?.length) {
      setFiles((p) => [...p, ...Array.from(e.dataTransfer.files)]);
    }
  };

  const submit = async () => {
    if (!files().length) { alert("Choose at least one file."); return; }
    setBusy(true); setSubmitErr("");
    let j;
    try {
      j = await createJob(author(), synth(), files());
    } catch (e) {
      setBusy(false);
      setSubmitErr(e.message || "Could not reach the server.");
      return;
    }
    setBusy(false);
    if (j.error) { setSubmitErr(j.error); return; }
    clearTimeout(runTimer);             // we have our own job now — stop watching
    localStorage.setItem(JOB_KEY, j.job_id);
    setJobId(j.job_id);
    setTab("train");
    poll(j.job_id);
  };

  const poll = (id) => {
    clearTimeout(pollTimer);
    getStatus(id).then((s) => {
      if (!s || !s.stage) {
        // Unexpected payload (not a clean 404) — keep the job, retry softly.
        setConn("reconnecting");
        pollTimer = setTimeout(() => poll(id), 3000);
        return;
      }
      setConn("ok");
      setStatus(s);
      // accumulate loss points for the live chart (dedup by step, persist)
      if (s.train_step != null && s.train_loss != null)
        _push(LOSS_KEY, setLossPts, s.train_step, s.train_loss);
      if (s.train_step != null && s.eval_loss != null)
        _push(EVAL_KEY, setEvalPts, s.train_step, s.eval_loss);
      if (s.stage !== "done" && s.stage !== "error") {
        pollTimer = setTimeout(() => poll(id), 2000);
      }
    }).catch((e) => {
      if (e.status === 404) {
        // Backend confirmed the job is gone (purged / ephemeral disk).
        setConn("gone");
        localStorage.removeItem(JOB_KEY);
        setJobId(null);
        setStatus(null);
        return;
      }
      // Network blip / cold-start 5xx / bad JSON — DO NOT drop the job.
      setConn("reconnecting");
      pollTimer = setTimeout(() => poll(id), 3000);
    });
  };

  const retry = async () => {
    if (!jobId()) return;
    await resumeJob(jobId());
    poll(jobId());
  };

  const isActive = () => jobId() && cur() && cur() !== "done" && cur() !== "error";

  // abandon this job in the UI (the backend keeps running). clears the saved
  // id so a refresh won't resurrect it, and resets the form + chart.
  const reset = () => {
    if (isActive() && !confirm("A job is still training. Abandon it and start over?")) return;
    clearTimeout(pollTimer);
    localStorage.removeItem(JOB_KEY);
    localStorage.removeItem(LOSS_KEY);
    localStorage.removeItem(EVAL_KEY);
    setJobId(null);
    setStatus(null);
    setConn("ok");
    setLossPts([]);
    setEvalPts([]);
    setSubmitErr("");
    watchForRuns();                     // cleared to start over — resume watching
  };

  onCleanup(() => { clearTimeout(pollTimer); clearTimeout(runTimer); });

  const st = () => status() || {};
  const cur = () => st().stage;
  const curIdx = () => STAGES.indexOf(cur());
  const stats = () => {
    const s = st();
    const out = [];
    if (s.n_samples != null) out.push(<><b>{s.n_samples}</b> samples</>);
    if (s.n_curated != null) out.push(<><b>{s.n_curated}</b> kept</>);
    if (s.n_pairs != null) out.push(<><b>{s.n_pairs}</b> pairs</>);
    if (s.train_step != null) out.push(<><b>step {s.train_step}</b></>);
    if (s.train_loss != null) out.push(<><b>{s.train_loss}</b> loss</>);
    if (s.eval_loss != null) out.push(<><b>{s.eval_loss}</b> eval loss</>);
    return out;
  };

  // hand-rolled monochrome sparkline — no chart lib. loss line + faint
  // vertical guides at eval steps. null until 2+ points exist.
  const chart = () => {
    const pts = lossPts();
    if (pts.length < 2) return null;
    const W = 100, H = 28;
    const ys = pts.map(([, l]) => l);
    const lo = Math.min(...ys), hi = Math.max(...ys), span = hi - lo || 1;
    const xs = pts.map(([s]) => s);
    const x0 = xs[0], xr = (xs[xs.length - 1] - x0) || 1;
    const xy = ([s, l]) =>
      [((s - x0) / xr) * W, H - ((l - lo) / span) * (H - 4) - 2];
    const line = pts.map(xy)
      .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
    return (
      <div class="loss" aria-label="training loss">
        <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
          {evalPts().map(([s], i) => (
            <line x1={((s - x0) / xr) * W} y1="0"
              x2={((s - x0) / xr) * W} y2={H}
              stroke="currentColor" stroke-width="0.5" opacity="0.18"
              vector-effect="non-scaling-stroke" />
          ))}
          <polyline points={line} fill="none" stroke="currentColor"
            vector-effect="non-scaling-stroke" />
        </svg>
        <div class="loss-legend">loss</div>
      </div>
    );
  };

  return (
    <div class="wrap">
      <nav class="topnav">
        <span class="brand" onClick={() => setTab("guide")}>
          <svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <rect x="0" y="13" width="3" height="7" />
            <rect x="5.7" y="6" width="3" height="14" />
            <rect x="11.3" y="9" width="3" height="11" />
            <rect x="17" y="14" width="3" height="6" />
          </svg>
          <span class="wm">TIMBRE</span>
        </span>
        <span class="menu" onClick={() => setTab(tab() === "guide" ? "train" : "guide")}>
          {tab() === "guide" ? "Train →" : "← Guide"}
        </span>
      </nav>

      <header class="hero">
        <h1>Write<br />like you.</h1>
        <p class="lede">
          Train a model on your own sent email. Get a file you run locally —
          free, private, yours. No data leaves your machine after download.
        </p>
      </header>

      <div class="tabs">
        <button class={`tab ${tab() === "guide" ? "active" : ""}`} onClick={() => setTab("guide")}>Guide</button>
        <button class={`tab ${tab() === "train" ? "active" : ""}`} onClick={() => setTab("train")}>Train<Show when={isActive()}><span class="live" /></Show></button>
      </div>

      <Show when={tab() === "guide"}>
        <Guide />
      </Show>

      <Show when={tab() === "train"}>
        <Show when={revealedKey()}>
          <div class="key-reveal">
            <span>Your API key — saved in this browser, shown once</span>
            <code>{revealedKey()}</code>
            <button type="button" onClick={() => setRevealedKey("")}>Dismiss</button>
          </div>
        </Show>
        <Show when={!jobId()}>
        <section style={{ "padding-top": "0" }}>
          <p class="label">Train</p>
          <h2 class="section-title">New job</h2>

          <div class="field">
            <label>Your email addresses <span class="opt">(only needed for .mbox/.eml)</span></label>
            <textarea value={author()} onInput={(e) => setAuthor(e.target.value)}
              placeholder="youremailaddress1@gmail.com, youremail2@icloud.com" />
          </div>
          <div class="field">
            <label>Teacher model</label>
            <select value={synth()} onChange={(e) => setSynth(e.target.value)}>
              <option value="anthropic/claude-opus-4.8">Claude Opus — best quality</option>
              <option value="anthropic/claude-sonnet-4">Claude Sonnet — balanced</option>
              <option value="google/gemini-2.5-flash">Gemini Flash — cheapest</option>
            </select>
          </div>
          <div class="field">
            <label>Your writing (.mbox, .eml, .txt, .md, .docx, .pdf)</label>
            <div class={`drop ${dragging() ? "over" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}>
              Drag a file or a mailbox folder here
              <div class="pick">
                <button type="button" class="pick-btn"
                  onClick={(e) => { e.stopPropagation(); document.getElementById("file").click(); }}>Choose files</button>
                <button type="button" class="pick-btn"
                  onClick={(e) => { e.stopPropagation(); document.getElementById("folder").click(); }}>Choose folder</button>
              </div>
              <input id="file" type="file" multiple hidden
                accept=".mbox,.eml,.txt,.md,.docx,.pdf" onChange={onPickFiles} />
              <input id="folder" type="file" webkitdirectory directory multiple hidden
                onChange={onPickFolder} />
            </div>
            <div class="flist">
              {files().map((f, i) => (
                <span>{f.name}
                  <button type="button" class="x" aria-label={`remove ${f.name}`} onClick={() => setFiles((p) => p.filter((_, j) => j !== i))}>×</button>
                </span>
              ))}
            </div>
            <Show when={pickHint()}>
              <div class="errbox" style={{ "margin-top": "16px" }}>{pickHint()}</div>
            </Show>
          </div>
          <button class="train-btn" disabled={busy()} onClick={submit}>
            {busy() && <span class="spin" aria-hidden="true" />}
            {busy() ? "Starting" : "Train →"}
          </button>
        </section>
        </Show>

        <Show when={jobId()}>
          <section style={{ "padding-top": "0" }}>
            <p class="label">Progress</p>
            <div class="stepper">
              {STAGES.map((name, i) => {
                let cls = "";
                if (cur() === "done" || i < curIdx()) cls = "done";
                else if (i === curIdx()) cls = "active";
                return <div class={`step ${cls}`}>{LABELS[name]}</div>;
              })}
            </div>
            <div class="bar"><i style={{ width: `${st().progress_pct || 0}%` }} /></div>
            <div class="msg">{st().message || LABELS[cur()] || ""}</div>
            <Show when={conn() === "reconnecting"}>
              <div class="msg" style={{ opacity: 0.6 }}>reconnecting…</div>
            </Show>
            <div class="stats">{stats()}</div>

            {chart()}

            <Show when={cur() === "done"}>
              <div class="dl">
                <a href="#" onClick={(e) => { e.preventDefault(); downloadFile(jobId(), "adapter.gguf").catch(() => alert("Download failed.")); }}>↓ adapter.gguf</a>
                <a href="#" onClick={(e) => { e.preventDefault(); downloadFile(jobId(), "Modelfile").catch(() => alert("Download failed.")); }}>↓ Modelfile</a>
              </div>
            </Show>

            <Show when={cur() === "error"}>
              <div class="errbox">{st().error || st().message || "Unknown error"}</div>
              <div style={{ "margin-top": "32px" }}>
                <button onClick={retry}>Retry →</button>
              </div>
            </Show>

            <div class="reset-row">
              <button class="reset" onClick={reset}>Start new</button>
            </div>
          </section>
        </Show>
      </Show>
    </div>
  );
}
