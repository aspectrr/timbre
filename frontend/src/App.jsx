import { createSignal, onCleanup, onMount, Show } from "solid-js";
import Guide from "./Guide.jsx";
import { createJob, getStatus, resumeJob, downloadUrl } from "./api.js";

// ponytail: one-slot localStorage — fine until a user needs a job history list.
const JOB_KEY = "styleclone:job_id";

const STAGES = ["ingesting", "curating", "synthesizing", "training", "exporting", "done"];
const LABELS = {
  ingesting: "Ingest", curating: "Curate", synthesizing: "Synthesize",
  training: "Train", exporting: "Export", done: "Done",
  queued: "Queued", error: "Error",
};

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
  const [busy, setBusy] = createSignal(false);
  // loss history for the live chart — in-session only. Cleared on reset.
  // After a refresh the graph rebuilds from subsequent polls (the backend
  // stores only the latest loss, not a history). upgrade: persist the array.
  const [lossPts, setLossPts] = createSignal([]);
  const [evalPts, setEvalPts] = createSignal([]);
  let pollTimer;

  onMount(() => { if (jobId()) poll(jobId()); });

  // append — a second pick/drop adds to the list, doesn't wipe the first.
  // reset the input value after so onChange always fires (without this, picking
  // a file with the same name as a prior pick is a no-op — value unchanged).
  const onPick = (e) => {
    setFiles((p) => [...p, ...Array.from(e.target.files)]);
    e.target.value = "";
  };
  const onDrop = (e) => {
    e.preventDefault(); setDragging(false);
    setFiles((p) => [...p, ...e.dataTransfer.files]);
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
      // accumulate loss points for the live chart (dedup by step)
      if (s.train_step != null && s.train_loss != null) {
        setLossPts((p) => p.length && p[p.length - 1][0] === s.train_step
          ? p : [...p, [s.train_step, s.train_loss]]);
      }
      if (s.eval_loss != null && s.train_step != null) {
        setEvalPts((p) => p.length && p[p.length - 1][0] === s.train_step
          ? p : [...p, [s.train_step, s.eval_loss]]);
      }
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
    setJobId(null);
    setStatus(null);
    setConn("ok");
    setLossPts([]);
    setEvalPts([]);
    setSubmitErr("");
  };

  onCleanup(() => clearTimeout(pollTimer));

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
        <Show when={!jobId()}>
        <section style={{ "padding-top": "0" }}>
          <p class="label">Train</p>
          <h2 class="section-title">New job</h2>

          <div class="field">
            <label>Your email addresses</label>
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
            <label>Your writing files (.mbox, .eml, .txt, .md)</label>
            <div class={`drop ${dragging() ? "over" : ""}`}
              onClick={() => document.getElementById("file").click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}>
              Drop files here or click to choose
              <input id="file" type="file" multiple hidden
                accept=".mbox,.eml,.txt,.md" onChange={onPick} />
            </div>
            <div class="flist">
              {files().map((f, i) => (
                <span>{f.name}
                  <button type="button" class="x" aria-label={`remove ${f.name}`} onClick={() => setFiles((p) => p.filter((_, j) => j !== i))}>×</button>
                </span>
              ))}
            </div>
          </div>
          <button disabled={busy()} onClick={submit}>
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
                <a href={downloadUrl(jobId(), "adapter.gguf")} download>↓ adapter.gguf</a>
                <a href={downloadUrl(jobId(), "Modelfile")} download>↓ Modelfile</a>
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
