import { createSignal, onCleanup, Show } from "solid-js";
import Guide from "./Guide.jsx";
import { createJob, getStatus, resumeJob, downloadUrl } from "./api.js";

const STAGES = ["ingesting", "curating", "synthesizing", "training", "exporting", "done"];
const LABELS = {
  ingesting: "Ingest", curating: "Curate", synthesizing: "Synthesize",
  training: "Train", exporting: "Export", done: "Done",
  queued: "Queued", error: "Error",
};

export default function App() {
  const [tab, setTab] = createSignal("guide");
  const [author, setAuthor] = createSignal("");
  const [synth, setSynth] = createSignal("anthropic/claude-opus-4.8");
  const [files, setFiles] = createSignal([]);
  const [dragging, setDragging] = createSignal(false);

  const [jobId, setJobId] = createSignal(null);
  const [status, setStatus] = createSignal(null);
  const [busy, setBusy] = createSignal(false);
  let pollTimer;

  const onPick = (e) => setFiles([...e.target.files]);
  const onDrop = (e) => {
    e.preventDefault(); setDragging(false);
    setFiles([...e.target.files]);
  };

  const submit = async () => {
    if (!files().length) { alert("Choose at least one file."); return; }
    setBusy(true);
    const j = await createJob(author(), synth(), files());
    setBusy(false);
    if (j.error) { alert(j.error); return; }
    setJobId(j.job_id);
    setTab("train");
    poll(j.job_id);
  };

  const poll = (id) => {
    clearTimeout(pollTimer);
    getStatus(id).then((s) => {
      setStatus(s);
      if (s.stage !== "done" && s.stage !== "error") {
        pollTimer = setTimeout(() => poll(id), 2000);
      }
    });
  };

  const retry = async () => {
    if (!jobId()) return;
    await resumeJob(jobId());
    poll(jobId());
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
    if (s.eval_loss != null) out.push(<><b>{s.eval_loss}</b> eval loss</>);
    return out;
  };

  return (
    <div class="wrap">
      <nav class="topnav">
        <span>Style Clone</span>
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
        <button class={`tab ${tab() === "train" ? "active" : ""}`} onClick={() => setTab("train")}>Train</button>
      </div>

      <Show when={tab() === "guide"}>
        <Guide />
      </Show>

      <Show when={tab() === "train"}>
        <section style={{ "padding-top": "0" }}>
          <p class="label">Train</p>
          <h2 class="section-title">New job</h2>

          <div class="field">
            <label>Your email addresses</label>
            <textarea value={author()} onInput={(e) => setAuthor(e.target.value)}
              placeholder="cpfeifer@madcactus.org, collinpfeifer@icloud.com" />
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
              {files().map((f) => <span>{f.name}</span>)}
            </div>
          </div>
          <button disabled={busy()} onClick={submit}>
            {busy() ? "Starting" : "Train →"}
          </button>
        </section>

        <Show when={jobId()}>
          <section>
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
            <div class="stats">{stats()}</div>

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
          </section>
        </Show>
      </Show>
    </div>
  );
}
