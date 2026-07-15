"""FastAPI app + routes. Imported only inside the container (web_app()), so
fastapi (not installed locally) never loads at deploy time.

Keeping routes here also fixes the closure type-hint resolution: FastAPI
resolves annotations against module globals, so UploadFile/Form must be
importable at THIS module's top level — which they are, in the container.
"""
from __future__ import annotations

import shutil

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .common import (jobs_vol, DATA_VOL, status_store, get_status,
                     new_job_id, job_dir, STAGES, MODEL_LABELS)
from .orchestrator import run_job


def create_app() -> FastAPI:
    web = FastAPI(title="Style Clone")

    @web.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    @web.post("/api/jobs")
    async def create_job(
        author: str = Form(...),
        synth_model: str = Form("anthropic/claude-opus-4.8"),
        base_model: str = Form("llama3.2-3b"),
        files: list[UploadFile] = File(...),
    ) -> JSONResponse:
        job_id = new_job_id()
        indir = job_dir(job_id) / "input"
        indir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for f in files:
            if not f.filename:
                continue
            with (indir / f.filename).open("wb") as out:
                shutil.copyfileobj(f.file, out)
            saved += 1
        jobs_vol.commit()

        if saved == 0:
            return JSONResponse({"error": "no files uploaded"}, status_code=400)

        addrs = [a.strip() for a in author.replace("\n", ",").split(",")
                 if a.strip()]
        status_store[job_id] = {
            "job_id": job_id, "stage": "queued", "message": "starting...",
            "progress_pct": 0, "n_files": saved, "synth_model": synth_model,
            "base_model": base_model, "author": addrs,
        }
        run_job.spawn(job_id, addrs, synth_model, base_model)
        return JSONResponse({"job_id": job_id})

    @web.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> JSONResponse:
        st = get_status(job_id)
        if not st:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(st)

    @web.get("/api/jobs/{job_id}/download/{which}")
    def download(job_id: str, which: str) -> Response:
        if which not in ("adapter.gguf", "Modelfile"):
            return JSONResponse({"error": "bad file"}, status_code=400)
        jobs_vol.reload()
        path = job_dir(job_id) / which
        if not path.exists():
            return JSONResponse({"error": "not ready"}, status_code=404)
        media = ("application/octet-stream" if which.endswith(".gguf")
                 else "text/plain")
        return Response(path.read_bytes(), media_type=media,
                        headers={"Content-Disposition":
                                 f'attachment; filename="{which}"'})

    return web


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Style Clone</title>
<style>
  :root{--bg:#0b0d10;--panel:#15181d;--ink:#e6e9ef;--mut:#8b93a1;--acc:#4f8cff;
        --acc2:#34d399;--err:#f87171;--bd:#232831;--code:#0e1116}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:780px;margin:0 auto;padding:36px 20px 90px}
  h1{font-size:27px;font-weight:700;margin:0 0 4px;letter-spacing:-.02em}
  .sub{color:var(--mut);margin:0 0 24px;font-size:15px}
  a{color:var(--acc)}
  .tabs{display:flex;gap:4px;border-bottom:1px solid var(--bd);margin-bottom:24px}
  .tab{padding:11px 18px;cursor:pointer;color:var(--mut);border:0;background:none;
    font:inherit;font-size:14px;font-weight:600;border-bottom:2px solid transparent;
    margin-bottom:-1px;border-radius:8px 8px 0 0}
  .tab:hover{color:var(--ink)}
  .tab.active{color:var(--ink);border-bottom-color:var(--acc)}
  .panel{display:none}.panel.active{display:block}
  .card{background:var(--panel);border:1px solid var(--bd);border-radius:14px;
    padding:22px;margin-bottom:18px}
  .card h2{font-size:18px;margin:0 0 6px;font-weight:650}
  .card h2+ p{margin-top:0}
  .hint{font-size:13px;color:var(--mut);margin:14px 0 0}
  .hint a{cursor:pointer;text-decoration:underline}
  label{display:block;font-size:13px;color:var(--mut);margin:0 0 6px;font-weight:500}
  input[type=text],select,textarea{width:100%;background:var(--code);color:var(--ink);
    border:1px solid var(--bd);border-radius:9px;padding:10px 12px;font:inherit}
  textarea{min-height:60px;resize:vertical}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .field{margin-bottom:16px}
  .drop{border:1.5px dashed var(--bd);border-radius:11px;padding:26px;text-align:
    center;color:var(--mut);cursor:pointer;transition:.15s}
  .drop:hover,.drop.over{border-color:var(--acc);color:var(--ink);background:#10141b}
  .drop input{display:none}
  .flist{margin:10px 0 0;font-size:13px;color:var(--mut)}
  .flist span{display:inline-block;background:var(--code);border:1px solid var(--bd);
    border-radius:6px;padding:2px 8px;margin:2px 4px 2px 0}
  button{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:12px 22px;
    font:inherit;font-weight:600;cursor:pointer;width:100%}
  button:disabled{opacity:.5;cursor:default}
  .stepper{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 14px}
  .step{flex:1;min-width:90px;padding:9px 10px;border-radius:9px;background:var(--code);
    border:1px solid var(--bd);font-size:12px;text-align:center;color:var(--mut)}
  .step.active{border-color:var(--acc);color:var(--ink)}
  .step.done{border-color:var(--acc2);color:var(--acc2)}
  .bar{height:7px;background:var(--code);border-radius:5px;overflow:hidden;margin:6px 0 10px}
  .bar>i{display:block;height:100%;width:0;background:var(--acc);transition:width .4s}
  .msg{color:var(--mut);font-size:13px;min-height:20px}
  .stats{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0;font-size:13px;color:var(--mut)}
  .stats b{color:var(--ink);font-weight:600}
  .dl{display:flex;gap:10px;margin-top:14px}
  .dl a{flex:1;text-align:center;background:var(--code);border:1px solid var(--bd);
    color:var(--ink);border-radius:9px;padding:11px;text-decoration:none;font-weight:600}
  .dl a.gguf{border-color:var(--acc2);color:var(--acc2)}
  .errbox{background:#2a1414;border:1px solid var(--err);color:var(--err);border-radius:9px;
    padding:12px;margin-top:10px;font-size:13px;white-space:pre-wrap}
  .hide{display:none}
  code,.cmd{background:var(--code);padding:1px 6px;border-radius:4px;font-size:13px;
    font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
  .cmd{display:flex;align-items:center;justify-content:space-between;gap:8px;
    padding:9px 12px;margin:6px 0;overflow-x:auto}
  .cmd code{background:none;padding:0;white-space:nowrap}
  .copy{font-size:11px;color:var(--mut);cursor:pointer;border:1px solid var(--bd);
    border-radius:5px;padding:2px 7px;background:transparent;flex-shrink:0}
  .copy:hover{color:var(--ink)}
  .copy.ok{color:var(--acc2);border-color:var(--acc2)}
  ol.steps{padding-left:22px;margin:12px 0}
  ol.steps li{margin:0 0 10px;color:var(--ink)}
  ol.steps li::marker{color:var(--acc);font-weight:700}
  .pill{display:inline-block;background:#10141b;border:1px solid var(--bd);border-radius:6px;
    padding:1px 7px;font-size:12px;color:var(--mut);margin-left:6px}
  .acc{color:var(--acc2)}
  details{border:1px solid var(--bd);border-radius:10px;padding:0;margin:10px 0;
    overflow:hidden;background:#10141b}
  details summary{padding:13px 16px;cursor:pointer;font-weight:600;color:var(--ink);
    list-style:none;display:flex;align-items:center;gap:8px}
  details summary::-webkit-details-marker{display:none}
  details summary::before{content:"›";color:var(--mut);transition:.15s;display:inline-block}
  details[open] summary::before{transform:rotate(90deg)}
  details .inner{padding:0 16px 16px}
  .note{font-size:13px;color:var(--mut);background:#10141b;border-left:3px solid var(--acc);
    border-radius:0 8px 8px 0;padding:10px 12px;margin:10px 0}
  .toc{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 16px}
  .toc a{font-size:13px;background:var(--code);border:1px solid var(--bd);border-radius:7px;
    padding:6px 11px;text-decoration:none;color:var(--mut)}
  .toc a:hover{color:var(--ink)}
</style></head><body><div class="wrap">
  <h1>Style Clone</h1>
  <p class="sub">Train a model that writes like you. Upload your email, get a model you run locally — free, private, yours.</p>

  <div class="tabs">
    <button class="tab active" data-tab="guide">Guide</button>
    <button class="tab" data-tab="train">Train</button>
  </div>

  <!-- ═══ TRAIN PANEL ═══ -->
  <div class="panel" id="train">

    <div class="card">
      <div class="field"><label>Your email addresses <span class="pill">used to find what YOU wrote</span></label>
        <textarea id="author" placeholder="cpfeifer@madcactus.org, collinpfeifer@icloud.com"></textarea></div>
      <div class="row">
        <div class="field"><label>Teacher model</label>
          <select id="synth">
            <option value="anthropic/claude-opus-4.8">Claude Opus (best quality)</option>
            <option value="anthropic/claude-sonnet-4">Claude Sonnet (balanced)</option>
            <option value="google/gemini-2.5-flash">Gemini Flash (cheapest)</option>
          </select></div>
        <div class="field"><label>Base model to fine-tune</label>
          <select id="base">
            <option value="llama3.2-3b">Llama 3.2 3B (runs on 16GB Mac)</option>
          </select></div>
      </div>
      <div class="field"><label>Your writing files (.mbox, .eml, .txt, .md)</label>
        <div class="drop" id="drop">
          Drop files here or click to choose
          <input type="file" id="file" multiple accept=".mbox,.eml,.txt,.md">
        </div>
        <div class="flist" id="flist"></div></div>
      <button id="go">Train my style model</button>
      <p class="hint">Not sure how to get your email into a file? <a onclick="showTab('guide')">Read the guide</a> — it covers Gmail, iCloud, Outlook, and more.</p>
    </div>

    <div class="card hide" id="jobcard">
      <div class="stepper" id="stepper"></div>
      <div class="bar"><i id="bar"></i></div>
      <div class="msg" id="msg"></div>
      <div class="stats" id="stats"></div>
      <div class="dl hide" id="dl">
        <a class="gguf" id="dl-gguf" download>&#11015; adapter.gguf</a>
        <a id="dl-mod" download>&#11015; Modelfile</a>
      </div>
      <div class="errbox hide" id="errbox"></div>
    </div>

  </div>

  <!-- ═══ GUIDE PANEL ═══ -->
  <div class="panel active" id="guide">

    <div class="toc">
      <a onclick="document.getElementById('g1').scrollIntoView({behavior:'smooth'})">1 · Get your email</a>
      <a onclick="document.getElementById('g2').scrollIntoView({behavior:'smooth'})">2 · Train</a>
      <a onclick="document.getElementById('g3').scrollIntoView({behavior:'smooth'})">3 · Use your model</a>
    </div>

    <div class="card" id="g1">
      <h2>1 · Get your email as a file</h2>
      <p style="color:var(--mut);margin:0 0 6px">This app learns from emails <b style="color:var(--ink)">you</b> wrote. Export your "Sent" mail into a file (called an mbox) and upload it. Nothing is deleted from your account — exporting just makes a copy.</p>

      <details open>
        <summary>Gmail — via Google Takeout <span class="pill">easiest</span></summary>
        <div class="inner">
          <ol class="steps">
            <li>Go to <a href="https://takeout.google.com" target="_blank">takeout.google.com</a> and sign in.</li>
            <li>Click <b>"Deselect all"</b> at the top of the product list.</li>
            <li>Scroll to <b>Mail</b> and tick its checkbox.</li>
            <li>Click <b>"All Mail data included"</b> → untick "All Mail" → tick just <b>"Sent"</b>. (Keeping it to Sent makes the file small and focused on what you wrote.)</li>
            <li>Scroll to the bottom → <b>"Next step"</b>.</li>
            <li>Leave it as <b>"Send download link via email"</b>, <b>"Export once"</b>, <b>.zip</b> → click <b>"Create export"</b>.</li>
            <li>Wait for Google's email (a few minutes for small inboxes, up to hours for large ones). Click the link and download the <code>.zip</code>.</li>
            <li>Unzip it. Inside you'll find a file like <code>Sent.mbox</code> (or <code>All mail Including Spam and Trash.mbox</code>).</li>
            <li>Upload that <code>.mbox</code> file here.</li>
          </ol>
          <div class="note">Exporting is a copy — your Gmail account, labels, and messages are untouched.</div>
        </div>
      </details>

      <details>
        <summary>iCloud — via Thunderbird</summary>
        <div class="inner">
          <p style="color:var(--mut)">iCloud has no "download your mail" button, so we use the free Thunderbird app to pull mail in, then export it.</p>
          <ol class="steps">
            <li>Install <a href="https://www.thunderbird.net" target="_blank">Thunderbird</a> (free).</li>
            <li><b>Make an app-specific password</b> (Apple requires this): go to <a href="https://account.apple.com" target="_blank">account.apple.com</a> → sign in → <b>App-Specific Passwords</b> → generate one, label it "Thunderbird". <i>(You'll need two-factor auth turned on.)</i> Copy that password.</li>
            <li>Open Thunderbird → add your mail account. Enter your name, your full <code>@icloud.com</code> address, and the <b>app-specific password</b> (not your Apple ID password). Thunderbird usually auto-detects the rest.</li>
            <li>If it asks for server settings, use: <b>IMAP server</b> <code>imap.mail.me.com</code>, <b>port</b> 993, <b>SSL</b> on.</li>
            <li>Let Thunderbird sync your folders (can take a while for big mailboxes).</li>
            <li>Install the free <b>ImportExportTools NG</b> add-on: Thunderbird menu → <i>Add-ons and Themes</i> → search "ImportExportTools NG" → Add.</li>
            <li>Right-click your <b>Sent</b> folder → <i>ImportExportTools NG</i> → <b>Export folder</b> → pick where to save.</li>
            <li>You'll get a <code>.mbox</code> file. Upload it here.</li>
          </ol>
          <div class="note">For the username, try your full email address first; if that fails, use just the part before <code>@</code>.</div>
        </div>
      </details>

      <details>
        <summary>Outlook, Yahoo, or other IMAP — via Thunderbird</summary>
        <div class="inner">
          <ol class="steps">
            <li>Install <a href="https://www.thunderbird.net" target="_blank">Thunderbird</a> (free).</li>
            <li>Most providers require an <b>app password</b> now (not your normal password). Generate one:
              <br>• <b>Outlook:</b> account.microsoft.com → Security → Advanced security options → App passwords
              <br>• <b>Yahoo:</b> account-info → Account security → Generate app password</li>
            <li>Add your account in Thunderbird using that app password. Auto-detect handles Outlook/Yahoo settings.</li>
            <li>Install the <b>ImportExportTools NG</b> add-on (Add-ons and Themes → search → Add).</li>
            <li>Right-click <b>Sent</b> → <i>ImportExportTools NG</i> → <b>Export folder</b> → save the <code>.mbox</code>.</li>
            <li>Upload the <code>.mbox</code> here.</li>
          </ol>
        </div>
      </details>

      <details>
        <summary>What about notes, docs, or other writing?</summary>
        <div class="inner">
          <p style="color:var(--mut)">Save them as <code>.txt</code> or <code>.md</code> files and upload those instead of (or alongside) an mbox. Each file becomes a writing sample. The more of your authentic voice, the better.</p>
          <p style="color:var(--mut)">Good sources: sent email, personal notes, blog posts, Slack exports, letters, journals. Aim for at least 50–100 messages/documents for a recognizable style.</p>
        </div>
      </details>
    </div>

    <div class="card" id="g2">
      <h2>2 · Train your model</h2>
      <ol class="steps">
        <li>Enter the email address <b>you send from</b> (so the app keeps only your writing).</li>
        <li>Pick a teacher model — <span class="acc">Claude Opus</span> gives the best result; Flash is cheaper.</li>
        <li>Upload your file(s).</li>
        <li>Click <b>Train my style model</b>.</li>
      </ol>
      <div class="note">Takes about 15–20 minutes. You'll see live progress: how many samples were kept, how many training pairs were generated, and the loss curve. Come back anytime — the page remembers your job.</div>
    </div>

    <div class="card" id="g3">
      <h2>3 · Run your model on your computer</h2>
      <p style="color:var(--mut);margin:0 0 8px">When training finishes, download <b>both</b> files: <code>adapter.gguf</code> and <code>Modelfile</code>. Then:</p>
      <ol class="steps">
        <li>Install <a href="https://ollama.com" target="_blank">Ollama</a> (free, Mac/Windows/Linux).</li>
        <li>Open <b>Terminal</b> (Mac: <code>Cmd + Space</code> → type "Terminal" → Enter).</li>
        <li>Download the base model once:
          <div class="cmd"><code>ollama pull llama3.2:3b</code><span class="copy">copy</span></div>
        </li>
        <li>Put <code>adapter.gguf</code> and <code>Modelfile</code> in the same folder — for example a new folder on your Desktop called <code>my-model</code>.</li>
        <li>In Terminal, go to that folder and build your model:
          <div class="cmd"><code>cd ~/Desktop/my-model</code><span class="copy">copy</span></div>
          <div class="cmd"><code>ollama create my-style -f Modelfile</code><span class="copy">copy</span></div>
        </li>
        <li>Run it:
          <div class="cmd"><code>ollama run my-style</code><span class="copy">copy</span></div>
        </li>
        <li>Type a request — e.g. <i>"write a quick follow-up to a client who hasn't replied"</i> — and it writes in your voice.</li>
      </ol>
      <div class="note">That's it. Anytime you want to write in your own voice, run <code>ollama run my-style</code> and ask it to draft whatever you need.</div>
    </div>

  </div>
</div>

<script>
const STAGES=["ingesting","curating","synthesizing","training","exporting","done"];
const LABELS={ingesting:"Ingest",curating:"Curate",synthesizing:"Synthesize",
  training:"Train",exporting:"Export",done:"Done",queued:"Queued",error:"Error"};
const $=id=>document.getElementById(id);

/* ── tabs ── */
function showTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.tab===name));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('active',p.id===name));
  window.scrollTo({top:0});
}
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>showTab(t.dataset.tab));

/* ── copy buttons ── */
document.addEventListener('click',e=>{
  const c=e.target.closest('.copy');if(!c)return;
  const code=c.parentElement.querySelector('code');if(!code)return;
  navigator.clipboard.writeText(code.textContent).then(()=>{
    c.textContent='copied';c.classList.add('ok');
    setTimeout(()=>{c.textContent='copy';c.classList.remove('ok');},1400);
  });
});

/* ── file picker ── */
const drop=$("drop"),file=$("file"),flist=$("flist");
let picked=[];
function renderFiles(){flist.innerHTML=picked.map(f=>`<span>${f}</span>`).join("");}
drop.onclick=()=>file.click();
file.onchange=e=>{picked=[...picked,...[...e.target.files].map(f=>f.name)];renderFiles();};
drop.ondragover=e=>{e.preventDefault();drop.classList.add("over");};
drop.ondragleave=()=>drop.classList.remove("over");
drop.ondrop=e=>{e.preventDefault();drop.classList.remove("over");
  picked=[...picked,...[...e.dataTransfer.files].map(f=>f.name)];
  file.files=e.dataTransfer.files;renderFiles();};

/* ── submit + poll ── */
$("go").onclick=async()=>{
  if(!file.files.length){alert("Choose at least one file.");return;}
  const fd=new FormData();
  fd.append("author",$("author").value);
  fd.append("synth_model",$("synth").value);
  fd.append("base_model",$("base").value);
  for(const f of file.files)fd.append("files",f);
  $("go").disabled=true;$("go").textContent="Starting job…";
  let r=await fetch("/api/jobs",{method:"POST",body:fd});
  let j=await r.json();
  $("go").disabled=false;$("go").textContent="Train my style model";
  if(j.error){alert(j.error);return;}
  $("jobcard").classList.remove("hide");
  poll(j.job_id);
};
function poll(id){
  fetch(`/api/jobs/${id}`).then(r=>r.json()).then(s=>{render(id,s);
    if(s.stage!=="done"&&s.stage!=="error")setTimeout(()=>poll(id),2000);});
}
function render(id,s){
  const cur=s.stage;
  $("stepper").innerHTML=STAGES.map(st=>{
    const idx=STAGES.indexOf(cur);
    const my=STAGES.indexOf(st);
    let cls="";if(my<idx||cur==="done")cls="done";
    else if(my===idx)cls="active";
    return `<div class="step ${cls}">${LABELS[st]}</div>`;}).join("");
  $("bar").style.width=(s.progress_pct||0)+"%";
  $("msg").textContent=s.message||LABELS[cur]||"";
  let st=[];
  if(s.n_samples!=null)st.push(`<b>${s.n_samples}</b> samples`);
  if(s.n_curated!=null)st.push(`<b>${s.n_curated}</b> kept`);
  if(s.n_pairs!=null)st.push(`<b>${s.n_pairs}</b> pairs`);
  if(s.train_loss!=null)st.push(`<b>${s.train_loss}</b> loss`);
  if(s.eval_loss!=null)st.push(`<b>${s.eval_loss}</b> eval loss`);
  $("stats").innerHTML=st.join(" · ");
  if(cur==="done"){
    $("dl").classList.remove("hide");
    $("dl-gguf").href=`/api/jobs/${id}/download/adapter.gguf`;
    $("dl-mod").href=`/api/jobs/${id}/download/Modelfile`;
    $("msg").innerHTML=`&#10003; Ready! Save <b>both</b> files in one folder, then follow <a onclick="showTab('guide');document.getElementById('g3').scrollIntoView()">step 3 of the guide</a>.`;
  }
  if(cur==="error"){$("errbox").classList.remove("hide");
    $("errbox").textContent=s.error||s.message||"Unknown error";}
}
</script></body></html>
"""
