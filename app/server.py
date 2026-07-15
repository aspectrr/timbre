"""FastAPI app + routes. Imported only inside the container (web_app()), so
fastapi (not installed locally) never loads at deploy time.

Keeping routes here also fixes the closure type-hint resolution: FastAPI
resolves annotations against module globals, so UploadFile/Form must be
importable at THIS module's top level — which they are, in the container.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import modal
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


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Style Clone</title>
<style>
  :root{--bg:#0b0d10;--panel:#15181d;--ink:#e6e9ef;--mut:#8b93a1;--acc:#4f8cff;
        --acc2:#34d399;--err:#f87171;--bd:#232831}
  *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:760px;margin:0 auto;padding:40px 20px 80px}
  h1{font-size:26px;font-weight:650;margin:0 0 4px;letter-spacing:-.01em}
  .sub{color:var(--mut);margin:0 0 28px}
  .card{background:var(--panel);border:1px solid var(--bd);border-radius:14px;
    padding:22px;margin-bottom:20px}
  label{display:block;font-size:13px;color:var(--mut);margin:0 0 6px}
  input[type=text],select,textarea{width:100%;background:#0e1116;color:var(--ink);
    border:1px solid var(--bd);border-radius:9px;padding:10px 12px;font:inherit}
  textarea{min-height:64px;resize:vertical}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .field{margin-bottom:16px}
  .drop{border:1.5px dashed var(--bd);border-radius:11px;padding:26px;text-align:
    center;color:var(--mut);cursor:pointer;transition:.15s}
  .drop:hover,.drop.over{border-color:var(--acc);color:var(--ink);background:#10141b}
  .drop input{display:none}
  .flist{margin:10px 0 0;font-size:13px;color:var(--mut)}
  .flist span{display:inline-block;background:#10141b;border:1px solid var(--bd);
    border-radius:6px;padding:2px 8px;margin:2px 4px 2px 0}
  button{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:12px 22px;
    font:inherit;font-weight:600;cursor:pointer;width:100%}
  button:disabled{opacity:.5;cursor:default}
  .stepper{display:flex;gap:6px;flex-wrap:wrap;margin:6px 0 14px}
  .step{flex:1;min-width:90px;padding:9px 10px;border-radius:9px;background:#10141b;
    border:1px solid var(--bd);font-size:12px;text-align:center;color:var(--mut)}
  .step.active{border-color:var(--acc);color:var(--ink)}
  .step.done{border-color:var(--acc2);color:var(--acc2)}
  .bar{height:7px;background:#10141b;border-radius:5px;overflow:hidden;margin:6px 0 10px}
  .bar>i{display:block;height:100%;width:0;background:var(--acc);transition:width .4s}
  .msg{color:var(--mut);font-size:13px;min-height:20px}
  .stats{display:flex;gap:18px;flex-wrap:wrap;margin:10px 0;font-size:13px;color:var(--mut)}
  .stats b{color:var(--ink);font-weight:600}
  .dl{display:flex;gap:10px;margin-top:14px}
  .dl a{flex:1;text-align:center;background:#10141b;border:1px solid var(--bd);
    color:var(--ink);border-radius:9px;padding:11px;text-decoration:none;font-weight:600}
  .dl a.gguf{border-color:var(--acc2);color:var(--acc2)}
  .errbox{background:#2a1414;border:1px solid var(--err);color:var(--err);border-radius:9px;
    padding:12px;margin-top:10px;font-size:13px;white-space:pre-wrap}
  .hide{display:none}
  code{background:#0e1116;padding:1px 5px;border-radius:4px;font-size:12px}
</style></head><body><div class="wrap">
  <h1>Style Clone</h1>
  <p class="sub">Upload your writing. Get a model that writes like you, for Ollama.</p>

  <div class="card">
    <div class="field"><label>Your email addresses (for author filtering)</label>
      <textarea id="author" placeholder="cpfeifer@madcactus.org, collinpfeifer@icloud.com"></textarea></div>
    <div class="row">
      <div class="field"><label>Teacher model (synthesis)</label>
        <select id="synth">
          <option value="anthropic/claude-opus-4.8">Claude Opus (best)</option>
          <option value="anthropic/claude-sonnet-4">Claude Sonnet (cheaper)</option>
          <option value="google/gemini-2.5-flash">Gemini Flash (cheapest)</option>
        </select></div>
      <div class="field"><label>Base model to fine-tune</label>
        <select id="base">
          <option value="llama3.2-3b">Llama 3.2 3B (runs on 16GB Mac)</option>
        </select></div>
    </div>
    <div class="field"><label>Writing samples (.mbox, .eml, .txt, .md)</label>
      <div class="drop" id="drop">
        Drop files here or click to choose
        <input type="file" id="file" multiple accept=".mbox,.eml,.txt,.md,.json">
      </div>
      <div class="flist" id="flist"></div></div>
    <button id="go">Train my style model</button>
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
<script>
const STAGES=["ingesting","curating","synthesizing","training","exporting","done"];
const LABELS={ingesting:"Ingest",curating:"Curate",synthesizing:"Synthesize",
  training:"Train",exporting:"Export",done:"Done",queued:"Queued",error:"Error"};
const $=id=>document.getElementById(id);
const drop=$("drop"),file=$("file"),flist=$("flist");
let picked=[];
function renderFiles(){flist.innerHTML=picked.map(f=>`<span>${f}</span>`).join("");}
drop.onclick=()=>file.click();
file.onchange=e=>{picked=[...picked,...[...e.target.files].map(f=>f.name)];renderFiles();};
drop.ondragover=e=>{e.preventDefault();drop.classList.add("over");};
drop.ondragleave=()=>drop.classList.remove("over");
drop.ondrop=e=>{e.preventDefault();drop.classList.remove("over");
  picked=[...picked,...[...e.target.files].map(f=>f.name)];
  file.files=e.dataTransfer.files;renderFiles();};
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
    $("msg").innerHTML=`Ready! Save both files in one folder, then run:<br>`
      +`<code>ollama create my-style -f Modelfile && ollama run my-style</code>`;
  }
  if(cur==="error"){$("errbox").classList.remove("hide");
    $("errbox").textContent=s.error||s.message||"Unknown error";}
}
</script></body></html>
"""
