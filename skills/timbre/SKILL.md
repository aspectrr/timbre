---
name: timbre
description: Clone your writing style into a local Ollama model via the Timbre service. Interview the user to gather their writing (sent email, Obsidian, Word docs, PDFs, plain text), start a training run, and guide them to run the model locally. Activate when the user says "train my voice", "clone my writing", "make a model of how I write", or references Timbre.
---

# Timbre — Train a model that writes like you

Timbre trains a LoRA adapter on the user's own writing, exports a GGUF + Modelfile, and they run it locally in Ollama. Your job is to act as the setup wizard: interview the user for where their writing lives, gather it, start the run, then point them at the web UI to watch.

## Architecture (context, not for the user)

- Backend (FastAPI + DBOS, durable pipeline): `ingest → curate → synthesize → train → finalize`.
- Hosted MCP server at `<TIMBRE_URL>/mcp` exposes the job lifecycle as tools.
- The user's **API key** is the shared identity: a job the agent starts here shows up in the web UI under the same key, so the user watches training live without any hand-off.

## Prerequisites

1. **API key.** Ask the user for theirs (from the "Let your agent do it" section of the web guide). If they don't have one, send them to the guide to generate it — **do not mint a key on their behalf** unless they explicitly ask; a key is how runs are tracked and must belong to them.
2. **Server URL.** Default `https://style-clone-backend.fly.dev`. Confirm with the user if unsure. Call it `<TIMBRE_URL>` below.
3. **MCP server configured.** The guide's config points the agent at `<TIMBRE_URL>/mcp` with the user's bearer key. If the `timbre` MCP tools are NOT present in your environment, fall back to the REST (curl) snippets at the end of each step.

## Procedure — run this as an interview

Don't dump all of this at once. Move through the stages below in order, waiting for the user between each.

### Stage 1 — What kind of writing?

Ask the user where their best writing lives. Present these options:

- **Sent email** (richest source for most people — needs their email address to filter).
- **Obsidian vault / notes** (`.md` — structural noise stripped automatically).
- **Word docs** (`.docx`), **PDFs** (`.pdf`), or **plain text** (`.txt`).

More than one is great. Aim for 50–100+ samples minimum; curation drops low-value ones and the job errors out under ~10 usable samples. Tell the user roughly how much they need based on what they picked.

### Stage 2 — Gather the data (walk them through export)

For each source the user named, give them the exact export steps. **Only cover the sources they picked** — don't recite all of these.

**Sent email — Gmail:**
1. takeout.google.com → sign in → **Deselect all** at the top.
2. Tick **Mail** → "All Mail data included" → untick "All Mail", tick just **Sent**.
3. Next step → "Send download link via email", "Export once", `.zip` → Create export.
4. Wait for Google's email (minutes to hours), download, unzip → `Sent.mbox`.

**Sent email — iCloud:** no direct export; use free Thunderbird + an Apple app-specific password, sync Sent, then ImportExportTools NG → right-click Sent → Export folder → `.mbox`.

**Sent email — Outlook/Yahoo/other IMAP:** Thunderbird + provider app password → ImportExportTools NG → export Sent folder → `.mbox`.

**Obsidian / notes:** copy `.md` files out of the vault (or point the agent at the vault path). Frontmatter, `[[wikilinks]]`, embeds, and tags are stripped automatically.

**Docs / PDFs / plain text:** save or export as `.docx` / `.pdf` / `.txt`. Anything the user actually wrote.

For email, also ask: **"Which email address(es) do you send from?"** That address filters sent mail — without it, email returns a 400.

Wait until the user confirms they have the file(s). If you have filesystem access and they've consented, read them directly; otherwise have the user confirm the path or drag them somewhere you can read.

### Stage 3 — Start the run

Use the MCP `create_job` tool (or the REST equivalent). For each file, base64-encode the bytes: `{"name": "<filename>", "content_b64": "<base64>"}`. Pass `author` as a comma-separated list of the user's sending addresses — only for email; leave empty for docs/Obsidian/plain text.

**MCP:**
```
create_job(author="<emails or empty>", synth_model="google/gemini-2.5-flash",
           files=[{"name":"Sent.mbox","content_b64":"..."}])
```

**REST fallback:**
```
curl -X POST <TIMBRE_URL>/api/jobs \
  -H "Authorization: Bearer <API_KEY>" \
  -F 'author=<emails or empty>' \
  -F 'synth_model=google/gemini-2.5-flash' \
  -F 'files=@/path/to/Sent.mbox'
```

`synth_model`: `anthropic/claude-opus-4.8` (best), `anthropic/claude-sonnet-4` (balanced), `google/gemini-2.5-flash` (cheapest). The teacher invents writing prompts for each sample; the user's real text is the training target.

You get back `{"job_id": "..."}`. **Tell the user the job started and to open the web UI** — because the agent and the web page share the same API key, their run appears there automatically with live progress.

### Stage 4 — Hand off to the web UI

Once `create_job` returns a job_id, say something like: *"Your run started. Open the Timbre web app and click the **Train** tab — you'll see it training live. It takes about 15 minutes. Come back when it hits Done."*

You can optionally poll status yourself (`get_job_status`) and notify the user when it finishes, but the web UI is the primary surface — the user doesn't need to stay in the chat.

### Stage 5 — Download and run locally (when done)

When `stage == "done"` (you polled, or the user returns):

```
curl <TIMBRE_URL>/api/jobs/<job_id>/download/adapter.gguf -o adapter.gguf \
  -H "Authorization: Bearer <API_KEY>"
curl <TIMBRE_URL>/api/jobs/<job_id>/download/Modelfile -o Modelfile \
  -H "Authorization: Bearer <API_KEY>"
```

Then with Ollama installed:
```
ollama pull llama3.2:3b          # base model, once
ollama create my-style -f Modelfile
ollama run my-style
```
Suggest a test prompt in their voice, e.g. *"write a quick follow-up to a client who hasn't replied."*

## Pitfalls

- **Author address for email**: `.mbox`/`.eml` without `author` returns a 400. Docs/Obsidian/plain text need none.
- **One active job per key**: a second concurrent job returns 409 `{"error":"job already running"}`. Wait for the current job to reach `done`/`error` before starting another, or tell the user to finish the existing run first.
- **Owner scoping**: a job under one key is invisible (404) to any other key. If the user can't see their run in the web UI, they're likely on a different key than the one in the agent's config — have them check the Guide section.
- **Errors are recoverable**: on `stage == "error"`, read the `error` field and call `resume_job` — pipeline steps are idempotent and skip completed work.
- **Key shown once**: if the user loses it, they generate a new one in the web guide and update the agent config.
- **Privacy**: after download, job data sits on the server until purged. Training runs on stateless GPU compute; only the backend holds job state.

## Verification

- Job reaches `done` with `adapter_mb` set.
- `ollama run my-style` loads and responds in the user's voice.
