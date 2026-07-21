---
name: timbre
description: Clone your writing style into a local Ollama model via the Timbre / Style Clone service. Train a model that writes like you from your sent email, Obsidian vault, Word docs, PDFs, or plain text. Activate when the user says "train my voice", "clone my writing", "make a model of how I write", or references Timbre/Style Clone.
---

# Timbre — Train a model that writes like you

Timbre trains a LoRA adapter on your own writing (email, Obsidian, Word, PDF, plain text), exports it as a GGUF + Modelfile, and you run it locally in Ollama. Your data never stays on the server after download.

## When to use

The user wants to clone their writing style — "train a model on my writing", "make my emails sound like me", "clone my voice", "train on my Obsidian vault / sent mail / docs".

## Prerequisites

- A **Timbre API key**. If the user doesn't have one, mint one (see below) and save it to memory immediately — the plaintext is shown only once.
- The **Timbre server URL** (the hosted backend). Default `https://style-clone-backend.fly.dev`. Confirm with the user if unsure.
- The user's **writing files**: `.mbox`/`.eml` (sent mail), `.txt`/`.md` (Obsidian/plain), `.docx`, or `.pdf`. For email, you also need their email address(es) to filter sent mail.

## Procedure

### 1. Get an API key (if needed)

```
curl -X POST <TIMBRE_URL>/api/keys -H 'Content-Type: application/json' -d '{"label":"my-agent"}'
# -> {"key":"<plaintext>","label":"my-agent"}
```
Save `<plaintext>` to memory as `timbre_api_key`. It is shown only this once.

### 2. Gather the user's writing

Collect their files. Sources that work well:
- **Sent mail**: Gmail Takeout `.mbox`, macOS Mail `.mbox` export, individual `.eml`.
- **Obsidian**: `.md` files from the vault (frontmatter/wikilinks/tags are auto-stripped).
- **Docs**: `.docx`, `.pdf`, `.txt` from a USB, Drive export, or local disk.

Only **email** needs the user's address to filter sent mail. Docs/Obsidian/plain text need nothing extra. Ask the user where their writing lives and read those files.

### 3. Create a training job

Use the MCP server (preferred) or the REST API.

**Via MCP** (add to the agent's MCP client config):
```json
{ "mcpServers": { "timbre": { "url": "<TIMBRE_URL>/mcp", "headers": { "Authorization": "Bearer <TIMBRE_API_KEY>" } } } }
```
Then call the `create_job` tool. Each file is `{"name": "<filename>", "content_b64": "<base64>"}`. Base64-encode the file bytes yourself. For email, pass `author` as a comma-separated list of the user's addresses.

**Via REST** (curl):
```
curl -X POST <TIMBRE_URL>/api/jobs \
  -H "Authorization: Bearer <TIMBRE_API_KEY>" \
  -F 'author=<email or empty>' \
  -F 'synth_model=google/gemini-2.5-flash' \
  -F 'files=@/path/to/file1' \
  -F 'files=@/path/to/file2'
# -> {"job_id":"<id>"}
```
`synth_model` options: `anthropic/claude-opus-4.8` (best), `anthropic/claude-sonnet-4` (balanced), `google/gemini-2.5-flash` (cheapest). The teacher model invents writing prompts for each sample; the user's real text becomes the training target.

### 4. Poll until done

```
curl <TIMBRE_URL>/api/jobs/<job_id> -H "Authorization: Bearer <TIMBRE_API_KEY>"
```
Stages: `queued → ingesting → curating → synthesizing → training → exporting → done`. Training on GPU takes ~5-10 min. Poll every few seconds. `stage == "done"` → ready. `stage == "error"` → read `error` field; you can `POST /api/jobs/<job_id>/resume` to retry (steps are idempotent).

### 5. Download and install locally

When `done`:
```
curl <TIMBRE_URL>/api/jobs/<job_id>/download/adapter.gguf -o adapter.gguf \
  -H "Authorization: Bearer <TIMBRE_API_KEY>"
curl <TIMBRE_URL>/api/jobs/<job_id>/download/Modelfile -o Modelfile \
  -H "Authorization: Bearer <TIMBRE_API_KEY>"
```
Then with Ollama installed:
```
ollama create my-style -f Modelfile   # Modelfile references ./adapter.gguf
ollama run my-style
```

## Pitfalls

- **Author address for email**: `.mbox`/`.eml` without an `author` address returns a 400. Docs/Obsidian need no author.
- **One job at a time per key**: a second concurrent job returns 409 `{"error":"job already running"}`. Wait for the current job to reach `done`/`error` before starting another.
- **Key shown once**: if the user loses it, mint a new one; the old one can't be recovered.
- **Owner scoping**: a job created under one key is invisible (404) to any other key.
- **Minimum data**: curation drops low-value samples and errors out if fewer than ~10 valuable samples remain. More writing = better clone. Aim for a few hundred samples.
- **Privacy**: after download, the job's data sits on the server until purged. The training itself runs on stateless GPU compute; only the Fly backend holds job state.

## Verification

- Job reaches `done` with `adapter_mb` set.
- `ollama run my-style` loads and responds in the user's voice.
