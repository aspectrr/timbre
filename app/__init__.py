"""Style-clone web app: ingest → curate → synthesize → train → export GGUF.

Single-user Stage 1. Upload writing samples (mbox/txt/md), a curation agent
keeps the valuable ones, a teacher model back-translates them into style pairs,
a LoRA is trained on Modal GPU, and the user downloads an Ollama-ready adapter.

Deploy:  uv run modal deploy app/web.py
Dev:     uv run modal serve  app/web.py
"""
