"""Hosted MCP server: mounted at /mcp, exposes the 5 tools, owner-scoped.

Tool *calls* are exercised through the plain `_*` helpers (the exact bodies
the @mcp.tool wrappers delegate to) with an explicit owner — the in-process
FastMCP Client has no header-injection hook, so this is how the owner context
is mocked. The tool registry itself is verified through the real Client.
"""
import asyncio

import auth
import main
import mcp_server
import status
from fastmcp import Client


def test_mcp_mounted_at_root():
    paths = {getattr(r, "path", None) for r in main.app.routes}
    assert "/mcp" in paths


def test_tool_registry():
    async def go():
        async with Client(mcp_server.mcp) as c:
            return await c.list_tools()
    names = {t.name for t in asyncio.run(go())}
    assert {"create_job", "get_job_status", "list_jobs",
            "resume_job", "download_model"} <= names


def _seed_owner() -> str:
    _, h = auth.new_key()
    status.create_key(h, "owner")
    return h


def test_list_jobs_owner_scoped():
    owner = _seed_owner()
    status.create_job("j-list", ["a@x.com"], "m", "llama3.2-3b", 1,
                      owner_key_hash=owner)
    out = mcp_server._list_jobs(owner)
    assert any(j["job_id"] == "j-list" for j in out)
    # a different owner sees nothing
    assert mcp_server._list_jobs(auth.hash_key("someone-else")) == []


def test_get_job_status_owner_scoped():
    owner = _seed_owner()
    status.create_job("j-get", ["a@x.com"], "m", "llama3.2-3b", 1,
                      owner_key_hash=owner)
    assert mcp_server._get_job_status(owner, "j-get")["job_id"] == "j-get"
    assert mcp_server._get_job_status(
        auth.hash_key("other"), "j-get") == {"error": "not found"}


def test_download_model_only_when_done():
    owner = _seed_owner()
    status.create_job("j-dl", ["a@x.com"], "m", "llama3.2-3b", 1,
                      owner_key_hash=owner)
    # job is still queued -> not ready
    assert mcp_server._download_model(owner, "j-dl") == {"error": "not ready"}
    # wrong owner -> not found (no leak)
    assert mcp_server._download_model(
        auth.hash_key("other"), "j-dl") == {"error": "not found"}
