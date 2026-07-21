"""Auth + concurrency guards via FastAPI TestClient.

No Modal/OpenRouter creds: start_job is stubbed so the guard is exercised
before any workflow would run.
"""
import auth
import main
import status
from fastapi.testclient import TestClient

# Stub the workflow start: tests assert the guards fire before this runs.
main.start_job = lambda *a, **k: None

client = TestClient(main.app)


def _auth(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


def _mint(label: str = "t") -> str:
    r = client.post("/api/keys", json={"label": label})
    assert r.status_code == 200, r.text
    return r.json()["key"]


def _create(key: str, name: str = "a.txt"):
    return client.post(
        "/api/jobs",
        headers=_auth(key),
        data={"author": "a@x.com", "synth_model": "m"},
        files={"files": (name, b"a writing sample", "text/plain")},
    )


# ── auth ──────────────────────────────────────────────────────────────────

def test_missing_bearer_is_401():
    r = client.get("/api/keys")
    assert r.status_code == 401
    assert r.json() == {"error": "unauthorized"}


def test_invalid_bearer_is_401():
    r = client.get("/api/keys", headers={"Authorization": "Bearer deadbeef"})
    assert r.status_code == 401
    assert r.json() == {"error": "unauthorized"}


def test_keys_post_is_unauthenticated():
    # POST /api/keys is the ONE open route — works with no auth header.
    r = client.post("/api/keys", json={"label": "anon"})
    assert r.status_code == 200
    assert "key" in r.json()


def test_valid_key_returns_label_not_plaintext():
    key = _mint("mykey")
    r = client.get("/api/keys", headers=_auth(key))
    assert r.status_code == 200
    data = r.json()
    assert data and data[0]["label"] == "mykey"
    assert "key" not in data[0]          # plaintext never returned again


# ── concurrency cap ───────────────────────────────────────────────────────

def test_second_concurrent_job_is_409():
    key = _mint()
    r1 = _create(key)
    assert r1.status_code == 200, r1.text
    r2 = _create(key, "b.txt")
    assert r2.status_code == 409
    assert r2.json() == {"error": "job already running"}


# ── owner scoping (404, never 403) ────────────────────────────────────────

def test_wrong_owner_job_access_is_404():
    key_a = _mint("a")
    key_b = _mint("b")
    jid = _create(key_a).json()["job_id"]
    # B cannot see A's job — 404, never 403 (no existence leak).
    assert client.get(f"/api/jobs/{jid}", headers=_auth(key_b)).status_code == 404
    # A sees its own job.
    assert client.get(f"/api/jobs/{jid}", headers=_auth(key_a)).status_code == 200


def test_list_jobs_scoped_to_owner():
    key_a = _mint("la")
    key_b = _mint("lb")
    assert _create(key_a, "la.txt").status_code == 200
    ra = client.get("/api/jobs", headers=_auth(key_a))
    rb = client.get("/api/jobs", headers=_auth(key_b))
    assert ra.status_code == 200 and len(ra.json()) >= 1
    assert rb.json() == []               # B sees none of A's jobs


def test_download_wrong_owner_is_404():
    key_a = _mint("da")
    key_b = _mint("db")
    jid = _create(key_a).json()["job_id"]
    assert client.get(f"/api/jobs/{jid}/download/adapter.gguf",
                      headers=_auth(key_b)).status_code == 404
