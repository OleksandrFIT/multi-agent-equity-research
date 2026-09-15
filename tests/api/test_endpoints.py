from fastapi.testclient import TestClient

import api.core as core
import api.main as main


def test_health(monkeypatch):
    monkeypatch.setattr(core, "health", lambda: {"ok": True, "model": "qwen2.5:7b", "ollama_reachable": True})
    client = TestClient(main.app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["ollama_reachable"] is True
    assert resp.json()["model"] == "qwen2.5:7b"
