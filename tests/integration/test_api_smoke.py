from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.infrastructure.database import DatabaseService
from src.main import create_app


@pytest.fixture
def client(tmp_path: Path):
    db = DatabaseService(str(tmp_path / "sigmaforge_smoke.duckdb"))
    db.initialize()
    app = create_app()
    app.state.db = db
    yield TestClient(app)


def _get_json(client: TestClient, url: str) -> dict:
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def test_chat_page(client: TestClient) -> None:
    response = client.get("/chat")
    assert response.status_code == 200
    assert "SigmaForge" in response.text


def test_config_page(client: TestClient) -> None:
    response = client.get("/config")
    assert response.status_code == 200
    assert "System Status" in response.text


def test_dashboard_page(client: TestClient) -> None:
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Database Dashboard" in response.text


def test_logs_page(client: TestClient) -> None:
    response = client.get("/logs")
    assert response.status_code == 200
    assert "System Logs" in response.text


def test_local_data_page(client: TestClient) -> None:
    response = client.get("/data/local")
    assert response.status_code == 200
    assert "Local Files" in response.text


def test_setup_redirects_to_config(client: TestClient) -> None:
    response = client.get("/setup", follow_redirects=False)
    assert response.status_code == 301
    assert response.headers["location"] == "/config"


def test_admin_status(client: TestClient) -> None:
    payload = _get_json(client, "/api/v1/admin/status")
    assert payload["data"]["llama_cpp"]["status"] in {"active", "inactive"}
    assert payload["data"]["qdrant"]["status"] in {"active", "inactive"}


def test_chat_history(client: TestClient) -> None:
    response = client.get("/api/v1/chat/history", headers={"X-Session-ID": "smoke"})
    assert response.status_code == 200
    assert response.json() == []


def test_prompts_list(client: TestClient) -> None:
    response = client.get("/api/v1/admin/prompts")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_dashboard_tables(client: TestClient) -> None:
    payload = _get_json(client, "/api/v1/dashboard/tables")
    assert "config" in payload["tables"]


def test_logging_config(client: TestClient) -> None:
    payload = _get_json(client, "/api/v1/config/logging")
    assert payload["status"] == "success"
    assert "level" in payload["data"]
