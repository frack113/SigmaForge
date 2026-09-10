"""Tests for chat API routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import create_app


def test_chat_page_loads() -> None:
    """Test that the chat page serves Jinja2 template."""
    app = create_app()
    client = TestClient(app)
    resp = client.get("/chat")
    assert resp.status_code == 200
    assert "SigmaForge" in resp.text
    assert "Chat" in resp.text


def test_send_chat_message() -> None:
    """Test sending a chat message via API."""
    app = create_app()
    client = TestClient(app)
    payload = {"message": "What is detection logic?", "mode": "search"}
    resp = client.post("/api/v1/chat/message", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "timestamp" in data
    assert data["mode"] == "search"


def test_send_empty_message() -> None:
    """Test sending empty message returns 400."""
    app = create_app()
    client = TestClient(app)
    payload = {"message": " ", "mode": "search"}
    resp = client.post("/api/v1/chat/message", json=payload)
    assert resp.status_code == 400


def test_upload_valid_yaml() -> None:
    """Test uploading a valid Sigma rule YAML file."""
    app = create_app()
    client = TestClient(app)
    yaml_content = b"""
id: test_rule_001
name: Test Sigma Rule
description: A test rule for detection
detection:
    selection:
        EventID: 4625
    condition: selection
"""
    resp = client.post(
        "/api/v1/chat/upload",
        files={"file": ("test.yaml", yaml_content, "application/x-yaml")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["validated"] is True
    assert data["rule_name"] == "Test Sigma Rule"


def test_upload_invalid_yaml() -> None:
    """Test uploading invalid YAML file returns error."""
    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/api/v1/chat/upload",
        files={"file": ("bad.yaml", b"not: [valid: yaml: broken", "application/x-yaml")},
    )
    assert resp.status_code == 422


def test_upload_wrong_extension() -> None:
    """Test uploading non-yaml file returns error."""
    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/api/v1/chat/upload",
        files={"file": ("test.txt", b"some content", "text/plain")},
    )
    assert resp.status_code == 400


def test_chat_page_in_nav() -> None:
    """Test that chat page is accessible and nav shows active state."""
    app = create_app()
    client = TestClient(app)
    resp = client.get("/chat")
    assert resp.status_code == 200
    assert 'class="active"' in resp.text or "active" in resp.text
