from __future__ import annotations

import pytest
from playwright.sync_api import Page

pytestmark = pytest.mark.e2e


def test_chat_page(page: Page, e2e_base_url: str) -> None:
    page.goto(f"{e2e_base_url}/chat")
    page.wait_for_selector("#message-input", state="visible")
    assert page.locator("#chat-welcome").is_visible()
    assert page.locator("#send-btn").is_visible()


def test_config_page(page: Page, e2e_base_url: str) -> None:
    page.goto(f"{e2e_base_url}/config")
    page.wait_for_selector("#status-grid", state="visible")
    assert page.locator("#status-banner").is_visible()
    assert page.locator("#general-section").is_visible()


def test_dashboard_page(page: Page, e2e_base_url: str) -> None:
    page.goto(f"{e2e_base_url}/dashboard")
    page.wait_for_selector("#duckdb-content", state="visible")


def test_logs_page(page: Page, e2e_base_url: str) -> None:
    page.goto(f"{e2e_base_url}/logs")
    page.wait_for_selector("#log-stats", state="visible")


def test_local_data_page(page: Page, e2e_base_url: str) -> None:
    page.goto(f"{e2e_base_url}/data/local")
    page.wait_for_selector("#local-files-body", state="visible")
    assert page.locator("#upload-zone").is_visible()


def test_setup_redirects_to_config(page: Page, e2e_base_url: str) -> None:
    page.goto(f"{e2e_base_url}/setup")
    assert page.url.endswith("/config")
