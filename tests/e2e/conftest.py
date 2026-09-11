from __future__ import annotations

import threading
import time
import urllib.request
from collections.abc import Generator
from typing import Any

import pytest
import uvicorn
from _pytest.monkeypatch import MonkeyPatch

from src import main as main_module
from src.infrastructure.database.core import DatabaseServiceCore

STARTUP_TIMEOUT_SECONDS = 60


def _wait_for_server(base_url: str) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/api/v1/admin/status", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(0.2)
    message = f"E2E server did not become ready at {base_url}: {last_error}"
    raise RuntimeError(message)


@pytest.fixture(scope="module")
def e2e_app(tmp_path_factory: pytest.TempPathFactory) -> Generator[Any, None, None]:
    sandbox = tmp_path_factory.mktemp("sigmaforge-e2e")
    monkey = MonkeyPatch()
    monkey.setenv("HF_HUB_OFFLINE", "1")
    monkey.setenv("HF_TOKEN", "")
    monkey.setenv("TQDM_DISABLE", "1")
    monkey.delenv("_SIGMA_SETUP_MODE", raising=False)
    monkey.setattr(main_module, "setup_mode", True)
    monkey.chdir(sandbox)

    app = main_module.create_app()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        lifespan="on",
    )
    config.load()
    sock = config.bind_socket()
    base_url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    server_thread.start()
    _wait_for_server(base_url)
    app.state.e2e_base_url = base_url
    try:
        yield app
    finally:
        server.should_exit = True
        server_thread.join(timeout=10)
        if sock.fileno() != -1:
            sock.close()
        monkey.undo()


@pytest.fixture(scope="module")
def e2e_base_url(e2e_app: Any) -> str:
    return e2e_app.state.e2e_base_url


@pytest.fixture(autouse=True)
def e2e_database(
    reset_modules: Generator[Any, None, None], e2e_app: Any
) -> Generator[None, None, None]:
    db = e2e_app.state.db
    DatabaseServiceCore._instance = db
    yield
    DatabaseServiceCore._instance = db
