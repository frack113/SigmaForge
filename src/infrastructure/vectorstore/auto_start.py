"""Qdrant auto-start on application launch."""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from src.shared.exceptions import ServiceStartError

logger = logging.getLogger(__name__)

_qdrant_started_by_us: bool = False
_started_binary_service: Any = None


def is_qdrant_running() -> bool:
    """Return whether Qdrant is running (HTTP health check)."""
    import httpx

    try:
        from src.config.settings import get_config

        config = get_config()
        port = config.qdrant_port
        response = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


async def start_qdrant(
    installer_service: Any | None = None,
    binary_service: Any | None = None,
    health_check: Callable[..., Awaitable[dict]] | None = None,
) -> None:
    from src.config.settings import get_config

    config = get_config()

    if not config.service_is_autostart("qdrant"):
        logger.info("Qdrant auto-start disabled (not internal or autorun=false) -- skipping")
        return

    if health_check is None:
        from src.infrastructure.vectorstore.health import check_health as health_check

    assert health_check is not None
    try:
        health = await health_check(timeout=2.0, port=config.qdrant_port)
        if health.get("status") == "active":
            logger.info("Qdrant is already running -- skipping auto-start")
            return
    except Exception:
        logger.warning("Qdrant health check failed -- will attempt start anyway")

    qdrant_bin = Path(config.qdrant_binary_path).resolve()
    if sys.platform == "win32":
        qdrant_candidates = ("qdrant.exe", "qdrant")
    else:
        qdrant_candidates = ("qdrant", "qdrant.exe")
    qdrant_exe: Path | None = None
    for name in qdrant_candidates:
        candidate = qdrant_bin / name
        if candidate.exists():
            qdrant_exe = candidate
            break

    if qdrant_exe is None:
        logger.warning("Qdrant binary not found at %s -- skipping auto-start", qdrant_bin)
        return

    if binary_service is None:
        from src.infrastructure.vectorstore.service import QdrantBinaryService

        binary_service = QdrantBinaryService()

    try:
        result = await binary_service.start()
        if not result.get("success"):
            raise ServiceStartError(f"Failed to start Qdrant: {result.get('error')}")
    except ServiceStartError:
        raise
    except Exception as e:
        raise ServiceStartError(f"Qdrant start raised an exception: {e}") from e

    global _qdrant_started_by_us, _started_binary_service
    _started_binary_service = binary_service

    assert health_check is not None
    for _ in range(30):
        try:
            health = await health_check(timeout=2.0, port=config.qdrant_port)
            if health.get("status") == "active":
                _qdrant_started_by_us = True
                logger.info("Qdrant started successfully and healthy")
                return
        except Exception:
            pass
        await asyncio.sleep(1)

    raise ServiceStartError("Qdrant process started but health check timed out after 30s")


async def stop_qdrant() -> None:
    global _qdrant_started_by_us
    if not _qdrant_started_by_us:
        return

    try:
        if _started_binary_service is not None:
            result = await _started_binary_service.stop()
        else:
            from src.infrastructure.vectorstore.service import QdrantBinaryService

            service = QdrantBinaryService()
            result = await service.stop()

        if result.get("success"):
            logger.info("Qdrant stopped (auto-started by app)")
        else:
            logger.warning(f"Failed to stop Qdrant: {result.get('error')}")
    except Exception as e:
        logger.warning(f"Qdrant stop raised an exception: {e}")
    finally:
        _qdrant_started_by_us = False
