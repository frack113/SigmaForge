"""API endpoint to view system logs."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from src.config.settings import get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/logs", tags=["v1-logs"])


def get_logs_dir() -> Path:
    return Path(get_config().paths_logs_dir).resolve()


LOG_FILES = {
    "system": "sigmahqrag.log",
    "llamacpp": "llama.cpp.log",
    "qdrant": "qdrant.log",
}

ENCODING_OPTIONS = ["utf-8", "latin-1", "cp1252", "ascii"]


def read_log_file(path: Path) -> list[str]:
    raw = path.read_bytes()
    for encoding in ENCODING_OPTIONS:
        try:
            return raw.decode(encoding).splitlines()
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").splitlines()


def _sse(event: str, data, **extra) -> str:
    payload: dict[str, Any] = {"type": event}
    if isinstance(data, list):
        payload["lines"] = data
    else:
        payload["message"] = data
    payload.update(extra)
    return "event: log\n" + "data: " + json.dumps(payload) + "\n\n"


@router.get("/stream")
async def stream_logs(
    source: str = Query("system", description="Log source: system, llamacpp, qdrant"),
    lines: int = Query(default=50, ge=1, le=500),
):
    """SSE endpoint for live log streaming (tail -f)."""

    if source not in LOG_FILES:
        return JSONResponse(status_code=400, content={"error": f"Invalid log source: {source}"})

    effective_lines = lines if lines > 0 else 0

    logs_dir = get_logs_dir()
    log_filename = LOG_FILES[source]
    log_path = logs_dir / log_filename

    async def event_generator():
        prev_total = 0
        prev_size = 0
        encoding = "utf-8"
        incomplete = ""

        while True:
            try:
                if not log_path.exists():
                    yield _sse("error", "Log file deleted")
                    break

                current_size = log_path.stat().st_size

                if current_size == prev_size and prev_size > 0:
                    await asyncio.sleep(0.5)
                    continue

                if prev_size == 0 or current_size < prev_size:
                    # First read or file truncated — full read with encoding detection
                    def _full_read(p: Path = log_path) -> tuple[str, str]:
                        for enc in ENCODING_OPTIONS:
                            try:
                                with open(p, encoding=enc, errors="strict") as f:
                                    return f.read(), enc
                            except UnicodeDecodeError:
                                continue
                        with open(p, encoding="utf-8", errors="replace") as f:
                            return f.read(), "utf-8"

                    all_text, encoding = await asyncio.to_thread(_full_read)
                    all_lines = all_text.splitlines()
                    total = len(all_lines)
                    recent = all_lines[-effective_lines:] if effective_lines > 0 else all_lines
                    entries = [line.strip() for line in recent]
                    yield _sse("init", entries, line_count=total)
                    prev_total = total
                    incomplete = ""
                else:
                    # File grew — read only the new bytes
                    def _tail_read(
                        p: Path = log_path, off: int = prev_size, enc: str = encoding
                    ) -> str:
                        with open(p, encoding=enc, errors="replace") as f:
                            f.seek(off)
                            return f.read()

                    new_text = await asyncio.to_thread(_tail_read)

                    combined = incomplete + new_text
                    lines = combined.splitlines()

                    # If the last char isn't a newline, the last line is partial
                    if new_text and new_text[-1] not in ("\n", "\r"):
                        incomplete = lines[-1] if lines else ""
                        lines = lines[:-1] if lines else []
                    else:
                        incomplete = ""

                    if lines:
                        entries = [line.strip() for line in lines]
                        yield _sse("update", entries, line_count=prev_total + len(lines))
                        prev_total += len(lines)

                prev_size = current_size
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("")
async def get_logs(
    source: str = Query("", description="Log source: system, llamacpp, qdrant"),
    lines: int = Query(default=0, ge=0, le=500),
    level: str = Query(default="", description="Filter by level: INFO, WARNING, ERROR"),
) -> JSONResponse:
    """Get recent log entries.

    Args:
        source: Log source (system, llamacpp, qdrant)
        lines: Number of lines (default from config)
        level: Filter by level (INFO, WARNING, ERROR)

    Returns:
        JSON with log entries
    """

    if not source:
        source = "system"

    if source not in LOG_FILES:
        return JSONResponse(content={"logs": [], "message": f"Invalid log source: {source}"})

    logs_dir = get_logs_dir()
    log_filename = LOG_FILES[source]
    log_path = logs_dir / log_filename

    if not log_path.exists():
        return JSONResponse(content={"logs": [], "message": "Log file not found"})

    try:
        all_lines = await asyncio.to_thread(read_log_file, log_path)

        if level:
            pattern = f" {level.upper()} "
            filtered = [line for line in all_lines if pattern in line.upper()]
        else:
            filtered = all_lines

        recent = filtered[-lines:]

        entries = []
        for line in recent:
            entries.append({"text": line.strip()})

        return JSONResponse(content={"logs": entries, "total": len(entries), "source": source})

    except Exception as e:
        import traceback

        logger.error(f"Failed to read logs: {e}\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": "An internal error occurred"})


@router.delete("")
async def clear_logs(
    source: str = Query("", description="Log source to clear: system, llamacpp, qdrant"),
) -> JSONResponse:
    """Clear log file contents.

    Args:
        source: Log source (system, llamacpp, qdrant). Defaults to system.

    Returns:
        JSON with success status
    """

    if not source:
        source = "system"

    if source not in LOG_FILES:
        return JSONResponse(content={"success": False, "message": f"Invalid log source: {source}"})

    logs_dir = get_logs_dir()
    log_filename = LOG_FILES[source]
    log_path = logs_dir / log_filename

    try:
        if log_path.exists():
            with open(log_path, "w") as f:
                f.write("")
            logger.info(f"Cleared log file: {log_path}")
            return JSONResponse(content={"success": True, "message": f"Cleared {log_filename}"})
        return JSONResponse(content={"success": False, "message": "Log file does not exist"})
    except Exception as e:
        logger.error(f"Failed to clear logs: {e}")
        return JSONResponse(status_code=500, content={"error": "An internal error occurred"})
