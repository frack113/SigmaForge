"""File Discovery and Embedding API v1."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Any

from src.shared.constants import NULL_UUID
from src.shared.utils.crypto_utils import compute_sha256_bytes, compute_sha256_str

from fastapi import APIRouter, Depends, UploadFile
from fastapi import File as FastAPIFile
from pydantic import BaseModel

from src.api.dependencies import get_dispatcher
from src.shared.utils.identify_file_type import SUPPORTED_DOC_EXTENSION_MAP, identify
from src.config.settings import get_config
from src.infrastructure.database.service import DatabaseService
from src.shared.utils import iso_now
from src.workers.enums import WorkerName

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/files", tags=["v1-files"])


class FileResponse(BaseModel):
    """Unified response for file operations."""

    success: bool
    message: str | None = None
    data: Any = None
    total: int = 0
    error: str | None = None


def _dispatch_workers(
    dispatcher,
    workers: list[tuple[WorkerName, str]],
) -> tuple[list[str], list[str]]:
    """Dispatch multiple workers and return (triggered_names, busy_names)."""
    triggered: list[str] = []
    busy: list[str] = []
    for worker, collection in workers:
        if dispatcher.ask_for_worker(worker, task_type=worker.value, collection_name=collection):
            triggered.append(worker.value)
        else:
            busy.append(worker.value)
    return triggered, busy


@router.post("/list", response_model=FileResponse)
async def file_list(
    dispatcher=Depends(get_dispatcher),
) -> FileResponse:
    """Trigger file discovery across all sources (GitHub, Local, SigmaRef)."""
    triggered, busy = _dispatch_workers(
        dispatcher,
        [
            (WorkerName.GITHUB_DISCOVERY, "all"),
            (WorkerName.LOCAL_DISCOVERY, "local"),
            (WorkerName.SPEC_DISCOVERY, "spec"),
        ],
    )
    # SIGMAREF_DISCOVERY is chained automatically after GITHUB_DISCOVERY
    # completes, so they are not dispatched in parallel.

    if busy:
        return FileResponse(
            success=False,
            error=f"Workers already busy: {', '.join(busy)}",
            data={"triggered": triggered} if triggered else None,
        )

    return FileResponse(
        success=True,
        message=f"Discovery queued for: {', '.join(triggered)}",
        data={"tasks": triggered},
    )


@router.post("/local/add", response_model=FileResponse)
async def add_local_file(
    file: UploadFile = FastAPIFile(...),
    collection_name: str = "local",
) -> FileResponse:
    """Upload a local file to configured documents path."""
    cfg = get_config()
    base_path = Path(cfg.local_documents_path)
    base_path.mkdir(parents=True, exist_ok=True)

    if not file.filename:
        return FileResponse(success=False, error="No filename provided.")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    if ext not in SUPPORTED_DOC_EXTENSION_MAP:
        return FileResponse(
            success=False,
            error=f"Unsupported file type: {ext}",
            data={"supported_extensions": list(SUPPORTED_DOC_EXTENSION_MAP.keys())},
        )

    dest_path = (base_path / file.filename).resolve()
    try:
        rel = dest_path.relative_to(base_path.resolve())
        if ".." in rel.parts:
            return FileResponse(success=False, error="Path traversal detected")
    except ValueError:
        return FileResponse(success=False, error="Path traversal detected")

    if dest_path.exists():
        return FileResponse(
            success=False,
            error=f"File already exists: {file.filename}",
        )

    def _write_and_hash() -> tuple[str, str, int]:
        with open(dest_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        try:
            content_type = identify(dest_path).value
            file_bytes = dest_path.read_bytes()
            content_hash = compute_sha256_bytes(file_bytes)
            file_size = dest_path.stat().st_size
        except Exception:
            logging.getLogger(__name__).error("Error reading file")
            return "", "", 0
        return content_type, content_hash, file_size

    content_type, content_hash, file_size = await asyncio.to_thread(_write_and_hash)

    file_rel_path = dest_path.relative_to(base_path).as_posix()
    url_hash = compute_sha256_str(f"local/{collection_name}/{file_rel_path}")
    title = dest_path.stem

    db = DatabaseService.get_instance()
    db.upsert_doc_registry(
        {
            "url_hash": url_hash,
            "org": "local",
            "repo": collection_name,
            "content_type": content_type,
            "file_name": file_rel_path,
            "content_sha256": content_hash,
            "file_size": file_size,
            "original_url": f"file://{dest_path.as_posix()}",
            "normalized_url": f"file://{dest_path.as_posix()}",
            "rule_id": NULL_UUID,
            "title": title,
            "timestamp": iso_now(),
            "last_seen": iso_now(),
            "embed_status": "discovery",
        }
    )

    return FileResponse(
        success=True,
        message=f"File '{file.filename}' added successfully.",
        data={
            "url_hash": url_hash,
            "file_name": file_rel_path,
            "content_type": content_type,
            "file_size": file_size,
            "embed_status": "discovery",
        },
    )


@router.delete("/local/delete", response_model=FileResponse)
async def delete_local_file(
    file_path: str,
) -> FileResponse:
    """Delete a local file from configured documents path and doc_registry."""
    cfg = get_config()
    base_path = Path(cfg.local_documents_path).resolve()
    fs_path = Path(file_path).resolve()

    try:
        fs_path.relative_to(base_path)
    except ValueError:
        return FileResponse(success=False, error="Path outside documents directory")

    if not fs_path.exists():
        return FileResponse(success=False, error=f"File does not exist: {file_path}")

    try:
        fs_path.unlink()
    except OSError as e:
        logging.getLogger(__name__).error(f"Error deleting file from filesystem: {e}")
        return FileResponse(success=False, error=f"Failed to delete file: {str(e)}")

    db = DatabaseService.get_instance()
    url_match = f"file://{fs_path.as_posix()}"
    db.delete_doc_registry_by_url(url_match)

    return FileResponse(
        success=True,
        message="File deleted successfully.",
    )


@router.get("/local/list", response_model=FileResponse)
async def list_local_files(
    limit: int = 1000,
    offset: int = 0,
) -> FileResponse:
    """List all local files from doc_registry."""
    db = DatabaseService.get_instance()
    files = db.get_local_files(limit=limit, offset=offset)
    total = db.get_local_file_count()

    return FileResponse(
        success=True,
        data=files,
        total=total,
    )


@router.post("/local/resync", response_model=FileResponse)
async def resync_local_file_sizes() -> FileResponse:
    """Resync local file sizes — runs blocking DB call in executor to avoid blocking event loop."""
    import asyncio

    cfg = get_config()
    base_path = cfg.local_documents_path

    if not base_path or not isinstance(base_path, str):
        return FileResponse(
            success=False,
            error="local_documents_path is not configured",
        )

    db = DatabaseService.get_instance()
    result = await asyncio.to_thread(db.resync_local_file_sizes, base_path)

    has_errors = result["error"] > 0
    has_incomplete = result.get("incomplete", 0) > 0
    all_skipped = result["updated"] == 0 and result["skipped"] > 0

    message = (
        f"Resync complete: {result['updated']} updated, "
        f"{result['skipped']} skipped, {result['error']} errors"
        + (f", {result['incomplete']} incomplete hashes" if has_incomplete else "")
    )

    success = not (has_errors or all_skipped) and not (has_incomplete and result["updated"] == 0)

    return FileResponse(
        success=success,
        message=message,
        data=result,
    )
