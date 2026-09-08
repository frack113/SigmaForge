"""LLM model API v1 routes."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.dependencies import get_database_service, get_embedding_manager, get_unified_registry
from src.application.models.download import HFDownloadService
from src.api.v1.models._models_shared import (
    _delete_all_models_of_type,
    _delete_llm_model_file,
    _download_progress,
)
from src.application.models import HFRepo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models", tags=["v1-models-llm"])


def set_progress(repo_id: str, progress: int, status: str = "downloading"):
    """Track download progress."""
    _download_progress[repo_id] = {"progress": progress, "status": status}


@router.get("/llm/progress")
async def get_download_progress(repo_id: str) -> JSONResponse:
    """Get download progress for a model."""
    progress = _download_progress.get(repo_id, {"progress": 0, "status": "idle"})
    return JSONResponse(content=progress)


@router.get("/llm/installed")
async def list_installed_llm_models() -> JSONResponse:
    """List installed LLM models."""
    try:
        db = get_database_service()
        reg = get_unified_registry()
        from src.config.settings import LLM_DIR

        reg.sync_llm_folder(LLM_DIR, db)
        llms = reg.list_llms(db)
        models = []
        for repo_id, data in llms.items():
            files = []
            for name, info in data.get("files", {}).items():
                files.append(
                    {
                        "filename": info.get("filename", name),
                        "path": info.get("local_path", ""),
                        "size": info.get("file_size", 0),
                        "status": info.get("status", "ready"),
                    }
                )
            models.append(
                {
                    "repo_id": repo_id,
                    "files": files,
                }
            )
        return JSONResponse(content={"models": models})
    except Exception as e:
        logger.error(f"Failed to list installed LLM models: {e}")
        return JSONResponse(status_code=500, content={"error": "An internal error occurred"})


@router.get("/llm/search")
async def search_llm_models(query: str, limit: int = 20) -> JSONResponse:
    """Search for LLM models on HuggingFace.

    No pipeline tag filter is applied because many GGUF repos
    (e.g. MaziyarPanahi/*, bartowski/*) do not set a pipeline_tag.
    """
    from src.application.models.exceptions import DownloadError

    try:
        service = HFDownloadService()
        results = await service.list_models(query)
        return JSONResponse(content={"models": [{"repo_id": r.full_id} for r in results[:limit]]})
    except DownloadError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"LLM search failed: {e}")
        return JSONResponse(status_code=500, content={"error": "An internal error occurred"})


@router.get("/llm/files")
async def list_llm_model_files(repo_id: str | None = None) -> JSONResponse:
    """List available GGUF files for a LLM model."""
    if not repo_id:
        return JSONResponse(status_code=400, content={"error": "repo_id is required"})

    try:
        mm = get_embedding_manager()
        files = await mm.download_service.list_gguf_files(HFRepo.from_string(repo_id))
        return JSONResponse(content={"files": files})
    except Exception as e:
        logger.error(f"Failed to list files for {repo_id}: {e}")
        return JSONResponse(status_code=500, content={"error": "An internal error occurred"})


@router.get("/llm/{repo_id}/info")
async def get_llm_model_info(repo_id: str) -> JSONResponse:
    """Get detailed information about a LLM model."""
    try:
        mm = get_embedding_manager()
        info = await mm.get_model_info(repo_id)
        if not info:
            return JSONResponse(
                status_code=404,
                content={"error": f"Model {repo_id} not found"},
            )

        def to_iso(val):
            if val is None:
                return None
            if hasattr(val, "isoformat"):
                return val.isoformat()
            return str(val)

        siblings = []
        if info.siblings:
            for f in info.siblings:
                if f.rfilename.endswith(".gguf"):
                    size = f.size if f.size is not None else 0
                    siblings.append(
                        {
                            "filename": f.rfilename,
                            "size": size,
                        }
                    )

        return JSONResponse(
            content={
                "repo_id": repo_id,
                "id": info.id,
                "author": info.author,
                "sha": info.sha,
                "last_modified": to_iso(info.last_modified),
                "tags": list(info.tags) if info.tags else [],
                "siblings": siblings,
            }
        )
    except Exception as e:
        logger.error(f"Failed to get info for {repo_id}: {e}")
        return JSONResponse(status_code=500, content={"error": "An internal error occurred"})


@router.post("/llm/download")
async def download_llm_model(
    repo_id: str,
    filename: str | None = None,
) -> JSONResponse:
    """Download a LLM model. Auto-deletes existing LLM model first.
    If filename is not provided, tries to auto-discover GGUF files."""
    import asyncio

    from huggingface_hub import hf_hub_download

    from src.config.settings import LLM_DIR
    from src.infrastructure.llm.llamacpp.auto_start import stop_llamacpp as _stop_llm
    from src.infrastructure.vectorstore.auto_start import stop_qdrant as _stop_qd

    await _stop_qd()
    await _stop_llm()
    _delete_all_models_of_type("llm")

    set_progress(repo_id, 0, "starting")

    async def download_in_background():
        try:
            resolved_filename = filename
            if not resolved_filename:
                mm = get_embedding_manager()
                files = await mm.download_service.list_gguf_files(HFRepo.from_string(repo_id))
                if files:
                    resolved_filename = files[0]["filename"]
            if not resolved_filename:
                raise ValueError("No GGUF file specified or discovered")

            set_progress(repo_id, 5, "downloading")

            repo = HFRepo.from_string(repo_id)
            dest_dir = LLM_DIR / repo.owner / repo.name
            dest_dir.mkdir(parents=True, exist_ok=True)

            from src.shared.utils.hf_hub import force_hf_online

            with force_hf_online():
                _raw_token = os.environ.get("HF_TOKEN")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=resolved_filename,
                    local_dir=dest_dir,
                    token=_raw_token if _raw_token else None,
                )

            db = get_database_service()
            reg = get_unified_registry()
            reg.sync_llm_folder(LLM_DIR, db)

            set_progress(repo_id, 100, "completed")
            logger.info(f"Download completed: {repo_id}")
        except Exception as e:
            set_progress(repo_id, 0, f"error: {str(e)}")
            logger.error(f"Download failed: {e}")
        finally:
            from src.infrastructure.llm.llamacpp.auto_start import start_llamacpp as _start_llm
            from src.infrastructure.vectorstore.auto_start import start_qdrant as _start_qd

            try:
                await _start_qd()
            except Exception:
                logger.exception("Failed to auto-start Qdrant after download")
            try:
                await _start_llm()
            except Exception:
                logger.exception("Failed to auto-start llama.cpp after download")

    asyncio.create_task(download_in_background())

    return JSONResponse(
        content={
            "success": True,
            "message": "Download started in background",
            "repo_id": repo_id,
        }
    )


@router.delete("/llm/{repo_id}")
async def delete_llm_model(repo_id: str) -> JSONResponse:
    """Delete a LLM model entry."""
    try:
        db = get_database_service()
        reg = get_unified_registry()
        record = reg.get_llm(repo_id, db)
        if not record:
            return JSONResponse(status_code=404, content={"error": f"Model {repo_id} not found"})
        reg.remove_llm(repo_id, db)
        return JSONResponse(content={"success": True, "repo_id": repo_id})
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return JSONResponse(status_code=500, content={"error": "An internal error occurred"})


@router.delete("/llm/{repo_id}/file/{filename}")
async def delete_llm_model_file(repo_id: str, filename: str) -> JSONResponse:
    """Delete a LLM model file."""
    result = _delete_llm_model_file(repo_id, filename)
    if result.get("success"):
        return JSONResponse(content={"success": True, "repo_id": repo_id, "filename": filename})

    status_code = result.get("status_code", 500)
    return JSONResponse(status_code=status_code, content={"error": result["error"]})
