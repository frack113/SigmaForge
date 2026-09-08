"""Sparse model API v1 routes — check status and trigger download of the SPLADE model."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.config.settings import SPARSE_MODEL_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/models/embedding-fast", tags=["v1-models-embedding-fast"])

_download_progress: dict[str, Any] = {}


@router.get("/installed")
async def list_installed_sparse_models() -> JSONResponse:
    """Check if the SPLADE sparse model is present in the local cache."""
    try:
        installed = SPARSE_MODEL_DIR.exists()
        return JSONResponse(
            content={"models": [{"repo_id": "prithivida/Splade_PP_en_v1", "installed": installed}]}
        )
    except Exception as e:
        logger.error("Failed to check sparse model: %s", e)
        return JSONResponse(status_code=500, content={"error": "An internal error occurred"})


@router.post("/download")
async def download_sparse_model() -> JSONResponse:
    """Download the SPLADE sparse model (prithivida/Splade_PP_en_v1) for hybrid search."""

    async def download_in_background() -> None:
        try:
            _download_progress["sparse"] = {"progress": 0, "status": "starting"}

            SPARSE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

            from src.shared.utils.hf_hub import force_hf_online

            with force_hf_online():
                from transformers import AutoModelForMaskedLM, AutoTokenizer

                _download_progress["sparse"] = {"progress": 10, "status": "downloading tokenizer"}
                tokenizer = AutoTokenizer.from_pretrained("prithivida/Splade_PP_en_v1")

                _download_progress["sparse"] = {"progress": 30, "status": "downloading model"}
                model = AutoModelForMaskedLM.from_pretrained("prithivida/Splade_PP_en_v1")

                _download_progress["sparse"] = {"progress": 70, "status": "saving to cache"}
                tokenizer.save_pretrained(str(SPARSE_MODEL_DIR))
                model.save_pretrained(str(SPARSE_MODEL_DIR))

            _download_progress["sparse"] = {"progress": 100, "status": "completed"}
        except Exception as e:
            logger.error("Failed to download sparse model: %s", e)
            _download_progress["sparse"] = {"progress": 0, "status": f"error: {e}"}

    asyncio.create_task(download_in_background())

    return JSONResponse(
        content={
            "success": True,
            "message": "Download started in background",
            "repo_id": "prithivida/Splade_PP_en_v1",
        }
    )


@router.get("/progress")
async def get_sparse_model_progress() -> JSONResponse:
    """Return the current download progress of the sparse model."""
    status = _download_progress.get("sparse", {"progress": 100, "status": "completed"})
    return JSONResponse(content=status)
