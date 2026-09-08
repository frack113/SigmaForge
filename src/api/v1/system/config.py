"""Centralized configuration API v1 — backend config (DuckDB) + remaining TOML settings."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from src.config.settings import get_config
from src.infrastructure.database.service import DatabaseService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["config-v1"])

LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class ConfigUpdateRequest(BaseModel):
    """Request model for config updates."""

    backend: dict[str, Any] | None = None


class LoggingConfigUpdateRequest(BaseModel):
    """Request model for logging config updates."""

    level: str | None = None
    log_max_size: str | None = None
    log_max_file: int | None = None
    clean_at_startup: bool | None = None

    @field_validator("level")
    @classmethod
    def validate_level(cls, v):
        if v is not None and v not in LOG_LEVELS:
            raise ValueError(f"Invalid log level: {v}")
        return v


@router.get("/config")
async def get_full_config() -> JSONResponse:
    """GET /api/v1/config — Return full application configuration."""
    try:
        config = get_config()
        return JSONResponse(content={"status": "success", "data": config.to_dict()})
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return JSONResponse(
            status_code=500, content={"status": "error", "error": "An internal error occurred"}
        )


@router.get("/config/logging")
async def get_logging_config() -> JSONResponse:
    """GET /api/v1/config/logging — Return logging configuration."""
    try:
        config = get_config()
        return JSONResponse(
            content={
                "status": "success",
                "data": {
                    "level": config.logging_level,
                    "log_max_size": config.logging_log_max_size,
                    "log_max_file": config.logging_log_max_file,
                    "clean_at_startup": config.logging_clean_at_startup,
                },
            }
        )
    except Exception as e:
        logger.error(f"Failed to load logging config: {e}")
        return JSONResponse(
            status_code=500, content={"status": "error", "error": "An internal error occurred"}
        )


@router.post("/config/logging")
async def update_logging_config(request: LoggingConfigUpdateRequest) -> JSONResponse:
    """POST /api/v1/config/logging — Update logging config persisted to DuckDB."""
    try:
        config = get_config()
        db = DatabaseService.get_instance()

        changed = False
        if request.level is not None:
            config.logging_level = request.level
            db.set_config("logging.level", request.level, persist=False)
            changed = True
        if request.log_max_size is not None:
            config.logging_log_max_size = request.log_max_size
            db.set_config("logging.log_max_size", request.log_max_size, persist=False)
            changed = True
        if request.log_max_file is not None:
            config.logging_log_max_file = request.log_max_file
            db.set_config("logging.log_max_file", request.log_max_file, persist=False)
            changed = True
        if request.clean_at_startup is not None:
            config.logging_clean_at_startup = request.clean_at_startup
            db.set_config("logging.clean_at_startup", request.clean_at_startup, persist=False)
            changed = True
        if changed:
            db.persist()

        return JSONResponse(
            content={
                "status": "success",
                "message": "Logging configuration updated",
                "data": {
                    "level": config.logging_level,
                    "log_max_size": config.logging_log_max_size,
                    "log_max_file": config.logging_log_max_file,
                    "clean_at_startup": config.logging_clean_at_startup,
                },
            }
        )
    except Exception as e:
        logger.error(f"Failed to update logging config: {e}")
        return JSONResponse(
            status_code=500, content={"status": "error", "error": "An internal error occurred"}
        )


@router.post("/config/fix-schema")
async def fix_schema_version() -> JSONResponse:
    """POST /api/v1/config/fix-schema — Update DuckDB schema_version to match code."""
    from src.config.constants import SCHEMA_VERSION

    try:
        db = DatabaseService.get_instance()
        db.set_config("schema_version", SCHEMA_VERSION)
        return JSONResponse(
            content={
                "status": "success",
                "message": f"Schema version updated to {SCHEMA_VERSION}",
            }
        )
    except Exception as e:
        logger.error(f"Failed to fix schema version: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "error": str(e)})


@router.post("/config")
async def update_config(request: ConfigUpdateRequest) -> JSONResponse:
    """POST /api/v1/config — Update backend config persisted to DuckDB."""
    try:
        config = get_config()
        if request.backend:
            os_val = request.backend.get("os")
            gpu_val = request.backend.get("gpu_type")
            services = request.backend.get("services", {}) or {}
            llama_base_url = request.backend.get("llama_base_url")
            if llama_base_url is None:
                llama_base_url = services.get("llama", {}).get("base_url")
            llama_manage = request.backend.get("llama_manage_internally")
            if llama_manage is None:
                llama_manage = services.get("llama", {}).get("manage_internally")
            llama_autorun = request.backend.get("llama_autorun_at_startup")
            if llama_autorun is None:
                llama_autorun = services.get("llama", {}).get("autorun_at_startup")
            qdrant_base_url = request.backend.get("qdrant_base_url")
            if qdrant_base_url is None:
                qdrant_base_url = services.get("qdrant", {}).get("base_url")
            qdrant_manage = request.backend.get("qdrant_manage_internally")
            if qdrant_manage is None:
                qdrant_manage = services.get("qdrant", {}).get("manage_internally")
            qdrant_autorun = request.backend.get("qdrant_autorun_at_startup")
            if qdrant_autorun is None:
                qdrant_autorun = services.get("qdrant", {}).get("autorun_at_startup")

            if os_val:
                config.os = os_val
            if gpu_val:
                config.gpu_type = gpu_val
            if llama_base_url is not None:
                config.llama_base_url = str(llama_base_url)
            if llama_manage is not None:
                config.llama_manage_internally = bool(llama_manage)
            if llama_autorun is not None:
                config.llama_autorun_at_startup = bool(llama_autorun)
            if qdrant_base_url is not None:
                config.qdrant_base_url = str(qdrant_base_url)
            if qdrant_manage is not None:
                config.qdrant_manage_internally = bool(qdrant_manage)
            if qdrant_autorun is not None:
                config.qdrant_autorun_at_startup = bool(qdrant_autorun)

            db = DatabaseService.get_instance()
            any_set = False
            if os_val is not None:
                db.set_config("backend.os", os_val, persist=False)
                any_set = True
            if gpu_val is not None:
                db.set_config("backend.gpu_type", gpu_val, persist=False)
                any_set = True
            if llama_base_url is not None:
                db.set_config("services.llama.base_url", llama_base_url, persist=False)
                any_set = True
            if llama_manage is not None:
                db.set_config("services.llama.manage_internally", llama_manage, persist=False)
                any_set = True
            if llama_autorun is not None:
                db.set_config("services.llama.autorun_at_startup", llama_autorun, persist=False)
                any_set = True
            if qdrant_base_url is not None:
                db.set_config("services.qdrant.base_url", qdrant_base_url, persist=False)
                any_set = True
            if qdrant_manage is not None:
                db.set_config("services.qdrant.manage_internally", qdrant_manage, persist=False)
                any_set = True
            if qdrant_autorun is not None:
                db.set_config("services.qdrant.autorun_at_startup", qdrant_autorun, persist=False)
                any_set = True
            if any_set:
                db.persist()

        return JSONResponse(
            content={
                "status": "success",
                "message": "Configuration updated and persisted",
                "data": config.to_dict(),
            }
        )
    except Exception as e:
        logger.error(f"Failed to update config: {e}")
        return JSONResponse(
            status_code=500, content={"status": "error", "error": "An internal error occurred"}
        )
