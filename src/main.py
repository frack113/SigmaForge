"""Main application entry point."""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from io import StringIO

# ruff: noqa: E402 — env vars must be set before HuggingFace/model imports
# Force air-gap mode: disable all HuggingFace Hub network calls
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Disable token warnings in air-gap mode
os.environ.setdefault("HF_TOKEN", "")
# Disable tqdm progress bars (Loading weights messages)
os.environ.setdefault("TQDM_DISABLE", "1")

# Suppress verbose model loading logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import URL

from src.api.routes.page_admin import router as admin_pages_router
from src.api.routes.page_chat import router as chat_page_router
from src.api.routes.page_data import router as data_page_router
from src.api.routes.page_dashboard import router as dashboard_page_router
from src.api.routes.page_logs import router as logs_page_router
from src.api.routes.page_setup import router as setup_page_router
from src.api.v1.models.admin_models import router as admin_models_router
from src.config.settings import Config
from src.api.v1.chat.chat import router as chat_v1_router
from src.api.v1.system.config import router as config_v1_router
from src.api.v1.sigma.coverage import router as coverage_v1_router
from src.api.v1.system.dispatcher import router as dispatcher_v1_router
from src.api.v1.system.infrastructure import router as system_infra_v1_router
from src.api.v1.documents.documents import router as documents_v1_router
from src.api.v1.system.dashboard import router as dashboard_v1_router
from src.api.v1.infrastructure.embeddings import router as embeddings_v1_router
from src.api.v1.sigma.explain import router as explain_v1_router
from src.api.v1.infrastructure.feedback import router as feedback_v1_router
from src.api.v1.documents.files import router as files_v1_router
from src.api.v1.infrastructure.github import router as github_v1_router
from src.api.v1.infrastructure.llamacpp import router as llama_router
from src.api.v1.system.logs import router as logs_v1_router
from src.api.v1.system.orchestration import router as orchestration_v1_router
from src.api.v1.models.models_embedding import router as models_embedding_router
from src.api.v1.models.models_sparse import router as models_sparse_router
from src.api.v1.models.models_llm import router as models_llm_router
from src.api.v1.infrastructure.qdrant import router as qdrant_router
from src.api.v1.infrastructure.releases import router as releases_v1_router
from src.api.v1.chat.search import router as search_v1_router
from src.api.v1.documents.spec import router as spec_v1_router
from src.api.v1.system.system_prompt import router as prompts_v1_router
from src.api.v1.sigma.translate import router as translate_v1_router
from src.shared.http import close_all_async_pooled_clients
from src.shared.service_manager import shutdown_all_services
from src.config.settings import TEMP_DIR
from src.presentation import STATIC_DIR
from src.infrastructure.database import DatabaseService
from src.infrastructure.llm.llamacpp.auto_start import start_llamacpp, stop_llamacpp
from src.infrastructure.vectorstore.auto_start import start_qdrant, stop_qdrant
from src.shared.exceptions import SigmaError
from src.workers.processor import TaskDispatcher

logger = logging.getLogger(__name__)
setup_mode = os.environ.get("_SIGMA_SETUP_MODE", "").lower() in ("1", "true", "yes")

# Configure root logger to suppress verbose output during startup
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Remove any existing handlers that might print to stderr
for handler in root_logger.handlers[:]:
    if hasattr(handler, "stream") and getattr(handler.stream, "name", None) == "<stderr>":
        root_logger.removeHandler(handler)

# Log air-gap status at startup
_airgap_status = (
    "AIR-GAP MODE ENABLED"
    if os.environ.get("HF_HUB_OFFLINE") == "1"
    else "Online mode (HF Hub accessible)"
)
logger.info(f"=== {_airgap_status} ===")


def apply_db_config_overrides(db) -> Config:
    """Apply DuckDB config overrides to the singleton Config."""
    from src.config.settings import get_config

    config = get_config()
    for key, attr in (
        ("backend.os", "os"),
        ("backend.gpu_type", "gpu_type"),
        ("llamacpp_version", "llamacpp_version"),
        ("qdrant_version", "qdrant_version"),
        ("qdrant_webui_version", "qdrant_webui_version"),
        ("logging.level", "logging_level"),
        ("logging.log_max_size", "logging_log_max_size"),
        ("logging.log_max_file", "logging_log_max_file"),
        ("logging.clean_at_startup", "logging_clean_at_startup"),
    ):
        val = db.get_config(key)
        if val is not None and isinstance(val, dict):
            val = val.get("value")
        if val is not None:
            current = getattr(config, attr, None)
            if isinstance(current, int) and isinstance(val, str):
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    pass
            setattr(config, attr, val)
    return config


def _parse_log_size(size_str: str) -> int:
    """Parse a human-readable size string (e.g. '5M', '10M', '1G') to bytes."""
    size_str = size_str.strip().upper()
    if size_str.endswith("G"):
        return int(float(size_str[:-1]) * 1024 * 1024 * 1024)
    if size_str.endswith("M"):
        return int(float(size_str[:-1]) * 1024 * 1024)
    if size_str.endswith("K"):
        return int(float(size_str[:-1]) * 1024)
    return int(size_str)


def _setup_logging(level: str = "INFO", max_size: str = "10M", max_files: int = 5) -> None:
    """Setup logging to file with rotation."""
    from src.config.settings import LOGS_DIR

    log_file = LOGS_DIR / "sigmahqrag.log"
    log_level = getattr(logging, level.upper(), logging.INFO)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        log_file, maxBytes=_parse_log_size(max_size), backupCount=max_files, encoding="utf-8"
    )
    handler.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    for noisy in ("httpx", "httpcore", "urllib3", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def _init_qdrant_collections() -> None:
    """Ensure all RAG collections exist with hybrid search enabled."""
    from src.infrastructure.vectorstore import QdrantVectorService

    for col in ("sigma_rules", "sigma_docs", "sigma_spec"):
        try:
            svc = QdrantVectorService(collection_name=col)
            await svc.create_collection(enable_hybrid=True)
        except Exception as e:
            logger.warning("Could not create Qdrant collection '%s': %s", col, e)


def _clean_at_startup() -> None:
    """Clean temp, pid, and log files at startup when clean_at_startup is enabled."""
    import shutil

    from src.config.settings import LOGS_DIR, PID_DIR

    for d in (TEMP_DIR, PID_DIR):
        if d.exists():
            for p in d.iterdir():
                try:
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)
                except Exception as e:
                    print(f"Could not clean {p}: {e}")

    # Clear all log files (they will be recreated by the handler)
    if LOGS_DIR.exists():
        for p in LOGS_DIR.iterdir():
            if p.is_file():
                try:
                    p.write_text("")
                except Exception as e:
                    print(f"Could not clear {p}: {e}")
        print("Cleanup at startup: all log files cleared.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    dispatcher = None
    db = None

    # Capture stderr to suppress tqdm/Loading weights messages during startup
    old_stderr = sys.stderr
    sys.stderr = StringIO()

    def _status(message: str) -> None:
        print(f"  {message}", file=old_stderr, flush=True)

    try:
        db = DatabaseService()
        db.initialize()
        app.state.db = db

        from src.application.system.prompts import sync_prompts_from_files

        sync_prompts_from_files()

        Config.init_app()
        config = apply_db_config_overrides(db)
        logger.info("DuckDB config overrides applied.")

        if config.logging_clean_at_startup:
            _clean_at_startup()

        _setup_logging(
            level=config.logging_level,
            max_size=config.logging_log_max_size,
            max_files=config.logging_log_max_file,
        )
        logger.info("=== Lifespan starting ===")
        logger.info("Database initialized.")
        logger.info("Config initialized.")

        # Start the background task dispatcher in its own thread
        dispatcher = TaskDispatcher(poll_interval=0.2, max_workers=4)
        app.state.dispatcher = dispatcher
        dispatcher.start()
        logger.info("Dispatcher started in background thread.")

        _validate_services()
        logger.info("Services validated.")

        if not setup_mode:
            _status("Starting llama.cpp...")
            try:
                await start_llamacpp()
                logger.info("llama.cpp started.")
            except Exception as e:
                logger.warning("llama.cpp auto-start skipped: %s", e)

            qdrant_started = False
            _status("Starting Qdrant...")
            try:
                await start_qdrant()
                qdrant_started = True
                logger.info("Qdrant started.")
            except Exception as e:
                logger.warning("Qdrant auto-start skipped: %s", e)

            if qdrant_started:
                _status("Initializing Qdrant collections...")
                await _init_qdrant_collections()
                logger.info("Qdrant collections initialized.")
        else:
            logger.info("Setup mode: skipping service startup.")

        logger.info("=== Application startup complete ===")
    except BaseException as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    finally:
        # Restore stderr
        sys.stderr = old_stderr

    yield
    if dispatcher:
        dispatcher.stop()
        logger.info("Dispatcher stopped.")
    await close_all_async_pooled_clients()
    await stop_llamacpp()
    await shutdown_all_services()
    await stop_qdrant()
    if db:
        try:
            db.persist()
        except Exception as exc:
            logger.warning("Failed to persist database on shutdown: %s", exc)
        db.close()


def _validate_services() -> None:
    """Validate required services are configured."""
    from src.config.settings import get_config

    config = get_config()
    if not config.llama_base_url:
        logger.warning("LLM service not configured (llama_base_url missing)")
    if not config.qdrant_collection_name:
        logger.warning("Qdrant service not configured (qdrant_collection_name missing)")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="SigmaHQ RAG",
        version="0.1.0",
        description="Local RAG system for Sigma rules",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def csrf_protection(request: Request, call_next):
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            origin = request.headers.get("origin", "")
            referer = request.headers.get("referer", "")
            if origin or referer:
                allowed = False
                for val in (origin, referer):
                    if not val:
                        continue
                    try:
                        parsed = URL(val)
                        if (
                            parsed.hostname in ("localhost", "127.0.0.1", "::1")
                            or parsed.scheme == "null"
                        ):
                            allowed = True
                            break
                    except Exception:
                        pass
                if not allowed:
                    return JSONResponse(
                        status_code=403, content={"error": "Cross-site request blocked"}
                    )
        return await call_next(request)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    app.include_router(setup_page_router)
    app.include_router(admin_pages_router)
    app.include_router(admin_models_router)
    app.include_router(dashboard_page_router)
    app.include_router(logs_page_router)
    app.include_router(dashboard_v1_router)
    app.include_router(config_v1_router)
    app.include_router(coverage_v1_router)
    app.include_router(dispatcher_v1_router)
    app.include_router(explain_v1_router)
    app.include_router(files_v1_router)
    app.include_router(github_v1_router)
    app.include_router(llama_router)
    app.include_router(logs_v1_router)
    app.include_router(orchestration_v1_router)
    app.include_router(system_infra_v1_router)
    app.include_router(models_llm_router)
    app.include_router(models_embedding_router)
    app.include_router(models_sparse_router)
    app.include_router(qdrant_router)
    app.include_router(releases_v1_router)
    app.include_router(search_v1_router)
    app.include_router(spec_v1_router)
    app.include_router(prompts_v1_router)
    app.include_router(translate_v1_router)
    app.include_router(chat_v1_router)
    app.include_router(chat_page_router)
    app.include_router(data_page_router)
    app.include_router(documents_v1_router)
    app.include_router(embeddings_v1_router)
    app.include_router(feedback_v1_router)

    @app.exception_handler(SigmaError)
    async def sigma_error_handler(request: Request, exc: SigmaError) -> JSONResponse:
        """Global handler for SigmaError exceptions."""
        logger.error(f"SigmaError ({exc.code}): {exc.message}")
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.to_dict(),
        )

    return app


app = create_app()
