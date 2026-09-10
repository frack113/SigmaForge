"""Admin page routes for Jinja2 templates."""

from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from src.config.settings import get_config
from src.presentation import TEMPLATES_DIR

templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter(prefix="", tags=["page-admin"])


@router.get("/config")
@router.get("/config/")
async def config_page(request: Request):
    """Serve the unified config dashboard page."""
    cfg = get_config().to_dict()
    # Escape "</" so embedded JSON cannot break out of the <script> tag.
    config_json = json.dumps(cfg).replace("</", "<\\/")
    return templates.TemplateResponse(
        request=request,
        name="config/config.html.j2",
        context={"config": cfg, "config_json": config_json},
    )


# Legacy redirects for backward compatibility
@router.get("/config/system")
@router.get("/config/system/")
async def config_system_redirect(request: Request):
    """Redirect to unified config page."""
    return RedirectResponse(url="/config", status_code=302)


@router.get("/config/backend")
@router.get("/config/backend/")
async def config_backend_redirect(request: Request):
    """Redirect to unified config page."""
    return RedirectResponse(url="/config", status_code=302)


@router.get("/config/llm")
@router.get("/config/llm/")
async def config_llm_redirect(request: Request):
    """Redirect to unified config page."""
    return RedirectResponse(url="/config", status_code=302)
