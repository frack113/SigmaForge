from __future__ import annotations

import json

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.config.settings import get_config
from src.presentation import TEMPLATES_DIR

templates = Jinja2Templates(directory=TEMPLATES_DIR)

router = APIRouter(prefix="", tags=["page-logs"])


@router.get("/logs", response_class=HTMLResponse)
async def logs_explorer(request: Request):
    cfg = get_config().to_dict()
    config_json = json.dumps(cfg).replace("</", "<\\/")
    return templates.TemplateResponse(
        request=request,
        name="logs/index.html.j2",
        context={"config": cfg, "config_json": config_json},
    )
