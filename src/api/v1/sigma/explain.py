"""Explain API for Sigma rules."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.infrastructure.llm.llamacpp import LlamaClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/explain", tags=["v1-explain"])


class ExplainRequest(BaseModel):
    """Request model for rule explanation."""

    rule_id: str = Field(..., description="Rule ID")
    text: str = Field(..., description="Rule content")


@router.post("/rule")
async def explain_rule(
    request: ExplainRequest,
) -> JSONResponse:
    """Explain a Sigma rule using the backend LLM service.

    Args:
        request: Rule ID and content

    Returns:
        JSON with rule explanation
    """
    if not request.rule_id or not request.text:
        return JSONResponse(
            status_code=400,
            content={"error": "Rule ID and text are required"},
        )

    system_prompt = (
        "You are a Sigma rule expert. Explain the given Sigma rule "
        "in plain language: what it detects, the log source, the selection "
        "logic, and any false positive considerations."
    )

    try:
        client = LlamaClient()
        explanation = await client.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.text},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return JSONResponse(
            content={
                "rule_id": request.rule_id,
                "explanation": explanation,
                "text": request.text,
            }
        )
    except Exception as e:
        logger.error(f"Explain error: {e}")
        return JSONResponse(status_code=500, content={"error": "An internal error occurred"})
