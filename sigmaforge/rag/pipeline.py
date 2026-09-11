from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Protocol

from sigmaforge.errors import LlamaError
from sigmaforge.rag.models import RAGAnswer
from sigmaforge.rag.prompts import render_search_prompt
from sigmaforge.search import SearchResult, format_context

logger = logging.getLogger(__name__)


class SearchEngineLike(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        collections: Sequence[str] | None = None,
        filters: Mapping[str, str] | None = None,
        extra_filter: Any | None = None,
    ) -> list[SearchResult]:
        ...


class LlamaClientLike(Protocol):
    model: str

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: str | Sequence[str] | None = None,
    ) -> str:
        ...

    def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: str | Sequence[str] | None = None,
    ) -> AsyncIterator[str]:
        ...


class RAGPipeline:
    def __init__(
        self,
        search_engine: SearchEngineLike,
        llm_client: LlamaClientLike,
        *,
        top_k: int = 5,
        context_max_results: int = 5,
        context_max_chars: int = 1200,
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> None:
        self._search_engine = search_engine
        self._llm_client = llm_client
        self._top_k = top_k
        self._context_max_results = context_max_results
        self._context_max_chars = context_max_chars
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def answer(
        self,
        query: str,
        *,
        top_k: int | None = None,
        collections: Sequence[str] | None = None,
        filters: Mapping[str, str] | None = None,
        extra_filter: Any | None = None,
    ) -> RAGAnswer:
        limit = top_k if top_k is not None else self._top_k
        results = self._search_engine.search(
            query,
            top_k=limit,
            collections=collections,
            filters=filters,
            extra_filter=extra_filter,
        )
        if not results:
            return RAGAnswer(
                query=query,
                answer=self._fallback_answer(results),
                sources=(),
                model=self._llm_client.model,
                fallback=True,
            )

        context = format_context(
            results,
            max_results=self._context_max_results,
            max_chars=self._context_max_chars,
        )
        prompt = render_search_prompt(context, query)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]

        try:
            response = await self._llm_client.chat(
                messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except LlamaError as exc:
            logger.warning("RAG LLM generation failed: %s", exc)
            return RAGAnswer(
                query=query,
                answer=self._fallback_answer(results),
                sources=tuple(results),
                model=self._llm_client.model,
                fallback=True,
            )

        return RAGAnswer(
            query=query,
            answer=response.strip(),
            sources=tuple(results),
            model=self._llm_client.model,
            fallback=False,
        )

    async def answer_stream(
        self,
        query: str,
        *,
        top_k: int | None = None,
        collections: Sequence[str] | None = None,
        filters: Mapping[str, str] | None = None,
        extra_filter: Any | None = None,
    ) -> AsyncIterator[str]:
        limit = top_k if top_k is not None else self._top_k
        results = self._search_engine.search(
            query,
            top_k=limit,
            collections=collections,
            filters=filters,
            extra_filter=extra_filter,
        )
        if not results:
            yield self._fallback_answer(results)
            return

        context = format_context(
            results,
            max_results=self._context_max_results,
            max_chars=self._context_max_chars,
        )
        prompt = render_search_prompt(context, query)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ]

        started = False
        try:
            async for chunk in self._llm_client.stream_chat(
                messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            ):
                started = True
                yield chunk
        except LlamaError as exc:
            logger.warning("RAG LLM stream failed: %s", exc)
            if not started:
                yield self._fallback_answer(results)

    def _fallback_answer(self, results: Sequence[SearchResult]) -> str:
        if not results:
            return "No matching Sigma rules or documents were found."
        lines = [
            f"{index}. {result.text[:200]}" for index, result in enumerate(results[:2], start=1)
        ]
        return "LLM unavailable. Top results:\n" + "\n".join(lines)
