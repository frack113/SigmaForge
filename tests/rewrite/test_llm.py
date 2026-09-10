from __future__ import annotations

import asyncio

import httpx
import pytest

from sigmaforge.errors import LlamaError
from sigmaforge.llm import LlamaClient


def test_chat_returns_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}}]})

    client = LlamaClient(
        "http://llm.test",
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(
        client.chat(
            [{"role": "user", "content": "hi"}],
        )
    )

    assert result == "hello"


def test_complete_returns_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/completions"
        return httpx.Response(200, json={"choices": [{"text": "completed"}]})

    client = LlamaClient(
        "http://llm.test",
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(client.complete("prompt"))

    assert result == "completed"


def test_stream_chat_yields_chunks() -> None:
    body = (
        b'data: {"choices": [{"delta": {"content": "hel"}}]}\n'
        b'data: {"choices": [{"delta": {"content": "lo"}}]}\n'
        b"data: [DONE]\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    client = LlamaClient(
        "http://llm.test",
        transport=httpx.MockTransport(handler),
    )

    async def collect() -> list[str]:
        chunks: list[str] = []
        async for chunk in client.stream_chat([{"role": "user", "content": "hi"}]):
            chunks.append(chunk)
        return chunks

    assert asyncio.run(collect()) == ["hel", "lo"]


def test_http_error_raises_llama_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    client = LlamaClient(
        "http://llm.test",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LlamaError):
        asyncio.run(client.chat([{"role": "user", "content": "hi"}]))
