from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any, Mapping

import httpx

from sigmaforge.errors import LlamaError


class LlamaClient:
    def __init__(
        self,
        base_url: str,
        model: str = "sigma",
        api_key: str = "sigma-key",
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout) as client:
                response = await client.post(url, json=payload, headers=self._headers())
                response.raise_for_status()
                data: dict[str, Any] = response.json()
                return data
        except httpx.HTTPError as exc:
            raise LlamaError(str(exc)) from exc
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise LlamaError(str(exc)) from exc

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: str | Sequence[str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop is not None:
            payload["stop"] = stop
        data = await self._post_json("/v1/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise LlamaError("completion response contains no choices")
        return str(choices[0].get("text", ""))

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: str | Sequence[str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop is not None:
            payload["stop"] = stop
        data = await self._post_json("/v1/chat/completions", payload)
        choices = data.get("choices") or []
        if not choices:
            raise LlamaError("chat response contains no choices")
        return str(choices[0].get("message", {}).get("content", ""))

    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: str | Sequence[str] | None = None,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if stop is not None:
            payload["stop"] = stop
        url = f"{self.base_url}/v1/chat/completions"
        try:
            async with httpx.AsyncClient(transport=self.transport, timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload, headers=self._headers()) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        choices = event.get("choices") or []
                        if not choices:
                            continue
                        chunk = choices[0].get("delta", {}).get("content")
                        if chunk:
                            yield str(chunk)
        except httpx.HTTPError as exc:
            raise LlamaError(str(exc)) from exc
