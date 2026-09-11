from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from sigmaforge.errors import LlamaError
from sigmaforge.llm import LlamaClient
from sigmaforge.rag import RAGAnswer, RAGPipeline, render_search_prompt
from sigmaforge.search import SearchEngine, SearchResult


def _result(index: int, text: str, source: str) -> SearchResult:
    return SearchResult(
        id=f"id-{index}",
        collection="sigma_rules",
        text=text,
        score=1.0 / index,
        payload={"source_file": source},
    )


class FakeSearchEngine:
    def __init__(self, results: Sequence[SearchResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, int | None]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        collections: Sequence[str] | None = None,
        filters: Mapping[str, str] | None = None,
        extra_filter: Any | None = None,
    ) -> list[SearchResult]:
        self.calls.append((query, top_k))
        if top_k is not None:
            return self._results[:top_k]
        return list(self._results)


class FakeLlamaClient:
    def __init__(
        self,
        response: str = "Default answer.",
        chunks: Sequence[str] | None = None,
        *,
        fail_chat: bool = False,
        fail_stream: bool = False,
    ) -> None:
        self.model = "fake-model"
        self._response = response
        self._chunks = list(chunks) if chunks is not None else [response]
        self._fail_chat = fail_chat
        self._fail_stream = fail_stream
        self.messages: list[dict[str, Any]] = []
        self.stream_messages: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: str | Sequence[str] | None = None,
    ) -> str:
        self.messages = [dict(message) for message in messages]
        if self._fail_chat:
            raise LlamaError("chat failed")
        return self._response

    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: str | Sequence[str] | None = None,
    ) -> AsyncIterator[str]:
        self.stream_messages = [dict(message) for message in messages]
        if self._fail_stream:
            raise LlamaError("stream failed")
        for chunk in self._chunks:
            yield chunk


class PartialFailLlamaClient:
    model = "fake-model"

    async def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: str | Sequence[str] | None = None,
    ) -> str:
        return "unused"

    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 512,
        stop: str | Sequence[str] | None = None,
    ) -> AsyncIterator[str]:
        yield "partial"
        raise LlamaError("stream failed after partial output")


def test_render_search_prompt_includes_context_and_question() -> None:
    prompt = render_search_prompt("search context", "question text")
    assert "search context" in prompt
    assert "question text" in prompt


def test_real_search_and_llm_types_are_accepted_by_pipeline() -> None:
    def create_pipeline(search_engine: SearchEngine, llm_client: LlamaClient) -> RAGPipeline:
        return RAGPipeline(search_engine, llm_client)

    assert create_pipeline is not None


def test_rag_answer_to_dict_includes_sources() -> None:
    result = _result(1, "rule alpha", "alpha.yml")
    answer = RAGAnswer(
        query="q",
        answer="a",
        sources=(result,),
        model="m",
        fallback=False,
    )

    data = answer.to_dict()

    assert data["query"] == "q"
    assert data["answer"] == "a"
    assert data["model"] == "m"
    assert data["fallback"] is False
    assert data["sources"][0]["id"] == "id-1"


async def test_answer_returns_llm_response_and_sources() -> None:
    results = [
        _result(1, "rule alpha detects powershell", "alpha.yml"),
        _result(2, "rule beta detects wmi", "beta.yml"),
    ]
    search = FakeSearchEngine(results)
    llm = FakeLlamaClient("Rule alpha detects PowerShell.")
    pipeline = RAGPipeline(search, llm)

    answer = await pipeline.answer("which rule detects powershell?")

    assert isinstance(answer, RAGAnswer)
    assert answer.answer == "Rule alpha detects PowerShell."
    assert answer.fallback is False
    assert answer.model == "fake-model"
    assert answer.sources == tuple(results)
    assert search.calls == [("which rule detects powershell?", 5)]

    system = str(llm.messages[0]["content"])
    user = str(llm.messages[1]["content"])
    assert "which rule detects powershell?" in system
    assert "rule alpha detects powershell" in system
    assert user == "which rule detects powershell?"


async def test_answer_passes_top_k_override_to_search() -> None:
    results = [
        _result(1, "rule alpha", "alpha.yml"),
        _result(2, "rule beta", "beta.yml"),
        _result(3, "rule gamma", "gamma.yml"),
    ]
    search = FakeSearchEngine(results)
    llm = FakeLlamaClient()
    pipeline = RAGPipeline(search, llm)

    answer = await pipeline.answer("query", top_k=2)

    assert search.calls == [("query", 2)]
    assert len(answer.sources) == 2


async def test_answer_without_results_returns_fallback_without_llm_call() -> None:
    search = FakeSearchEngine([])
    llm = FakeLlamaClient()
    pipeline = RAGPipeline(search, llm)

    answer = await pipeline.answer("missing")

    assert answer.fallback is True
    assert answer.sources == ()
    assert "No matching" in answer.answer
    assert llm.messages == []


async def test_answer_llm_failure_returns_fallback_with_sources() -> None:
    results = [
        _result(1, "rule alpha detects powershell", "alpha.yml"),
        _result(2, "rule beta detects wmi", "beta.yml"),
    ]
    search = FakeSearchEngine(results)
    llm = FakeLlamaClient(fail_chat=True)
    pipeline = RAGPipeline(search, llm)

    answer = await pipeline.answer("which rule detects powershell?")

    assert answer.fallback is True
    assert answer.sources == tuple(results)
    assert "LLM unavailable" in answer.answer
    assert "rule alpha detects powershell" in answer.answer


async def test_answer_stream_yields_llm_chunks() -> None:
    results = [_result(1, "rule alpha detects powershell", "alpha.yml")]
    search = FakeSearchEngine(results)
    llm = FakeLlamaClient(chunks=["Pow", "erShell"])
    pipeline = RAGPipeline(search, llm)

    chunks = [chunk async for chunk in pipeline.answer_stream("powershell")]

    assert chunks == ["Pow", "erShell"]
    system = str(llm.stream_messages[0]["content"])
    assert "powershell" in system
    assert "rule alpha detects powershell" in system


async def test_answer_stream_without_results_yields_fallback() -> None:
    search = FakeSearchEngine([])
    llm = FakeLlamaClient()
    pipeline = RAGPipeline(search, llm)

    chunks = [chunk async for chunk in pipeline.answer_stream("missing")]

    assert chunks == ["No matching Sigma rules or documents were found."]


async def test_answer_stream_failure_before_output_yields_fallback() -> None:
    results = [_result(1, "rule alpha detects powershell", "alpha.yml")]
    search = FakeSearchEngine(results)
    llm = FakeLlamaClient(fail_stream=True)
    pipeline = RAGPipeline(search, llm)

    chunks = [chunk async for chunk in pipeline.answer_stream("powershell")]

    assert len(chunks) == 1
    assert "LLM unavailable" in chunks[0]


async def test_answer_stream_failure_after_partial_output_does_not_add_fallback() -> None:
    results = [_result(1, "rule alpha detects powershell", "alpha.yml")]
    search = FakeSearchEngine(results)
    llm = PartialFailLlamaClient()
    pipeline = RAGPipeline(search, llm)

    chunks = [chunk async for chunk in pipeline.answer_stream("powershell")]

    assert chunks == ["partial"]
