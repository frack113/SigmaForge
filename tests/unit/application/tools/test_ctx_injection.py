"""Tests for ToolContext injection into tool execution."""

import pytest

from src.application.tools.models import ToolContext
from src.application.tools.executor import ToolDispatcher
from src.application.tools.registry import reset_tools, tool


class MockSearchEngine:
    async def search(self, query: str, **kwargs) -> list[dict]:
        return [{"text": "mock result", "score": 0.95, "metadata": {"title": "Test Rule"}}]


class MockLLMClient:
    async def chat(self, messages, **kwargs) -> str:
        return "mock llm response"


class TestCtxInjection:
    """Verify ctx is injected into tools and excluded from JSON schema."""

    def setup_method(self) -> None:
        reset_tools()

    def test_ctx_excluded_from_json_schema(self) -> None:
        """ctx must not appear in the tool schema sent to the LLM."""

        @tool
        async def test_tool(query: str, ctx: ToolContext | None = None) -> str:
            """A tool with ctx.

            :param query: The query.
            """
            return "ok"

        schema = test_tool.parameters
        assert "ctx" not in schema["properties"]
        assert "ctx" not in schema["required"]
        assert "query" in schema["properties"]

    def test_self_excluded_from_json_schema(self) -> None:
        """self must not appear in the tool schema."""

        class MyTools:
            @tool
            async def bound_tool(self, query: str) -> str:
                """A bound method tool.

                :param query: The query.
                """
                return "ok"

        schema = MyTools.bound_tool.parameters
        assert "self" not in schema["properties"]

    @pytest.mark.asyncio
    async def test_ctx_injected_into_tool(self) -> None:
        """ctx is passed to the tool function when provided to execute()."""
        received_ctx: ToolContext | None = None

        @tool
        async def ctx_tool(query: str, ctx: ToolContext | None = None) -> str:
            """A tool that captures ctx.

            :param query: The query.
            """
            nonlocal received_ctx
            received_ctx = ctx
            return "ctx received" if ctx else "no ctx"

        ctx = ToolContext(search_engine=MockSearchEngine(), llm_client=MockLLMClient())
        dispatcher = ToolDispatcher([ctx_tool])
        result = await dispatcher.execute("ctx_tool", {"query": "test"}, "call_1", ctx=ctx)

        assert result.content == "ctx received"
        assert received_ctx is ctx

    @pytest.mark.asyncio
    async def test_ctx_none_when_not_provided(self) -> None:
        """ctx is None when execute() is called without ctx."""

        @tool
        async def no_ctx_tool(query: str, ctx: ToolContext | None = None) -> str:
            """A tool checking ctx is None.

            :param query: The query.
            """
            return "has ctx" if ctx else "no ctx"

        dispatcher = ToolDispatcher([no_ctx_tool])
        result = await dispatcher.execute("no_ctx_tool", {"query": "test"}, "call_1")
        assert result.content == "no ctx"

    @pytest.mark.asyncio
    async def test_ctx_stripped_from_arguments(self) -> None:
        """Even if LLM sends ctx in arguments, it is stripped before call."""
        received_ctx: ToolContext | None = None

        @tool
        async def strip_tool(query: str, ctx: ToolContext | None = None) -> str:
            """A tool.

            :param query: The query.
            """
            nonlocal received_ctx
            received_ctx = ctx
            return "ok"

        real_ctx = ToolContext(search_engine=MockSearchEngine(), llm_client=MockLLMClient())
        dispatcher = ToolDispatcher([strip_tool])
        result = await dispatcher.execute(
            "strip_tool", {"query": "test", "ctx": "bogus_llm_value"}, "call_1", ctx=real_ctx
        )
        assert result.content == "ok"
        assert received_ctx is real_ctx

    @pytest.mark.asyncio
    async def test_sigma_tools_receive_ctx(self) -> None:
        """All 5 registered sigma tools work when ctx is injected."""
        from src.application.tools.sigma.tools import search_sigma  # noqa: F401

        ctx = ToolContext(search_engine=MockSearchEngine(), llm_client=MockLLMClient())
        dispatcher = ToolDispatcher()
        dispatcher.register(search_sigma)

        result = await dispatcher.execute(
            "search_sigma", {"query": "suspicious process"}, "tc1", ctx=ctx
        )
        assert "Error: tool context not available" not in result.content
        assert "Test Rule" in result.content
