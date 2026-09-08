"""Chat service for processing messages and managing sessions."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from src.application.chat.rag import RAGPipeline
from src.application.sigma.validator import SigmaValidator
from src.application.sigma.translate import (
    detect_sigma_yaml,
    extract_yaml_block,
    translate_detection,
)
from src.application.tools import ToolContext, ToolDispatcher, get_tools
from src.core.search.engine import SearchEngine, format_result_by_collection
from src.core.sigma.models import SigmaRule
from src.api.v1.chat.schemas import ChatMode
from src.shared.session import SessionStore

logger = logging.getLogger(__name__)

MAX_HISTORY = 50
MAX_TOOL_CALLS = 5

_session_store = SessionStore()


def get_session_store() -> SessionStore:
    return _session_store


class ChatService:
    """Service layer for chat operations.

    Session-scoped state (history, uploaded rule, citations) is stored in
    a shared :class:`SessionStore` keyed by *session_id*.  Callers should
    pass a unique *session_id* (e.g. from ``X-Session-ID`` header) to
    avoid state leaking between users.
    """

    def __init__(self, use_router: bool = True) -> None:
        self.search_engine = SearchEngine(use_router=use_router)
        self.rag_pipeline = RAGPipeline()
        self.validator = SigmaValidator()

        # Tool-calling setup
        self._tool_executor = ToolDispatcher(get_tools())
        self._tool_context = ToolContext(
            search_engine=self.search_engine,
            llm_client=self.rag_pipeline.llm_client,
            rag_pipeline=self.rag_pipeline,
        )

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------

    def _sid(self, session_id: str | None) -> str:
        return session_id or "_default"

    def _get_rule(self, session_id: str | None) -> SigmaRule | None:
        return get_session_store().get(self._sid(session_id), "_uploaded_rule")

    def _set_rule(self, session_id: str | None, rule: SigmaRule | None) -> None:
        get_session_store().set(self._sid(session_id), "_uploaded_rule", rule)

    def _get_history(self, session_id: str | None) -> list[dict[str, str]]:
        return get_session_store().get(self._sid(session_id), "_history", [])

    def _set_history(self, session_id: str | None, history: list[dict[str, str]]) -> None:
        get_session_store().set(self._sid(session_id), "_history", history)

    def _get_citations(self, session_id: str | None) -> list[str]:
        return get_session_store().get(self._sid(session_id), "_last_citations", [])

    def _set_citations(self, session_id: str | None, citations: list[str]) -> None:
        get_session_store().set(self._sid(session_id), "_last_citations", citations)

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas for the LLM."""
        return self._tool_executor.list_tools()

    async def process_message(
        self,
        message: str,
        mode: str = ChatMode.SEARCH.value,
        model: str = "",
        prompt_id: str = "",
        session_id: str | None = None,
    ) -> str:
        """Process a chat message based on the current mode.

        Args:
            message: User message text
            mode: Chat mode (search, coverage, explain)
            model: Selected LLM model path
            prompt_id: Selected system prompt ID
            session_id: Session identifier for state isolation.

        Returns:
            AI response text
        """
        if not message.strip():
            return ""

        sid = self._sid(session_id)
        self._add_to_history("user", message, sid)

        try:
            if mode == ChatMode.EXPLAIN.value:
                response = await self._handle_explain(message, prompt_id, sid)
            elif mode == ChatMode.COVERAGE.value:
                response = await self._handle_coverage(message, prompt_id, sid)
            else:
                response = await self._handle_search(message, prompt_id)
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            response = f"Error processing message: {str(e)}"

        self._add_to_history("assistant", response, sid)
        return response

    async def process_message_stream(
        self,
        message: str,
        mode: str = ChatMode.SEARCH.value,
        model: str = "",
        prompt_id: str = "",
        session_id: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat message response based on the current mode."""
        if not message.strip():
            return

        sid = self._sid(session_id)
        self._add_to_history("user", message, sid)
        if model:
            logger.info("Selected model: %s", model)
        if prompt_id:
            logger.info("Selected prompt_id: %s", prompt_id)

        accumulated: list[str] = []
        try:
            if mode == ChatMode.EXPLAIN.value:
                async for token in self._handle_explain_stream(message, prompt_id, sid):
                    accumulated.append(token)
                    yield token
            elif mode == ChatMode.COVERAGE.value:
                async for token in self._handle_coverage_stream(message, prompt_id, sid):
                    accumulated.append(token)
                    yield token
            else:
                async for token in self._handle_search_stream(message, prompt_id, sid):
                    accumulated.append(token)
                    yield token
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            error_text = f"Error processing message: {str(e)}"
            accumulated.append(error_text)
            yield error_text

        self._add_to_history("assistant", "".join(accumulated), sid)

    async def _execute_tool_calls(self, messages: list[dict[str, Any]]) -> str:
        """Execute tool calls returned by the LLM in a multi-turn loop.

        1. Send messages + tool schemas to LLM
        2. If LLM returns tool_calls → execute them → append results → repeat
        3. If LLM returns text → return it
        4. Bail out after MAX_TOOL_CALLS iterations

        Returns the final assistant text response.
        """
        schemas = self._get_tool_schemas()
        turns = 0

        while turns < MAX_TOOL_CALLS:
            turns += 1

            # Send chat request with tools and get raw response
            tool_response = await self._send_chat_with_tools(messages, schemas)
            choices = tool_response.get("choices") or []
            if not choices:
                return "No response from model."

            message = choices[0].get("message") or {}
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                # LLM returned text, done
                return message.get("content", "")  # type: ignore[no-any-return]

            # Execute each tool call
            tool_messages = [message] if "content" in message else []
            for tc in tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                fn_name = func.get("name", "")
                fn_args_str = func.get("arguments", "{}")

                try:
                    fn_args = json.loads(fn_args_str) if fn_args_str else {}
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                try:
                    result = await self._tool_executor.execute(
                        fn_name, fn_args, tc_id, ctx=self._tool_context
                    )
                    tool_result = {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result.content,
                    }
                except Exception as e:
                    logger.warning("Tool '%s' failed: %s", fn_name, e)
                    tool_result = {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": f"Error: {e}",
                    }

                tool_messages.append(tool_result)

            messages.extend(tool_messages)

        logger.warning("Max tool call iterations (%d) reached", MAX_TOOL_CALLS)
        return "Too many tool calls. Please simplify your request."

    async def _send_chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send a chat request with tools and return the raw JSON response."""
        try:
            return await self.rag_pipeline.llm_client.chat_raw(
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                tools=tools,
            )
        except Exception as e:
            logger.error("Tool-calling chat failed: %s", e)
            raise

    async def _handle_search(self, message: str, prompt_id: str) -> str:
        """Handle search mode: multi-turn tool-calling loop."""
        prompt_content = self.rag_pipeline._resolve_prompt(prompt_id, mode="search")
        assistant_message = {
            "role": "assistant",
            "content": message,
        }
        system_msg = {
            "role": "system",
            "content": prompt_content,
        }
        messages = [system_msg, assistant_message]
        return await self._execute_tool_calls(messages)

    async def _handle_explain(self, message: str, prompt_id: str, sid: str) -> str:
        """Handle explain mode: analyze uploaded Sigma rule."""
        rule = self._get_rule(sid)
        if not rule:
            return "No Sigma rule uploaded. Please upload a .yaml file first."

        related = await self.search_engine.search(rule.name)
        return await self.rag_pipeline.explain_rule(rule, related, system_prompt_id=prompt_id)

    async def _handle_explain_stream(
        self,
        message: str,
        prompt_id: str,
        sid: str,
    ) -> AsyncGenerator[str, None]:
        """Handle explain mode with streaming."""
        rule = self._get_rule(sid)
        if not rule:
            yield "No Sigma rule uploaded. Please upload a .yaml file first."
            return

        related = await self.search_engine.search(rule.name)
        async for token in self.rag_pipeline.explain_rule_stream(
            rule, related, system_prompt_id=prompt_id
        ):
            yield token

    async def _handle_coverage(self, message: str, prompt_id: str, sid: str) -> str:
        """Handle coverage mode."""
        rule = self._get_rule(sid)
        if not rule:
            return "No Sigma rule uploaded. Upload a .yaml file to check coverage."

        results = await self.search_engine.search(message)
        if not results:
            return "No related rules found for coverage analysis."

        try:
            return await self.rag_pipeline.analyze_coverage(
                rule, results, system_prompt_id=prompt_id
            )
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")
            return f"Found {len(results)} related rules for coverage comparison."

    async def _handle_search_stream(
        self,
        message: str,
        prompt_id: str,
        sid: str,
    ) -> AsyncGenerator[str, None]:
        """Handle search mode with streaming: semantic search + LLM token stream.

        If the message contains Sigma YAML, auto-translates and includes
        the translation as context.
        """
        # Auto-detect and translate Sigma YAML
        translation = ""
        if detect_sigma_yaml(message):
            yaml_block = extract_yaml_block(message)
            if yaml_block:
                logger.info("Detected Sigma YAML in chat message — auto-translating")
                translation = await translate_detection(yaml_block, self.rag_pipeline)
                try:
                    await self.rag_pipeline.llm_client.erase_slot_cache()
                except Exception:
                    logger.warning("Failed to clear KV cache after translate")

        results = await self.search_engine.search(message)

        if not results and not translation:
            yield "No matching Sigma rules found. Try a different query."
            return

        citations = [
            self.search_engine.get_citation(r)
            for r in results[:5]
            if self.search_engine.get_citation(r)
        ]
        self._set_citations(sid, citations)

        if citations:
            yield f"__CITATIONS__:{json.dumps(citations)}"

        # If we have a translation, prepend it as context
        augmented_message = message
        if translation:
            augmented_message = (
                f"[Auto-translated detection]\n{translation}\n\n[Original YAML]\n{message}"
            )

        formatted_results = [format_result_by_collection(r) for r in results]

        try:
            found = False
            async for token in self.rag_pipeline.answer_search_query_stream(
                augmented_message, formatted_results, system_prompt_id=prompt_id
            ):
                yield token
                found = True
            if not found:
                logger.warning("RAG stream returned no tokens")
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")

    async def _handle_coverage_stream(
        self,
        message: str,
        prompt_id: str,
        sid: str,
    ) -> AsyncGenerator[str, None]:
        """Handle coverage mode with streaming."""
        rule = self._get_rule(sid)
        if not rule:
            yield "No Sigma rule uploaded. Upload a .yaml file to check coverage."
            return

        results = await self.search_engine.search(message)
        if not results:
            yield "No related rules found for coverage analysis."
            return

        formatted_results = [format_result_by_collection(r) for r in results]

        try:
            async for token in self.rag_pipeline.analyze_coverage_stream(
                rule, formatted_results, system_prompt_id=prompt_id
            ):
                yield token
        except Exception as e:
            logger.error(f"RAG pipeline failed: {e}")

    async def validate_and_store_yaml(
        self, content: bytes, session_id: str | None = None
    ) -> SigmaRule:
        """Validate YAML content and store rule data in session.

        Args:
            content: Raw YAML file content
            session_id: Session identifier for state isolation.

        Returns:
            Parsed and validated SigmaRule
        """
        rule = self.validator.validate(content)
        self._set_rule(session_id, rule)
        self.rag_pipeline.cache.invalidate()
        return rule

    def get_last_citations(self, session_id: str | None = None) -> list[str]:
        """Get citations from the last search response."""
        return self._get_citations(session_id).copy()

    def _add_to_history(self, role: str, content: str, sid: str) -> None:
        """Add a message to chat history (max 50 messages)."""
        history = self._get_history(sid)
        history.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]
        self._set_history(sid, history)

    def get_history(self, session_id: str | None = None) -> list[dict[str, str]]:
        """Get chat history."""
        return self._get_history(session_id).copy()

    async def clear_history(self, session_id: str | None = None) -> None:
        """Clear chat history, uploaded rule, and llama.cpp KV cache."""
        sid = self._sid(session_id)
        self._set_history(sid, [])
        self._set_rule(sid, None)
        self._set_citations(sid, [])
        try:
            await self.rag_pipeline.llm_client.erase_slot_cache()
        except Exception:
            logger.warning("Failed to clear llama.cpp KV cache")
