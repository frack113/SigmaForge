"""Tool executor that dispatches calls to registered ToolDefs."""

from __future__ import annotations

import logging
from typing import Any

from .models import ToolContext, ToolDef, ToolExecutor, ToolResult

logger = logging.getLogger(__name__)


class ToolExecutionError(Exception):
    """Raised when a tool raises during execution."""

    def __init__(self, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class ToolDispatcher(ToolExecutor):
    """Executes tools by name, returning structured results."""

    def __init__(self, tools: list[ToolDef] | None = None) -> None:
        self._tools: dict[str, ToolDef] = {}
        for t in tools or []:
            self._tools[t.name] = t

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
        ctx: ToolContext | None = None,
    ) -> ToolResult:
        tool = self._tools.get(tool_name)
        if not tool:
            raise ToolExecutionError(
                tool_name,
                f"Unknown tool '{tool_name}'. Available tools: {list(self._tools.keys())}",
            )

        arguments = {k: v for k, v in arguments.items() if k != "ctx"}
        if ctx is not None and tool.has_ctx:
            arguments["ctx"] = ctx
        try:
            result = await tool.fn(**arguments)
            return ToolResult(content=str(result), tool_call_id=tool_call_id)
        except ToolExecutionError:
            raise
        except ValueError as e:
            raise ToolExecutionError(tool_name, f"Validation error: {e}") from e
        except Exception as e:
            logger.error("Tool '%s' execution failed: %s", tool_name, e, exc_info=True)
            raise ToolExecutionError(tool_name, str(e)) from e

    def list_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schemas."""
        return [t.to_json_schema() for t in self._tools.values()]
