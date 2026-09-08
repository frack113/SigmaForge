"""Tool registry with automatic JSON schema generation.

The :class:`@tool` decorator introspects function signatures and
docstrings to produce OpenAI-compatible JSON schemas automatically,
eliminating manual schema maintenance.
"""

from __future__ import annotations

import inspect
import re
import textwrap
from typing import Any, get_args, get_origin

from .models import ToolDef

_tools: list[ToolDef] = []


def _python_to_json_type(annotation: Any) -> str:
    """Convert a Python type hint to a JSON schema type string."""
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if annotation is list or annotation is list[str]:
        return "array"
    if annotation is dict or annotation is dict[str, Any]:
        return "object"

    origin = get_origin(annotation)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"

    # Default to string for Any, Unknown, or complex types
    return "string"


def _build_json_schema(fn: Any) -> dict[str, Any]:
    """Build a JSON schema from a function's type hints and docstring.

    Parameters follow OpenAI's ``chat.completions`` function-calling format.
    """
    sig = inspect.signature(fn)
    schema: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    _skipped = {"self", "ctx"}

    for name, param in sig.parameters.items():
        if name in _skipped:
            continue

        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            annotation = str

        json_type = _python_to_json_type(annotation)

        # Extract description from docstring using "@param" or ":param" format
        doc = inspect.getdoc(fn) or ""
        desc = ""
        pattern = rf"(?:@|:)param\s+{re.escape(name)}\s*:\s*(.+?)(?=\n\s*(?:@|:)param|\Z)"
        for match in re.finditer(
            pattern,
            doc,
            re.MULTILINE | re.DOTALL,
        ):
            desc = textwrap.dedent(match.group(1)).strip()
            break

        prop: dict[str, Any] = {"type": json_type}
        if desc:
            prop["description"] = desc

        schema["properties"][name] = prop

        if param.default is inspect.Parameter.empty:
            schema["required"].append(name)

    # Add enum validation for parameters that are typed as Literal or str with limited values
    _enrich_enums(sig, schema)

    return schema


def _enrich_enums(sig: inspect.Signature, schema: dict[str, Any]) -> None:
    """Add ``enum`` constraints for string parameters with known valid values."""
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        prop = schema["properties"].get(name)
        if not prop:
            continue

        # Literal types: extract allowed values
        annotation = param.annotation
        origin = get_origin(annotation)
        if origin is None:
            continue
        if origin.__name__ == "Literal":
            args = get_args(annotation)
            if args and all(isinstance(a, str) for a in args):
                prop["enum"] = list(args)
                prop["type"] = "string"


# Valid metadata filter fields for filter_metadata tool
_VALID_FILTER_FIELDS: frozenset[str] = frozenset(
    {
        "author",
        "cve",
        "mitre",
        "level",
        "status",
        "logsource_product",
        "logsource_category",
        "source",
    }
)

# Mapping from user-facing field names to Qdrant column names
_FIELD_MAP: dict[str, str] = {
    "author": "author",
    "cve": "tags",
    "mitre": "tags",
    "level": "level",
    "status": "status",
    "logsource_product": "product",
    "logsource_category": "category",
    "source": "collection",
}


def tool(fn: Any | None = None, *, description_override: str | None = None) -> Any:
    """Decorator that registers an async function as an available tool.

    The function signature + docstring are automatically converted into
    an OpenAI-compatible JSON schema.

    Usage::

        @tool
        async def search_sigma(query: str) -> str:
            \"\"\"Search Sigma rules by natural language query.

            :param query: Search query describing the threat or behavior.
            \"\"\"
            ...
    """

    def decorator(fn: Any) -> ToolDef:
        doc = inspect.getdoc(fn) or ""
        params = _build_json_schema(fn)

        # Override description if explicitly provided
        if description_override:
            params["description"] = description_override
        elif "description" not in params and doc:
            # First sentence of docstring as description
            first_sentence = doc.strip().split("\n")[0].strip()
            params["description"] = first_sentence

        has_ctx = "ctx" in inspect.signature(fn).parameters
        td = ToolDef(
            name=fn.__name__,
            description=params.get("description", ""),
            parameters=params,
            fn=fn,
            has_ctx=has_ctx,
        )
        _tools.append(td)
        return td

    if fn is None:
        return decorator
    return decorator(fn)


def get_tools() -> list[ToolDef]:
    """Return all registered tools."""
    return list(_tools)


def get_tool_by_name(name: str) -> ToolDef | None:
    """Find a tool by name."""
    for t in _tools:
        if t.name == name:
            return t
    return None


def register_field_enums() -> list[ToolDef]:
    """Register dynamic tools that depend on module-level constants."""
    # filter_metadata already has its enum baked in via schema override
    return []


def reset_tools() -> None:
    """Clear all registered tools (primarily for testing)."""
    _tools.clear()
