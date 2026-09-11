from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml

from sigmaforge.ingestion.files import iter_files
from sigmaforge.ingestion.models import Chunk

_YAML_EXTENSIONS: tuple[str, ...] = (".yaml", ".yml")
_MAX_RULE_BYTES: int = 1024 * 1024


@dataclass(slots=True)
class SigmaRule:
    id: str
    title: str
    detection: dict[str, Any] = field(default_factory=dict)
    condition: str = ""
    status: str | None = None
    level: str | None = None
    tags: list[str] = field(default_factory=list)
    falsepositives: list[str] = field(default_factory=list)
    description: str | None = None
    author: str | None = None
    date: str | None = None
    modified: str | None = None
    references: list[str] = field(default_factory=list)
    logsource: dict[str, Any] = field(default_factory=dict)
    license: str | None = None
    file_path: str | None = None
    line_number: int | None = None
    fields: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_sigma_rule_dict(data: object) -> bool:
    return isinstance(data, dict) and isinstance(data.get("detection"), dict)


def iter_sigma_rule_files(
    directory: str | Path,
    recursive: bool = True,
    selected_dirs: Sequence[str] = (),
) -> list[Path]:
    return iter_files(
        directory,
        extensions=_YAML_EXTENSIONS,
        recursive=recursive,
        selected_dirs=selected_dirs,
    )


def parse_sigma_rule(path: str | Path) -> SigmaRule | None:
    path = Path(path)
    if path.suffix.lower() not in _YAML_EXTENSIONS or not path.is_file():
        return None
    try:
        if path.stat().st_size > _MAX_RULE_BYTES:
            return None
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    if not is_sigma_rule_dict(data):
        return None
    detection = _as_mapping(data.get("detection"))
    return SigmaRule(
        id=str(data.get("id") or path.stem),
        title=str(data.get("title") or path.stem),
        detection=detection,
        condition=str(data.get("condition") or detection.get("condition") or ""),
        status=_as_optional_str(data.get("status")),
        level=_as_optional_str(data.get("level")),
        tags=_as_str_list(data.get("tags")),
        falsepositives=_as_str_list(data.get("falsepositives")),
        description=_as_optional_str(data.get("description")),
        author=_as_optional_str(data.get("author")),
        date=_normalize_date(data.get("date")),
        modified=_normalize_date(data.get("modified")),
        references=_as_str_list(data.get("references")),
        logsource=_as_mapping(data.get("logsource")),
        license=_as_optional_str(data.get("license")),
        file_path=str(path),
        line_number=_as_optional_int(data.get("line_number")),
        fields=_as_str_list(data.get("fields")),
    )


def flat_rule_text(rule: SigmaRule) -> str:
    lines = [f"Title: {rule.title}", f"Rule ID: {rule.id}"]
    if rule.status:
        lines.append(f"Status: {rule.status}")
    if rule.level:
        lines.append(f"Level: {rule.level}")
    if rule.description:
        lines.append(f"Description: {rule.description}")
    if rule.author:
        lines.append(f"Author: {rule.author}")
    if rule.date:
        lines.append(f"Date: {rule.date}")
    if rule.modified:
        lines.append(f"Modified: {rule.modified}")
    if rule.tags:
        lines.append(f"Tags: {', '.join(rule.tags)}")
    if rule.references:
        lines.append(f"References: {', '.join(rule.references)}")
    if rule.logsource:
        lines.append(f"Logsource: {format_logsource(rule.logsource)}")
    if rule.condition:
        lines.append(f"Condition: {rule.condition}")
    detection_lines = [
        f"  {key}: {_format_value(value)}"
        for key, value in rule.detection.items()
        if key != "condition" and _format_value(value)
    ]
    if detection_lines:
        lines.append("Detection:")
        lines.extend(detection_lines)
    if rule.falsepositives:
        lines.append(f"False positives: {'; '.join(rule.falsepositives)}")
    return "\n".join(lines)


def rule_metadata(rule: SigmaRule) -> dict[str, Any]:
    logsource = rule.logsource
    return {
        "rule_id": rule.id,
        "title": rule.title,
        "status": rule.status,
        "level": rule.level,
        "tags": list(rule.tags),
        "product": logsource.get("product"),
        "category": logsource.get("category"),
        "service": logsource.get("service"),
        "definition": logsource.get("definition"),
        "logsource": dict(logsource),
        "description": rule.description,
        "author": rule.author,
        "date": rule.date,
        "modified": rule.modified,
        "falsepositives": list(rule.falsepositives),
        "references": list(rule.references),
    }


def rich_rule_chunks(rule: SigmaRule) -> list[Chunk]:
    metadata = rule_metadata(rule)
    source = rule.file_path or rule.id
    chunks: list[Chunk] = []

    def add(chunk_type: str, text: str) -> None:
        text = text.strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    source_file=source,
                    chunk_type=chunk_type,
                    chunk_index=len(chunks),
                    metadata=metadata,
                )
            )

    logsource = format_logsource(rule.logsource) if rule.logsource else ""
    executive = [f"Title: {rule.title}", f"Rule ID: {rule.id}"]
    if rule.status:
        executive.append(f"Status: {rule.status}")
    if rule.level:
        executive.append(f"Level: {rule.level}")
    if rule.description:
        executive.append(f"Description: {rule.description}")
    if logsource:
        executive.append(f"Logsource: {logsource}")
    if rule.condition:
        executive.append(f"Condition: {rule.condition}")
    if rule.tags:
        executive.append(f"Tags: {', '.join(rule.tags)}")
    add("executive_summary", "\n".join(executive))

    lifecycle = [f"Status: {rule.status or 'unknown'}"]
    if rule.author:
        lifecycle.append(f"Author: {rule.author}")
    if rule.date:
        lifecycle.append(f"Date: {rule.date}")
    if rule.modified:
        lifecycle.append(f"Modified: {rule.modified}")
    if rule.license:
        lifecycle.append(f"License: {rule.license}")
    if rule.references:
        lifecycle.append(f"References: {', '.join(rule.references)}")
    add("metadata_lifecycle", "\n".join(lifecycle))

    if rule.logsource:
        context = [f"Logsource: {logsource}"]
        for key in ("product", "category", "service", "definition"):
            value = rule.logsource.get(key)
            if value:
                context.append(f"{key.capitalize()}: {value}")
        add("logsource_context", "\n".join(context))

    condition_lines: list[str] = []
    if rule.condition:
        condition_lines.append(f"Condition: {rule.condition}")
    detection_keys = [str(key) for key in rule.detection if key != "condition"]
    if detection_keys:
        condition_lines.append(f"Detection blocks: {', '.join(detection_keys)}")
    if condition_lines:
        add("condition", "\n".join(condition_lines))

    for name, value in rule.detection.items():
        if name == "condition":
            continue
        rendered = _format_value(value)
        if not rendered:
            continue
        add(
            f"detection_block:{name}",
            f"Rule: {rule.title}\nDetection block: {name}\n{rendered}",
        )

    if rule.falsepositives:
        add("false_positives", "False positives: " + "; ".join(rule.falsepositives))
    return chunks


def chunk_rule(rule: SigmaRule, mode: str = "flat") -> list[Chunk]:
    if mode == "flat":
        text = flat_rule_text(rule).strip()
        if not text:
            return []
        return [
            Chunk(
                text=text,
                source_file=rule.file_path or rule.id,
                chunk_type="rule",
                chunk_index=0,
                metadata=rule_metadata(rule),
            )
        ]
    if mode == "rich":
        return rich_rule_chunks(rule)
    raise ValueError("mode must be 'flat' or 'rich'")


def format_logsource(logsource: Mapping[str, Any]) -> str:
    ordered = ("product", "category", "service", "definition")
    parts = [f"{key}={logsource[key]}" for key in ordered if logsource.get(key)]
    if not parts:
        parts = [f"{key}={value}" for key, value in logsource.items() if value not in (None, "")]
    return " ".join(str(part) for part in parts) or "unknown"


def _as_mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _normalize_date(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ", ".join(_format_value(item) for item in value)
    if isinstance(value, Mapping):
        return " ".join(
            f"{key} {_format_value(item)}".strip() for key, item in value.items()
        )
    return str(value)
