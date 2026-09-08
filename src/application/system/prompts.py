"""Admin prompts management."""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

from src.infrastructure.database import DatabaseService

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "templates" / "prompts" / "chat"

_prompts: dict[str, Prompt] = {}


def _load_all() -> dict[str, Prompt]:
    db = DatabaseService.get_instance()
    if db is None:
        return {}
    prompts_list = db.get_prompts()
    return {
        p["id"]: Prompt(
            prompt_id=p["id"],
            name=p["name"],
            description=p.get("description", ""),
            content=p["content"],
            is_active=p.get("is_active", False),
        )
        for p in prompts_list
    }


def _save_all(prompts: dict[str, Prompt]) -> None:
    db = DatabaseService.get_instance()
    for p in prompts.values():
        db.upsert_prompt(
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "content": p.content,
                "is_active": p.is_active,
            }
        )
    db.persist()


def _ensure_loaded() -> None:
    global _prompts
    if not _prompts:
        _prompts = _load_all()


def validate_name(name: str) -> None:
    """Validates that the name is kebab-case and max 25 characters."""
    if len(name) > 25:
        raise ValueError("Name must be 25 characters or less.")
    if not re.match(r"^[a-z0-9]+(?:-[a-z0-9]+)*$", name):
        raise ValueError("Name must be in kebab-case (e.g., 'my-prompt').")


def validate_description(description: str) -> None:
    """Validates that the description is max 100 characters."""
    if len(description) > 100:
        raise ValueError("Description must be 100 characters or less.")


class Prompt:
    """Represents a system prompt."""

    def __init__(
        self,
        prompt_id: str,
        name: str,
        description: str,
        content: str,
        is_active: bool = False,
    ):
        self.id = prompt_id
        self.name = name
        self.description = description
        self.content = content
        self.is_active = is_active

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Prompt:
        return cls(
            prompt_id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            content=data["content"],
            is_active=data.get("is_active", False),
        )


def list_prompts() -> list[dict]:
    """List all prompts."""
    _ensure_loaded()
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "is_active": p.is_active,
        }
        for p in _prompts.values()
    ]


def get_prompt_by_id(prompt_id: str) -> Prompt | None:
    """Get a prompt by ID."""
    _ensure_loaded()
    return _prompts.get(prompt_id)


def get_prompt_by_name(name: str) -> Prompt | None:
    """Get a prompt by its name."""
    _ensure_loaded()
    for prompt in _prompts.values():
        if prompt.name == name:
            return prompt
    return None


def get_active_prompt() -> Prompt | None:
    """Get the active prompt."""
    _ensure_loaded()
    for prompt in _prompts.values():
        if prompt.is_active:
            return prompt
    return None


def add_prompt(name: str, description: str, content: str) -> Prompt:
    """Add a new prompt."""
    validate_name(name)
    validate_description(description)

    prompt = Prompt(
        prompt_id=str(uuid.uuid4()), name=name, description=description, content=content
    )
    _prompts[prompt.id] = prompt
    _save_all(_prompts)
    return prompt


def update_prompt(
    prompt_id: str,
    name: str | None = None,
    description: str | None = None,
    content: str | None = None,
) -> bool:
    """Update an existing prompt."""
    _ensure_loaded()
    prompt = _prompts.get(prompt_id)
    if not prompt:
        return False

    if name is not None:
        validate_name(name)
        prompt.name = name
    if description is not None:
        validate_description(description)
        prompt.description = description
    if content is not None:
        prompt.content = content

    _save_all(_prompts)
    return True


def set_active_prompt(prompt_id: str) -> bool:
    """Set a prompt as active."""
    _ensure_loaded()
    for p in _prompts.values():
        p.is_active = p.id == prompt_id
    _save_all(_prompts)
    return prompt_id in _prompts


def delete_prompt(prompt_id: str) -> None:
    """Delete a prompt by ID."""
    _ensure_loaded()
    if prompt_id in _prompts:
        del _prompts[prompt_id]
        db = DatabaseService.get_instance()
        if db:
            db.delete_prompt(prompt_id)
    else:
        raise ValueError(f"Prompt ID '{prompt_id}' not found")


def sync_prompts_from_files() -> int:
    """Sync .md prompt files from templates/prompts/ into DuckDB.

    Existing active prompt is preserved. API-created prompts (UUID id)
    are not overwritten. Returns number of prompts synced.
    """
    md_files = sorted(PROMPTS_DIR.glob("*.md"))
    if not md_files:
        logger.warning("No .md prompt files found in %s", PROMPTS_DIR)
        return 0

    db = DatabaseService.get_instance()
    if db is None:
        logger.error("Database not available for prompt sync")
        return 0

    _ensure_loaded()
    # Collect all file-synced names for this sync cycle
    file_names = {md_file.stem.replace("_", "-") for md_file in md_files}

    active = get_active_prompt()
    current_active_id = active.id if active else None

    count = 0
    # If the active prompt was API-created (UUID outside file set), preserve it.
    _DEFAULT_PROMPT_NAME = "general-search"
    preferred_active_name = (
        current_active_id
        if current_active_id and current_active_id not in file_names
        else _DEFAULT_PROMPT_NAME
    )
    for i, md_file in enumerate(md_files):
        name = md_file.stem.replace("_", "-")

        existing = get_prompt_by_name(name)
        if existing and existing.id != name:
            logger.warning(
                "Skipping sync for '%s': already exists as API-created prompt (id=%s)",
                name,
                existing.id,
            )
            continue

        content = md_file.read_text(encoding="utf-8")
        desc = f"Prompt template: {name}"
        is_active = name == preferred_active_name
        db.upsert_prompt(
            {
                "id": name,
                "name": name,
                "description": desc,
                "content": content,
                "is_active": is_active,
            }
        )
        count += 1

    global _prompts
    _prompts = _load_all()

    db.persist()
    logger.info("Synced %d prompt files from %s", count, PROMPTS_DIR)
    return count
