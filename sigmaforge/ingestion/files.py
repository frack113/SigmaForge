from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path


def iter_files(
    directory: str | Path,
    extensions: Sequence[str] = (),
    recursive: bool = True,
    selected_dirs: Sequence[str] = (),
) -> list[Path]:
    root = Path(directory)
    if not root.exists() or not root.is_dir():
        return []
    normalized = {str(extension).lower() for extension in extensions}
    pattern = "**/*" if recursive else "*"
    files: list[Path] = []
    for path in root.glob(pattern):
        if not path.is_file():
            continue
        if normalized and path.suffix.lower() not in normalized:
            continue
        if selected_dirs and not _matches_selected_dirs(path, root, selected_dirs):
            continue
        files.append(path)
    return sorted(files)


def _matches_selected_dirs(path: Path, root: Path, selected_dirs: Iterable[str]) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    rel_parts = rel.parts
    parent_parts = rel.parent.parts
    for raw in selected_dirs:
        parts = Path(str(raw)).parts
        if not parts:
            continue
        if parts[-1] in parent_parts:
            return True
        if parent_parts[: len(parts)] == parts:
            return True
        if rel_parts[: len(parts)] == parts:
            return True
    return False
