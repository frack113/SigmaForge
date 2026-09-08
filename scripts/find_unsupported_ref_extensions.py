#!/usr/bin/env python3
"""Scan Sigma rules and report unsupported reference URL extensions.

Usage:
    uv run python scripts/find_unsupported_ref_extensions.py [rules_dir]

Default rules_dir: data/github/sigmahq/sigma
"""

from __future__ import annotations

import sys
import argparse
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import yaml

from src.shared.utils.identify_file_type import SUPPORTED_DOC_EXTENSION_MAP


def _extract_extension(url: str) -> str | None:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    suffix = Path(path).suffix.lower()
    return suffix if suffix else None


def scan(rules_dir: Path) -> None:
    yml_files = list(rules_dir.rglob("*.yaml")) + list(rules_dir.rglob("*.yml"))
    print(f"Scanning {len(yml_files)} YAML files in {rules_dir}...")

    total_rules = 0
    total_refs = 0
    ext_counter: Counter[str] = Counter()
    examples: dict[str, str] = {}

    for fpath in yml_files:
        try:
            data = yaml.safe_load(fpath.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue

        refs = data.get("references")
        if not isinstance(refs, list):
            continue

        has_refs = False
        for ref in refs:
            if not isinstance(ref, str):
                continue
            if not ref.lower().startswith(("http://", "https://")):
                continue

            ext = _extract_extension(ref)
            if ext is None:
                ext = "<no extension>"
            ext_counter[ext] += 1
            if ext not in examples:
                examples[ext] = ref
            has_refs = True

        if has_refs:
            total_rules += 1
        total_refs += len(refs)

    total_distinct = len(ext_counter)
    supported_exts = set(SUPPORTED_DOC_EXTENSION_MAP.keys())
    unsupported = {ext for ext in ext_counter if ext not in supported_exts}

    print(f"  Rules with HTTP(S) references: {total_rules}")
    print(f"  Total references:              {total_refs}")
    print(f"  Distinct extensions found:      {total_distinct}")
    print()

    if not ext_counter:
        print("No reference URLs found.")
        return

    print(f"  Supported extensions ({len(supported_exts & set(ext_counter.keys()))}):")
    for ext in sorted(ext_counter):
        if ext in supported_exts:
            ft = SUPPORTED_DOC_EXTENSION_MAP[ext].value
            print(f"    {ext:15s}  x{ext_counter[ext]:4d}  ({ft})")
    print()

    if unsupported:
        print(f"  Unsupported extensions ({len(unsupported)}):")
        for ext in sorted(unsupported):
            print(f"    {ext:15s}  x{ext_counter[ext]:4d}")
            print(f"      example: {examples[ext]}")
    else:
        print("  No unsupported extensions found.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Find unsupported Sigma reference URL extensions")
    parser.add_argument(
        "rules_dir",
        nargs="?",
        default="data/github/sigmahq/sigma",
        help="Path to Sigma rules directory (default: data/github/sigmahq/sigma)",
    )
    args = parser.parse_args()

    rules_dir = Path(args.rules_dir)
    if not rules_dir.is_dir():
        print(f"Error: '{rules_dir}' is not a directory", file=sys.stderr)
        return 1

    scan(rules_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
