"""Download Sigma rule reference documents for offline RAG indexing."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import yaml

from src.shared.constants import NULL_UUID
from src.shared.http import download_file as http_download_file
from src.shared.http import head_url as http_head_url
from src.shared.utils.registry_utils import build_registry_entry
from src.shared.utils.crypto_utils import (
    compute_sha256_file,
    compute_sha256_str,
)
from src.shared.utils.identify_file_type import (
    SUPPORTED_DOC_EXTENSION_MAP,
    SUPPORTED_REFERENCE_DOC_TYPES,
    filetype_ext,
)
from src.shared.utils.url_utils import is_private_url, normalize_url
from src.infrastructure.database import DatabaseService
from src.core.sigma.models import is_sigma_rule_dict
from src.shared.utils.sigma_utils import extract_sigma_references
from src.config.settings import get_config

# Local helper modules
from .sigma_ref_paths import (
    resolve_rule_path as _resolve_rule_path,
    sigmaref_resolve_path as _sigmaref_resolve_path,
    sigmaref_write_path as _sigmaref_write_path,
)
from .sigma_ref_url import detect_url_type as _detect_url_type
from .sigma_ref_registry import (
    load_registry as _load_registry,
    load_error_registry as _load_error_registry,
    maybe_record_error as _maybe_record_error,
    save_registry as _save_registry,
)

logger = logging.getLogger(__name__)
DEFAULT_REQUEST_DELAY = 0.5
DEFAULT_MAX_WORKERS = 5
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ext: ft.value for ext, ft in SUPPORTED_DOC_EXTENSION_MAP.items()
}
_registry_lock = threading.Lock()


def download_sigma_references(
    db: DatabaseService,
    output_dir: str,
    mode: str = "scan",
    rules_dir: str | None = None,
    supported_types: set[str] | None = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    progress_callback: Callable[[int, int, str], None] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    selected_dirs: list[str] | None = None,
) -> dict[str, Any]:
    """Download Sigma rule references using the specified mode.

    Two modes are supported:

    - **scan** (default): Scans a local directory of Sigma rule YAML files,
      extracts reference URLs, and downloads matching documents.
      Requires ``rules_dir``.

    - **registry**: Reads pending Sigma rule entries from the doc_registry,
      resolves their local file paths, and downloads referenced documents.
      Does **not** require ``rules_dir``.

    Args:
        db: Database service instance.
        output_dir: Path to the output directory for downloaded files.
        mode: ``"scan"`` or ``"registry"``.
        rules_dir: Path to the directory containing Sigma rule YAML files
            (only used in ``"scan"`` mode).
        supported_types: Set of FileType values to accept (e.g. {"markdown"}).
            Defaults to all reference doc types.
        request_delay: Seconds to wait between download requests.
        progress_callback: Optional callback(current, total, phase).
        max_workers: Max concurrent HTTP download threads.
        selected_dirs: Optional list of relative directory paths to scan
            (only used in ``"scan"`` mode).

    Returns:
        Dict with summary stats: total_rules, total_refs, downloaded, skipped, failed.
    """
    if supported_types is None:
        supported_types = SUPPORTED_REFERENCE_DOC_TYPES

    if mode == "scan":
        if not rules_dir:
            raise ValueError("rules_dir is required in scan mode")
        return _download_scan_mode(
            rules_dir=rules_dir,
            output_dir=output_dir,
            db=db,
            supported_types=supported_types,
            request_delay=request_delay,
            progress_callback=progress_callback,
            max_workers=max_workers,
            selected_dirs=selected_dirs,
        )
    if mode == "registry":
        return _download_registry_mode(
            output_dir=output_dir,
            db=db,
            supported_types=supported_types,
            request_delay=request_delay,
            progress_callback=progress_callback,
            max_workers=max_workers,
        )
    msg = f"Unknown mode: {mode!r} (expected 'scan' or 'registry')"
    raise ValueError(msg)


def download_references(
    rules_dir: str,
    output_dir: str,
    db: DatabaseService,
    supported_types: set[str] | None = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    progress_callback: Callable[[int, int, str], None] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    selected_dirs: list[str] | None = None,
) -> dict[str, Any]:
    """Download all Sigma rule references matching supported document types.

    Scans all Sigma rules in the given directory, extracts reference URLs,
    filters by supported document types, and downloads matching files.

    Delegates to :func:`download_sigma_references` with ``mode="scan"``.
    """
    return download_sigma_references(
        db=db,
        output_dir=output_dir,
        mode="scan",
        rules_dir=rules_dir,
        supported_types=supported_types,
        request_delay=request_delay,
        progress_callback=progress_callback,
        max_workers=max_workers,
        selected_dirs=selected_dirs,
    )


def _download_scan_mode(
    rules_dir: str,
    output_dir: str,
    db: DatabaseService,
    supported_types: set[str] | None = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    progress_callback: Callable[[int, int, str], None] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    selected_dirs: list[str] | None = None,
) -> dict[str, Any]:
    """Scan-mode implementation — see :func:`download_sigma_references`."""
    if supported_types is None:
        supported_types = SUPPORTED_REFERENCE_DOC_TYPES

    rules_path = Path(rules_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not rules_path.is_dir():
        logger.warning("Rules directory does not exist: %s", rules_dir)
        return _empty_summary()

    with _registry_lock:
        registry = _load_registry(output_path, db)
        error_registry = _load_error_registry(db)

    total_rules = 0
    total_refs = 0
    downloaded = 0
    skipped = 0
    failed = 0

    _EXCLUDE_DIRS = {".github", ".git", "__pycache__", ".vscode", ".idea", ".pytest_cache"}

    def _is_selected_dir(yml_file: Path) -> bool:
        """Check if the file is within any of the selected directories."""
        if not selected_dirs:
            return True
        try:
            rel_path = yml_file.relative_to(rules_path).with_suffix("")
            rel_str = rel_path.as_posix()
            for sd in selected_dirs:
                clean_sd = sd.lstrip("./").rstrip("/")
                if not clean_sd:
                    return True
                if rel_str == clean_sd or rel_str.startswith(clean_sd + "/"):
                    return True
        except (ValueError, TypeError):
            pass
        return False

    def _collect_yaml_files() -> list[Path]:
        yml_files: list[Path] = []
        for pattern in ("*.yaml", "*.yml"):
            for yml_file in rules_path.rglob(pattern):
                parts = yml_file.parts
                if any(part in _EXCLUDE_DIRS for part in parts):
                    continue
                if not _is_selected_dir(yml_file):
                    continue
                yml_files.append(yml_file)
        return yml_files

    yml_files = _collect_yaml_files()

    # Phase 1: scan YAML files, classify refs
    download_queue: list[dict[str, Any]] = []
    head_pending: list[dict[str, Any]] = []
    rule_refs: list[dict[str, str]] = []

    total_files = len(yml_files)
    for file_idx, yml_file in enumerate(yml_files):
        if progress_callback:
            progress_callback(file_idx + 1, total_files, "scanning")
        if not yml_file.is_file():
            continue
        try:
            data = yaml.safe_load(yml_file.read_text(encoding="utf-8", errors="replace"))
        except Exception as exc:
            logger.warning("Failed to parse YAML %s: %s", yml_file, exc)
            continue
        if not isinstance(data, dict):
            continue
        if not is_sigma_rule_dict(data):
            continue

        total_rules += 1
        refs = data.get("references", [])
        if not isinstance(refs, list):
            continue

        rule_id = data.get("id", yml_file.stem)
        rule_title = data.get("title", "")

        for ref in refs:
            if not isinstance(ref, str):
                continue
            if not ref.lower().startswith(("http://", "https://")):
                logger.debug("Non-HTTP ref skipped: %s", ref)
                continue

            if is_private_url(ref):
                logger.warning("Skipping private URL ref: %s", ref)
                skipped += 1
                continue

            total_refs += 1
            normalized = normalize_url(ref)

            url_hash = compute_sha256_str(normalized)

            rule_refs.append({"rule_id": rule_id, "url_hash": url_hash, "ref_url": ref})

            if url_hash in error_registry:
                logger.debug("Skipping previously failed URL: %s", normalized)
                skipped += 1
                continue

            # Fast path: already in registry with file_name and valid file on disk
            if url_hash in registry:
                fname = registry[url_hash].get("file_name", "")
                if fname:
                    content_type_for_subdir = registry[url_hash].get("content_type", "")
                    output_file = _sigmaref_resolve_path(
                        output_path, content_type_for_subdir, fname
                    )
                    if output_file.exists():
                        existing_sha = registry[url_hash].get("content_sha256")
                        if (
                            existing_sha is not None
                            and compute_sha256_file(output_file) != existing_sha
                        ):
                            download_queue.append(
                                {
                                    "url_hash": url_hash,
                                    "original_url": ref,
                                    "normalized_url": normalized,
                                    "output_file": _sigmaref_write_path(
                                        output_path, content_type_for_subdir, fname
                                    ),
                                    "content_type": registry[url_hash].get(
                                        "content_type", "markdown"
                                    ),
                                    "rule_id": rule_id,
                                    "rule_title": rule_title,
                                    "is_redownload": True,
                                }
                            )
                            continue
                        skipped += 1
                        continue
                # File missing or no file_name — fall through to re-download

            # Determine extension and content type
            ext = filetype_ext(normalized)
            ftype = _detect_url_type(normalized)
            if ftype is None and url_hash in registry:
                ct = registry[url_hash].get("content_type")
                if ct:
                    ftype = ct

            if ftype is None:
                head_pending.append(
                    {
                        "normalized": normalized,
                        "url_hash": url_hash,
                        "ext": ext,
                        "original_url": ref,
                        "rule_id": rule_id,
                        "rule_title": rule_title,
                    }
                )
                continue

            if not ext and ftype is not None:
                ext = filetype_ext(ftype)

            if not ext:
                ext = ".md"

            fname = f"{url_hash}{ext}"
            output_file = _sigmaref_resolve_path(output_path, ftype, fname)

            if output_file.exists():
                content_hash = compute_sha256_file(output_file)
                if url_hash in registry:
                    existing_sha = registry[url_hash].get("content_sha256")
                    if existing_sha is not None and content_hash != existing_sha:
                        download_queue.append(
                            {
                                "url_hash": url_hash,
                                "original_url": ref,
                                "normalized_url": normalized,
                                "output_file": _sigmaref_write_path(output_path, ftype, fname),
                                "content_type": ftype or "markdown",
                                "rule_id": rule_id,
                                "rule_title": rule_title,
                                "is_redownload": True,
                            }
                        )
                        continue
                registry[url_hash] = build_registry_entry(
                    url_hash=url_hash,
                    original_url=ref,
                    normalized_url=normalized,
                    content_type=ftype or "markdown",
                    rule_id=rule_id,
                    title=rule_title,
                    content_sha256=content_hash,
                    file_name=output_file.name,
                    file_size=output_file.stat().st_size,
                )
                skipped += 1
                continue

            if ftype is None or ftype not in supported_types:
                skipped += 1
                continue

            download_queue.append(
                {
                    "url_hash": url_hash,
                    "original_url": ref,
                    "normalized_url": normalized,
                    "output_file": _sigmaref_write_path(output_path, ftype, fname),
                    "content_type": ftype,
                    "rule_id": rule_id,
                    "rule_title": rule_title,
                    "is_redownload": False,
                }
            )

    # Phase 1.5: batch HEAD requests in parallel for URLs with unknown type
    if head_pending:
        logger.info("Resolving %d unknown URL types via HEAD requests", len(head_pending))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map: dict[Future, dict[str, Any]] = {}
            for item in head_pending:
                future = executor.submit(http_head_url, item["normalized"], 10, check_ssrf=False)
                future_map[future] = item
            resolved = 0
            for future in as_completed(future_map):
                resolved += 1
                item = future_map[future]
                if progress_callback:
                    progress_callback(resolved, len(head_pending), "downloading")
                head_ct, _, _ = future.result()
                ftype = _detect_url_type(item["normalized"], content_type=head_ct)
                ext = item["ext"]
                if not ext and ftype is not None:
                    ext = filetype_ext(ftype)
                if not ext:
                    ext = ".md"
                fname = f"{item['url_hash']}{ext}"
                output_file = _sigmaref_resolve_path(output_path, ftype, fname)
                if ftype is None or ftype not in supported_types:
                    skipped += 1
                    continue
                download_queue.append(
                    {
                        "url_hash": item["url_hash"],
                        "original_url": item["original_url"],
                        "normalized_url": item["normalized"],
                        "output_file": _sigmaref_write_path(output_path, ftype, fname),
                        "content_type": ftype,
                        "rule_id": item["rule_id"],
                        "rule_title": item["rule_title"],
                        "is_redownload": False,
                    }
                )

    # Phase 2: download all queued refs in parallel
    queue_size = len(download_queue)
    if download_queue:
        logger.info("Downloading %d refs with %d workers", queue_size, max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for item in download_queue:
                future = executor.submit(
                    http_download_file,  # type: ignore[arg-type]
                    item["normalized_url"],
                    item["output_file"],
                    check_ssrf=False,
                )
                future_map[future] = item

            for idx, future in enumerate(as_completed(future_map)):
                if progress_callback:
                    progress_callback(idx + 1, queue_size, "downloading")
                item = future_map[future]
                success, status_code = cast(tuple[bool, int | None], future.result())
                if success:
                    output_file = item["output_file"]
                    content_hash = compute_sha256_file(output_file) if output_file.exists() else ""
                    registry[item["url_hash"]] = build_registry_entry(
                        url_hash=item["url_hash"],
                        original_url=item["original_url"],
                        normalized_url=item["normalized_url"],
                        content_type=item["content_type"],
                        rule_id=item["rule_id"],
                        title=item["rule_title"],
                        content_sha256=content_hash,
                        file_name=output_file.name,
                        file_size=output_file.stat().st_size if output_file.exists() else None,
                    )
                    downloaded += 1
                else:
                    logger.debug(
                        "Reference download failed | url=%s rule_id=%s status=%s",
                        item["original_url"],
                        item["rule_id"],
                        status_code,
                    )
                    _maybe_record_error(
                        db,
                        item["url_hash"],
                        item["original_url"],
                        item["normalized_url"],
                        status_code,
                        item["rule_id"],
                        item["rule_title"],
                    )
                    failed += 1

    with _registry_lock:
        _save_registry(registry, output_path, db)

    if rule_refs:
        db.batch_upsert_rule_references(rule_refs)

    summary: dict[str, Any] = {
        "total_rules": total_rules,
        "total_refs": total_refs,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }
    _log_download_summary(summary)
    return summary


def _log_download_summary(summary: dict[str, Any]) -> None:
    """Log download summary and warn if failure rate exceeds threshold."""
    logger.info("Download complete: %s", summary)
    total_refs = summary.get("total_refs", 0)
    failed = summary.get("failed", 0)
    if total_refs > 0:
        fail_rate = failed / total_refs
        if fail_rate > 0.05:
            logger.warning(
                "High reference failure rate: %.1f%% (%d/%d) — check network or URL validity",
                fail_rate * 100,
                failed,
                total_refs,
            )


def _empty_summary() -> dict[str, Any]:
    """Return an empty summary dict."""
    return {
        "total_rules": 0,
        "total_refs": 0,
        "downloaded": 0,
        "skipped": 0,
        "failed": 0,
    }


# ------------------------------------------------------------------
# Registry mode — reads pending Sigma rules from doc_registry and
# downloads their referenced documents.
# ------------------------------------------------------------------


def _download_registry_mode(
    output_dir: str,
    db: DatabaseService,
    supported_types: set[str] | None = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    progress_callback: Callable[[int, int, str], None] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    """Registry-mode implementation — see :func:`download_sigma_references`."""
    if supported_types is None:
        supported_types = SUPPORTED_REFERENCE_DOC_TYPES

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    entries = db.get_pending_registry_all()
    sigma_entries = [
        e
        for e in entries
        if e.get("content_type") == "sigma_rule" and e.get("embed_status") == "discovery"
    ]

    total_rules = len(sigma_entries)
    total_refs = 0
    downloaded = 0
    skipped = 0
    failed = 0

    cfg = get_config()

    with _registry_lock:
        registry = _load_registry(output_path, db)

    head_queue: list[dict[str, Any]] = []
    download_ready: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    rule_refs: list[dict[str, str]] = []

    for rule_entry in sigma_entries:
        rule_id: str = rule_entry.get("rule_id", NULL_UUID)
        original_url = rule_entry.get("original_url", "")
        file_name = rule_entry.get("file_name", "")

        file_path = _resolve_rule_path(rule_entry, cfg)
        if not file_path or not file_path.exists():
            logger.warning("Rule file not found, skipping: %s (url=%s)", file_name, original_url)
            skipped += 1
            continue

        refs = extract_sigma_references(file_path)
        rule_title = rule_entry.get("title", file_name)
        logger.info("Rule %s processed: %d reference(s) found", rule_id, len(refs))

        if not refs:
            continue

        total_refs += len(refs)

        for ref_url in refs:
            ref_url_clean = ref_url.strip()
            if not ref_url_clean:
                continue

            norm_url = normalize_url(ref_url_clean)
            url_hash = compute_sha256_str(norm_url)

            if url_hash in seen_urls:
                logger.debug("Reference already queued this run: %s", ref_url_clean)
                skipped += 1
                continue
            seen_urls.add(url_hash)

            rule_refs.append({"rule_id": rule_id, "url_hash": url_hash, "ref_url": ref_url_clean})

            existing = registry.get(url_hash)

            if existing and existing.get("content_sha256"):
                skipped += 1
                continue

            if existing:
                download_ready.append(
                    {
                        "url": ref_url_clean,
                        "rule_id": rule_id,
                        "rule_title": rule_title,
                        "url_hash": url_hash,
                        "final_url": existing.get("normalized_url", ref_url_clean),
                        "content_type": existing.get("content_type", ""),
                    }
                )
            else:
                head_queue.append(
                    {
                        "url": ref_url_clean,
                        "rule_id": rule_id,
                        "rule_title": rule_title,
                    }
                )

    total_head = len(head_queue)
    total_download_ready = len(download_ready)

    # Phase 1: parallel HEAD requests
    head_pending: list[dict[str, Any]] = []
    head_completed = 0
    if head_queue:
        with ThreadPoolExecutor(max_workers=max_workers) as head_executor:
            head_futures: dict[Future, dict[str, Any]] = {}
            for item in head_queue:
                future = head_executor.submit(http_head_url, item["url"], 15.0)
                head_futures[future] = item

            for future in as_completed(head_futures):
                head_completed += 1
                if progress_callback:
                    progress_callback(
                        head_completed,
                        total_head + total_download_ready,
                        "resolving URLs",
                    )
                item = head_futures[future]
                url = item["url"]
                try:
                    content_type, size, final_url = future.result()
                    norm_url = normalize_url(final_url or url)
                    url_hash = compute_sha256_str(norm_url)

                    if content_type not in supported_types:
                        db.batch_upsert_doc_registry(
                            [
                                build_registry_entry(
                                    normalized_url=norm_url,
                                    content_type=content_type or "unknown",
                                    rule_id=item["rule_id"],
                                    title=item["rule_title"],
                                    embed_status="head_verified",
                                )
                            ]
                        )
                        skipped += 1
                        continue

                    db.batch_upsert_doc_registry(
                        [
                            build_registry_entry(
                                normalized_url=norm_url,
                                content_type=content_type,
                                rule_id=item["rule_id"],
                                title=item["rule_title"],
                                file_size=size,
                                embed_status="head_verified",
                            )
                        ]
                    )
                    head_pending.append(
                        {
                            **item,
                            "content_type": content_type,
                            "size": size,
                            "final_url": final_url or url,
                            "url_hash": url_hash,
                        }
                    )
                except Exception as e:
                    logger.warning("HEAD failed for %s: %s", url, e)
                    failed += 1

    # Phase 2: merge & download
    all_to_download = head_pending + download_ready
    total_to_download = len(all_to_download)

    def _download_one(item: dict[str, Any]) -> tuple[str, str, str, int] | None:
        url = item["final_url"]
        content_type = item.get("content_type", "")
        ext = filetype_ext(content_type)
        url_hash = item.get("url_hash") or compute_sha256_str(normalize_url(url))
        fname = f"{url_hash}{ext}"
        existing_path = _sigmaref_resolve_path(output_path, content_type, fname)
        if existing_path.exists():
            existing_entry = registry.get(url_hash)
            if existing_entry and existing_entry.get("content_sha256"):
                return None

        file_path = _sigmaref_write_path(output_path, content_type, fname)

        ok, _ = http_download_file(url, file_path, check_ssrf=False)
        if ok:
            content_hash = compute_sha256_file(file_path)
            file_size = file_path.stat().st_size
            return ("ok", url_hash, content_hash, file_size)
        logger.error("Reference download failed: %s", url)
        return ("fail", "", "", 0)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict[Future, dict[str, Any]] = {
            executor.submit(_download_one, item): item for item in all_to_download
        }

        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                if result is None:
                    skipped += 1
                elif result is not None and result[0] == "ok":
                    _, url_hash, content_hash, size = cast(tuple[str, str, str, int], result)
                    entry = build_registry_entry(
                        url_hash=url_hash,
                        normalized_url=item.get("final_url", item["url"]),
                        content_type=item["content_type"],
                        rule_id=item["rule_id"],
                        title=item["rule_title"],
                        content_sha256=content_hash,
                        file_name=f"{url_hash}{filetype_ext(item['content_type'])}",
                        file_size=size,
                        embed_status="discovery",
                    )
                    db.batch_upsert_doc_registry([entry])
                    downloaded += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error("Download task failed: %s", e)
                failed += 1

            if progress_callback:
                completed = downloaded + failed
                progress_callback(
                    total_head + completed,
                    total_head + total_to_download,
                    "downloading",
                )

    if rule_refs:
        db.batch_upsert_rule_references(rule_refs)

    summary = {
        "total_rules": total_rules,
        "total_refs": total_refs,
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
    }
    _log_download_summary(summary)
    return summary
