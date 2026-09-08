"""DuckDB document operations — unified on doc_registry and doc_error tables."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.shared.constants import NULL_UUID
from src.shared.utils.crypto_utils import compute_sha256_file
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import threading

    import duckdb


logger = logging.getLogger(__name__)


class DatabaseServiceDocOps:
    """Document registry operations (doc_registry) and error tracking (doc_error)."""

    # Provided by DatabaseServiceCore mixin
    _lock: threading.RLock
    _writer_conn: duckdb.DuckDBPyConnection

    def _safe_query(self, query: str, params: tuple = ()) -> Any: ...

    # ------------------------------------------------------------------
    # DOC_REGISTRY table (unified document registry)
    # ------------------------------------------------------------------

    def get_entries_by_org(self, org: str, limit: int = 100, offset: int = 0) -> list[dict]:
        """Return entries for a specific org."""
        with self._lock:
            query = (
                "SELECT url_hash, org, repo, content_type, file_name, content_sha256, file_size, "
                "original_url, normalized_url, rule_id, title, timestamp, last_seen, embed_status "
                "FROM doc_registry WHERE org = ? ORDER BY url_hash"
            )
            params: list[int | str] = [org]
            if limit > 0:
                query += " LIMIT ?"
                params.append(limit)
            if offset > 0:
                query += " OFFSET ?"
                params.append(offset)
            results = self._writer_conn.execute(query, params).fetchall()
            col_names = [desc[0] for desc in self._writer_conn.description]
        return [dict(zip(col_names, row)) for row in results]

    def get_pending_entries(self, org: str | None = None, repo: str | None = None) -> list[dict]:
        """Return pending entries, optionally filtered by org/repo."""
        with self._lock:
            conditions = ["embed_status = 'discovery'"]
            params: list[str] = []
            if org:
                conditions.append("org = ?")
                params.append(org)
            if repo:
                conditions.append("repo = ?")
                params.append(repo)
            where_clause = " AND ".join(conditions)

            results = self._writer_conn.execute(
                f"""SELECT url_hash, org, repo, content_type, file_name, content_sha256, file_size,
                    original_url, normalized_url, rule_id, title, timestamp, last_seen, embed_status
                    FROM doc_registry WHERE {where_clause} ORDER BY url_hash""",
                params,
            ).fetchall()
            col_names = [desc[0] for desc in self._writer_conn.description]
        return [dict(zip(col_names, row)) for row in results]

    def update_embed_status(self, url_hash: str, status: str) -> None:
        with self._lock:
            self._writer_conn.execute(
                "UPDATE doc_registry SET embed_status = ? WHERE url_hash = ?",
                (status, url_hash),
            )
            self._writer_conn.commit()

    def upsert_doc_registry(self, data: dict) -> None:
        with self._lock:
            self._writer_conn.execute(
                """INSERT INTO doc_registry (
                    url_hash, org, repo, content_type, file_name, content_sha256, file_size,
                    original_url, normalized_url, rule_id, title, timestamp, last_seen, embed_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (url_hash) DO UPDATE SET
                    org = EXCLUDED.org,
                    repo = EXCLUDED.repo,
                    content_type = EXCLUDED.content_type,
                    file_name = EXCLUDED.file_name,
                    content_sha256 = EXCLUDED.content_sha256,
                    file_size = EXCLUDED.file_size,
                    original_url = EXCLUDED.original_url,
                    normalized_url = EXCLUDED.normalized_url,
                    rule_id = EXCLUDED.rule_id,
                    title = EXCLUDED.title,
                    timestamp = EXCLUDED.timestamp,
                    last_seen = EXCLUDED.last_seen,
                    embed_status = CASE
                        WHEN excluded.content_sha256 IS NOT NULL
                             AND excluded.content_sha256 != doc_registry.content_sha256
                        THEN 'discovery'
                        ELSE doc_registry.embed_status
                    END""",
                (
                    data.get("url_hash"),
                    data.get("org"),
                    data.get("repo"),
                    data.get("content_type"),
                    data.get("file_name"),
                    data.get("content_sha256"),
                    data.get("file_size"),
                    data.get("original_url"),
                    data.get("normalized_url"),
                    data.get("rule_id", NULL_UUID),
                    data.get("title"),
                    data.get("timestamp"),
                    data.get("last_seen"),
                    data.get("embed_status", "discovery"),
                ),
            )
            self._writer_conn.commit()

    def batch_upsert_doc_registry(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self._lock:
            self._writer_conn.executemany(
                """INSERT INTO doc_registry (
                    url_hash, org, repo, content_type, file_name, content_sha256, file_size,
                    original_url, normalized_url, rule_id, title, timestamp, last_seen, embed_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (url_hash) DO UPDATE SET
                    org = EXCLUDED.org,
                    repo = EXCLUDED.repo,
                    content_type = EXCLUDED.content_type,
                    file_name = EXCLUDED.file_name,
                    content_sha256 = EXCLUDED.content_sha256,
                    file_size = EXCLUDED.file_size,
                    original_url = EXCLUDED.original_url,
                    normalized_url = EXCLUDED.normalized_url,
                    rule_id = EXCLUDED.rule_id,
                    title = EXCLUDED.title,
                    timestamp = EXCLUDED.timestamp,
                    last_seen = EXCLUDED.last_seen,
                    embed_status = CASE
                        WHEN excluded.content_sha256 IS NOT NULL
                             AND excluded.content_sha256 != doc_registry.content_sha256
                        THEN 'discovery'
                        ELSE doc_registry.embed_status
                    END""",
                [
                    (
                        r.get("url_hash"),
                        r.get("org"),
                        r.get("repo"),
                        r.get("content_type"),
                        r.get("file_name"),
                        r.get("content_sha256"),
                        r.get("file_size"),
                        r.get("original_url"),
                        r.get("normalized_url"),
                        r.get("rule_id", NULL_UUID),
                        r.get("title"),
                        r.get("timestamp"),
                        r.get("last_seen"),
                        r.get("embed_status", "discovery"),
                    )
                    for r in rows
                ],
            )
            self._writer_conn.commit()

    def entry_exists(self, url_hash: str) -> bool:
        result = self._safe_query("SELECT 1 FROM doc_registry WHERE url_hash = ?", (url_hash,))
        return result is not None

    def get_entry(self, url_hash: str) -> dict | None:
        result = self._safe_query(
            "SELECT url_hash, org, repo, content_type, file_name, content_sha256, "
            "original_url, normalized_url, rule_id, title, embed_status "
            "FROM doc_registry WHERE url_hash = ?",
            (url_hash,),
        )
        if result is None:
            return None
        return {
            "url_hash": result[0],
            "org": result[1],
            "repo": result[2],
            "content_type": result[3],
            "file_name": result[4],
            "content_sha256": result[5] or "",
            "original_url": result[6],
            "normalized_url": result[7],
            "rule_id": result[8],
            "title": result[9],
            "embed_status": result[10],
        }

    def delete_entries_by_org_repo(self, org: str, repo: str) -> None:
        with self._lock:
            self._writer_conn.execute(
                "DELETE FROM doc_registry WHERE org = ? AND repo = ?", (org, repo)
            )
            self._writer_conn.commit()

    # ------------------------------------------------------------------
    # SIGMA_SPEC table (dedicated spec documents)
    # ------------------------------------------------------------------

    def get_pending_sigma_spec(self, org: str | None = None, repo: str | None = None) -> list[dict]:
        """Return pending entries from sigma_spec table, optionally filtered by org/repo."""
        with self._lock:
            conditions = ["embed_status = 'discovery'"]
            params: list[str] = []
            if org:
                conditions.append("org = ?")
                params.append(org)
            if repo:
                conditions.append("repo = ?")
                params.append(repo)
            where = " AND ".join(conditions)
            results = self._writer_conn.execute(
                f"SELECT url_hash, org, repo, content_type, file_name, content_sha256, file_size, "
                f"original_url, normalized_url, title, timestamp, last_seen, embed_status "
                f"FROM sigma_spec WHERE {where} ORDER BY url_hash",
                params,
            ).fetchall()
            col_names = [desc[0] for desc in self._writer_conn.description]
        return [dict(zip(col_names, row)) for row in results]

    def update_spec_status(self, url_hash: str, status: str) -> None:
        with self._lock:
            self._writer_conn.execute(
                "UPDATE sigma_spec SET embed_status = ? WHERE url_hash = ?",
                (status, url_hash),
            )
            self._writer_conn.commit()

    def batch_upsert_sigma_spec(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self._lock:
            self._writer_conn.executemany(
                """INSERT INTO sigma_spec (
                    url_hash, org, repo, content_type, file_name, content_sha256, file_size,
                    original_url, normalized_url, title, timestamp, last_seen, embed_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (url_hash) DO UPDATE SET
                    org = EXCLUDED.org,
                    repo = EXCLUDED.repo,
                    content_type = EXCLUDED.content_type,
                    file_name = EXCLUDED.file_name,
                    content_sha256 = EXCLUDED.content_sha256,
                    file_size = EXCLUDED.file_size,
                    original_url = EXCLUDED.original_url,
                    normalized_url = EXCLUDED.normalized_url,
                    title = EXCLUDED.title,
                    timestamp = EXCLUDED.timestamp,
                    last_seen = EXCLUDED.last_seen,
                    embed_status = CASE
                        WHEN excluded.content_sha256 IS NOT NULL
                             AND excluded.content_sha256 != sigma_spec.content_sha256
                        THEN 'discovery'
                        ELSE sigma_spec.embed_status
                    END""",
                [
                    (
                        r.get("url_hash"),
                        r.get("org") or "",
                        r.get("repo") or "",
                        r.get("content_type"),
                        r.get("file_name"),
                        r.get("content_sha256"),
                        r.get("file_size"),
                        r.get("original_url"),
                        r.get("normalized_url"),
                        r.get("title"),
                        r.get("timestamp"),
                        r.get("last_seen"),
                        r.get("embed_status", "discovery"),
                    )
                    for r in rows
                ],
            )
            self._writer_conn.commit()

    def delete_sigma_spec_by_org_repo(self, org: str, repo: str) -> None:
        """Delete all sigma_spec entries for a given org/repo."""
        with self._lock:
            self._writer_conn.execute(
                "DELETE FROM sigma_spec WHERE org = ? AND repo = ?", (org, repo)
            )
            self._writer_conn.commit()

    # ------------------------------------------------------------------
    # DOC_ERROR table (failed URLs — 30x/40x errors to skip on retry)
    # ------------------------------------------------------------------

    def upsert_doc_error(self, data: dict) -> None:
        with self._lock:
            self._writer_conn.execute(
                """INSERT INTO doc_error (
                    url_hash, original_url, normalized_url,
                    error_code, error_message, org, repo, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (url_hash) DO UPDATE SET
                    error_code = EXCLUDED.error_code,
                    error_message = EXCLUDED.error_message,
                    timestamp = EXCLUDED.timestamp""",
                (
                    data.get("url_hash"),
                    data.get("original_url"),
                    data.get("normalized_url"),
                    data.get("error_code"),
                    data.get("error_message"),
                    data.get("org"),
                    data.get("repo"),
                    data.get("timestamp"),
                ),
            )
            self._writer_conn.commit()

    def get_doc_errors(self, limit: int = 1000, offset: int = 0) -> list[dict]:
        with self._lock:
            results = self._writer_conn.execute(
                "SELECT url_hash, original_url, normalized_url, error_code, error_message, org, repo, timestamp "
                "FROM doc_error ORDER BY url_hash LIMIT ? OFFSET ?",
                [limit, offset],
            ).fetchall()
            col_names = [desc[0] for desc in self._writer_conn.description]
        return [dict(zip(col_names, row)) for row in results]

    def doc_error_exists(self, url_hash: str) -> bool:
        result = self._safe_query("SELECT 1 FROM doc_error WHERE url_hash = ?", (url_hash,))
        return result is not None

    def delete_doc_error(self, url_hash: str) -> None:
        with self._lock:
            self._writer_conn.execute("DELETE FROM doc_error WHERE url_hash = ?", (url_hash,))
            self._writer_conn.commit()

    def get_doc_registry(self, limit: int = 100, offset: int = 0) -> list[dict]:
        with self._lock:
            results = self._writer_conn.execute(
                "SELECT url_hash, org, repo, content_type, file_name, content_sha256, file_size, "
                "original_url, normalized_url, rule_id, title, timestamp, last_seen, embed_status "
                "FROM doc_registry LIMIT ? OFFSET ?",
                [limit, offset],
            ).fetchall()
            col_names = [desc[0] for desc in self._writer_conn.description]
        return [dict(zip(col_names, row)) for row in results]

    def get_pending_doc_registry(self, org: str, repo: str) -> list[dict]:
        with self._lock:
            results = self._writer_conn.execute(
                "SELECT url_hash, org, repo, content_type, file_name, content_sha256, file_size, "
                "original_url, normalized_url, rule_id, title, timestamp, last_seen, embed_status "
                "FROM doc_registry WHERE org = ? AND repo = ? AND embed_status = 'discovery' ORDER BY url_hash",
                (org, repo),
            ).fetchall()
            col_names = [desc[0] for desc in self._writer_conn.description]
        return [dict(zip(col_names, row)) for row in results]

    def get_pending_registry_all(self) -> list[dict]:
        """Return all pending entries from doc_registry."""
        return self.get_pending_entries()

    def get_pending_by_content_type(self, content_type: str | None = None) -> list[dict]:
        """Return pending entries from doc_registry, optionally filtered by content_type."""
        with self._lock:
            if content_type:
                results = self._writer_conn.execute(
                    "SELECT url_hash, org, repo, content_type, file_name, content_sha256, file_size, "
                    "original_url, normalized_url, rule_id, title, timestamp, last_seen, embed_status "
                    "FROM doc_registry WHERE embed_status = 'discovery' AND content_type = ? ORDER BY url_hash",
                    (content_type,),
                ).fetchall()
            else:
                results = self._writer_conn.execute(
                    "SELECT url_hash, org, repo, content_type, file_name, content_sha256, file_size, "
                    "original_url, normalized_url, rule_id, title, timestamp, last_seen, embed_status "
                    "FROM doc_registry WHERE embed_status = 'discovery' "
                    "AND (content_type IS NULL OR content_type != 'sigma_rule') ORDER BY url_hash"
                ).fetchall()
            col_names = [desc[0] for desc in self._writer_conn.description]
        return [dict(zip(col_names, row)) for row in results]

    def reset_github_all(self) -> None:
        """Reset embed_status to 'discovery' for all GitHub entries in doc_registry."""
        with self._lock:
            self._writer_conn.execute(
                "UPDATE doc_registry SET embed_status = 'discovery' WHERE org NOT IN ('local', 'sigmaref')"
            )
            self._writer_conn.commit()

    def update_doc_registry_embed_status(self, url_hash: str, status: str) -> None:
        with self._lock:
            self._writer_conn.execute(
                "UPDATE doc_registry SET embed_status = ? WHERE url_hash = ?",
                (status, url_hash),
            )
            self._writer_conn.commit()

    def reset_embed_status_for_collection(self, collection_name: str) -> None:
        with self._lock:
            if collection_name == "sigmaref":
                self._writer_conn.execute(
                    "UPDATE doc_registry SET embed_status = 'discovery' WHERE org = 'sigmaref'"
                )
            elif collection_name == "github":
                self._writer_conn.execute(
                    "UPDATE doc_registry SET embed_status = 'discovery' WHERE org NOT IN ('local', 'sigmaref')"
                )
            elif collection_name == "local":
                self._writer_conn.execute(
                    "UPDATE doc_registry SET embed_status = 'discovery' WHERE org = 'local'"
                )
            elif collection_name == "sigma_spec":
                self._writer_conn.execute("UPDATE sigma_spec SET embed_status = 'discovery'")
            elif "/" in collection_name:
                parts = collection_name.split("/", 2)
                if len(parts) == 2:
                    org, repo = parts
                    self._writer_conn.execute(
                        "UPDATE doc_registry SET embed_status = 'discovery' WHERE org = ? AND repo = ?",
                        (org, repo),
                    )
            self._writer_conn.commit()

    def delete_doc_registry_by_repo(self, org: str, repo: str) -> None:
        with self._lock:
            self._writer_conn.execute(
                "DELETE FROM doc_registry WHERE org = ? AND repo = ?", (org, repo)
            )
            self._writer_conn.commit()

    def delete_doc_registry_by_url(self, original_url: str) -> None:
        with self._lock:
            self._writer_conn.execute(
                "DELETE FROM doc_registry WHERE original_url = ?", (original_url,)
            )
            self._writer_conn.commit()

    def delete_doc_registry_by_url_hashes(self, url_hashes: list[str]) -> None:
        if not url_hashes:
            return
        with self._lock:
            placeholders = ",".join("?" for _ in url_hashes)
            self._writer_conn.execute(
                f"DELETE FROM doc_registry WHERE url_hash IN ({placeholders})",
                url_hashes,
            )
            self._writer_conn.commit()

    def get_doc_registry_url_hashes_by_repo(self, org: str, repo: str) -> list[str]:
        with self._lock:
            results = self._writer_conn.execute(
                "SELECT url_hash FROM doc_registry WHERE org = ? AND repo = ?",
                (org, repo),
            ).fetchall()
        return [row[0] for row in results]

    def get_local_files(self, limit: int = 1000, offset: int = 0) -> list[dict]:
        with self._lock:
            results = self._writer_conn.execute(
                "SELECT url_hash, org, repo, content_type, file_name, content_sha256, file_size, "
                "original_url, normalized_url, rule_id, title, timestamp, last_seen, embed_status "
                "FROM doc_registry WHERE org = 'local' AND repo = 'local' LIMIT ? OFFSET ?",
                [limit, offset],
            ).fetchall()
            col_names = [desc[0] for desc in self._writer_conn.description]
        return [dict(zip(col_names, row)) for row in results]

    def get_local_file_count(self) -> int:
        with self._lock:
            result = self._writer_conn.execute(
                "SELECT COUNT(*) FROM doc_registry WHERE org = 'local' AND repo = 'local'"
            ).fetchone()
        return result[0] if result else 0

    def resync_local_file_sizes(self, base_path: str) -> dict[str, int]:
        updated = 0
        skipped = 0
        errors = 0
        incomplete = 0

        try:
            base_dir = Path(base_path).resolve()
        except Exception as e:
            logger.warning(f"[resync_local_file_sizes] Invalid base path '{base_path}': {e}")
            return {"updated": 0, "skipped": 0, "error": 0, "incomplete": 0}

        if not base_dir.exists():
            msg = f"[resync_local_file_sizes] Base directory does not exist: {base_dir}"
            logger.warning(msg)
            return {"updated": 0, "skipped": 0, "error": 0, "incomplete": 0}

        def _hash_file(path: Path) -> str | None:
            digest = compute_sha256_file(path)
            if not digest:
                logger.warning(f"[resync_local_file_sizes] Cannot read {path}")
                return None
            return digest

        snapshot: list[tuple[str, str | None, str | None]] = []
        with self._lock:
            results = self._writer_conn.execute(
                "SELECT url_hash, file_name, content_sha256 FROM doc_registry "
                "WHERE org = 'local' AND repo = 'local' "
                "AND (file_size IS NULL OR file_size = 0) ORDER BY url_hash"
            ).fetchall()
            for row in results:
                snapshot.append((row[0], row[1], row[2]))

        updates_file_size: list[tuple[int, str]] = []
        updates_content_sha256: list[tuple[str, str]] = []

        for url_hash, file_name, existing_hash in snapshot:
            if not file_name or not isinstance(file_name, str):
                skipped += 1
                continue

            try:
                file_path = (base_dir / Path(file_name)).resolve()
            except Exception:
                errors += 1
                continue

            try:
                if os.path.commonpath([str(base_dir), str(file_path)]) != str(base_dir):
                    logger.warning(
                        f"[resync_local_file_sizes] Path traversal detected for {file_name}"
                    )
                    skipped += 1
                    continue
            except ValueError:
                logger.warning(
                    f"[resync_local_file_sizes] Cross-drive path detected for {file_name}"
                )
                skipped += 1
                continue

            try:
                stat = file_path.stat()
                new_size = stat.st_size
            except OSError as e:
                logger.warning(f"[resync_local_file_sizes] Cannot stat {file_name}: {e}")
                errors += 1
                continue

            new_hash = existing_hash
            if not existing_hash:
                computed = _hash_file(file_path)
                if computed:
                    new_hash = computed
                else:
                    logger.warning(
                        f"[resync_local_file_sizes] Hash failed for {file_name} "
                        f"(url_hash={url_hash}), file_size updated but content_sha256 unchanged"
                    )

            updates_file_size.append((new_size, url_hash))

            if new_hash and new_hash != existing_hash:
                updates_content_sha256.append((new_hash, url_hash))

            if existing_hash or new_hash:
                updated += 1
            else:
                incomplete += 1

        with self._lock:
            try:
                if updates_file_size:
                    self._writer_conn.executemany(
                        "UPDATE doc_registry SET file_size = ? WHERE url_hash = ?",
                        updates_file_size,
                    )
                if updates_content_sha256:
                    self._writer_conn.executemany(
                        "UPDATE doc_registry SET content_sha256 = ? WHERE url_hash = ?",
                        updates_content_sha256,
                    )
                self._writer_conn.commit()
            except Exception as e:
                logger.error(f"[resync_local_file_sizes] Batch update failed, rolling back: {e}")

        return {"updated": updated, "skipped": skipped, "error": errors, "incomplete": incomplete}

    # ------------------------------------------------------------------
    # RULE_REFERENCES table (junction rule_id ↔ url_hash)
    # ------------------------------------------------------------------

    def upsert_rule_reference(self, rule_id: str, url_hash: str, ref_url: str) -> None:
        """Record that *rule_id* references *ref_url* (identified by *url_hash*)."""
        with self._lock:
            self._writer_conn.execute(
                "INSERT INTO rule_references (rule_id, url_hash, ref_url) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT (rule_id, url_hash) DO NOTHING",
                (rule_id, url_hash, ref_url),
            )
            self._writer_conn.commit()

    def batch_upsert_rule_references(self, rows: list[dict[str, str]]) -> None:
        """Batch upsert rule_references rows.

        Each row must have keys: rule_id, url_hash, ref_url.
        """
        with self._lock:
            self._writer_conn.executemany(
                "INSERT INTO rule_references (rule_id, url_hash, ref_url) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT (rule_id, url_hash) DO NOTHING",
                [(r["rule_id"], r["url_hash"], r["ref_url"]) for r in rows],
            )
            self._writer_conn.commit()

    def get_referencing_rules(self, url_hash: str) -> list[dict]:
        """Return all rules that reference the document identified by *url_hash*."""
        with self._lock:
            results = self._writer_conn.execute(
                "SELECT rule_id, ref_url, created FROM rule_references WHERE url_hash = ? ORDER BY rule_id",
                (url_hash,),
            ).fetchall()
            col_names = [desc[0] for desc in self._writer_conn.description]
        return [dict(zip(col_names, row)) for row in results]

    def get_rule_references(self, rule_id: str) -> list[dict]:
        """Return all reference documents for a given *rule_id*."""
        with self._lock:
            results = self._writer_conn.execute(
                "SELECT r.url_hash, r.ref_url, r.created, "
                "d.content_type, d.file_name, d.embed_status, d.content_sha256 "
                "FROM rule_references r "
                "LEFT JOIN doc_registry d ON r.url_hash = d.url_hash "
                "WHERE r.rule_id = ? ORDER BY r.ref_url",
                (rule_id,),
            ).fetchall()
            col_names = [desc[0] for desc in self._writer_conn.description]
        return [dict(zip(col_names, row)) for row in results]

    def delete_rule_references_by_url_hash(self, url_hash: str) -> None:
        """Delete all rule_references entries for a given *url_hash*."""
        with self._lock:
            self._writer_conn.execute("DELETE FROM rule_references WHERE url_hash = ?", (url_hash,))
            self._writer_conn.commit()

    def get_rule_id_by_url_hash(self, url_hash: str) -> str | None:
        """Return the rule_id associated with a doc_registry entry."""
        with self._lock:
            result = self._writer_conn.execute(
                "SELECT rule_id FROM doc_registry WHERE url_hash = ?", (url_hash,)
            ).fetchone()
        return result[0] if result else None

    def get_rule_reference_paths(self, rule_id: str) -> list[Path]:
        """Return the on-disk paths for all reference documents of a rule."""
        with self._lock:
            results = self._writer_conn.execute(
                "SELECT r.ref_url, d.file_name, d.content_type "
                "FROM rule_references r "
                "JOIN doc_registry d ON r.url_hash = d.url_hash "
                "WHERE r.rule_id = ?",
                (rule_id,),
            ).fetchall()
        paths: list[Path] = []
        for ref_url, file_name, content_type in results:
            try:
                from src.application.documents.sigma_ref_paths import sigmaref_resolve_path
                from src.config.settings import get_config

                cfg = get_config()
                path = sigmaref_resolve_path(
                    Path(cfg.sigmaref_documents_path).resolve(),
                    content_type,
                    file_name or "",
                )
                if path.exists():
                    paths.append(path)
            except Exception:
                continue
        return paths

    # ------------------------------------------------------------------
    # R1.4 — cleanup orphaned head_verified entries (no content_sha256)
    # ------------------------------------------------------------------

    def delete_head_verified_orphans(self, grace_days: int = 7) -> int:
        """Delete doc_registry entries stuck in 'head_verified' without content.

        These are entries created by a HEAD request whose content type was not
        in the supported set — they will never transition to 'embedded'.
        Also removes corresponding rule_references rows.

        Args:
            grace_days: Delete entries older than N days (default 7).

        Returns:
            Number of deleted entries.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=grace_days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        with self._lock:
            # Find orphan url_hashes
            orphans = self._writer_conn.execute(
                "SELECT url_hash FROM doc_registry "
                "WHERE embed_status = 'head_verified' "
                "AND (content_sha256 IS NULL OR content_sha256 = '') "
                "AND (last_seen IS NULL "
                "     OR last_seen < ?)",
                [cutoff],
            ).fetchall()
            orphan_hashes = [row[0] for row in orphans]

            if not orphan_hashes:
                return 0

            # Delete from rule_references first (FK-like cleanup)
            placeholders = ",".join("?" for _ in orphan_hashes)
            self._writer_conn.execute(
                f"DELETE FROM rule_references WHERE url_hash IN ({placeholders})",
                orphan_hashes,
            )
            # Delete from doc_registry
            self._writer_conn.execute(
                f"DELETE FROM doc_registry WHERE url_hash IN ({placeholders})",
                orphan_hashes,
            )
            self._writer_conn.commit()

        logger.info("Deleted %d orphan head_verified entries", len(orphan_hashes))
        return len(orphan_hashes)

    def delete_unreferenced_entries(self) -> int:
        """Delete sigmaref entries whose url_hash is no longer in rule_references.

        These are documents downloaded for rules that no longer exist or whose
        references have been removed.  Only affects entries with ``org='sigmaref'``
        to avoid touching local/GitHub entries.
        """
        with self._lock:
            orphans = self._writer_conn.execute(
                "SELECT url_hash FROM doc_registry "
                "WHERE org = 'sigmaref' "
                "AND url_hash NOT IN (SELECT DISTINCT url_hash FROM rule_references)",
            ).fetchall()
            orphan_hashes = [row[0] for row in orphans]

            if not orphan_hashes:
                return 0

            placeholders = ",".join("?" for _ in orphan_hashes)
            self._writer_conn.execute(
                f"DELETE FROM doc_registry WHERE url_hash IN ({placeholders})",
                orphan_hashes,
            )
            self._writer_conn.commit()

        logger.info("Deleted %d unreferenced sigmaref entries", len(orphan_hashes))
        return len(orphan_hashes)
