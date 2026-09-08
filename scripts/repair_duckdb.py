#!/usr/bin/env python3
# mypy: ignore-errors
"""Repair / reset the DuckDB database after a crash or stale state.

Usage:
    uv run python scripts/repair_duckdb.py          # check + repair
    uv run python scripts/repair_duckdb.py --reset   # reset all data (fresh db)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import duckdb


def _db_path() -> Path:
    """Resolve DuckDB path the same way the app does."""
    try:
        from src.config.settings import get_config

        return Path(get_config().paths_duckdb_path)
    except Exception:
        return Path("data/duckdb/sigmahq.duckdb")


def _initdb_sql() -> str:
    sql_path = Path(__file__).parent.parent / "src" / "infrastructure" / "database" / "initdb.sql"
    return sql_path.read_text(encoding="utf-8")


# Tables the application expects (subset from _VALID_TABLES)
EXPECTED_TABLES: set[str] = {
    "config",
    "embedding_config",
    "system_prompts",
    "models",
    "doc_registry",
    "sigma_spec",
    "doc_error",
    "git_metadata",
    "git_selected_dirs",
    "worker_state",
    "release_cache",
}


def check_integrity(conn: duckdb.DuckDBPyConnection) -> list[str]:
    issues: list[str] = []
    try:
        row = conn.execute("SELECT * FROM pragma_database_size()").fetchone()
        if row:
            print(f"  Database size: {row[3] if len(row) > 3 else '?'}")
    except Exception as e:
        issues.append(f"Cannot read database size: {e}")
    return issues


def check_tables(conn: duckdb.DuckDBPyConnection) -> list[str]:
    issues: list[str] = []
    existing: set[str] = set()
    for row in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_type='BASE TABLE'"
    ).fetchall():
        existing.add(row[0])
    missing = EXPECTED_TABLES - existing
    if missing:
        issues.append(f"Missing tables: {sorted(missing)}")
    else:
        print("  All expected tables present.")
    extra = existing - EXPECTED_TABLES
    if extra:
        print(f"  Extra tables (not in expected set): {sorted(extra)}")
    for t in sorted(existing & EXPECTED_TABLES):
        row = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()
        count = row[0] if row else 0  # type: ignore[index]
        print(f"    {t}: {count} rows")
    return issues


def fix_missing_tables(conn: duckdb.DuckDBPyConnection) -> int:
    existing: set[str] = set()
    for row in conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_type='BASE TABLE'"
    ).fetchall():
        existing.add(row[0])
    missing = EXPECTED_TABLES - existing
    if not missing:
        return 0
    print(f"  Creating missing tables: {sorted(missing)}")
    conn.execute(_initdb_sql())
    conn.commit()
    return len(missing)


def fix_stale_workers(conn: duckdb.DuckDBPyConnection) -> int:
    result = conn.execute(
        """UPDATE worker_state
           SET status = 'idle',
               progress_percent = 0.0,
               current_file = '',
               current_task_id = '',
               last_heartbeat = strftime(now()::TIMESTAMP, '%Y-%m-%dT%H:%M:%S')
           WHERE status IN ('running', 'waiting')
        """
    )
    affected = result.fetchone()
    count = affected[0] if affected else 0
    if count:
        conn.commit()
        print(f"  Reset {count} stale worker(s) to idle.")
    else:
        print("  No stale workers found.")
    return count


def fix_stuck_embeddings(conn: duckdb.DuckDBPyConnection) -> int:
    total = 0
    affected = conn.execute(
        """UPDATE doc_registry
            SET embed_status = 'discovery'
            WHERE embed_status = 'embedding'
        """
    ).fetchone()
    count = affected[0] if affected else 0
    if count:
        conn.commit()
        print(f"  Reset {count} stuck 'embedding' entries to 'discovery' in doc_registry.")
    total += count
    if not total:
        print("  No stuck embeddings found.")
    return total


def fix_orphaned_error_entries(conn: duckdb.DuckDBPyConnection) -> int:
    affected = conn.execute(
        """DELETE FROM doc_error
           WHERE url_hash IN (
               SELECT e.url_hash
               FROM doc_error e
               LEFT JOIN doc_registry r ON e.url_hash = r.url_hash
               WHERE r.url_hash IS NOT NULL
                 AND r.embed_status = 'discovery'
           )
        """
    ).fetchone()
    count = affected[0] if affected else 0
    if count:
        conn.commit()
        print(f"  Removed {count} orphaned error entries (re-discovered URLs).")
    return count


def resync_local_file_sizes(conn: duckdb.DuckDBPyConnection) -> int:
    base_path = Path("data/github")
    updated = 0
    rows = conn.execute(
        """SELECT url_hash, file_name, original_url
            FROM doc_registry
            WHERE org = 'local' AND repo = 'local'
              AND (file_size IS NULL OR file_size = 0)
        """
    ).fetchall()
    for url_hash, file_name, original_url in rows:
        if not file_name:
            continue
        fpath: Path = (
            Path(original_url) if original_url.startswith("file://") else base_path / file_name
        )
        if not fpath.exists():
            local = base_path / file_name
            if local.exists():
                fpath = local
            else:
                continue
        try:
            fsize = fpath.stat().st_size
            sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
            conn.execute(
                "UPDATE doc_registry SET file_size = ?, content_sha256 = ? WHERE url_hash = ?",
                [fsize, sha, url_hash],
            )
            updated += 1
        except Exception:
            continue
    if updated:
        conn.commit()
        print(f"  Resynced {updated} local file(s).")
    else:
        print("  No local files needed resync.")
    return updated


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def fix_missing_last_seen(conn: duckdb.DuckDBPyConnection) -> int:
    updated = 0
    count = conn.execute("SELECT COUNT(*) FROM doc_registry WHERE last_seen IS NULL").fetchone()
    n = count[0] if count else 0
    if n:
        now = _iso_now()
        conn.execute("UPDATE doc_registry SET last_seen = ? WHERE last_seen IS NULL", [now])
        conn.commit()
        print(f"  Set last_seen on {n} row(s) in doc_registry.")
        updated += n
    if not updated:
        print("  No missing last_seen values.")
    return updated


def _find_sigma_ref_dirs() -> list[Path]:
    candidates = [Path("data/sigma_ref_docs"), Path("data/documents/sigmaref")]
    return [d for d in candidates if d.is_dir()]


def reconcile_sigma_ref_files(conn: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """Cross-check sigma ref doc files on disk against the doc_registry table (org='sigmaref').

    Returns dict with 'disk_not_db' (files on disk not in DB) and 'db_not_disk' (DB entries missing files).
    """
    ref_dirs = _find_sigma_ref_dirs()
    db_not_disk: list[str] = []
    disk_not_db: list[str] = []

    if not ref_dirs:
        print("  No sigma ref documents directory found on disk.")
        return {"db_not_disk": db_not_disk, "disk_not_db": disk_not_db}

    db_hashes: set[str] = set()
    for row in conn.execute("SELECT url_hash FROM doc_registry WHERE org = 'sigmaref'").fetchall():
        db_hashes.add(row[0])

    disk_hashes: set[str] = set()
    for ref_dir in ref_dirs:
        for f in ref_dir.iterdir():
            if not f.is_file():
                continue
            stem = f.stem
            if len(stem) == 64:
                try:
                    _ = int(stem, 16)  # validate hex
                    disk_hashes.add(stem)
                except ValueError:
                    pass
            # Files not named by hash (e.g. registry.json)
            if stem not in db_hashes and f.suffix.lower() in (
                ".md",
                ".txt",
                ".html",
                ".pdf",
                ".docx",
                ".htm",
            ):
                disk_not_db.append(f.name)

    for h in sorted(db_hashes - disk_hashes):
        db_not_disk.append(h)

    if disk_not_db:
        print(f"  Sigma ref files on disk NOT in doc_registry: {len(disk_not_db)}")
    if db_not_disk:
        print(f"  doc_registry sigma ref entries with missing files: {len(db_not_disk)}")
    if not disk_not_db and not db_not_disk:
        print("  Sigma ref files are in sync with DB.")
    return {"db_not_disk": db_not_disk, "disk_not_db": disk_not_db}


def reconcile_github_files(conn: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """Cross-check github repo files on disk against the doc_registry table.

    Matches by (org, repo, relative_file_path) rather than content hashing.
    Returns dict with 'disk_not_db' and 'db_not_disk' lists.
    """
    gh_dir = Path("data/github")
    db_not_disk: list[tuple[str, str, str]] = []
    disk_not_db: list[str] = []

    if not gh_dir.is_dir():
        return {"db_not_disk": [str(e) for e in db_not_disk], "disk_not_db": disk_not_db}

    # Build (org, repo, rel_path) set from DB
    db_paths: set[tuple[str, str, str]] = set()
    for row in conn.execute(
        "SELECT org, repo, file_name FROM doc_registry WHERE org != 'local' AND org != 'sigmaref'"
    ).fetchall():
        org, repo, fname = row[0], row[1], row[2]
        if fname:
            db_paths.add((org, repo, fname))

    # Walk disk and match
    disk_paths: set[tuple[str, str, str]] = set()
    for org_dir in gh_dir.iterdir():
        if not org_dir.is_dir():
            continue
        for repo_dir in org_dir.iterdir():
            if not repo_dir.is_dir():
                continue
            org = org_dir.name
            repo = repo_dir.name
            for f in repo_dir.rglob("*"):
                if not f.is_file():
                    continue
                rel = f.relative_to(org_dir / repo).as_posix()
                disk_paths.add((org, repo, rel))
                if (org, repo, rel) not in db_paths:
                    disk_not_db.append(f"{org}/{repo}/{rel}")

    for org, repo, rel in sorted(db_paths - disk_paths):
        db_not_disk.append((org, repo, rel))

    if disk_not_db:
        print(f"  Files on disk NOT in doc_registry: {len(disk_not_db)}")
    if db_not_disk:
        print(f"  DB entries in doc_registry missing on disk: {len(db_not_disk)}")
    if not disk_not_db and not db_not_disk:
        print("  GitHub repo files are in sync with DB.")
    return {
        "db_not_disk": [f"{o}/{r}/{p}" for o, r, p in db_not_disk],
        "disk_not_db": disk_not_db,
    }


def compact_database(conn: duckdb.DuckDBPyConnection, db_path: Path) -> bool:
    try:
        conn.execute("CHECKPOINT")
        print("  Database checkpoint completed.")
        return True
    except Exception as e:
        print(f"  Checkpoint failed: {e}")
        return False


def reset_database(db_path: Path) -> bool:
    if db_path.exists():
        db_path.unlink()
    tmp = db_path.with_suffix(".duckdb.tmp")
    if tmp.exists():
        tmp.unlink()
    conn = duckdb.connect(str(db_path))
    conn.execute(_initdb_sql())
    conn.commit()
    conn.close()
    return True


def cleanup_temp_files(db_path: Path) -> int:
    removed = 0
    tmp = db_path.with_suffix(".duckdb.tmp")
    if tmp.exists():
        try:
            tmp.unlink()
            print(f"  Removed stale temp file: {tmp}")
            removed += 1
        except Exception as e:
            print(f"  Failed to remove {tmp}: {e}")
    return removed


def register_orphan_sigmaref_files(conn: duckdb.DuckDBPyConnection) -> int:
    """Register orphan sigma ref doc files (on disk but not in doc_registry) into the DB."""
    ref_dirs = _find_sigma_ref_dirs()
    if not ref_dirs:
        return 0
    db_hashes = {
        row[0]
        for row in conn.execute(
            "SELECT url_hash FROM doc_registry WHERE org = 'sigmaref'"
        ).fetchall()
    }
    now = _iso_now()
    rows = []
    for ref_dir in ref_dirs:
        for f in ref_dir.iterdir():
            if not f.is_file():
                continue
            stem = f.stem
            if len(stem) != 64:
                continue
            try:
                int(stem, 16)
            except ValueError:
                continue
            if stem in db_hashes:
                continue
            ext = f.suffix.lower().lstrip(".")
            ct_map = {
                "md": "markdown",
                "html": "html",
                "htm": "html",
                "txt": "plain_text",
                "pdf": "pdf",
                "docx": "office_document",
            }
            content_type = ct_map.get(ext, "markdown")
            content_sha = hashlib.sha256(f.read_bytes()).hexdigest()
            rows.append(
                {
                    "url_hash": stem,
                    "org": "sigmaref",
                    "repo": "references",
                    "content_type": content_type,
                    "file_name": f.name,
                    "content_sha256": content_sha,
                    "file_size": f.stat().st_size,
                    "original_url": f"unknown://{stem}",
                    "normalized_url": f"unknown://{stem}",
                    "rule_id": "00000000-0000-0000-0000-000000000000",
                    "title": "",
                    "timestamp": now,
                    "last_seen": now,
                    "embed_status": "discovery",
                }
            )
    if not rows:
        print("  No orphan files to register.")
        return 0
    conn.executemany(
        """INSERT INTO doc_registry (
            url_hash, org, repo, content_type, file_name, content_sha256, file_size,
            original_url, normalized_url, rule_id, title, timestamp, last_seen, embed_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                r["url_hash"],
                r["org"],
                r["repo"],
                r["content_type"],
                r["file_name"],
                r["content_sha256"],
                r["file_size"],
                r["original_url"],
                r["normalized_url"],
                r["rule_id"],
                r["title"],
                r["timestamp"],
                r["last_seen"],
                r["embed_status"],
            )
            for r in rows
        ],
    )
    conn.commit()
    print(f"  Registered {len(rows)} orphan file(s) in doc_registry.")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair DuckDB database")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the database from scratch (all data lost)",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register orphan files on disk into the database",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Override DuckDB path (default: from config or data/duckdb/sigmahq.duckdb)",
    )
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else _db_path()
    print(f"DuckDB path: {db_path.resolve()}")

    if args.reset:
        print("\n--- RESET ---")
        if reset_database(db_path):
            print("Database reset successfully.")
        return 0

    if not db_path.exists():
        print(f"Database file not found at {db_path}")
        print("Creating fresh database...")
        conn = duckdb.connect(str(db_path))
        conn.execute(_initdb_sql())
        conn.commit()
        conn.close()
        print("Fresh database created.")
        return 0

    issues_found = False

    # Open database directly (bypass app)
    conn = duckdb.connect(str(db_path))

    print("\n--- Integrity check ---")
    issues = check_integrity(conn)
    if issues:
        issues_found = True
        for i in issues:
            print(f"  ISSUE: {i}")

    print("\n--- Tables ---")
    issues = check_tables(conn)
    if issues:
        issues_found = True

    print("\n--- Fixes ---")
    n = fix_missing_tables(conn)
    if n:
        print(f"  Created {n} missing table(s).")
    fix_stale_workers(conn)
    fix_stuck_embeddings(conn)
    fix_orphaned_error_entries(conn)
    fix_missing_last_seen(conn)
    resync_local_file_sizes(conn)
    cleanup_temp_files(db_path)
    compact_database(conn, db_path)

    print("\n--- Reconciliation disk <-> DB ---")
    ref_result = reconcile_sigma_ref_files(conn)
    if ref_result["disk_not_db"] or ref_result["db_not_disk"]:
        issues_found = True

    if args.register:
        print("\n--- Register orphan files ---")
        register_orphan_sigmaref_files(conn)

    conn.close()

    print("\n--- Summary ---")
    if issues_found:
        print("Issues found (see above).")
    else:
        print("Database looks healthy.")


if __name__ == "__main__":
    sys.exit(main())
