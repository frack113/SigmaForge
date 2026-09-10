"""DuckDB database lifecycle manager."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import NamedTuple


class DuckDbHealth(Enum):
    """DuckDB database health state."""

    MISSING = "missing"
    DIRTY_TABLES = "dirty_tables"
    EXCESS_TABLES = "excess_tables"
    HEALTHY = "healthy"


class DuckDbStatus(NamedTuple):
    path: Path
    state: DuckDbHealth
    file_size: int
    tables_missing: list[str]
    tables_excess: list[str]
    needs_fix: bool
    needs_clean: bool


# Project root: goes up from src/application/system/ to SigmaForge/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Tables that must exist per initdb.sql (CREATE TABLE IF NOT EXISTS order)
_EXPECTED_TABLES = [
    "config",
    "system_prompts",
    "models",
    "doc_registry",
    "git_metadata",
    "git_selected_dirs",
    "sigma_spec",
    "rule_references",
    "doc_error",
    "worker_state",
    "release_cache",
    "installed_versions",
]


class DuckDbManager:
    """Manages the DuckDB database file: creation, cleanup, hard reset, schema validation."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            db_path = Path("data/duckdb/sigmahq.duckdb")
        else:
            db_path = Path(db_path)
        if not db_path.is_absolute():
            db_path = (_PROJECT_ROOT / db_path).resolve()
        else:
            db_path = db_path.resolve()
        self._db_path = db_path

    @property
    def db_path(self) -> Path:
        return self._db_path

    @classmethod
    def default(cls) -> DuckDbManager:
        return cls()

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    @staticmethod
    def _get_existing_tables(db_path: Path) -> list[str]:
        """Return list of existing table names in the DuckDB database."""
        try:
            import duckdb

            with duckdb.connect(str(db_path)) as conn:
                rows = conn.execute("SELECT table_name FROM duckdb_tables()").fetchall()
                return sorted([r[0] for r in rows])
        except Exception:
            return []

    @staticmethod
    def _inspect_db(path: Path) -> DuckDbStatus:
        exists = path.exists() and path.is_file()
        if not exists:
            return DuckDbStatus(
                path=path,
                state=DuckDbHealth.MISSING,
                file_size=0,
                tables_missing=list(_EXPECTED_TABLES),
                tables_excess=[],
                needs_fix=True,
                needs_clean=False,
            )

        file_size = path.stat().st_size
        existing = DuckDbManager._get_existing_tables(path)
        existing_set = set(existing)
        expected_set = set(_EXPECTED_TABLES)

        tables_missing = sorted(expected_set - existing_set)
        tables_excess = sorted(existing_set - expected_set)

        if tables_missing:
            state = DuckDbHealth.DIRTY_TABLES
            needs_fix = True
            needs_clean = False
        elif tables_excess:
            state = DuckDbHealth.EXCESS_TABLES
            needs_fix = False
            needs_clean = True
        else:
            state = DuckDbHealth.HEALTHY
            needs_fix = False
            needs_clean = False

        return DuckDbStatus(
            path=path,
            state=state,
            file_size=file_size,
            tables_missing=tables_missing,
            tables_excess=tables_excess,
            needs_fix=needs_fix,
            needs_clean=needs_clean,
        )

    # ------------------------------------------------------------------
    # Status (API)
    # ------------------------------------------------------------------

    def status(self) -> dict:
        s = self._inspect_db(self._db_path)
        return {
            "path": str(self._db_path),
            "relative": str(s.path),
            "state": s.state.value,
            "file_size": s.file_size,
            "tables_missing": s.tables_missing,
            "tables_excess": s.tables_excess,
            "needs_fix": s.needs_fix,
            "needs_clean": s.needs_clean,
            "is_healthy": s.state == DuckDbHealth.HEALTHY,
        }

    # ------------------------------------------------------------------
    # Fix — create DB or restore missing tables
    # ------------------------------------------------------------------

    def create_missing(self) -> str | None:
        """Create the database file if missing, or run initdb.sql to restore missing tables."""
        import duckdb

        if not self._db_path.exists():
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            keep = self._db_path.parent / ".keep"
            if not keep.exists():
                keep.touch()

        initdb_path = _PROJECT_ROOT / "src" / "infrastructure" / "database" / "initdb.sql"
        if initdb_path.exists():
            with duckdb.connect(str(self._db_path)) as conn:
                conn.execute(initdb_path.read_text(encoding="utf-8"))
            return str(self._db_path)

        return None

    def ensure(self) -> str | None:
        return self.create_missing()

    # ------------------------------------------------------------------
    # Clean — drop excess tables
    # ------------------------------------------------------------------

    def clean(self) -> dict:
        import duckdb

        if not self._db_path.exists():
            return {"status": "ok", "message": "DuckDB database does not exist"}

        status = self._inspect_db(self._db_path)
        if not status.tables_excess:
            return {"status": "ok", "message": "No excess tables to remove"}

        with duckdb.connect(str(self._db_path)) as conn:
            for table in status.tables_excess:
                conn.execute(f'DROP TABLE IF EXISTS "{table}"')

        return {
            "status": "ok",
            "message": f"Removed {len(status.tables_excess)} excess table(s): {', '.join(status.tables_excess)}",
        }

    # ------------------------------------------------------------------
    # Hard reset
    # ------------------------------------------------------------------

    def hard_reset(self) -> dict:
        """Delete and recreate the database from scratch."""
        import duckdb

        if self._db_path.exists():
            self._db_path.unlink()

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        initdb_path = _PROJECT_ROOT / "src" / "infrastructure" / "database" / "initdb.sql"
        if initdb_path.exists():
            with duckdb.connect(str(self._db_path)) as conn:
                conn.execute(initdb_path.read_text(encoding="utf-8"))

        return {"status": "ok", "message": "DuckDB database hard reset"}

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        s = self._inspect_db(self._db_path)
        return {
            "healthy": 1 if s.state == DuckDbHealth.HEALTHY else 0,
            "dirty": 1 if s.state == DuckDbHealth.DIRTY_TABLES else 0,
            "missing": 1 if s.state == DuckDbHealth.MISSING else 0,
            "excess": 1 if s.state == DuckDbHealth.EXCESS_TABLES else 0,
        }
