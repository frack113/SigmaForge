"""Shared repository management router factory.

Replaces the duplicated github.py / spec.py patterns.
"""

import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from src.infrastructure.github.git import (
    _get_repo_path,
    _is_valid_repo,
    clone_repo,
    delete_repo,
    get_last_commit_date,
    get_metadata,
    get_selected_dirs,
    is_repo_outdated,
    list_directory_tree,
    list_repos,
    save_metadata,
    save_selected_dirs,
    update_repo,
)
from src.workers.enums import WorkerName

logger = logging.getLogger(__name__)

_VALID_ORG_NAME_RE = re.compile(r"^(?!\.\.?$)[\w.-]+$")


# ── Shared Pydantic models ──


class RepositoryAddRequest(BaseModel):
    url: str = Field(..., description="Git repository URL")
    branch: str = Field(default="main", description="Branch to clone")


class RepositoryResponse(BaseModel):
    success: bool
    message: str | None = None
    data: Any = None
    error: str | None = None


class RepositoryStatus(BaseModel):
    org: str
    name: str
    repo_status: str | None = None
    last_synced: datetime | str | None = None
    url: str | None = None
    branch: str | None = None
    last_commit: str | None = None
    sync_class: str = Field(
        default="btn-success",
        description="btn-success|btn-warning|btn-unknown",
    )


class DirectoryTreeResponse(BaseModel):
    success: bool
    tree: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class SelectDirsRequest(BaseModel):
    selected: list[str] = Field(default_factory=list, description="List of folder paths")


class SelectDirsResponse(BaseModel):
    success: bool
    message: str | None = None
    error: str | None = None
    data: Any = None


# ── Shared helpers ──


def _validate_org_name(org: str, name: str) -> None:
    """Validate org/name to prevent path traversal. Raises HTTPException if invalid."""
    if not _VALID_ORG_NAME_RE.match(org) or not _VALID_ORG_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid org/name: '{org}/{name}' contains path traversal characters",
        )


def _extract_org_name(url: str) -> tuple[str, str]:
    """Extract org and name from URL."""
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parts = url.split("/")
    if len(parts) < 2:
        raise ValueError("Invalid URL format")
    return parts[-2].lower(), parts[-1].lower()


def _mark_selected(node: dict[str, Any], selected: list[str]) -> None:
    """Mark tree nodes as selected based on their path."""
    if node.get("path") in selected:
        node["selected"] = True
    if "children" in node:
        for child in node["children"]:
            _mark_selected(child, selected)


# ── Factory ──


def create_repo_router(
    *,
    prefix: str,
    tags: list[str],
    repos_dir_getter: Callable[[], Path],
    include_detail_endpoint: bool = False,
    include_status_endpoint: bool = False,
    include_selected_dirs_endpoint: bool = False,
    include_outdated_check: bool = True,
    include_scan_endpoint: bool = True,
    on_delete_cleanup: Callable[[str, str], None] | None = None,
    use_get_for_sync: bool = False,
    select_dirs_worker: WorkerName = WorkerName.GITHUB_DISCOVERY,
    scan_workers: list[WorkerName] | None = None,
    source_type: str = "",
) -> APIRouter:
    """Create a configured repository management router.

    Parameters
    ----------
    prefix : str
        FastAPI router prefix (e.g. "/api/v1/github").
    tags : list[str]
        FastAPI router tags.
    repos_dir_getter : Callable[[], Path]
        Function returning the base directory for repo storage.
    include_detail_endpoint : bool
        Add ``GET /repos/{org}/{name}`` detail endpoint.
    include_status_endpoint : bool
        Add ``GET /repos/{org}/{name}/status`` endpoint.
    include_selected_dirs_endpoint : bool
        Add ``GET /repos/{org}/{name}/selected-dirs`` endpoint.
    include_scan_endpoint : bool
        Add ``POST /repos/{org}/{name}/scan`` endpoint (sync + file discovery).
    on_delete_cleanup : Callable | None
        Extra cleanup callback on repo deletion (receives ``org, name``).
    use_get_for_sync : bool
        Use GET instead of POST for the single-repo sync endpoint.
    select_dirs_worker : WorkerName
        Worker type to dispatch after saving selected dirs (default: GITHUB_DISCOVERY).
    scan_workers : list[WorkerName] | None
        Worker types to dispatch after a scan sync. When ``None``, fires all three discovery
        workers (GITHUB, LOCAL, SPEC) for backward compatibility.
    source_type : str
        Source type identifier stored with selected dirs (e.g. 'github', 'spec').
    """
    router = APIRouter(prefix=prefix, tags=tags)  # type: ignore[arg-type]
    _sync_lock = Lock()

    def _get_repos_dir() -> Path:
        return repos_dir_getter().resolve()

    def _sync_single_repo(org: str, name: str, branch: str | None = None) -> dict[str, Any]:
        """Sync a single repository and save metadata. Runs under _sync_lock."""
        with _sync_lock:
            repos_dir = _get_repos_dir()
            meta = get_metadata(org, name, repos_dir=repos_dir) or {}
            resolved_branch = branch or meta.get("branch", "main")
            result = update_repo(org=org, name=name, branch=resolved_branch, repos_dir=repos_dir)
            if result.get("success"):
                merged = {
                    **meta,
                    "org": org,
                    "name": name,
                    "branch": resolved_branch,
                    "status": "synced",
                    "last_synced": datetime.now().isoformat(),
                    "remote_head": result.get("remote_head"),
                }
                save_metadata(org, name, merged, repos_dir=repos_dir)
        return result

    # ── LIST repos ──

    @router.get("/repos", response_model=list[RepositoryStatus])
    async def list_repos_handler() -> list[RepositoryStatus]:
        """List all repositories."""

        def _build_repo_list() -> list[RepositoryStatus]:
            repos_dir = _get_repos_dir()
            repos = list_repos(repos_dir=repos_dir)
            result: list[RepositoryStatus] = []
            for repo in repos:
                metadata = get_metadata(repo["org"], repo["name"]) or {}
                stored_status = metadata.get("status", "synced")

                if stored_status in ("cloning", "syncing"):
                    sync_class = "btn-warning"
                elif stored_status == "error":
                    sync_class = "btn-unknown"
                elif include_outdated_check and is_repo_outdated(repo["org"], repo["name"]):
                    sync_class = "btn-danger"
                else:
                    sync_class = "btn-success"

                last_commit = get_last_commit_date(repo["org"], repo["name"], repos_dir=repos_dir)
                result.append(
                    RepositoryStatus(
                        org=repo["org"],
                        name=repo["name"],
                        repo_status=metadata.get("status", "synced"),
                        last_synced=metadata.get("last_synced"),
                        url=metadata.get("url"),
                        branch=metadata.get("branch"),
                        last_commit=last_commit,
                        sync_class=sync_class,
                    )
                )
            return result

        return await asyncio.to_thread(_build_repo_list)

    # ── ADD repo ──

    @router.post("/repos", response_model=RepositoryResponse)
    async def add_repo(
        request: RepositoryAddRequest,
        background_tasks: BackgroundTasks,
    ) -> RepositoryResponse:
        """Add a new repository."""
        try:
            org, name = _extract_org_name(request.url)
        except ValueError:
            return RepositoryResponse(success=False, error="Invalid URL format")

        repos_dir = _get_repos_dir()
        existing = list_repos(repos_dir=repos_dir)
        if any(r["org"] == org and r["name"] == name for r in existing):
            return RepositoryResponse(
                success=False, error=f"Repository '{org}/{name}' already exists"
            )

        def clone_with_status() -> None:
            with _sync_lock:
                rd = _get_repos_dir()
                result = clone_repo(url=request.url, repos_dir=rd, branch=request.branch)
                if result.get("success"):
                    save_metadata(
                        org,
                        name,
                        {
                            "org": org,
                            "name": name,
                            "url": request.url,
                            "branch": request.branch,
                            "status": "synced",
                            "last_synced": datetime.now().isoformat(),
                            "created_at": datetime.now().isoformat(),
                            "remote_head": result.get("remote_head"),
                        },
                        repos_dir=rd,
                    )
                    save_selected_dirs(org, name, [], repos_dir=rd, source_type=source_type)
                else:
                    save_metadata(
                        org,
                        name,
                        {
                            "org": org,
                            "name": name,
                            "url": request.url,
                            "branch": request.branch,
                            "status": "error",
                            "error": result.get("error"),
                        },
                        repos_dir=rd,
                    )

        background_tasks.add_task(clone_with_status)

        return RepositoryResponse(
            success=True,
            message=f"Cloning repository '{org}/{name}' in background",
            data={"org": org, "name": name, "status": "cloning"},
        )

    # ── DELETE repo ──

    @router.delete("/repos/{org}/{name}", response_model=RepositoryResponse)
    async def delete_repo_handler(
        org: str,
        name: str,
    ) -> RepositoryResponse:
        """Delete a repository."""
        try:
            _validate_org_name(org, name)
        except HTTPException as e:
            return RepositoryResponse(success=False, error=e.detail)
        result = delete_repo(org, name, repos_dir=_get_repos_dir())
        if result.get("success"):
            if on_delete_cleanup:
                try:
                    on_delete_cleanup(org, name)
                except Exception as e:
                    logger.error("Cleanup failed for %s/%s: %s", org, name, e)
            return RepositoryResponse(success=True, message=f"Repository '{org}/{name}' deleted")
        return RepositoryResponse(success=False, error=result.get("error", "Delete failed"))

    # ── SYNC single repo ──

    @router.get(
        "/repos/{org}/{name}/sync",
        response_model=RepositoryResponse,
        include_in_schema=use_get_for_sync,
    )
    @router.post(
        "/repos/{org}/{name}/sync",
        response_model=RepositoryResponse,
        include_in_schema=not use_get_for_sync,
    )
    async def sync_repo(
        org: str,
        name: str,
        background_tasks: BackgroundTasks,
        branch: str | None = None,
    ) -> RepositoryResponse:
        """Sync a repository."""
        _validate_org_name(org, name)
        repos_dir = _get_repos_dir()
        metadata = get_metadata(org, name, repos_dir=repos_dir)
        if branch is None:
            branch = (metadata or {}).get("branch", "main")

        if branch and (".." in branch or branch.startswith("/") or branch.endswith("/")):
            return RepositoryResponse(success=False, error="Invalid branch name")
        if len(branch) > 255:
            return RepositoryResponse(success=False, error="Branch name too long")

        repos = list_repos(repos_dir=repos_dir)
        if not any(r["org"] == org and r["name"] == name for r in repos):
            return RepositoryResponse(success=False, error=f"Repository '{org}/{name}' not found")

        background_tasks.add_task(_sync_single_repo, org, name, branch)

        return RepositoryResponse(success=True, message="Sync started in background")

    # ── SYNC ALL repos ──

    @router.post("/repos/sync-all", response_model=RepositoryResponse)
    async def sync_all_repos(
        background_tasks: BackgroundTasks,
    ) -> RepositoryResponse:
        """Sync all registered repositories."""
        repos_dir = _get_repos_dir()
        repos = list_repos(repos_dir=repos_dir)
        if not repos:
            return RepositoryResponse(success=True, message="No repositories to sync")

        def sync_all_task() -> None:
            for repo_data in list(repos):
                try:
                    _sync_single_repo(repo_data["org"], repo_data["name"])
                except Exception as e:
                    logger.error(f"Failed to sync {repo_data['org']}/{repo_data['name']}: {e}")

        background_tasks.add_task(sync_all_task)

        return RepositoryResponse(success=True, message="Sync started for all repositories")

    # ── TREE listing ──

    @router.get("/repos/{org}/{name}/tree", response_model=DirectoryTreeResponse)
    async def get_repo_tree(
        org: str,
        name: str,
        max_depth: int = 5,
    ) -> DirectoryTreeResponse:
        """Get directory tree for a repository."""
        try:
            _validate_org_name(org, name)
        except HTTPException as e:
            return DirectoryTreeResponse(success=False, error=e.detail)
        try:
            repos_dir = _get_repos_dir()
            repo_path = _get_repo_path(repos_dir, org, name)
            if not repo_path.exists() or not _is_valid_repo(repo_path):
                try:
                    metadata = get_metadata(org, name, repos_dir=repos_dir)
                    if metadata:
                        return DirectoryTreeResponse(
                            success=False,
                            error=f"Repository '{org}/{name}' not cloned yet",
                        )
                except Exception:
                    pass
                return DirectoryTreeResponse(
                    success=False,
                    error=f"Repository '{org}/{name}' not found or not cloned yet",
                )

            tree = list_directory_tree(org, name, repos_dir=repos_dir, max_depth=max_depth)

            # When a separate selected-dirs endpoint exists, the frontend fetches it
            # separately — no need to mark selections inline.
            if not include_selected_dirs_endpoint:
                selected: list[str] = []
                try:
                    selected = get_selected_dirs(org, name) or []
                except Exception:
                    pass
                for node in tree:
                    _mark_selected(node, selected)

            return DirectoryTreeResponse(success=True, tree=tree)
        except PermissionError:
            return DirectoryTreeResponse(
                success=False,
                error="Permission denied accessing repository directory",
            )
        except Exception as e:
            logger.error(f"Failed to get tree for {org}/{name}: {type(e).__name__}: {e}")
            return DirectoryTreeResponse(
                success=False,
                error=f"Error loading directory structure: {type(e).__name__}",
            )

    # ── SAVE selected dirs ──

    @router.post("/repos/{org}/{name}/select-dirs", response_model=SelectDirsResponse)
    async def select_dirs(
        org: str,
        name: str,
        request_body: SelectDirsRequest,
        background_tasks: BackgroundTasks,
        request: Request,
    ) -> SelectDirsResponse:
        """Save selected directories for a repository and trigger discovery."""
        try:
            _validate_org_name(org, name)
        except HTTPException as e:
            return SelectDirsResponse(success=False, error=e.detail)
        repos_dir = _get_repos_dir()
        try:
            repo_path = _get_repo_path(repos_dir, org, name)
            if not repo_path.exists() or not _is_valid_repo(repo_path):
                return SelectDirsResponse(
                    success=False, error=f"Repository '{org}/{name}' not found"
                )
        except Exception as e:
            return SelectDirsResponse(success=False, error=str(e))
        result = save_selected_dirs(
            org, name, request_body.selected, repos_dir=repos_dir, source_type=source_type
        )
        if result.get("success"):
            dispatcher = request.app.state.dispatcher
            background_tasks.add_task(
                dispatcher.ask_for_worker,
                select_dirs_worker,
                task_type=select_dirs_worker.value,
                collection_name="all",
                repo_key=f"{org}/{name}",
            )
            return SelectDirsResponse(
                success=True,
                message=f"Saved {len(request_body.selected)} selected directories for {org}/{name}",
            )
        return SelectDirsResponse(
            success=False,
            error=result.get("error", "Failed to save selections"),
        )

    # ── SCAN repo (sync + file discovery) ──

    if include_scan_endpoint:

        @router.post("/repos/{org}/{name}/scan", response_model=RepositoryResponse)
        async def scan_repo(
            org: str,
            name: str,
            background_tasks: BackgroundTasks,
            request: Request,
        ) -> RepositoryResponse:
            """Sync a repository and trigger file discovery.

            This combines ``sync`` + the global file-list trigger so users
            don't have to click ``List Document`` separately.
            """
            _validate_org_name(org, name)
            repos_dir = _get_repos_dir()
            repos = list_repos(repos_dir=repos_dir)
            if not any(r["org"] == org and r["name"] == name for r in repos):
                return RepositoryResponse(
                    success=False,
                    error=f"Repository '{org}/{name}' not found",
                )

            dispatcher = request.app.state.dispatcher

            def _run_scan() -> None:
                with _sync_lock:
                    _sync_single_repo(org, name, None)
                _workers = scan_workers or [
                    WorkerName.GITHUB_DISCOVERY,
                    WorkerName.LOCAL_DISCOVERY,
                    WorkerName.SPEC_DISCOVERY,
                ]
                _collection_map = {
                    WorkerName.GITHUB_DISCOVERY: "all",
                    WorkerName.LOCAL_DISCOVERY: "local",
                    WorkerName.SPEC_DISCOVERY: "spec",
                }
                for wc in _workers:
                    coll = _collection_map.get(wc, "all")
                    dispatcher.ask_for_worker(wc, task_type=wc.value, collection_name=coll)

            background_tasks.add_task(_run_scan)

            return RepositoryResponse(
                success=True,
                message=f"Scan started for '{org}/{name}'",
            )

    # ── OPTIONAL endpoints ──

    if include_selected_dirs_endpoint:

        @router.get("/repos/{org}/{name}/selected-dirs", response_model=SelectDirsResponse)
        async def get_selected_dirs_endpoint(
            org: str,
            name: str,
        ) -> SelectDirsResponse:
            """Get saved selected directories for a repository."""
            try:
                _validate_org_name(org, name)
            except HTTPException as e:
                return SelectDirsResponse(success=False, error=e.detail)
            selected = get_selected_dirs(org, name, repos_dir=_get_repos_dir())
            return SelectDirsResponse(
                success=True,
                message=f"Loaded {len(selected)} selected directories for {org}/{name}",
                data={"selected": selected},
            )

    if include_detail_endpoint:

        @router.get("/repos/{org}/{name}", response_model=RepositoryStatus)
        async def get_repo(org: str, name: str) -> RepositoryStatus:
            """Get repository details."""
            _validate_org_name(org, name)
            repos_dir = _get_repos_dir()
            repos = list_repos(repos_dir=repos_dir)
            if not any(r["org"] == org and r["name"] == name for r in repos):
                raise HTTPException(
                    status_code=404,
                    detail=f"Repository '{org}/{name}' not found",
                )
            metadata = get_metadata(org, name) or {}
            last_commit = get_last_commit_date(org, name)
            return RepositoryStatus(
                org=org,
                name=name,
                repo_status=metadata.get("status", "synced"),
                last_synced=metadata.get("last_synced"),
                url=metadata.get("url"),
                branch=metadata.get("branch"),
                last_commit=last_commit,
            )

    if include_status_endpoint:

        @router.get("/repos/{org}/{name}/status", response_model=RepositoryStatus)
        async def get_repo_status(org: str, name: str) -> RepositoryStatus:
            """Get repository sync status."""
            _validate_org_name(org, name)
            repos_dir = _get_repos_dir()
            repos = list_repos(repos_dir=repos_dir)
            if not any(r["org"] == org and r["name"] == name for r in repos):
                return RepositoryStatus(org=org, name=name, repo_status="error")
            metadata = get_metadata(org, name) or {}
            return RepositoryStatus(
                org=org,
                name=name,
                repo_status=metadata.get("status", "synced"),
                last_synced=metadata.get("last_synced"),
                url=metadata.get("url"),
                branch=metadata.get("branch"),
            )

    return router
