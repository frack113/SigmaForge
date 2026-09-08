"""Git repository local management."""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from git import Repo
from git.exc import InvalidGitRepositoryError

from src.config.settings import get_config
from src.infrastructure.database import DatabaseService

logger = logging.getLogger(__name__)

LOCKFILE = ".cloning"

# Valid org/name pattern: alphanumeric, hyphens, underscores, dots (no path separators)
_VALID_ORG_NAME_RE = re.compile(r"^(?!\.\.?$)[\w.-]+$")


def _validate_org_name(org: str, name: str) -> None:
    """Validate org/name to prevent path traversal."""
    if not _VALID_ORG_NAME_RE.match(org) or not _VALID_ORG_NAME_RE.match(name):
        raise ValueError(f"Invalid org/name: '{org}/{name}' contains path traversal characters")


def _get_repo_path(repos_dir: Path, org: str, name: str) -> Path:
    """Get the full path for a repository."""
    _validate_org_name(org, name)
    return repos_dir / org / name


def _is_valid_repo(path: Path) -> bool:
    """Check if a path is a valid Git repository using gitpython."""
    try:
        Repo(path)
        return True
    except InvalidGitRepositoryError:
        return False


def _get_or_create_repo(path: Path) -> Repo | None:
    """Get Repo instance or None if invalid."""
    try:
        return Repo(path)
    except InvalidGitRepositoryError:
        return None


def _validate_git_url(url: str) -> None:
    """Validate a git URL to prevent SSRF. Only https:// and git:// schemes, block private IPs."""
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "git"):
        raise ValueError(
            f"Unsupported URL scheme: {parsed.scheme} — only https:// and git:// are allowed"
        )
    host = parsed.hostname or ""
    if host in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("URL points to localhost, which is not allowed")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return
    if ip.is_private or ip.is_loopback or ip.is_link_local:
        raise ValueError(f"URL points to a private/reserved IP address: {host}")


def clone_repo(
    url: str | None = None,
    org: str | None = None,
    name: str | None = None,
    repos_dir: Path | None = None,
    branch: str | None = None,
    depth: int | None = None,
) -> dict[str, Any]:
    """Clone a Git repository.

    Args:
        url: Git repository URL. If None, constructed from org/name.
        org: Organization/user. Auto-extracted from URL if None.
        name: Repository name. Auto-extracted from URL if None.
        repos_dir: Base directory for cloned repos.
        branch: Branch to clone (default: None = default branch).
        depth: Shallow clone depth (default: None = full clone).
    """
    if url is not None:
        url = url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        try:
            _validate_git_url(url)
        except ValueError as e:
            return {"success": False, "error": str(e)}
    if url is None:
        if not org or not name:
            return {
                "success": False,
                "error": "org and name required when url is not provided",
            }
        url = f"https://github.com/{org}/{name}.git"

    repos_dir = Path(repos_dir or get_config().paths_github_dir).resolve()
    repos_dir.mkdir(parents=True, exist_ok=True)

    # Defensive: a __temp__ left over from a previous crashed/failed clone
    # makes git reject every subsequent attempt with
    # `fatal: destination path '...' already exists and is not an empty directory`.
    # Always prune it before we try to use it.
    temp_dir = repos_dir / "__temp__"
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Extract org/name from URL using gitpython's remote
    if org is None or name is None:
        try:
            temp_clone = Repo.clone_from(url, temp_dir / "__extract__", depth=1)
            remote_url = temp_clone.remote().url
            # git@github.com:org/name.git or https://github.com/org/name.git
            parts = remote_url.rstrip("/").replace(".git", "").split("/")
            if remote_url.startswith("git@"):
                org = parts[-1].lower()
                name = parts[-2].lower()
            else:
                org = parts[-2].lower()
                name = parts[-1].lower()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    if org is None or name is None:
        return {"success": False, "error": "Could not determine org/name from URL"}

    dest_path = _get_repo_path(repos_dir, org, name)

    if dest_path.exists():
        if _is_valid_repo(dest_path):
            return {
                "success": False,
                "error": f"Repository '{org}/{name}' already exists",
            }
        shutil.rmtree(dest_path, ignore_errors=True)

    lockfile = dest_path / LOCKFILE
    if lockfile.exists():
        return {
            "success": False,
            "error": f"Repository '{org}/{name}' is currently being cloned",
        }

    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {}
        if branch:
            kwargs["branch"] = branch
        if depth:
            kwargs["depth"] = depth

        cloned = Repo.clone_from(url, dest_path, **kwargs)

        # gitpython extracts this automatically
        current_branch = cloned.active_branch.name if cloned.active_branch else None
        head_hash = cloned.head.commit.hexsha

        logger.info(f"Cloned repository: {url} -> {dest_path}")
        return {
            "success": True,
            "org": org,
            "name": name,
            "path": str(dest_path),
            "branch": current_branch,
            "remote_head": head_hash,
        }
    except Exception as e:
        logger.error(f"Failed to clone {url}: {e}")
        if dest_path.exists():
            shutil.rmtree(dest_path, ignore_errors=True)
        return {"success": False, "error": str(e)}


def update_repo(
    org: str,
    name: str,
    branch: str = "main",
    repos_dir: Path | None = None,
) -> dict[str, Any]:
    """Update a local Git repository by pulling latest changes."""
    repos_dir = Path(repos_dir or get_config().paths_github_dir).resolve()
    repo_path = _get_repo_path(repos_dir, org, name)

    repo = _get_or_create_repo(repo_path)
    if repo is None:
        return {"success": False, "error": f"Repository '{org}/{name}' not found"}

    try:
        # gitpython provides remotes and fetch/pull natively
        origin = repo.remotes.origin
        fetch_info = origin.fetch()
        # Try to find the remote head from fetch info
        remote_head = None
        for info in fetch_info:
            ref = info.remote_ref_path
            if ref == branch or ref == f"refs/heads/{branch}":
                remote_head = info.commit.hexsha
                break
        # Fallback: access the remote tracking ref after fetch
        if remote_head is None:
            try:
                remote_head = repo.refs[f"origin/{branch}"].commit.hexsha
            except Exception:
                pass
        # Pull with rebase or merge based on config
        origin.pull(branch)
        logger.info(f"Updated repository: {org}/{name} on branch {branch}")
        return {
            "success": True,
            "org": org,
            "name": name,
            "path": str(repo_path),
            "branch": branch,
            "remote_head": remote_head or "",
        }
    except Exception as e:
        logger.error(f"Failed to update {org}/{name}: {e}")
        return {"success": False, "error": str(e)}


def _remove_readonly(
    func: Any,
    path: str,
    excinfo: Any,
) -> None:
    """Clear read-only bit and retry deletion (Windows fix for .git/pack files)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def delete_repo(org: str, name: str, repos_dir: Path | None = None) -> dict[str, Any]:
    """Delete a local repository."""
    repos_dir = Path(repos_dir or get_config().paths_github_dir).resolve()
    repo_path = _get_repo_path(repos_dir, org, name)
    repo_key = _get_repo_key(org, name)

    if not repo_path.exists():
        return {"success": False, "error": f"Repository '{org}/{name}' not found"}

    try:
        shutil.rmtree(repo_path, onerror=_remove_readonly)
        logger.info(f"Deleted repository: {org}/{name}")

        org_path = repos_dir / org
        if org_path.exists() and not any(org_path.iterdir()):
            org_path.rmdir()

        db = DatabaseService.get_instance()
        if db:
            db.delete_git_metadata(repo_key)
            db.delete_selected_dirs(repo_key)

        return {"success": True}
    except ValueError as e:
        logger.error(f"Delete failed: {e}")
        return {"success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Failed to delete {org}/{name}: {e}")
        return {"success": False, "error": str(e)}


def list_repos(
    repos_dir: Path | None = None,
    org_filter: str | None = None,
    *,
    fetch_remote: bool = False,
) -> list[dict[str, Any]]:
    """List all cloned repositories with their metadata.

    Args:
        repos_dir: Base directory for cloned repos.
        org_filter: If provided, only list repos under this org directory.
        fetch_remote: If True, perform a network fetch per repo to get
            the latest remote HEAD. Default False (local refs only).
    """
    repos_dir = Path(repos_dir or get_config().paths_github_dir).resolve()
    repos_dir.mkdir(parents=True, exist_ok=True)

    repos = []
    for org_dir in repos_dir.iterdir():
        if not org_dir.is_dir():
            continue
        if org_filter and org_dir.name != org_filter:
            continue
        for repo_dir in org_dir.iterdir():
            repo = _get_or_create_repo(repo_dir)
            if repo is not None:
                info: dict[str, Any] = {
                    "org": org_dir.name,
                    "name": repo_dir.name,
                    "path": str(repo_dir),
                }
                info["branch"] = repo.active_branch.name if repo.active_branch else None
                try:
                    info["remote_url"] = repo.remote().url
                except Exception:
                    info["remote_url"] = None
                info["remote_head"] = ""
                if fetch_remote:
                    try:
                        origin = repo.remotes.origin
                        origin.fetch()
                        remote_ref = f"origin/{info.get('branch')}"
                        info["remote_head"] = (
                            repo.refs[remote_ref].commit.hexsha if remote_ref in repo.refs else ""
                        )
                    except Exception as e:
                        logger.warning("Fetch failed for %s/%s: %s", org_dir.name, repo_dir.name, e)
                repos.append(info)

    return sorted(repos, key=lambda r: (r["org"], r["name"]))


def save_metadata(
    org: str, name: str, metadata: dict[str, Any], repos_dir: Path | None = None
) -> None:
    """Save metadata for a repository to DuckDB."""
    db = DatabaseService.get_instance()
    repo_key = f"{org}/{name}"
    db.set_git_metadata(repo_key, metadata)


def get_metadata(org: str, name: str, repos_dir: Path | None = None) -> dict[str, Any] | None:
    """Get metadata for a repository from DuckDB."""
    db = DatabaseService.get_instance()
    repo_key = f"{org}/{name}"
    return db.get_git_metadata(repo_key)


def _get_repo_key(org: str, name: str) -> str:
    return f"{org}/{name}"


def list_directory_tree(
    org: str, name: str, repos_dir: Path | None = None, max_depth: int = 5
) -> list[dict[str, Any]]:
    """List directory tree for a repository.

    Args:
        org: Organization/user
        name: Repository name
        repos_dir: Base directory for cloned repos
        max_depth: Maximum depth to traverse

    Returns:
        List of folder nodes with 'path', 'name', and optional 'children'
    """
    repos_dir = Path(repos_dir or get_config().paths_github_dir).resolve()
    repo_path = _get_repo_path(repos_dir, org, name)

    if not repo_path.exists() or not _is_valid_repo(repo_path):
        return []

    def _walk_dir(path: Path, current_depth: int) -> list[dict[str, Any]]:
        if current_depth > max_depth:
            return []
        results = []
        try:
            for entry in sorted(path.iterdir()):
                if entry.is_dir():
                    if entry.name.startswith(".") or entry.name.startswith("__"):
                        continue
                    children = _walk_dir(entry, current_depth + 1)
                    node: dict[str, Any] = {
                        "name": entry.name,
                        "path": entry.relative_to(repo_path).as_posix(),
                    }
                    if children:
                        node["children"] = children
                    results.append(node)
        except PermissionError:
            pass
        return results

    return _walk_dir(repo_path, 1)


def save_selected_dirs(
    org: str,
    name: str,
    selected: list[str],
    repos_dir: Path | None = None,
    source_type: str = "",
) -> dict[str, Any]:
    """Save selected directories for a repository to DuckDB.

    Args:
        org: Organization/user
        name: Repository name
        selected: List of relative folder paths
        repos_dir: Base directory for cloned repos
        source_type: Source type identifier (e.g. 'github', 'spec')

    Returns:
        Result dict with success status
    """
    db = DatabaseService.get_instance()
    repo_key = _get_repo_key(org, name)
    try:
        db.set_selected_dirs(repo_key, selected, source_type=source_type)
        db.persist()
        logger.info(f"Saved selection for {org}/{name}: {selected}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to save selection for {org}/{name}: {e}")
        return {"success": False, "error": str(e)}


def get_selected_dirs(org: str, name: str, repos_dir: Path | None = None) -> list[str]:
    """Get selected directories for a repository from DuckDB.

    Args:
        org: Organization/user
        name: Repository name
        repos_dir: Base directory for cloned repos

    Returns:
        List of selected folder paths, empty if not found
    """
    db = DatabaseService.get_instance()
    repo_key = _get_repo_key(org, name)
    try:
        return db.get_selected_dirs(repo_key)
    except Exception as e:
        logger.error(f"Failed to read selection for {org}/{name}: {e}")
        return []


def get_last_commit_date(org: str, name: str, repos_dir: Path | None = None) -> str | None:
    """Get the date of the last commit in the repository."""
    repos_dir = Path(repos_dir or get_config().paths_github_dir).resolve()
    repo_path = _get_repo_path(repos_dir, org, name)

    repo = _get_or_create_repo(repo_path)
    if repo is None:
        return None

    try:
        commit = repo.head.commit
        return str(commit.committed_datetime.isoformat())
    except Exception:
        return None


def is_repo_outdated(org: str, name: str, repos_dir: Path | None = None) -> bool:
    """Check if local repo is behind remote by comparing against the tracking ref."""
    repos_dir = Path(repos_dir or get_config().paths_github_dir).resolve()
    repo_path = _get_repo_path(repos_dir, org, name)

    repo = _get_or_create_repo(repo_path)
    if repo is None:
        return False

    try:
        local_commit = repo.head.commit.hexsha
        branch = repo.active_branch.name
        # Compare against origin tracking ref (always up-to-date after fetch in list_repos)
        remote_ref = f"origin/{branch}"
        if remote_ref not in repo.refs:
            return True
        remote_commit = repo.refs[remote_ref].commit.hexsha
        return local_commit != remote_commit
    except Exception:
        return False
