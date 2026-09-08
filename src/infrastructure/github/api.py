"""GitHub API interactions."""

from __future__ import annotations

from src.shared.http import get_async_pooled_client

GITHUB_API_URL = "https://api.github.com/repos"


def _github_headers(github_token: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/vnd.github+json"}
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"
    return headers


async def list_releases(owner: str, repo: str, github_token: str | None = None) -> list[dict]:
    """List all releases for a repository."""
    url = f"{GITHUB_API_URL}/{owner}/{repo}/releases"
    client = get_async_pooled_client(timeout=30.0)
    response = await client.get(url, headers=_github_headers(github_token))
    response.raise_for_status()
    data = response.json()
    return [
        {
            "tag_name": r.get("tag_name"),
            "name": r.get("name"),
            "published_at": r.get("published_at"),
            "prerelease": r.get("prerelease"),
            "draft": r.get("draft"),
            "assets_count": len(r.get("assets", [])),
        }
        for r in data
    ]


async def info_release(owner: str, repo: str, tag: str, github_token: str | None = None) -> dict:
    """Get release info by tag."""
    url = f"{GITHUB_API_URL}/{owner}/{repo}/releases/tags/{tag}"
    client = get_async_pooled_client(timeout=30.0)
    response = await client.get(url, headers=_github_headers(github_token))
    response.raise_for_status()
    return response.json()  # type: ignore[no-any-return]


async def list_release_files(
    owner: str, repo: str, tag: str, github_token: str | None = None
) -> list[dict]:
    """List all files (assets) of a release."""
    url = f"{GITHUB_API_URL}/{owner}/{repo}/releases/tags/{tag}"
    client = get_async_pooled_client(timeout=30.0)
    response = await client.get(url, headers=_github_headers(github_token))
    response.raise_for_status()
    data = response.json()
    return [
        {
            "name": a["name"],
            "size": a["size"],
            "download_url": a["browser_download_url"],
            "content_type": a["content_type"],
        }
        for a in data.get("assets", [])
    ]


async def download_release_file(
    owner: str,
    repo: str,
    filename: str,
    tag: str,
    github_token: str | None = None,
) -> dict:
    """Download a specific file from a release."""
    url = f"{GITHUB_API_URL}/{owner}/{repo}/releases/tags/{tag}"
    client = get_async_pooled_client(timeout=30.0)
    response = await client.get(url, headers=_github_headers(github_token))
    response.raise_for_status()
    data = response.json()

    for asset in data.get("assets", []):
        if asset["name"] == filename:
            return {
                "name": asset["name"],
                "size": asset["size"],
                "download_url": asset["browser_download_url"],
                "content_type": asset["content_type"],
            }

    raise ValueError(f"File '{filename}' not found in release '{tag}'")
