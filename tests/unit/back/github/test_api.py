"""Tests for GitHub API interactions."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.infrastructure.github.api import (
    download_release_file,
    info_release,
    list_release_files,
    list_releases,
)

MOCK_TARGET = "src.infrastructure.github.api.get_async_pooled_client"


def _mock_response(data):
    """Create a mock httpx response with synchronous json() method."""
    resp = MagicMock()
    resp.json.return_value = data
    resp.raise_for_status.return_value = None
    return resp


@pytest.fixture
def mock_client():
    client = AsyncMock()
    return client


@pytest.fixture
def sample_releases() -> list[dict]:
    return [
        {
            "tag_name": "v1.0.0",
            "name": "Release 1.0.0",
            "published_at": "2024-01-01T00:00:00Z",
            "prerelease": False,
            "draft": False,
            "assets": [
                {"name": "asset1.zip", "size": 100, "browser_download_url": "https://example.com/1"}
            ],
        }
    ]


@pytest.fixture
def sample_release_detail() -> dict:
    return {
        "tag_name": "v1.0.0",
        "name": "Release 1.0.0",
        "assets": [
            {
                "name": "asset1.zip",
                "size": 100,
                "browser_download_url": "https://example.com/1",
                "content_type": "application/zip",
            }
        ],
    }


class TestListReleases:
    @pytest.mark.asyncio
    async def test_success(self, mock_client: AsyncMock, sample_releases: list[dict]) -> None:
        mock_client.get.return_value = _mock_response(sample_releases)
        with patch(MOCK_TARGET, return_value=mock_client):
            result = await list_releases("owner", "repo")
            assert len(result) == 1
            assert result[0]["tag_name"] == "v1.0.0"
            assert result[0]["assets_count"] == 1

    @pytest.mark.asyncio
    async def test_with_token(self, mock_client: AsyncMock, sample_releases: list[dict]) -> None:
        mock_client.get.return_value = _mock_response(sample_releases)
        with patch(MOCK_TARGET, return_value=mock_client):
            await list_releases("owner", "repo", github_token="token-123")
            call_kwargs = mock_client.get.call_args
            headers = call_kwargs[1]["headers"]
            assert "Authorization" in headers
            assert headers["Authorization"] == "Bearer token-123"

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self, mock_client: AsyncMock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )
        mock_client.get.return_value = mock_response
        with patch(MOCK_TARGET, return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                await list_releases("owner", "repo")


class TestInfoRelease:
    @pytest.mark.asyncio
    async def test_success(self, mock_client: AsyncMock, sample_release_detail: dict) -> None:
        mock_client.get.return_value = _mock_response(sample_release_detail)
        with patch(MOCK_TARGET, return_value=mock_client):
            result = await info_release("owner", "repo", "v1.0.0")
            assert result["tag_name"] == "v1.0.0"

    @pytest.mark.asyncio
    async def test_uses_tag_as_is_without_v_prefix(
        self, mock_client: AsyncMock, sample_release_detail: dict
    ) -> None:
        mock_client.get.return_value = _mock_response(sample_release_detail)
        with patch(MOCK_TARGET, return_value=mock_client):
            await info_release("owner", "repo", "1.0.0")
            call_url = mock_client.get.call_args[0][0]
            assert "1.0.0" in call_url
            assert "v1.0.0" not in call_url

    @pytest.mark.asyncio
    async def test_handles_b_prefix_tag(
        self, mock_client: AsyncMock, sample_release_detail: dict
    ) -> None:
        mock_client.get.return_value = _mock_response(sample_release_detail)
        with patch(MOCK_TARGET, return_value=mock_client):
            await info_release("owner", "repo", "b9601")
            call_url = mock_client.get.call_args[0][0]
            assert "b9601" in call_url

    @pytest.mark.asyncio
    async def test_preserves_v_prefix(
        self, mock_client: AsyncMock, sample_release_detail: dict
    ) -> None:
        mock_client.get.return_value = _mock_response(sample_release_detail)
        with patch(MOCK_TARGET, return_value=mock_client):
            await info_release("owner", "repo", "v2.0.0")
            call_url = mock_client.get.call_args[0][0]
            assert "v2.0.0" in call_url

    @pytest.mark.asyncio
    async def test_with_token(self, mock_client: AsyncMock, sample_release_detail: dict) -> None:
        mock_client.get.return_value = _mock_response(sample_release_detail)
        with patch(MOCK_TARGET, return_value=mock_client):
            await info_release("owner", "repo", "v1.0.0", github_token="token-789")
            call_kwargs = mock_client.get.call_args
            headers = call_kwargs[1]["headers"]
            assert headers["Authorization"] == "Bearer token-789"


class TestListReleaseFiles:
    @pytest.mark.asyncio
    async def test_success(self, mock_client: AsyncMock) -> None:
        data = {
            "tag_name": "v1.0.0",
            "assets": [
                {
                    "name": "file1.zip",
                    "size": 100,
                    "browser_download_url": "https://example.com/1",
                    "content_type": "application/zip",
                },
                {
                    "name": "file2.tar.gz",
                    "size": 200,
                    "browser_download_url": "https://example.com/2",
                    "content_type": "application/gzip",
                },
            ],
        }
        mock_client.get.return_value = _mock_response(data)
        with patch(MOCK_TARGET, return_value=mock_client):
            result = await list_release_files("owner", "repo", "v1.0.0")
            assert len(result) == 2
            assert result[0]["name"] == "file1.zip"
            assert result[1]["name"] == "file2.tar.gz"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_assets(self, mock_client: AsyncMock) -> None:
        data = {"tag_name": "v1.0.0", "assets": []}
        mock_client.get.return_value = _mock_response(data)
        with patch(MOCK_TARGET, return_value=mock_client):
            result = await list_release_files("owner", "repo", "v1.0.0")
            assert result == []

    @pytest.mark.asyncio
    async def test_with_token(self, mock_client: AsyncMock) -> None:
        data = {"tag_name": "v1.0.0", "assets": []}
        mock_client.get.return_value = _mock_response(data)
        with patch(MOCK_TARGET, return_value=mock_client):
            await list_release_files("owner", "repo", "v1.0.0", github_token="token-abc")
            call_kwargs = mock_client.get.call_args
            headers = call_kwargs[1]["headers"]
            assert headers["Authorization"] == "Bearer token-abc"


class TestDownloadReleaseFile:
    @pytest.mark.asyncio
    async def test_finds_matching_asset(self, mock_client: AsyncMock) -> None:
        data = {
            "tag_name": "v1.0.0",
            "assets": [
                {
                    "name": "target.zip",
                    "size": 500,
                    "browser_download_url": "https://example.com/target",
                    "content_type": "application/zip",
                }
            ],
        }
        mock_client.get.return_value = _mock_response(data)
        with patch(MOCK_TARGET, return_value=mock_client):
            result = await download_release_file("owner", "repo", "target.zip", "v1.0.0")
            assert result["name"] == "target.zip"
            assert result["size"] == 500

    @pytest.mark.asyncio
    async def test_raises_when_not_found(self, mock_client: AsyncMock) -> None:
        data = {"tag_name": "v1.0.0", "assets": []}
        mock_client.get.return_value = _mock_response(data)
        with patch(MOCK_TARGET, return_value=mock_client):
            with pytest.raises(ValueError, match="not found"):
                await download_release_file("owner", "repo", "missing.zip", "v1.0.0")

    @pytest.mark.asyncio
    async def test_with_token(self, mock_client: AsyncMock, sample_release_detail: dict) -> None:
        mock_client.get.return_value = _mock_response(sample_release_detail)
        with patch(MOCK_TARGET, return_value=mock_client):
            result = await download_release_file(
                "owner", "repo", "asset1.zip", "v1.0.0", github_token="token-456"
            )
            assert result["name"] == "asset1.zip"
            call_kwargs = mock_client.get.call_args
            headers = call_kwargs[1]["headers"]
            assert headers["Authorization"] == "Bearer token-456"
