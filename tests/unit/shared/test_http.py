"""Tests for shared HTTP utilities."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from src.shared.http import (
    close_all_pooled_clients,
    create_client,
    download_file,
    get_pooled_client,
    head_url,
    _backoff_delay,
    _get_retry_after,
)


def _mock_stream_client(content: bytes, side_effect_error: Exception | None = None):
    """Create a mock client whose .stream() returns a context manager yielding a response."""
    mock_resp = MagicMock(spec=httpx.Response)
    if side_effect_error:
        mock_resp.raise_for_status.side_effect = side_effect_error
    else:
        mock_resp.raise_for_status.return_value = None

    def iter_bytes(chunk_size: int = 65536):
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]

    mock_resp.iter_bytes = iter_bytes

    @contextmanager
    def stream(method: str, url: str):
        if side_effect_error is None:
            yield mock_resp
        else:
            raise side_effect_error

    mock_client = MagicMock(spec=httpx.Client)
    mock_client.stream.side_effect = stream
    return mock_client


class TestCreateClient:
    def test_returns_httpx_client(self) -> None:
        client = create_client()
        assert isinstance(client, httpx.Client)
        client.close()

    def test_sets_user_agent(self) -> None:
        client = create_client()
        assert client.headers.get("User-Agent") == "SigmaRAG/1.0"
        client.close()

    def test_merges_custom_headers(self) -> None:
        client = create_client(headers={"X-Custom": "value"})
        assert client.headers.get("User-Agent") == "SigmaRAG/1.0"
        assert client.headers.get("X-Custom") == "value"
        client.close()

    def test_follow_redirects_default_true(self) -> None:
        client = create_client()
        assert client.follow_redirects is True
        client.close()

    def test_follow_redirects_false(self) -> None:
        client = create_client(follow_redirects=False)
        assert client.follow_redirects is False
        client.close()

    def test_timeout_float(self) -> None:
        client = create_client(timeout=15.0)
        assert client.timeout == httpx.Timeout(15.0)
        client.close()


class TestHeadUrl:
    def test_returns_content_type_size_url(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.headers = {
            "content-type": "text/html; charset=utf-8",
            "content-length": "1234",
        }
        mock_resp.url = httpx.URL("https://example.com/doc")
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.head.return_value = mock_resp

        with patch("src.shared.http.get_pooled_client", return_value=mock_client):
            ctype, size, final_url = head_url("https://example.com/doc")

        assert ctype == "text/html"
        assert size == 1234
        assert final_url == "https://example.com/doc"

    def test_returns_none_on_http_error(self) -> None:
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.head.side_effect = httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock()
        )

        with patch("src.shared.http.get_pooled_client", return_value=mock_client):
            result = head_url("https://example.com/404")

        assert result == (None, None, None)

    def test_returns_none_on_connection_error(self) -> None:
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.head.side_effect = httpx.ConnectError("connection refused")

        with patch("src.shared.http.get_pooled_client", return_value=mock_client):
            result = head_url("https://example.com/down")

        assert result == (None, None, None)

    def test_returns_none_for_private_url(self) -> None:
        with patch("src.shared.http.is_private_url", return_value=True):
            result = head_url("http://localhost:8080/secret")

        assert result == (None, None, None)

    def test_returns_none_for_empty_content_type(self) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.headers = {}
        mock_resp.url = httpx.URL("https://example.com/doc")
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.head.return_value = mock_resp

        with patch("src.shared.http.get_pooled_client", return_value=mock_client):
            ctype, size, final_url = head_url("https://example.com/doc")

        assert ctype is None
        assert size == 0

    def test_private_url_skip_by_default(self) -> None:
        with patch("src.shared.http.is_private_url", return_value=True):
            result = head_url("http://localhost:8080")

        assert result == (None, None, None)

    def test_private_url_not_skipped_when_check_ssrf_false(self) -> None:
        mock_client = MagicMock(spec=httpx.Client)
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.headers = {"content-type": "text/plain"}
        mock_resp.url = httpx.URL("http://localhost:8080/doc")
        mock_client.head.return_value = mock_resp

        with (
            patch("src.shared.http.get_pooled_client", return_value=mock_client),
            patch("src.shared.http.is_private_url", return_value=True),
        ):
            ctype, size, url = head_url("http://localhost:8080/doc", check_ssrf=False)

        assert ctype == "text/plain"


class TestDownloadFile:
    def test_successful_download(self, tmp_path: Path) -> None:
        output = tmp_path / "doc.md"
        mock_client = _mock_stream_client(b"hello world")

        with patch("src.shared.http.get_pooled_client", return_value=mock_client):
            ok, status = download_file("https://example.com/doc", output)

        assert ok is True
        assert status is None
        assert output.read_bytes() == b"hello world"

    def test_skips_private_url(self, tmp_path: Path) -> None:
        output = tmp_path / "secret.md"
        with patch("src.shared.http.is_private_url", return_value=True):
            ok, status = download_file("http://localhost:8080/secret", output)

        assert ok is False
        assert status is None
        assert not output.exists()

    def test_retries_on_http_429(self, tmp_path: Path) -> None:
        output = tmp_path / "retry.md"

        mock_resp_429 = MagicMock(spec=httpx.Response)
        mock_resp_429.status_code = 429
        mock_resp_429.headers = {}

        @contextmanager
        def _cm_429(method: str, url: str):
            yield mock_resp_429

        @contextmanager
        def _cm_ok(method: str, url: str):
            mock_resp_ok = MagicMock(spec=httpx.Response)
            mock_resp_ok.raise_for_status.return_value = None
            mock_resp_ok.iter_bytes = lambda chunk_size=65536: iter([b"content"])
            yield mock_resp_ok

        cm_sequence = iter([_cm_429, _cm_ok])
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.stream.side_effect = lambda m, u: next(cm_sequence)(m, u)

        with (
            patch("src.shared.http.get_pooled_client", return_value=mock_client),
            patch("time.sleep"),
        ):
            ok, status = download_file("https://example.com/doc", output)

        assert ok is True

    def test_retries_on_network_error(self, tmp_path: Path) -> None:
        output = tmp_path / "retry_net.md"

        @contextmanager
        def _cm_fail(method: str, url: str):
            raise httpx.ConnectError("timeout")
            yield

        @contextmanager
        def _cm_ok(method: str, url: str):
            mock_resp = MagicMock(spec=httpx.Response)
            mock_resp.raise_for_status.return_value = None
            mock_resp.iter_bytes = lambda chunk_size=65536: iter([b"ok"])
            yield mock_resp

        cm_sequence = iter([_cm_fail, _cm_ok])
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.stream.side_effect = lambda m, u: next(cm_sequence)(m, u)

        with (
            patch("src.shared.http.get_pooled_client", return_value=mock_client),
            patch("time.sleep"),
        ):
            ok, status = download_file("https://example.com/doc", output)

        assert ok is True

    def test_gives_up_after_max_retries(self, tmp_path: Path) -> None:
        output = tmp_path / "fail.md"

        @contextmanager
        def _cm_fail(method: str, url: str):
            raise httpx.ConnectError("always fails")
            yield

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.stream.side_effect = _cm_fail

        with (
            patch("src.shared.http.get_pooled_client", return_value=mock_client),
            patch("time.sleep"),
        ):
            ok, status = download_file("https://example.com/doc", output, max_retries=2)

        assert ok is False
        assert status is None

    def test_non_retryable_http_status(self, tmp_path: Path) -> None:
        output = tmp_path / "forbidden.md"
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 403
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "403", request=MagicMock(), response=mock_resp
        )

        @contextmanager
        def _cm_forbidden(method: str, url: str):
            yield mock_resp

        mock_client = MagicMock(spec=httpx.Client)
        mock_client.stream.side_effect = _cm_forbidden

        with patch("src.shared.http.get_pooled_client", return_value=mock_client):
            ok, status = download_file("https://example.com/forbidden", output)

        assert ok is False
        assert status == 403

    def test_filesystem_error(self, tmp_path: Path) -> None:
        output = Path("/nonexistent/path/doc.md")
        mock_client = _mock_stream_client(b"data")

        with patch("src.shared.http.get_pooled_client", return_value=mock_client):
            ok, status = download_file("https://example.com/doc", output)

        assert ok is False

    def test_respects_retry_after_header(self, tmp_path: Path) -> None:
        output = tmp_path / "retry_after.md"
        mock_resp_429 = MagicMock(spec=httpx.Response)
        mock_resp_429.status_code = 429
        mock_resp_429.headers = {"Retry-After": "5"}
        mock_resp_429.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429", request=MagicMock(), response=mock_resp_429
        )

        @contextmanager
        def _cm_429(method: str, url: str):
            yield mock_resp_429

        @contextmanager
        def _cm_ok(method: str, url: str):
            mock_resp_ok = MagicMock(spec=httpx.Response)
            mock_resp_ok.raise_for_status.return_value = None
            mock_resp_ok.iter_bytes = lambda chunk_size=65536: iter([b"content"])
            yield mock_resp_ok

        cm_sequence = iter([_cm_429, _cm_ok])
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.stream.side_effect = lambda m, u: next(cm_sequence)(m, u)

        with (
            patch("src.shared.http.get_pooled_client", return_value=mock_client),
            patch("time.sleep") as mock_sleep,
        ):
            ok, status = download_file("https://example.com/doc", output)

        assert ok is True
        mock_sleep.assert_called_once_with(5.0)


class TestBackoffDelay:
    def test_first_attempt(self) -> None:
        assert _backoff_delay(1) == 1.0

    def test_second_attempt(self) -> None:
        assert _backoff_delay(2) == 4.0

    def test_third_attempt(self) -> None:
        assert _backoff_delay(3) == 9.0

    def test_beyond_list_falls_back(self) -> None:
        assert _backoff_delay(10) == 9.0


class TestGetRetryAfter:
    def test_valid_header(self) -> None:
        resp = MagicMock(spec=httpx.Response)
        resp.headers = {"Retry-After": "30"}
        assert _get_retry_after(resp) == 30

    def test_missing_header(self) -> None:
        resp = MagicMock(spec=httpx.Response)
        resp.headers = {}
        assert _get_retry_after(resp) is None

    def test_invalid_value(self) -> None:
        resp = MagicMock(spec=httpx.Response)
        resp.headers = {"Retry-After": "invalid"}
        assert _get_retry_after(resp) is None


class TestPooledClient:
    def test_returns_same_client_for_same_params(self) -> None:
        c1 = get_pooled_client(timeout=30.0, follow_redirects=True)
        c2 = get_pooled_client(timeout=30.0, follow_redirects=True)
        assert c1 is c2

    def test_returns_different_client_for_different_timeout(self) -> None:
        c1 = get_pooled_client(timeout=30.0)
        c2 = get_pooled_client(timeout=45.0)
        assert c1 is not c2

    def test_returns_different_client_for_different_redirects(self) -> None:
        c1 = get_pooled_client(follow_redirects=True)
        c2 = get_pooled_client(follow_redirects=False)
        assert c1 is not c2

    def test_close_all_clears_pool(self) -> None:
        c1 = get_pooled_client(timeout=30.0)
        close_all_pooled_clients()
        c2 = get_pooled_client(timeout=30.0)
        assert c1 is not c2
