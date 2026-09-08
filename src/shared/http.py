"""Shared HTTP utilities — HEAD, download with retry/backoff, and client factory.

Consolidates the HTTP logic duplicated across:
- sigma_ref_downloader.py (``_head_content_type``, ``_download_file``)
- sigma_ref_processor.py (``_head_request``, ``_download_one``)
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from src.shared.utils.url_utils import is_private_url

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
DEFAULT_USER_AGENT = "SigmaRAG/1.0"
MAX_RETRIES = 3
BACKOFF_DELAYS = [1.0, 4.0, 9.0]
RETRY_STATUSES = {429, 500, 502, 503, 504}


# ------------------------------------------------------------------
# Connection pool
# ------------------------------------------------------------------

_pool_lock = threading.Lock()
_pool: dict[str, httpx.Client] = {}


def _pool_key(timeout: float, follow_redirects: bool) -> str:
    return f"{timeout:.1f}:{follow_redirects}"


def get_pooled_client(
    timeout: float = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
    follow_redirects: bool = True,
) -> httpx.Client:
    """Return a pooled ``httpx.Client`` keyed by (timeout, follow_redirects).

    Connections are reused across requests to the same host, reducing TCP
    handshake overhead for repeated downloads (e.g. reference documents).
    """
    key = _pool_key(timeout, follow_redirects)
    with _pool_lock:
        client = _pool.get(key)
        if client is not None:
            return client

    merged: dict[str, str] = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        merged.update(headers)

    transport = httpx.HTTPTransport(
        retries=2,
        limits=httpx.Limits(
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        ),
    )

    new_client = httpx.Client(
        timeout=httpx.Timeout(timeout),
        headers=merged,
        follow_redirects=follow_redirects,
        transport=transport,
    )

    with _pool_lock:
        _pool[key] = new_client
    return new_client


def close_all_pooled_clients() -> None:
    """Close all pooled HTTP clients. Call at shutdown."""
    with _pool_lock:
        for client in _pool.values():
            try:
                client.close()
            except Exception:
                pass
        _pool.clear()


def create_client(
    timeout: float = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
    follow_redirects: bool = True,
) -> httpx.Client:
    """Create a properly configured ``httpx.Client``.

    Parameters
    ----------
    timeout :
        Request timeout in seconds.
    headers :
        Extra HTTP headers.  ``User-Agent`` is set automatically.
    follow_redirects :
        Whether to follow redirects automatically.

    Returns
    -------
    httpx.Client
    """
    merged: dict[str, Any] = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        merged.update(headers)
    return httpx.Client(
        timeout=httpx.Timeout(timeout),
        headers=merged,
        follow_redirects=follow_redirects,
    )


def head_url(
    url: str,
    timeout: float = 10.0,
    *,
    check_ssrf: bool = True,
) -> tuple[str | None, int | None, str | None]:
    """HEAD request to discover content type, size, and final URL.

    Parameters
    ----------
    url :
        The URL to check.
    timeout :
        Request timeout in seconds.
    check_ssrf :
        If ``True`` (default), reject private/reserved IPs before connecting.

    Returns
    -------
    tuple of (content_type, content_length, final_url)
    ``(None, None, None)`` on failure or SSRF rejection.
    """
    if check_ssrf and is_private_url(url):
        logger.warning("Skipping private URL: %s", url)
        return None, None, None

    try:
        client = get_pooled_client(timeout=timeout)
        resp = client.head(url)
        resp.raise_for_status()
        ctype = resp.headers.get("content-type")
        if ctype:
            ctype = ctype.split(";")[0].strip()
        size_str = resp.headers.get("content-length", "0")
        size = int(size_str) if size_str else 0
        return ctype, size, str(resp.url)
    except Exception:
        return None, None, None


def download_file(
    url: str,
    output_path: str | Path,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    *,
    check_ssrf: bool = True,
) -> tuple[bool, int | None]:
    """Download a file with retry and backoff.

    Parameters
    ----------
    url :
        The URL to download.
    output_path :
        Local filesystem path to save the file.
    timeout :
        HTTP request timeout in seconds.
    max_retries :
        Maximum number of retry attempts.  0 means no retries.
    check_ssrf :
        If ``True`` (default), reject private/reserved IPs before connecting.

    Returns
    -------
    tuple of (success, http_status_or_None)
    ``http_status`` is ``None`` for non-HTTP errors.
    """
    if check_ssrf and is_private_url(url):
        logger.warning("Skipping private URL (SSRF): %s", url)
        return False, None

    path = Path(output_path)
    client = get_pooled_client(timeout=timeout)

    for attempt in range(1, max_retries + 1):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
            return True, None

        except OSError as exc:
            logger.warning("Filesystem error for %s: %s — skipping", url, exc)
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass
            return False, None

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in RETRY_STATUSES and attempt < max_retries:
                retry_after = _get_retry_after(exc.response)
                if retry_after is not None:
                    wait = min(float(retry_after), 120.0)
                else:
                    wait = _backoff_delay(attempt)
                logger.warning(
                    "HTTP %d for %s — retrying in %.1fs (attempt %d/%d)",
                    status,
                    url,
                    wait,
                    attempt,
                    max_retries,
                )
                time.sleep(wait)
                continue
            logger.warning("HTTP %d for %s — giving up", status, url)
            return False, status

        except (
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
        ) as exc:
            if attempt < max_retries:
                wait = _backoff_delay(attempt)
                logger.warning(
                    "Network error for %s — retrying in %.1fs (attempt %d/%d): %s",
                    url,
                    wait,
                    attempt,
                    max_retries,
                    exc,
                )
                time.sleep(wait)
                continue
            logger.warning(
                "Network error for %s after %d attempts: %s",
                url,
                max_retries,
                exc,
            )
            return False, None

    return False, None


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _backoff_delay(attempt: int) -> float:
    """Return the backoff delay for the given attempt number (1-indexed)."""
    idx = attempt - 1
    if idx < len(BACKOFF_DELAYS):
        return BACKOFF_DELAYS[idx]
    return BACKOFF_DELAYS[-1]


def _get_retry_after(response: httpx.Response) -> int | None:
    """Extract Retry-After header value as seconds."""
    raw = response.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None
