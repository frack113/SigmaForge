"""Context manager to temporarily force HuggingFace Hub online mode."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


@contextmanager
def force_hf_online() -> Iterator[None]:
    """Temporarily disable HF_HUB_OFFLINE for network operations.

    Usage:
        with force_hf_online():
            snapshot_download(...)
    """
    import huggingface_hub.constants as hc

    was_offline = hc.HF_HUB_OFFLINE
    hc.HF_HUB_OFFLINE = False
    try:
        yield
    finally:
        hc.HF_HUB_OFFLINE = was_offline
