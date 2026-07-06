"""Integration tests for catalog: verifies featured models against real HF API.

Run with:
    uv run pytest tests/integration/test_catalog_integration.py -v -m slow
"""

from __future__ import annotations

import httpx
import pytest

from lilbee.catalog import FEATURED_ALL
from lilbee.catalog.hf_client import DEFAULT_TIMEOUT, HF_API_URL, hf_headers

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    "entry",
    FEATURED_ALL,
    ids=[e.hf_repo for e in FEATURED_ALL],
)
def test_featured_models_all_have_gguf(entry) -> None:
    """Each featured model's HF repo must contain at least one .gguf file."""
    # Authenticated request (HF_TOKEN) lifts the unauthenticated rate limit that
    # otherwise 429s the featured-model sweep from a shared CI IP.
    resp = httpx.get(
        f"{HF_API_URL}/{entry.hf_repo}",
        timeout=DEFAULT_TIMEOUT,
        headers=hf_headers(),
    )
    resp.raise_for_status()
    siblings = resp.json().get("siblings", [])
    gguf_files = [s["rfilename"] for s in siblings if s.get("rfilename", "").endswith(".gguf")]
    assert gguf_files, f"{entry.hf_repo} has no .gguf files in siblings"
