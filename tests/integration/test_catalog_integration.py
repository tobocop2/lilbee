"""Integration tests for catalog — verifies featured models against real HF API.

Run with:
    uv run pytest tests/integration/test_catalog_integration.py -v -m slow
"""

from __future__ import annotations

import httpx
import pytest

from lilbee.catalog import FEATURED_ALL

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    "entry",
    FEATURED_ALL,
    ids=[e.name for e in FEATURED_ALL],
)
def test_featured_models_all_have_gguf(entry) -> None:
    """Each featured model's HF repo must contain at least one .gguf file."""
    resp = httpx.get(
        f"https://huggingface.co/api/models/{entry.hf_repo}",
        timeout=30.0,
    )
    resp.raise_for_status()
    siblings = resp.json().get("siblings", [])
    gguf_files = [s["rfilename"] for s in siblings if s.get("rfilename", "").endswith(".gguf")]
    assert gguf_files, f"{entry.name} ({entry.hf_repo}) has no .gguf files in siblings"
