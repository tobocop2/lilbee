"""Integration tests for catalog: verifies featured models against real HF API.

Run with:
    uv run pytest tests/integration/test_catalog_integration.py -v -m slow
"""

from __future__ import annotations

import fnmatch

import httpx
import pytest

from lilbee.catalog import FEATURED_ALL
from lilbee.catalog.hf_client import DEFAULT_TIMEOUT, HF_API_URL, hf_headers

pytestmark = pytest.mark.slow


def _repo_gguf_files(hf_repo: str) -> list[str]:
    # Authenticated request (HF_TOKEN) lifts the unauthenticated rate limit that
    # otherwise 429s the featured-model sweep from a shared CI IP.
    resp = httpx.get(
        f"{HF_API_URL}/{hf_repo}",
        timeout=DEFAULT_TIMEOUT,
        headers=hf_headers(),
    )
    resp.raise_for_status()
    siblings = resp.json().get("siblings", [])
    return [s["rfilename"] for s in siblings if s.get("rfilename", "").endswith(".gguf")]


@pytest.mark.parametrize(
    "entry",
    FEATURED_ALL,
    ids=[e.hf_repo for e in FEATURED_ALL],
)
def test_featured_models_all_have_gguf(entry) -> None:
    """Each featured model's HF repo must contain at least one .gguf file."""
    assert _repo_gguf_files(entry.hf_repo), f"{entry.hf_repo} has no .gguf files in siblings"


@pytest.mark.parametrize(
    "entry",
    FEATURED_ALL,
    ids=[e.hf_repo for e in FEATURED_ALL],
)
def test_featured_gguf_filename_exists_in_repo(entry) -> None:
    """The exact filename a featured entry names must be published by its repo.

    Asserting only that a repo has some .gguf let an entry pin a quant the repo
    never published: the featured Gemma 4 E4B pick named a Q4_K_M against a repo
    offering BF16/Q4_0/Q8_0, so pulling it failed with 'Entry Not Found' while
    this file stayed green. Matched with fnmatch to accept the glob filenames
    the download path resolves the same way.
    """
    gguf_files = _repo_gguf_files(entry.hf_repo)
    matches = fnmatch.filter(gguf_files, entry.gguf_filename)
    assert matches, (
        f"{entry.hf_repo} publishes no file matching {entry.gguf_filename!r}; "
        f"available: {sorted(gguf_files)}"
    )
