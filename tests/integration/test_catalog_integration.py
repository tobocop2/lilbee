"""Integration tests for catalog: verifies live picks against the real HF API.

Run with:
    uv run pytest tests/integration/test_catalog_integration.py -v -m slow
"""

from __future__ import annotations

import httpx
import pytest

from lilbee.catalog.hf_client import DEFAULT_TIMEOUT, HF_API_URL, hf_headers
from lilbee.catalog.picks import get_picks, reset_picks
from lilbee.catalog.query import reclassify_by_name, size_bucket
from lilbee.catalog.types import CatalogSize, ModelTask

pytestmark = [pytest.mark.slow, pytest.mark.live_picks]


@pytest.fixture(scope="module")
def live_picks():
    """Resolve picks once against the real trending ranking."""
    reset_picks()
    picks = get_picks()
    if not picks:
        pytest.skip("HuggingFace unreachable")
    yield picks
    reset_picks()


def test_every_pick_has_a_gguf(live_picks) -> None:
    """A pick the user cannot actually pull is worse than no pick."""
    # Authenticated request (HF_TOKEN) lifts the unauthenticated rate limit that
    # otherwise 429s a full sweep from a shared CI IP.
    for entry in live_picks:
        resp = httpx.get(
            f"{HF_API_URL}/{entry.hf_repo}",
            timeout=DEFAULT_TIMEOUT,
            headers=hf_headers(),
        )
        resp.raise_for_status()
        siblings = resp.json().get("siblings", [])
        gguf = [s["rfilename"] for s in siblings if s.get("rfilename", "").endswith(".gguf")]
        assert gguf, f"{entry.hf_repo} has no .gguf files in siblings"


def test_chat_picks_span_the_parameter_tiers(live_picks) -> None:
    """The spread is the whole point: a small machine must find something."""
    tiers = {size_bucket(m.params) for m in live_picks if m.task == ModelTask.CHAT and m.params > 0}
    assert tiers == set(CatalogSize), f"chat picks only covered {sorted(t.value for t in tiers)}"


def test_every_role_is_represented(live_picks) -> None:
    for task in (ModelTask.CHAT, ModelTask.EMBEDDING, ModelTask.VISION, ModelTask.RERANK):
        assert [m for m in live_picks if m.task == task], f"no {task} pick"


def test_role_picks_are_named_like_their_role(live_picks) -> None:
    """Guards the mistagging that put a chat model in the reranker slot."""
    for task in (ModelTask.EMBEDDING, ModelTask.RERANK):
        for entry in (m for m in live_picks if m.task == task):
            assert reclassify_by_name(entry.hf_repo, ModelTask.CHAT) == task, (
                f"{entry.hf_repo} is shown as {task} but is not named like one"
            )
