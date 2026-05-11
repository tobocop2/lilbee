"""T5 wiki build / synthesize E2E.

Wiki is the headline feature; today's QA only hits empty-store reads. This
file exercises the actual generation pipeline end to end.

Two tiers:

* `test_wiki_build_dry_run_extracts_entities`. Fast, deterministic. Runs
  `lilbee wiki build --dry-run` which executes the NER entity extractor
  without making any LLM calls. Asserts the extractor produces ≥1 entity
  candidate from the seeded corpus. Catches regressions in the extraction
  pipeline without depending on model behavior.

* `test_wiki_build_full_runs_clean`. Slow, exercises the full LLM-curated
  build path with the QA-matrix Qwen 0.6B chat model. Assertion is "command
  exits 0 and emits a JSON envelope with the expected keys"; we don't pin
  the count of generated pages because that's content-dependent. Times out
  generously to absorb cold-start LLM warmup on bare runners.

Both require `LILBEE_WIKI=1` to flip `cfg.wiki` on (off by default).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import SYNC_TIMEOUT, WIKI_FAST_TIMEOUT, Lane, run_lilbee_with_env, seed_fixture_corpus

_BUILD_FULL_TIMEOUT = 720.0


def _wiki_env(base: dict[str, str]) -> dict[str, str]:
    """Copy of the base env with wiki enabled."""
    out = dict(base)
    out["LILBEE_WIKI"] = "1"
    return out


def _skip_unless_spacy_bundled(lane: Lane) -> None:
    """Wiki NER concept extraction needs spaCy + en_core_web_sm. Only the
    standalone binary bundles them; the bare pip wheel ships neither (spaCy
    lives in the [graph] extra), so the wiki pipeline degrades to zero
    entities there rather than failing. Skip on non-binary lanes."""
    if not lane.is_binary:
        pytest.skip(
            "wiki NER needs spaCy + en_core_web_sm; only the standalone binary bundles them"
        )


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(360)
def test_wiki_build_dry_run_extracts_entities(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
) -> None:
    """`lilbee --json wiki build --dry-run` extracts at least one NER entity
    candidate from the seeded corpus, without making any LLM calls."""
    _skip_unless_spacy_bundled(lane)
    env = _wiki_env(lilbee_env_with_models)
    seed_fixture_corpus(lilbee_data)

    sync = run_lilbee_with_env(lane, ["sync"], env=env, timeout=SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    result = run_lilbee_with_env(
        lane,
        ["--json", "wiki", "build", "--dry-run"],
        env=env,
        timeout=WIKI_FAST_TIMEOUT,
    )
    assert result.returncode == 0, (
        f"wiki build --dry-run failed: rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    payload = json.loads(result.stdout)
    assert payload.get("command") == "wiki_build", payload
    assert payload.get("dry_run") is True, payload
    entities = payload.get("entities", [])
    assert isinstance(entities, list), payload
    assert len(entities) >= 1, (
        f"NER extraction produced zero entities from a 2-doc corpus; "
        f"that's a regression in the entity extractor: {payload}"
    )
    # Each entity row should have the documented shape.
    sample = entities[0]
    for required_key in ("slug", "label", "kind", "mentions", "sources"):
        assert required_key in sample, f"entity row missing {required_key!r}: {sample}"
    # Bind the assertion to the seeded corpus so a stub or hallucinated
    # entity list can't satisfy the count assertion. The fixtures cover EV
    # batteries and coffee, so at least one extracted entity should mention
    # one of those topics in its label or its sources.
    corpus_haystack = " ".join(
        f"{e.get('label', '')} {' '.join(e.get('sources', []))}".lower() for e in entities
    )
    corpus_keywords = ("battery", "coffee", "ev-notes", "coffee-notes", "lithium", "espresso")
    assert any(kw in corpus_haystack for kw in corpus_keywords), (
        f"extracted entities do not reference the seeded fixture corpus; "
        f"entity labels/sources: {corpus_haystack[:500]!r}"
    )


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(_BUILD_FULL_TIMEOUT + 60)
def test_wiki_build_full_runs_clean(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
) -> None:
    """`lilbee --json wiki build` runs to completion and emits a structured
    result with paths/count/entities keys.

    With a 2-doc corpus and a 0.6B chat model the count may be zero (the
    LLM may decline to propose concepts spanning <3 sources); that's a
    content-dependent outcome, not a regression. We assert on the envelope,
    not the page count.
    """
    _skip_unless_spacy_bundled(lane)
    env = _wiki_env(lilbee_env_with_models)
    # Allow the LLM more time per call on slow runners; the build serially
    # invokes the chat model per source.
    env.setdefault("LILBEE_WIKI_BUILD_PER_SOURCE_TIMEOUT_SECS", "300")

    seed_fixture_corpus(lilbee_data)
    sync = run_lilbee_with_env(lane, ["sync"], env=env, timeout=SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    result = run_lilbee_with_env(
        lane, ["--json", "wiki", "build"], env=env, timeout=_BUILD_FULL_TIMEOUT
    )
    assert result.returncode == 0, (
        f"wiki build failed: rc={result.returncode}\n"
        f"stdout tail: {result.stdout[-1500:]}\nstderr tail: {result.stderr[-1500:]}"
    )
    payload = json.loads(result.stdout)
    assert payload.get("command") == "wiki_build", payload
    for required_key in ("paths", "count", "entities"):
        assert required_key in payload, f"wiki_build response missing {required_key!r}: {payload}"
    assert isinstance(payload["paths"], list), payload
    assert isinstance(payload["count"], int), payload


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(360)
def test_wiki_synthesize_runs_clean(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
) -> None:
    """`lilbee --json wiki synthesize` runs to completion on a small corpus
    and returns a structured envelope.

    Synthesis requires concept clusters spanning ≥3 sources; the 2-doc
    fixture corpus rarely produces any. This test asserts the command
    runs without crashing and returns the documented JSON shape, which
    is the regression we want to catch (broken synth code path), not page
    count; a larger fixture corpus would let an ordering / paths-non-empty
    assertion go in.
    """
    _skip_unless_spacy_bundled(lane)
    env = _wiki_env(lilbee_env_with_models)
    seed_fixture_corpus(lilbee_data)

    sync = run_lilbee_with_env(lane, ["sync"], env=env, timeout=SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    result = run_lilbee_with_env(
        lane, ["--json", "wiki", "synthesize"], env=env, timeout=WIKI_FAST_TIMEOUT
    )
    assert result.returncode == 0, (
        f"wiki synthesize failed: rc={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    payload = json.loads(result.stdout)
    assert payload.get("command") == "wiki_synthesize", payload
    for required_key in ("paths", "count"):
        assert required_key in payload, f"wiki_synthesize response missing {required_key!r}"
    assert isinstance(payload["paths"], list), payload
    assert isinstance(payload["count"], int), payload
