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
import shutil
import subprocess
from pathlib import Path

import pytest

from conftest import Lane

_FIXTURES = Path(__file__).parent / "fixtures" / "notes"
_SYNC_TIMEOUT = 240.0
_BUILD_DRY_RUN_TIMEOUT = 240.0
_BUILD_FULL_TIMEOUT = 720.0


def _seed_corpus(lilbee_data: Path) -> None:
    documents = lilbee_data / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    for path in _FIXTURES.glob("*.md"):
        shutil.copy(path, documents / path.name)


def _wiki_env(base: dict[str, str]) -> dict[str, str]:
    """Copy of the base env with wiki enabled."""
    out = dict(base)
    out["LILBEE_WIKI"] = "1"
    return out


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(360)
def test_wiki_build_dry_run_extracts_entities(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
    models_pulled: dict[str, str],
) -> None:
    """`lilbee --json wiki build --dry-run` extracts at least one NER entity
    candidate from the seeded corpus, without making any LLM calls."""
    env = _wiki_env(lilbee_env_with_models)
    _seed_corpus(lilbee_data)

    sync = subprocess.run(
        [lane.lilbee_bin, "sync"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_SYNC_TIMEOUT,
        check=False,
    )
    assert sync.returncode == 0, sync.stderr

    result = subprocess.run(
        [lane.lilbee_bin, "--json", "wiki", "build", "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_BUILD_DRY_RUN_TIMEOUT,
        check=False,
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
    models_pulled: dict[str, str],
) -> None:
    """`lilbee --json wiki build` runs to completion and emits a structured
    result with paths/count/entities keys.

    With a 2-doc corpus and a 0.6B chat model the count may be zero (the
    LLM may decline to propose concepts spanning <3 sources); that's a
    content-dependent outcome, not a regression. We assert on the envelope,
    not the page count.
    """
    env = _wiki_env(lilbee_env_with_models)
    # Allow the LLM more time per call on slow runners; the build serially
    # invokes the chat model per source.
    env.setdefault("LILBEE_WIKI_BUILD_PER_SOURCE_TIMEOUT_SECS", "300")

    _seed_corpus(lilbee_data)
    sync = subprocess.run(
        [lane.lilbee_bin, "sync"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_SYNC_TIMEOUT,
        check=False,
    )
    assert sync.returncode == 0, sync.stderr

    result = subprocess.run(
        [lane.lilbee_bin, "--json", "wiki", "build"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_BUILD_FULL_TIMEOUT,
        check=False,
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
    models_pulled: dict[str, str],
) -> None:
    """`lilbee --json wiki synthesize` runs to completion on a small corpus
    and returns a structured envelope.

    Synthesis requires concept clusters spanning ≥3 sources; the 2-doc
    fixture corpus rarely produces any. This test asserts the command
    runs without crashing and returns the documented JSON shape, which
    is the regression we want to catch (broken synth code path), not page
    count. A future test with a larger fixture corpus can assert on
    paths != [].
    """
    env = _wiki_env(lilbee_env_with_models)
    _seed_corpus(lilbee_data)

    sync = subprocess.run(
        [lane.lilbee_bin, "sync"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_SYNC_TIMEOUT,
        check=False,
    )
    assert sync.returncode == 0, sync.stderr

    result = subprocess.run(
        [lane.lilbee_bin, "--json", "wiki", "synthesize"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_BUILD_DRY_RUN_TIMEOUT,
        check=False,
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
