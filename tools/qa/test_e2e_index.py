"""T5 e2e indexing. Drop fixtures into documents/, sync, search returns them.

Exercises the full ingest path: file watch / hash check / chunker / embedder /
LanceDB write. Requires both chat and embedding models to be pulled.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import (
    SEARCH_TIMEOUT,
    SYNC_TIMEOUT,
    Lane,
    run_lilbee_with_env,
    seed_fixture_corpus,
)


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(420)
def test_sync_indexes_fixture_corpus(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
) -> None:
    """`lilbee sync` indexes both notes; status reports the right counts."""
    seed_fixture_corpus(lilbee_data)

    sync = run_lilbee_with_env(lane, ["sync"], env=lilbee_env_with_models, timeout=SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr
    assert "Failed: 0" in sync.stdout, sync.stdout
    assert "Added: 2" in sync.stdout, sync.stdout

    status = run_lilbee_with_env(
        lane, ["--json", "status"], env=lilbee_env_with_models, timeout=60.0
    )
    assert status.returncode == 0, status.stderr
    payload = json.loads(status.stdout)
    sources = payload["sources"]
    filenames = {src["filename"] for src in sources}
    assert filenames == {"ev-notes.md", "coffee-notes.md"}, payload
    assert payload["total_chunks"] >= 2


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(420)
def test_search_finds_battery_query_in_ev_notes(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
) -> None:
    """Semantic search routes a battery query to ev-notes, not coffee-notes."""
    seed_fixture_corpus(lilbee_data)
    sync = run_lilbee_with_env(lane, ["sync"], env=lilbee_env_with_models, timeout=SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    search = run_lilbee_with_env(
        lane,
        ["--json", "search", "lithium-ion battery technology", "--top-k", "3"],
        env=lilbee_env_with_models,
        timeout=SEARCH_TIMEOUT,
    )
    assert search.returncode == 0, search.stderr
    payload = json.loads(search.stdout)
    sources = [r["source"] for r in payload["results"]]
    assert "ev-notes.md" in sources, payload
    # The top hit should be ev-notes; coffee-notes can be in the result list.
    assert sources[0] == "ev-notes.md", sources


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(420)
def test_search_finds_coffee_query_in_coffee_notes(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
) -> None:
    """Mirror of the battery query: french press routes to coffee-notes."""
    seed_fixture_corpus(lilbee_data)
    sync = run_lilbee_with_env(lane, ["sync"], env=lilbee_env_with_models, timeout=SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    search = run_lilbee_with_env(
        lane,
        ["--json", "search", "french press extraction time", "--top-k", "3"],
        env=lilbee_env_with_models,
        timeout=SEARCH_TIMEOUT,
    )
    assert search.returncode == 0, search.stderr
    payload = json.loads(search.stdout)
    sources = [r["source"] for r in payload["results"]]
    assert sources[0] == "coffee-notes.md", sources


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(420)
def test_remove_clears_indexed_source(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
) -> None:
    """`lilbee remove <name>` purges chunks for that source."""
    seed_fixture_corpus(lilbee_data)
    sync = run_lilbee_with_env(lane, ["sync"], env=lilbee_env_with_models, timeout=SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    remove = run_lilbee_with_env(
        lane, ["remove", "coffee-notes.md"], env=lilbee_env_with_models, timeout=60.0
    )
    assert remove.returncode == 0, remove.stderr

    status = run_lilbee_with_env(
        lane, ["--json", "status"], env=lilbee_env_with_models, timeout=60.0
    )
    payload = json.loads(status.stdout)
    filenames = {src["filename"] for src in payload["sources"]}
    assert "coffee-notes.md" not in filenames, payload
    assert "ev-notes.md" in filenames, payload
