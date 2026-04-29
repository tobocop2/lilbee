"""T5 e2e indexing. Drop fixtures into documents/, sync, search returns them.

Exercises the full ingest path: file watch / hash check / chunker / embedder /
LanceDB write. Requires both chat and embedding models to be pulled.
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
_SEARCH_TIMEOUT = 90.0


def _seed_corpus(lilbee_data: Path) -> Path:
    documents = lilbee_data / "documents"
    documents.mkdir(parents=True, exist_ok=True)
    for path in _FIXTURES.glob("*.md"):
        shutil.copy(path, documents / path.name)
    return documents


def _run(
    lane: Lane, args: list[str], env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [lane.lilbee_bin, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


@pytest.mark.wiki
@pytest.mark.writer
@pytest.mark.timeout(420)
def test_sync_indexes_fixture_corpus(
    lane: Lane,
    lilbee_data: Path,
    lilbee_env_with_models: dict[str, str],
    models_pulled: dict[str, str],
) -> None:
    """`lilbee sync` indexes both notes; status reports the right counts."""
    _seed_corpus(lilbee_data)

    sync = _run(lane, ["sync"], lilbee_env_with_models, _SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr
    assert "Failed: 0" in sync.stdout, sync.stdout
    assert "Added: 2" in sync.stdout, sync.stdout

    status = _run(lane, ["--json", "status"], lilbee_env_with_models, 60.0)
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
    models_pulled: dict[str, str],
) -> None:
    """Semantic search routes a battery query to ev-notes, not coffee-notes."""
    _seed_corpus(lilbee_data)
    sync = _run(lane, ["sync"], lilbee_env_with_models, _SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    search = _run(
        lane,
        ["--json", "search", "lithium-ion battery technology", "--top-k", "3"],
        lilbee_env_with_models,
        _SEARCH_TIMEOUT,
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
    models_pulled: dict[str, str],
) -> None:
    """Mirror of the battery query: french press routes to coffee-notes."""
    _seed_corpus(lilbee_data)
    sync = _run(lane, ["sync"], lilbee_env_with_models, _SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    search = _run(
        lane,
        ["--json", "search", "french press extraction time", "--top-k", "3"],
        lilbee_env_with_models,
        _SEARCH_TIMEOUT,
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
    models_pulled: dict[str, str],
) -> None:
    """`lilbee remove <name>` purges chunks for that source."""
    _seed_corpus(lilbee_data)
    sync = _run(lane, ["sync"], lilbee_env_with_models, _SYNC_TIMEOUT)
    assert sync.returncode == 0, sync.stderr

    remove = _run(lane, ["remove", "coffee-notes.md"], lilbee_env_with_models, 60.0)
    assert remove.returncode == 0, remove.stderr

    status = _run(lane, ["--json", "status"], lilbee_env_with_models, 60.0)
    payload = json.loads(status.stdout)
    filenames = {src["filename"] for src in payload["sources"]}
    assert "coffee-notes.md" not in filenames, payload
    assert "ev-notes.md" in filenames, payload
