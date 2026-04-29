"""T1 CLI status. JSON shape, config keys, and text output against an empty data dir."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import Lane, run_lilbee


@pytest.mark.cli
def test_status_json_has_command_key(lane: Lane, lilbee_data: Path) -> None:
    result = run_lilbee(lane, ["--json", "status"], data_dir=lilbee_data, timeout=60)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["command"] == "status"


@pytest.mark.cli
def test_status_json_exposes_config_block(lane: Lane, lilbee_data: Path) -> None:
    result = run_lilbee(lane, ["--json", "status"], data_dir=lilbee_data, timeout=60)
    payload = json.loads(result.stdout)
    config = payload.get("config")
    assert isinstance(config, dict), payload
    for key in ("documents_dir", "data_dir", "chat_model", "embedding_model"):
        assert key in config, f"config missing {key}: {config}"


@pytest.mark.cli
def test_status_json_documents_dir_under_lilbee_data(lane: Lane, lilbee_data: Path) -> None:
    """LILBEE_DATA env var should resolve documents_dir under it."""
    result = run_lilbee(lane, ["--json", "status"], data_dir=lilbee_data, timeout=60)
    payload = json.loads(result.stdout)
    documents_dir = Path(payload["config"]["documents_dir"])
    # Path comparison, not string startswith, so symlinked tmp dirs match.
    assert lilbee_data.resolve() in documents_dir.resolve().parents, (
        f"documents_dir {documents_dir} not under lilbee_data {lilbee_data}"
    )


@pytest.mark.cli
def test_status_text_mentions_chat_and_embedding_models(lane: Lane, lilbee_data: Path) -> None:
    """Default (non-JSON) status text exposes Chat model and Embeddings rows."""
    result = run_lilbee(lane, ["status"], data_dir=lilbee_data, timeout=60)
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "Chat model" in output
    assert "Embeddings" in output


@pytest.mark.cli
def test_status_reports_zero_chunks(lane: Lane, lilbee_data: Path) -> None:
    result = run_lilbee(lane, ["--json", "status"], data_dir=lilbee_data, timeout=60)
    payload = json.loads(result.stdout)
    assert payload["total_chunks"] == 0
    assert payload["sources"] == []


@pytest.mark.cli
def test_data_dir_flag_resolves_documents_dir_under_it(lane: Lane, tmp_path: Path) -> None:
    """`--data-dir <path>` puts documents_dir under the supplied path."""
    explicit = tmp_path / "explicit-dir"
    explicit.mkdir()
    result = run_lilbee(
        lane,
        ["--data-dir", str(explicit), "--json", "status"],
        data_dir=explicit,
        timeout=60,
    )
    payload = json.loads(result.stdout)
    documents_dir = Path(payload["config"]["documents_dir"])
    assert explicit.resolve() in documents_dir.resolve().parents
