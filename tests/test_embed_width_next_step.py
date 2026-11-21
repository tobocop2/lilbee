"""An embedder of another width fails with a message that names the rebuild.

After a switch to an embedder whose vectors are wider than ``embedding_dim``, the
width check fires on the embed path. Every surface shows that one message, so the
message itself carries the next step for each of them.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from litestar.testing import TestClient
from typer.testing import CliRunner

from lilbee.app import services as svc_mod
from lilbee.cli.app import app
from lilbee.core.config import cfg
from lilbee.retrieval.embedder import Embedder
from tests.conftest import make_mock_services

STORED_WIDTH = 768
SERVED_WIDTH = 1024
MISMATCH = f"Embedding dimension mismatch: expected {STORED_WIDTH}, got {SERVED_WIDTH}"


class _WideProvider:
    """An engine whose model answers with SERVED_WIDTH, whatever the config says."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.5] * SERVED_WIDTH for _ in texts]


@pytest.fixture(autouse=True)
def switched(tmp_path, monkeypatch):
    """A memory-enabled config at STORED_WIDTH served by a wider embedder."""
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.data_dir = tmp_path / "data"
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.json_mode = False
    cfg.memory_enabled = True
    cfg.embedding_dim = STORED_WIDTH
    monkeypatch.setattr("lilbee.retrieval.embedder.is_model_available", lambda *_: True)
    embedder = Embedder(cfg, _WideProvider())
    svc_mod.set_services(make_mock_services(store=MagicMock(), embedder=embedder))
    yield
    svc_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


FULL_MESSAGE = (
    f"{MISMATCH}. The embedding model changed width, so re-embed the index under it: "
    "run `lilbee rebuild` (CLI), `/rebuild` (TUI), POST /api/sync with force_rebuild "
    "true (REST), or the sync tool with force_rebuild true (MCP)."
)


def _assert_names_every_fix(text: str) -> None:
    assert text == FULL_MESSAGE


def test_the_embedder_names_the_rebuild_on_every_surface():
    embedder = Embedder(cfg, _WideProvider())

    with pytest.raises(ValueError, match="force_rebuild") as raised:
        embedder.embed_query("what do bees eat")

    _assert_names_every_fix(str(raised.value))


def test_a_vector_of_the_right_width_with_a_bad_value_names_the_index():
    embedder = Embedder(cfg, _WideProvider())
    vector = [0.5] * STORED_WIDTH
    vector[3] = float("nan")

    with pytest.raises(ValueError, match="invalid value at index 3") as raised:
        embedder.validate_vector(vector)

    assert str(raised.value) == "Embedding contains invalid value at index 3: nan"


def test_serve_answers_400_with_the_rebuild_step():
    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    auth_mod.session_manager.disable()
    try:
        resp = TestClient(create_app()).post("/api/ask", json={"question": "what do bees eat"})
    finally:
        auth_mod.session_manager.cleanup()

    assert resp.status_code == 400
    _assert_names_every_fix(resp.json()["detail"])


def test_the_cli_shows_the_rebuild_step(monkeypatch):
    monkeypatch.setattr("lilbee.cli.commands.search_chat._prepare_ask", lambda *, no_sync: None)

    result = CliRunner().invoke(app, ["ask", "--no-sync", "what do bees eat"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    _assert_names_every_fix(str(result.exception))
