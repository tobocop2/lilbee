"""A model ref the config rejects reads as an unknown engine on every status surface."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee import mcp_server
from lilbee.app import services as svc_mod
from lilbee.app.status import gather_status, serving_roles
from lilbee.cli.helpers import render_status_result
from lilbee.cli.tui.screens.status import _build_config_content
from lilbee.core.config import cfg
from lilbee.providers.roles import WorkerRole
from lilbee.providers.serving import (
    UNKNOWN_ENGINE,
    UNKNOWN_VERSION,
    ServingEngine,
    served_by,
)
from lilbee.server import handlers
from tests.conftest import make_mock_services

INVALID_REF = "nosuchengine"


@pytest.fixture(autouse=True)
def invalid_chat_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    monkeypatch.setattr(cfg, "server_presets", None)
    monkeypatch.setitem(cfg.__dict__, "chat_model", INVALID_REF)
    monkeypatch.setattr(cfg, "embedding_model", "")
    monkeypatch.setattr(cfg, "reranker_model", "")
    monkeypatch.setattr(cfg, "vision_model", "")


def _store() -> MagicMock:
    store = MagicMock()
    store.get_meta.return_value = None
    store.source_summary.return_value = MagicMock(newest=[], kinds={}, document_count=0)
    store.total_chunks = 0
    store.held_records.return_value = {}
    store.health_warnings.return_value = []
    store.upgrade_progress.return_value = None
    return store


@pytest.fixture()
def services() -> Iterator[None]:
    svc_mod.set_services(make_mock_services(store=_store()))
    yield
    svc_mod.set_services(None)


def test_the_shared_lookup_reads_a_rejected_ref_as_unknown() -> None:
    assert served_by(INVALID_REF, None, native=False) == ServingEngine(
        UNKNOWN_ENGINE, UNKNOWN_VERSION
    )


def test_the_status_rows_read_a_rejected_ref_as_unknown() -> None:
    assert [(row.role, row.engine, row.version) for row in serving_roles()] == [
        (WorkerRole.CHAT, UNKNOWN_ENGINE, UNKNOWN_VERSION)
    ]


@pytest.mark.usefixtures("services")
async def test_the_rest_status_and_health_read_a_rejected_ref_as_unknown() -> None:
    status = await handlers.status()
    assert [(row.engine, row.version) for row in status.serving] == [
        (UNKNOWN_ENGINE, UNKNOWN_VERSION)
    ]
    assert (await handlers.health()).chat_engine == UNKNOWN_ENGINE


@pytest.mark.usefixtures("services")
def test_the_mcp_status_reads_a_rejected_ref_as_unknown() -> None:
    rows = mcp_server.status()["serving"]
    assert [(row["engine"], row["version"]) for row in rows] == [(UNKNOWN_ENGINE, UNKNOWN_VERSION)]


@pytest.mark.usefixtures("services")
def test_the_cli_status_reads_a_rejected_ref_as_unknown() -> None:
    lines = [str(line) for line in render_status_result(gather_status())]
    engine_lines = [line for line in lines if "Chat engine" in line]
    assert len(engine_lines) == 1
    assert f"{UNKNOWN_ENGINE} {UNKNOWN_VERSION}" in engine_lines[0]


def test_the_tui_config_section_reads_a_rejected_ref_as_unknown() -> None:
    plain = _build_config_content().plain
    assert "Chat engine" in plain
    assert f"{UNKNOWN_ENGINE} {UNKNOWN_VERSION}" in plain
