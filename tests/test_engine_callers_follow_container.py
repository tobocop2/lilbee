"""The pull callers name the engine of their container's config, not the global config's."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from litestar.exceptions import HTTPException

from lilbee.app import engine_choice
from lilbee.app.engine_choice import EngineBook
from lilbee.catalog.compat import UnsupportedArchError
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelTask
from lilbee.core.config import Config, cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import HostFacts
from lilbee.server import handlers
from tests._lilbee_app_test_host import LilbeeAppHost
from tests.conformance.fake_adapter import FakeAdapter

REF = "org/repo/file.gguf"


class _GlobalEngine(FakeAdapter):
    name = "llama.cpp"
    formats = frozenset({"gguf"})
    architectures = frozenset({"global_arch"})


class _LibraryEngine(FakeAdapter):
    name = "openvino"
    formats = frozenset({"gguf"})
    architectures = frozenset({"library_arch"})


GLOBAL_ENGINE = _GlobalEngine()
LIBRARY_ENGINE = _LibraryEngine()


@pytest.fixture(autouse=True)
def _two_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        engine_choice, "host_facts", lambda: HostFacts(system="darwin", machine="arm64")
    )
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, GLOBAL_ENGINE)  # type: ignore[arg-type]
    monkeypatch.setitem(choice.ADAPTERS, EngineName.OPENVINO, LIBRARY_ENGINE)  # type: ignore[arg-type]


@pytest.fixture
def library_book() -> EngineBook:
    """The engine book of a container whose config names an engine the global config does not."""
    library = Config()
    library.preferred_engine = EngineSetting.OPENVINO
    library.engine_choices = json.dumps({REF: "openvino"})
    assert cfg.preferred_engine is EngineSetting.AUTO
    assert cfg.engine_choices is None
    return EngineBook(library)


def test_the_two_configs_name_different_engines(library_book: EngineBook) -> None:
    assert library_book.for_ref(REF) is LIBRARY_ENGINE
    assert EngineBook(cfg).for_ref(REF) is GLOBAL_ENGINE


@pytest.mark.asyncio
async def test_the_mcp_pull_error_lists_the_architectures_of_the_container_engine(
    monkeypatch: pytest.MonkeyPatch, library_book: EngineBook
) -> None:
    from lilbee import mcp_server

    def _raise(*_a: object, **_kw: object) -> object:
        raise UnsupportedArchError(REF, "kimi_k2")

    monkeypatch.setattr("lilbee.app.models.pull_model_data", _raise)
    monkeypatch.setattr(mcp_server, "get_services", lambda: SimpleNamespace(engines=library_book))
    result = await mcp_server.model_pull(model=REF)
    assert result["error"]["supported_examples"] == ["library_arch"]
    assert result["error"]["total_supported"] == 1


@pytest.mark.asyncio
async def test_the_rest_pull_refusal_lists_the_architectures_of_the_container_engine(
    monkeypatch: pytest.MonkeyPatch, library_book: EngineBook
) -> None:
    manager = MagicMock()
    manager.enforce_arch_compat.side_effect = UnsupportedArchError(REF, "kimi_k2")
    monkeypatch.setattr(
        "lilbee.server.handlers.models.get_services",
        lambda: SimpleNamespace(model_manager=manager, engines=library_book),
    )
    with pytest.raises(HTTPException) as excinfo:
        await handlers.enforce_pull_arch_compat(REF, source="native")
    assert excinfo.value.extra["supported_examples"] == ["library_arch"]
    assert excinfo.value.extra["total_supported"] == 1


class _Host(LilbeeAppHost):
    pass


async def test_the_catalog_screen_resolves_the_file_with_the_container_engine(
    monkeypatch: pytest.MonkeyPatch, library_book: EngineBook
) -> None:
    from lilbee.cli.tui.screens import catalog as catalog_screen
    from lilbee.cli.tui.screens.catalog import CatalogScreen

    model = CatalogModel(
        hf_repo=REF,
        gguf_filename="model.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
    )
    async with _Host().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        seen: list[object] = []

        def _resolve(_model: CatalogModel, engine: object) -> str:
            seen.append(engine)
            return "model.gguf"

        monkeypatch.setattr(catalog_screen, "resolve_filename", _resolve)
        monkeypatch.setattr(catalog_screen, "download_bytes", lambda *_a, **_kw: 1)
        monkeypatch.setattr(catalog_screen, "disk_shortfall", lambda *_a, **_kw: None)
        monkeypatch.setattr(
            catalog_screen, "get_services", lambda: SimpleNamespace(engines=library_book)
        )
        monkeypatch.setattr(screen, "_enqueue_download", lambda _m: None)
        monkeypatch.setattr(screen, "notify", lambda *_a, **_kw: None)
        screen._install_model(model)
        assert seen == [LIBRARY_ENGINE]
