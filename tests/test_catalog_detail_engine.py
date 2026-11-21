"""The catalog detail panel names the engine on the supported line and lists the formats."""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from lilbee.app import engine_service
from lilbee.app.engine_service import EngineOption
from lilbee.catalog.formats import ModelFormat
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelCompat, ModelTask
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.engine_mark import engine_mark
from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
from lilbee.cli.tui.widgets import catalog_detail
from lilbee.cli.tui.widgets.catalog_detail import (
    CatalogDetailDrawer,
    _compat_sentence,
    _formats_block,
)
from lilbee.runtime.hardware import FitChip, FitLevel
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost

_BOTH = frozenset({ModelFormat.GGUF, ModelFormat.MLX})
_FITS = FitChip(level=FitLevel.FITS, headroom_gb=4.0)


def _row(
    formats: frozenset[ModelFormat] = _BOTH,
    *,
    compat: ModelCompat = ModelCompat.SUPPORTED,
    fmt: ModelFormat = ModelFormat.GGUF,
    fit: FitChip | None = _FITS,
) -> LocalCatalogRow:
    model = CatalogModel(
        hf_repo="acme/foo",
        gguf_filename="*.gguf",
        size_gb=4.6,
        min_ram_gb=8.0,
        description="",
        featured=False,
        downloads=0,
        task=ModelTask.CHAT,
        compat=compat,
        format=fmt,
        formats=formats,
    )
    return LocalCatalogRow(
        name="foo",
        task="chat",
        params="",
        size="4.6 GB",
        quant="",
        downloads="",
        featured=False,
        installed=False,
        sort_downloads=0,
        sort_size=4.6,
        ref="acme/foo",
        backend="native",
        compat=compat,
        catalog_model=model,
        fit=fit,
    )


def _engines(monkeypatch: pytest.MonkeyPatch, installed: dict[ModelFormat, str]) -> None:
    monkeypatch.setattr(catalog_detail, "_engine_name", lambda fmt: installed.get(fmt))


def _options(monkeypatch: pytest.MonkeyPatch, options: list[EngineOption]) -> list[str]:
    asked: list[str] = []

    def _fake(ref: str) -> list[EngineOption]:
        asked.append(ref)
        return [EngineOption("auto"), *options]

    monkeypatch.setattr(engine_service, "engine_options", _fake)
    return asked


def test_the_supported_line_uses_the_engine_mark(monkeypatch: pytest.MonkeyPatch) -> None:
    _engines(monkeypatch, {ModelFormat.GGUF: "llama.cpp"})
    line = _compat_sentence(_row(frozenset({ModelFormat.GGUF})))
    assert line.plain == "Supported by your ■ llama.cpp build."
    assert line.spans == _expected_supported_line("llama.cpp").spans


def test_the_supported_line_names_the_engine_of_the_row_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engines(monkeypatch, {ModelFormat.GGUF: "llama.cpp", ModelFormat.MLX: "mlx"})
    line = _compat_sentence(_row(fmt=ModelFormat.MLX))
    assert line.plain == "Supported by your ■ mlx build."


def test_the_supported_line_without_an_engine_names_no_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engines(monkeypatch, {})
    assert _compat_sentence(_row()).plain == msg.COMPAT_DETAIL_SENTENCE_SUPPORTED_NO_ENGINE


def test_the_unsupported_and_unknown_lines_do_not_ask_for_an_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_fmt: ModelFormat) -> str:
        raise AssertionError("no engine lookup")

    monkeypatch.setattr(catalog_detail, "_engine_name", _boom)
    assert "Architecture unknown" in _compat_sentence(_row(compat=ModelCompat.UNKNOWN)).plain


def _expected_supported_line(engine: str):  # type: ignore[no-untyped-def]
    from textual.content import Content

    return Content.assemble("Supported by your ", engine_mark(engine), " build.")


def test_one_format_shows_no_formats_section(monkeypatch: pytest.MonkeyPatch) -> None:
    _engines(monkeypatch, {ModelFormat.GGUF: "llama.cpp"})
    assert _formats_block(_row(frozenset({ModelFormat.GGUF}))).plain == ""


def test_a_row_with_no_catalog_model_shows_no_formats_section() -> None:
    row = replace(_row(), catalog_model=None)
    assert _formats_block(row).plain == ""


def test_the_formats_section_lists_each_format_with_size_fit_and_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engines(monkeypatch, {ModelFormat.GGUF: "llama.cpp", ModelFormat.MLX: "mlx"})
    block = _formats_block(_row())
    assert block.plain == "Formats\n  gguf  4.6 GB  ✓  ■ llama.cpp\n  mlx  ■ mlx"


def test_a_row_without_a_fit_shows_the_size_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    _engines(monkeypatch, {ModelFormat.GGUF: "llama.cpp", ModelFormat.MLX: "mlx"})
    block = _formats_block(_row(fit=None))
    assert block.plain.splitlines()[1] == "  gguf  4.6 GB  ■ llama.cpp"


def test_a_missing_engine_shows_its_download_size(monkeypatch: pytest.MonkeyPatch) -> None:
    _engines(monkeypatch, {ModelFormat.GGUF: "llama.cpp"})
    asked = _options(
        monkeypatch,
        [EngineOption("mlx", installed=False, download_bytes=180_000_000)],
    )
    block = _formats_block(_row())
    assert block.plain.splitlines()[2] == "  mlx  ■ mlx (180 MB download)"
    assert asked == ["acme/foo#mlx"]


def test_a_missing_engine_with_no_known_size_shows_the_name_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engines(monkeypatch, {ModelFormat.GGUF: "llama.cpp"})
    _options(monkeypatch, [EngineOption("mlx", installed=False)])
    assert _formats_block(_row()).plain.splitlines()[2] == "  mlx  ■ mlx"


def test_a_format_no_engine_can_load_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    _engines(monkeypatch, {ModelFormat.GGUF: "llama.cpp"})
    _options(monkeypatch, [EngineOption("mlx", installed=True)])
    assert _formats_block(_row()).plain.splitlines()[2] == f"  mlx  {msg.CATALOG_FORMAT_NO_ENGINE}"


class _Drawer(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield CatalogDetailDrawer(id="catalog-detail-drawer")


async def test_the_drawer_shows_and_clears_the_formats_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lilbee.catalog.types import KeyStatus
    from lilbee.cli.tui.screens.catalog_utils import FrontierCatalogRow

    _engines(monkeypatch, {ModelFormat.GGUF: "llama.cpp", ModelFormat.MLX: "mlx"})
    frontier = FrontierCatalogRow(
        name="f",
        ref="p/f",
        task="chat",
        provider="Gemini",
        provider_id="gemini",
        key_status=KeyStatus.READY,
    )
    async with _Drawer().run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        drawer = pilot.app.query_one(CatalogDetailDrawer)
        formats = drawer.query_one("#catalog-detail-formats", Static)
        compat = drawer.query_one("#catalog-detail-compat", Static)
        drawer.update_for_row(_row())
        assert await wait_until(pilot, lambda: "Formats" in str(formats.render()))
        assert "mlx" in str(formats.render())
        assert "Supported by your ■ llama.cpp build." in str(compat.render())
        drawer.update_for_row(frontier)
        assert await wait_until(pilot, lambda: str(formats.render()).strip() == "")
        drawer.update_for_row(_row())
        assert await wait_until(pilot, lambda: "Formats" in str(formats.render()))
        drawer.update_for_row(None)
        assert await wait_until(pilot, lambda: str(formats.render()).strip() == "")
