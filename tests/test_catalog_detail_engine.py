"""The catalog detail panel shows the engine line and lists the formats."""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.app import ComposeResult
from textual.widgets import Static

from lilbee.app import engine_service
from lilbee.app.engine_service import EngineOption
from lilbee.catalog import runnable
from lilbee.catalog.formats import ModelFormat
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.refs import model_format
from lilbee.catalog.types import ModelCompat, ModelTask
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.engine_mark import engine_mark
from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow, SizeVariant
from lilbee.cli.tui.widgets import catalog_detail
from lilbee.cli.tui.widgets.catalog_detail import (
    CatalogDetailDrawer,
    _compat_sentence,
    _engine_line,
    _formats_block,
)
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.host_caps import collect
from lilbee.providers.roles import WorkerRole
from lilbee.runtime.hardware import FitChip, FitLevel
from tests._async_wait import wait_until
from tests._lilbee_app_test_host import LilbeeAppHost
from tests.conformance.fake_adapter import BundledFakeAdapter

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
    monkeypatch.setattr(
        catalog_detail, "_engine_name", lambda ref: installed.get(model_format(ref))
    )


def _options(monkeypatch: pytest.MonkeyPatch, options: list[EngineOption]) -> list[str]:
    asked: list[str] = []

    def _fake(ref: str) -> list[EngineOption]:
        asked.append(ref)
        return [EngineOption("auto"), *options]

    monkeypatch.setattr(engine_service, "engine_options", _fake)
    return asked


def _expected_engine_line(engine: str):  # type: ignore[no-untyped-def]
    from textual.content import Content

    return Content.assemble("Engine: ", engine_mark(engine), " (chosen for this file)")


def test_the_engine_line_is_labelled_and_uses_the_engine_mark(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engines(monkeypatch, {ModelFormat.GGUF: "llama.cpp"})
    line = _engine_line(_row(frozenset({ModelFormat.GGUF})))
    assert line.plain == "Engine: ■ llama.cpp (chosen for this file)"
    assert line.spans == _expected_engine_line("llama.cpp").spans


def test_the_engine_line_names_the_engine_of_the_row_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engines(monkeypatch, {ModelFormat.GGUF: "llama.cpp", ModelFormat.MLX: "mlx"})
    line = _engine_line(_row(fmt=ModelFormat.MLX))
    assert line.plain == "Engine: ■ mlx (chosen for this file)"


def test_the_engine_line_with_no_installed_engine_says_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _engines(monkeypatch, {})
    assert _engine_line(_row()).plain == "Engine: not installed"


@pytest.mark.parametrize("compat", [ModelCompat.UNSUPPORTED, ModelCompat.UNKNOWN])
def test_the_engine_line_shows_on_a_row_that_is_not_marked_supported(
    monkeypatch: pytest.MonkeyPatch, compat: ModelCompat
) -> None:
    _engines(monkeypatch, {ModelFormat.GGUF: "llama.cpp"})
    assert _engine_line(_row(compat=compat)).plain == "Engine: ■ llama.cpp (chosen for this file)"


def test_the_compat_sentence_does_not_ask_for_an_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_ref: str) -> str:
        raise AssertionError("no engine lookup")

    monkeypatch.setattr(catalog_detail, "_engine_name", _boom)
    assert "Architecture unknown" in _compat_sentence(_row(compat=ModelCompat.UNKNOWN)).plain
    assert _compat_sentence(_row()).plain == msg.COMPAT_DETAIL_SENTENCE_SUPPORTED


class _GgufEngine(BundledFakeAdapter):
    name = "llama.cpp"
    formats = frozenset({"gguf"})
    served_roles = frozenset(WorkerRole)


class _OpenvinoEngine(BundledFakeAdapter):
    name = "openvino"
    formats = frozenset({"gguf"})
    served_roles = frozenset(WorkerRole)


@pytest.fixture
def real_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two installed engines that load GGUF; the lookup under test is the real one."""
    mac = collect(system="Darwin", machine="arm64", cpu_info={}, nvidia=None)
    monkeypatch.setattr(runnable, "session_host", lambda: mac)
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, _GgufEngine())  # type: ignore[arg-type]
    monkeypatch.setitem(choice.ADAPTERS, EngineName.OPENVINO, _OpenvinoEngine())  # type: ignore[arg-type]


def test_the_engine_line_follows_the_default_engine_through_the_real_lookup(
    real_lookup: None,
) -> None:
    assert _engine_line(_row(frozenset({ModelFormat.GGUF}))).plain == (
        "Engine: ■ llama.cpp (chosen for this file)"
    )


def test_a_stored_engine_choice_shows_in_the_engine_line(real_lookup: None) -> None:
    engine_service.use("acme/foo", "openvino")
    assert _engine_line(_row(frozenset({ModelFormat.GGUF}))).plain == (
        "Engine: ■ openvino (chosen for this file)"
    )


_FILE_REF = "acme/foo/foo-q4_k_m.gguf"


def test_a_choice_stored_under_the_installed_file_shows_for_an_installed_row(
    real_lookup: None,
) -> None:
    engine_service.use(_FILE_REF, "openvino")
    row = _row(frozenset({ModelFormat.GGUF}))
    assert _engine_line(row, _FILE_REF).plain == "Engine: ■ openvino (chosen for this file)"
    assert _formats_block(_row(), _FILE_REF).plain.splitlines()[1].endswith("■ openvino")


def test_a_choice_stored_under_the_repo_does_not_show_for_an_installed_row(
    real_lookup: None,
) -> None:
    engine_service.use("acme/foo", "openvino")
    row = _row(frozenset({ModelFormat.GGUF}))
    assert _engine_line(row, _FILE_REF).plain == "Engine: ■ llama.cpp (chosen for this file)"


def test_a_row_that_is_not_installed_keeps_the_repo_lookup(real_lookup: None) -> None:
    engine_service.use(_FILE_REF, "openvino")
    row = _row(frozenset({ModelFormat.GGUF}))
    assert _engine_line(row).plain == "Engine: ■ llama.cpp (chosen for this file)"
    engine_service.use("acme/foo", "openvino")
    assert _engine_line(row).plain == "Engine: ■ openvino (chosen for this file)"


def test_a_stored_engine_choice_shows_in_the_formats_section(real_lookup: None) -> None:
    engine_service.use("acme/foo", "openvino")
    assert _formats_block(_row()).plain.splitlines()[1].endswith("■ openvino")


def test_an_engine_that_cannot_be_found_shows_the_not_installed_state(
    real_lookup: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(choice, "ADAPTERS", {})
    line = _engine_line(_row(frozenset({ModelFormat.GGUF})))
    assert line.plain == "Engine: not installed"
    assert "llama.cpp" not in line.plain


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


@pytest.mark.parametrize(
    ("level", "glyph"),
    [(FitLevel.FITS, "✓"), (FitLevel.TIGHT, "⚠"), (FitLevel.WONT_RUN, "✗")],
)
def test_each_fit_level_has_its_own_glyph(level: FitLevel, glyph: str) -> None:
    assert catalog_detail._fit_glyph(FitChip(level=level, headroom_gb=1.0)) == glyph


@pytest.mark.parametrize("compat", [ModelCompat.UNSUPPORTED, ModelCompat.UNKNOWN])
def test_a_row_with_no_catalog_model_says_the_architecture_is_unknown(
    compat: ModelCompat,
) -> None:
    row = replace(_row(compat=compat), catalog_model=None)
    sentence = _compat_sentence(row).plain
    if compat is ModelCompat.UNSUPPORTED:
        assert sentence == (
            "Architecture unknown is not in the supported set. Pull may fail at load."
        )
    else:
        assert sentence == msg.COMPAT_DETAIL_SENTENCE_UNKNOWN


def test_a_size_variant_with_no_fit_has_no_glyph() -> None:
    variant = SizeVariant(label="Q4_K_M", quant="Q4_K_M", size_gb=4.6, ref="r/q4", fit=None)
    assert catalog_detail._render_sizes_block([variant]) == "Sizes\n  Q4_K_M  4.6 GB"


async def test_the_engine_and_formats_widgets_carry_their_css_classes() -> None:
    async with _Drawer().run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        drawer = pilot.app.query_one(CatalogDetailDrawer)
        engine = drawer.query_one("#catalog-detail-engine", Static)
        formats = drawer.query_one("#catalog-detail-formats", Static)
        assert engine.classes == frozenset({"catalog-detail-engine"})
        assert formats.classes == frozenset({"catalog-detail-formats"})


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
        engine = drawer.query_one("#catalog-detail-engine", Static)
        drawer.update_for_row(_row())
        assert await wait_until(pilot, lambda: "Formats" in str(formats.render()))
        assert "mlx" in str(formats.render())
        assert "Engine: ■ llama.cpp (chosen for this file)" in str(engine.render())
        drawer.update_for_row(frontier)
        assert await wait_until(pilot, lambda: str(formats.render()).strip() == "")
        assert str(engine.render()).strip() == ""
        drawer.update_for_row(_row())
        assert await wait_until(pilot, lambda: "Formats" in str(formats.render()))
        drawer.update_for_row(None)
        assert await wait_until(pilot, lambda: str(formats.render()).strip() == "")
