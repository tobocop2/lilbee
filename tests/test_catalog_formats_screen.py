"""The catalog screen shows hidden rows and lets the user pick a weight format."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from textual.app import ComposeResult
from textual.widgets import OptionList, Static

from _async_wait import wait_until
from conftest import stub_hf_listing
from lilbee.catalog import query as _query
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.cli.tui.screens.catalog import CatalogScreen
from lilbee.cli.tui.screens.catalog_utils import LocalCatalogRow
from lilbee.cli.tui.widgets.catalog_detail import CatalogDetailDrawer
from lilbee.cli.tui.widgets.format_picker import FormatPicker
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from tests._lilbee_app_test_host import LilbeeAppHost
from tests.conformance.fake_adapter import BundledFakeAdapter


class _App(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Static("host")


class _SafetensorsEngine(BundledFakeAdapter):
    name = "llama.cpp"
    formats = frozenset({"gguf", "safetensors"})
    architectures = frozenset({"gguf_arch"})


@pytest.fixture
def safetensors_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install an engine that loads safetensors, so a pull of that format has an engine."""
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, _SafetensorsEngine())  # type: ignore[arg-type]


def _row(repo: str, formats: frozenset[ModelFormat]) -> CatalogModel:
    return CatalogModel(
        hf_repo=repo,
        gguf_filename="*.gguf",
        size_gb=1.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=10,
        task=ModelTask.CHAT,
        formats=formats,
    )


def _screen_text(screen: CatalogScreen) -> str:
    return "\n".join(str(w.render()) for w in screen.query(Static))


def _hidden_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_query, "get_picks", lambda: [])
    stub_hf_listing(
        monkeypatch,
        {
            "text-generation": [
                _row("a/both", frozenset({ModelFormat.GGUF})),
                _row("b/mlx-only", frozenset({ModelFormat.MLX})),
            ]
        },
    )


async def test_grid_scroll_hint_says_how_many_rows_are_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hidden_listing(monkeypatch)
    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        shown = await wait_until(
            pilot,
            lambda: (
                "1 hidden (no engine for mlx)" in _screen_text(screen)
                and "All 1 models loaded" in _screen_text(screen)
            ),
        )
        assert shown, _screen_text(screen)
        assert "b/mlx-only" not in _screen_text(screen)


async def test_list_view_count_line_says_how_many_rows_are_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hidden_listing(monkeypatch)
    async with _App().run_test(size=(160, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await wait_until(pilot, lambda: "1 hidden" in _screen_text(screen))
        await pilot.press("v")
        shown = await wait_until(
            pilot,
            lambda: "models · 1 hidden (no engine for mlx)  |  Press v" in _screen_text(screen),
        )
        assert shown, _screen_text(screen)


async def test_a_repeated_hidden_row_counts_once_on_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two pages that both hide the same repo give one count, as the server does."""
    _hidden_listing(monkeypatch)
    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await wait_until(pilot, lambda: "1 hidden" in _screen_text(screen))
        screen._fetch_more_hf_for_task(ModelTask.CHAT)
        await pilot.pause()
        assert "2 hidden" not in _screen_text(screen)


def _record_downloads(pilot_app: LilbeeAppHost) -> list[CatalogModel]:
    started: list[CatalogModel] = []

    def _record(model: CatalogModel, **_kw: object) -> str:
        started.append(model)
        return "task-id"

    pilot_app.task_bar.start_download = _record  # type: ignore[method-assign]
    return started


def _quiet(monkeypatch: pytest.MonkeyPatch, screen: CatalogScreen) -> None:
    monkeypatch.setattr(screen, "notify", lambda message, **_kw: None)
    monkeypatch.setattr("lilbee.cli.tui.screens.catalog.disk_shortfall", lambda *_a, **_kw: None)


async def test_install_of_a_two_format_row_offers_the_formats_and_pulls_the_chosen_one(
    monkeypatch: pytest.MonkeyPatch, safetensors_engine: None
) -> None:
    model = _row("acme/two", frozenset({ModelFormat.GGUF, ModelFormat.SAFETENSORS}))
    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        started = _record_downloads(pilot.app)
        _quiet(monkeypatch, screen)

        screen._install_model(model)
        assert await wait_until(
            pilot,
            lambda: (
                isinstance(pilot.app.screen, FormatPicker)
                and bool(pilot.app.screen.query(OptionList))
            ),
        )
        picker = pilot.app.screen
        assert [
            str(picker.query_one(OptionList).get_option_at_index(i).prompt) for i in range(2)
        ] == [
            "gguf",
            "safetensors",
        ]
        assert started == []
        await pilot.press("down", "enter")
        assert await wait_until(pilot, lambda: bool(started))
        assert [(m.format, m.pull_ref) for m in started] == [
            (ModelFormat.SAFETENSORS, "acme/two#safetensors")
        ]


async def test_install_of_a_format_no_engine_loads_refuses_and_starts_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a gguf engine is installed, so the safetensors pick is refused before any download."""
    model = _row("acme/two", frozenset({ModelFormat.GGUF, ModelFormat.SAFETENSORS}))
    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        started = _record_downloads(pilot.app)
        _quiet(monkeypatch, screen)
        notices: list[str] = []
        monkeypatch.setattr(screen, "notify", lambda message, **_kw: notices.append(str(message)))

        screen._install_model(model)
        assert await wait_until(pilot, lambda: isinstance(pilot.app.screen, FormatPicker))
        await pilot.press("down", "enter")
        assert await wait_until(pilot, lambda: bool(notices))
        assert "No installed engine loads the safetensors format" in notices[0]
        assert started == []


async def test_escape_in_the_format_picker_installs_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _row("acme/two", frozenset({ModelFormat.GGUF, ModelFormat.SAFETENSORS}))
    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        started = _record_downloads(pilot.app)
        _quiet(monkeypatch, screen)

        screen._install_model(model)
        assert await wait_until(pilot, lambda: isinstance(pilot.app.screen, FormatPicker))
        await pilot.press("escape")
        assert await wait_until(pilot, lambda: pilot.app.screen is screen)
        assert started == []


async def test_install_of_a_one_format_row_asks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _row("acme/one", frozenset({ModelFormat.GGUF}))
    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        started = _record_downloads(pilot.app)
        _quiet(monkeypatch, screen)
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.catalog.resolve_filename", lambda *_a, **_kw: "x.gguf"
        )
        monkeypatch.setattr("lilbee.cli.tui.screens.catalog.download_bytes", lambda *_a: 1)

        screen._install_model(model)
        assert await wait_until(pilot, lambda: bool(started))
        assert pilot.app.screen is screen
        assert [m.pull_ref for m in started] == ["acme/one"]


def _detail_row(formats: frozenset[ModelFormat]) -> LocalCatalogRow:
    return LocalCatalogRow(
        name="Two",
        task=ModelTask.CHAT,
        params="8B",
        size="4.6 GB",
        quant="Q4_K_M",
        downloads="--",
        featured=False,
        installed=False,
        sort_downloads=0,
        sort_size=4.6,
        ref="acme/two",
        backend="native",
        catalog_model=_row("acme/two", formats),
    )


@pytest.mark.parametrize(
    ("formats", "shown"),
    [
        (frozenset({ModelFormat.GGUF, ModelFormat.SAFETENSORS}), True),
        (frozenset({ModelFormat.GGUF}), False),
    ],
    ids=["two-formats", "one-format"],
)
async def test_detail_pane_lists_formats_only_when_there_is_a_choice(
    formats: frozenset[ModelFormat], shown: bool
) -> None:
    class _Drawer(LilbeeAppHost):
        def compose(self) -> ComposeResult:
            yield CatalogDetailDrawer(id="catalog-detail-drawer")

    async with _Drawer().run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        drawer = pilot.app.query_one(CatalogDetailDrawer)
        drawer.update_for_row(_detail_row(formats))
        compat = drawer.query_one("#catalog-detail-compat", Static)
        await wait_until(pilot, lambda: bool(str(compat.render()).strip()))
        assert ("Formats: gguf, safetensors" in str(compat.render())) is shown


async def test_search_result_hides_count_in_the_same_note(monkeypatch: pytest.MonkeyPatch) -> None:
    from lilbee.app.services import get_services
    from lilbee.catalog.models import HfPage

    monkeypatch.setattr(_query, "get_picks", lambda: [])

    def _fetch(**kwargs: object) -> HfPage:
        if kwargs.get("search"):
            return HfPage(models=[_row("s/mlx-found", frozenset({ModelFormat.MLX}))])
        return HfPage(models=[_row("a/ok", frozenset({ModelFormat.GGUF}))])

    monkeypatch.setattr(get_services().hf_client, "fetch_models", _fetch)
    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await wait_until(pilot, lambda: "All 1 models loaded" in _screen_text(screen))
        assert "hidden" not in _screen_text(screen)
        screen._trigger_remote_search("found")
        shown = await wait_until(
            pilot, lambda: "1 hidden (no engine for mlx)" in _screen_text(screen)
        )
        assert shown, _screen_text(screen)


async def test_cut_short_hint_keeps_the_hidden_note(monkeypatch: pytest.MonkeyPatch) -> None:
    """A scan that stops at its bound with only hidden rows says so and counts them."""
    from lilbee.app.services import get_services
    from lilbee.catalog.models import HfPage

    monkeypatch.setattr(_query, "get_picks", lambda: [])
    seen: list[str] = []

    def _fetch(**kwargs: object) -> HfPage:
        repo = f"m/mlx-{len(seen)}"
        seen.append(repo)
        return HfPage(models=[_row(repo, frozenset({ModelFormat.MLX}))], next_cursor=str(len(seen)))

    monkeypatch.setattr(get_services().hf_client, "fetch_models", _fetch)
    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        shown = await wait_until(
            pilot,
            lambda: (
                "the listing stops here" in _screen_text(screen)
                and " hidden (" in _screen_text(screen)
            ),
        )
        assert shown, _screen_text(screen)


async def test_install_of_a_file_already_on_disk_says_so_and_pulls_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from lilbee.core.config import cfg

    (tmp_path / "x.gguf").write_bytes(b"x")
    monkeypatch.setattr(cfg, "models_dir", tmp_path)
    monkeypatch.setattr(
        "lilbee.cli.tui.screens.catalog.resolve_filename", lambda *_a, **_kw: "x.gguf"
    )
    async with _App().run_test(size=(120, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        started = _record_downloads(pilot.app)
        notices: list[str] = []
        monkeypatch.setattr(screen, "notify", lambda message, **_kw: notices.append(str(message)))

        screen._install_model(_row("acme/one", frozenset({ModelFormat.GGUF})))

        assert started == []
        assert any("already installed" in n for n in notices)


def test_the_download_task_pulls_by_the_format_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    from lilbee.cli.tui.widgets.task_bar_controller import _download_target

    pulled: list[str] = []
    monkeypatch.setattr(
        "lilbee.app.models.pull_model_data", lambda ref, *_a, **_kw: pulled.append(ref)
    )
    model = _row("acme/two", frozenset({ModelFormat.GGUF, ModelFormat.MLX}))
    _download_target(object(), replace(model, format=ModelFormat.MLX))  # type: ignore[arg-type]
    _download_target(object(), model)  # type: ignore[arg-type]
    assert pulled == ["acme/two#mlx", "acme/two"]
