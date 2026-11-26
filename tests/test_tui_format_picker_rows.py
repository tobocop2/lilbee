"""The install format picker shows size, fit and engine, and installs a missing engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from textual.app import ComposeResult
from textual.content import Content
from textual.widgets import OptionList, Static

from lilbee.app.engine_service import EngineOption
from lilbee.catalog import runnable
from lilbee.catalog.models import CatalogModel
from lilbee.catalog.pull_fit import PullFit
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.cli.tui import format_rows as rows_module
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.format_rows import FormatRow, format_prompt, format_rows
from lilbee.cli.tui.screens.catalog import CatalogScreen
from lilbee.cli.tui.widgets.format_picker import FormatPicker
from lilbee.core.config import cfg
from lilbee.core.config.enums import EngineSetting
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.host_caps import collect
from lilbee.providers.roles import WorkerRole
from lilbee.runtime.hardware import FitLevel
from tests._async_wait import wait_for_pull_check, wait_until
from tests._lilbee_app_test_host import LilbeeAppHost
from tests.conformance.fake_adapter import BundledFakeAdapter

_GIB = 1024**3
_ENGINE_MB = 180_000_000


class _App(LilbeeAppHost):
    def compose(self) -> ComposeResult:
        yield Static("host")


def _model(
    default: ModelFormat = ModelFormat.GGUF,
    formats: frozenset[ModelFormat] = frozenset({ModelFormat.GGUF, ModelFormat.SAFETENSORS}),
    task: ModelTask = ModelTask.CHAT,
) -> CatalogModel:
    return CatalogModel(
        hf_repo="acme/two",
        gguf_filename="*.gguf",
        size_gb=2.0,
        min_ram_gb=2.0,
        description="",
        featured=False,
        downloads=10,
        task=task,
        format=default,
        formats=formats,
    )


def _options(
    table: dict[str, list[EngineOption]],
) -> Callable[[str], list[EngineOption]]:
    """An ``engine_options`` stand-in: the rows by the format named in the ref."""

    def options(ref: str) -> list[EngineOption]:
        fmt = ref.rpartition("#")[2] if "#" in ref else "gguf"
        return table[fmt]

    return options


_LLAMA_NOW = [
    EngineOption(choice.AUTO, auto_now="llama.cpp"),
    EngineOption("llama.cpp", bundled=True),
]
_MLX_MISSING = [
    EngineOption(choice.AUTO),
    EngineOption("mlx", installed=False, download_bytes=_ENGINE_MB),
]


@pytest.fixture
def engines(monkeypatch: pytest.MonkeyPatch) -> Callable[[dict[str, list[EngineOption]]], None]:
    def install(table: dict[str, list[EngineOption]]) -> None:
        monkeypatch.setattr(rows_module.engine_service, "engine_options", _options(table))

    return install


@pytest.fixture
def hub_fit(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Stand in for the Hub read of a safetensors model; records each (repo, engine) asked."""
    asked: list[tuple[str, str]] = []

    def fit(repo: str, engine: object, **_kw: object) -> PullFit:
        asked.append((repo, str(engine.name)))  # type: ignore[attr-defined]
        return PullFit(device_bytes=3 * _GIB, ctx=4096, size_known=True, upper_bound=False)

    monkeypatch.setattr("lilbee.catalog.pull_fit.chat_pull_fit", fit)
    return asked


def test_default_format_comes_first_and_says_so(
    engines: Callable[[dict[str, list[EngineOption]]], None], hub_fit: list[tuple[str, str]]
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _LLAMA_NOW})
    rows = format_rows(_model(default=ModelFormat.SAFETENSORS), 16 * _GIB)
    assert [(r.fmt, r.default) for r in rows] == [
        (ModelFormat.SAFETENSORS, True),
        (ModelFormat.GGUF, False),
    ]
    assert str(format_prompt(rows[0])).startswith("safetensors " + msg.FORMAT_ROW_DEFAULT)
    assert msg.FORMAT_ROW_DEFAULT not in str(format_prompt(rows[1]))


def test_a_gguf_row_has_size_fit_and_engine(
    engines: Callable[[dict[str, list[EngineOption]]], None], hub_fit: list[tuple[str, str]]
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _LLAMA_NOW})
    gguf = format_rows(_model(), 16 * _GIB)[0]
    assert (gguf.size_bytes, gguf.fit, gguf.engine, gguf.missing) == (
        2 * _GIB,
        FitLevel.FITS,
        "llama.cpp",
        None,
    )
    text = str(format_prompt(gguf))
    assert "2.1 GB" in text
    assert msg.FORMAT_FIT_FITS in text
    assert "■ llama.cpp" in text


def test_a_safetensors_chat_row_takes_its_size_from_the_hub_fit(
    engines: Callable[[dict[str, list[EngineOption]]], None], hub_fit: list[tuple[str, str]]
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _LLAMA_NOW})
    safetensors = format_rows(_model(), 16 * _GIB)[1]
    assert safetensors.size_bytes == 3 * _GIB
    assert hub_fit == [("acme/two", "llama.cpp")]


def test_a_hub_row_with_an_unknown_size_says_so(
    engines: Callable[[dict[str, list[EngineOption]]], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _LLAMA_NOW})
    monkeypatch.setattr(
        "lilbee.catalog.pull_fit.chat_pull_fit",
        lambda *_a, **_kw: PullFit(
            device_bytes=_GIB, ctx=4096, size_known=False, upper_bound=False
        ),
    )
    row = format_rows(_model(), 16 * _GIB)[1]
    assert (row.size_bytes, row.fit) == (None, None)
    assert msg.FORMAT_SIZE_UNKNOWN in str(format_prompt(row))


def test_a_non_default_format_of_another_task_has_no_size(
    engines: Callable[[dict[str, list[EngineOption]]], None], hub_fit: list[tuple[str, str]]
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _LLAMA_NOW})
    row = format_rows(_model(task=ModelTask.EMBEDDING), 16 * _GIB)[1]
    assert row.size_bytes is None
    assert hub_fit == []


@pytest.mark.parametrize(
    ("available", "level", "text"),
    [
        (16 * _GIB, FitLevel.FITS, msg.FORMAT_FIT_FITS),
        (int(2.5 * _GIB), FitLevel.TIGHT, msg.FORMAT_FIT_TIGHT),
        (_GIB, FitLevel.WONT_RUN, msg.FORMAT_FIT_WONT_RUN),
    ],
)
def test_the_fit_follows_the_memory_available(
    engines: Callable[[dict[str, list[EngineOption]]], None],
    hub_fit: list[tuple[str, str]],
    available: int,
    level: FitLevel,
    text: str,
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _LLAMA_NOW})
    gguf = format_rows(_model(), available)[0]
    assert gguf.fit is level
    assert text in str(format_prompt(gguf))


def test_no_fit_when_memory_cannot_be_read(
    engines: Callable[[dict[str, list[EngineOption]]], None], hub_fit: list[tuple[str, str]]
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _LLAMA_NOW})
    gguf = format_rows(_model(), None)[0]
    assert gguf.fit is None
    assert gguf.size_bytes == 2 * _GIB


def test_a_missing_engine_adds_the_muted_download_size(
    engines: Callable[[dict[str, list[EngineOption]]], None], hub_fit: list[tuple[str, str]]
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _MLX_MISSING})
    row = format_rows(_model(), 16 * _GIB)[1]
    assert row.engine == "mlx"
    assert row.missing is _MLX_MISSING[1]
    prompt = format_prompt(row)
    assert str(prompt).endswith("engine download 180 MB")
    muted = [span for span in prompt.spans if span.style == "$text-muted"]
    assert muted
    assert "engine download 180 MB" in str(prompt)[muted[-1].start : muted[-1].end]


def test_a_missing_engine_without_a_pack_size_names_no_size() -> None:
    row = FormatRow(ModelFormat.MLX, False, "mlx", None, None, EngineOption("mlx", installed=False))
    assert str(format_prompt(row)).endswith(msg.FORMAT_ENGINE_DOWNLOAD_NO_SIZE)


def test_an_installed_engine_without_a_default_is_shown_without_a_download(
    engines: Callable[[dict[str, list[EngineOption]]], None], hub_fit: list[tuple[str, str]]
) -> None:
    engines(
        {
            "gguf": _LLAMA_NOW,
            "safetensors": [EngineOption(choice.AUTO), EngineOption("mlx", backend="metal")],
        }
    )
    row = format_rows(_model(), 16 * _GIB)[1]
    assert (row.engine, row.missing) == ("mlx", None)


def test_a_format_no_engine_loads_says_no_engine(
    engines: Callable[[dict[str, list[EngineOption]]], None], hub_fit: list[tuple[str, str]]
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": [EngineOption(choice.AUTO)]})
    row = format_rows(_model(), 16 * _GIB)[1]
    assert (row.engine, row.missing, row.size_bytes) == (None, None, None)
    assert msg.FORMAT_NO_ENGINE in str(format_prompt(row))
    assert hub_fit == []


class _Llama(BundledFakeAdapter):
    name = "llama.cpp"
    formats = frozenset({"gguf"})
    served_roles = frozenset(WorkerRole)


class _Openvino(BundledFakeAdapter):
    name = "openvino"
    formats = frozenset({"gguf"})
    served_roles = frozenset(WorkerRole)


@pytest.fixture
def two_installed_engines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Two installed engines that load GGUF. The engine lookup under test is the real one."""
    mac = collect(system="Darwin", machine="arm64", cpu_info={}, nvidia=None)
    monkeypatch.setattr(runnable, "session_host", lambda: mac)
    monkeypatch.setattr(cfg, "models_dir", tmp_path / "models")
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, _Llama())  # type: ignore[arg-type]
    monkeypatch.setitem(choice.ADAPTERS, EngineName.OPENVINO, _Openvino())  # type: ignore[arg-type]


def test_an_installed_default_engine_is_the_engine_of_the_row_through_the_real_lookup(
    two_installed_engines: None,
) -> None:
    row = format_rows(_model(formats=frozenset({ModelFormat.GGUF})), 16 * _GIB)[0]
    assert (row.engine, row.missing) == ("llama.cpp", None)


def test_the_row_names_the_engine_auto_picks_not_the_first_installed_engine(
    two_installed_engines: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cfg, "preferred_engine", EngineSetting.OPENVINO)
    row = format_rows(_model(formats=frozenset({ModelFormat.GGUF})), 16 * _GIB)[0]
    assert (row.engine, row.missing) == ("openvino", None)


def _record_downloads(app: LilbeeAppHost) -> list[CatalogModel]:
    started: list[CatalogModel] = []

    def record(model: CatalogModel, **_kw: object) -> str:
        started.append(model)
        return "task-id"

    app.task_bar.start_download = record  # type: ignore[method-assign]
    return started


async def _open_picker(
    pilot: object, monkeypatch: pytest.MonkeyPatch, model: CatalogModel
) -> tuple[CatalogScreen, list[CatalogModel], list[tuple[str, Callable[[], object]]]]:
    app = pilot.app  # type: ignore[attr-defined]
    screen = CatalogScreen()
    await app.push_screen(screen)
    await pilot.pause()  # type: ignore[attr-defined]
    started = _record_downloads(app)
    installs: list[tuple[str, Callable[[], object]]] = []
    monkeypatch.setattr(screen, "notify", lambda message, **_kw: None)
    monkeypatch.setattr("lilbee.cli.tui.screens.catalog.disk_shortfall", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "lilbee.cli.tui.screens.catalog.install_then",
        lambda _app, name, then: installs.append((name, then)),
    )
    screen._install_model(model)
    assert await wait_until(pilot, lambda: isinstance(app.screen, FormatPicker))  # type: ignore[arg-type]
    return screen, started, installs


async def test_the_picker_lists_one_described_row_for_each_format(
    monkeypatch: pytest.MonkeyPatch,
    engines: Callable[[dict[str, list[EngineOption]]], None],
    hub_fit: list[tuple[str, str]],
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _MLX_MISSING})
    async with _App().run_test(size=(140, 40)) as pilot:
        await _open_picker(pilot, monkeypatch, _model())
        picker = pilot.app.screen
        options = picker.query_one(OptionList)
        prompts = [str(options.get_option_at_index(i).prompt) for i in range(2)]
        assert prompts[0].startswith("gguf (default)")
        assert "■ llama.cpp" in prompts[0]
        assert prompts[1].startswith("safetensors")
        assert prompts[1].endswith("engine download 180 MB")
        title = picker.query_one("#format-title", Static)
        assert str(title.render()) == msg.FORMAT_PICKER_TITLE.format(name="Two")
        assert msg.FORMAT_PICKER_HINT in str(picker.query_one("#format-hint", Static).render())


async def test_a_format_whose_engine_is_missing_installs_the_engine_then_downloads(
    monkeypatch: pytest.MonkeyPatch,
    engines: Callable[[dict[str, list[EngineOption]]], None],
    hub_fit: list[tuple[str, str]],
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _MLX_MISSING})
    async with _App().run_test(size=(140, 40)) as pilot:
        _screen, started, installs = await _open_picker(pilot, monkeypatch, _model())
        await pilot.press("down", "enter")
        assert await wait_until(pilot, lambda: bool(installs))
        assert [name for name, _then in installs] == ["mlx"]
        assert started == []
        installs[0][1]()
        await wait_for_pull_check(pilot.app)
        assert [(m.format, m.pull_ref) for m in started] == [
            (ModelFormat.SAFETENSORS, "acme/two#safetensors")
        ]


async def test_a_format_whose_engine_is_installed_downloads_at_once(
    monkeypatch: pytest.MonkeyPatch,
    engines: Callable[[dict[str, list[EngineOption]]], None],
    hub_fit: list[tuple[str, str]],
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _LLAMA_NOW})
    async with _App().run_test(size=(140, 40)) as pilot:
        _screen, started, installs = await _open_picker(pilot, monkeypatch, _model())
        await pilot.press("down", "enter")
        assert await wait_until(pilot, lambda: bool(started))
        assert installs == []


def test_the_hub_size_request_carries_the_hub_token(
    engines: Callable[[dict[str, list[EngineOption]]], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _LLAMA_NOW})
    seen: list[object] = []

    def fit(_repo: str, _engine: object, **kw: object) -> PullFit:
        seen.append(kw.get("token"))
        return PullFit(device_bytes=_GIB, ctx=4096, size_known=True, upper_bound=False)

    monkeypatch.setattr("lilbee.catalog.hf_client.hf_token", lambda: "tok")
    monkeypatch.setattr("lilbee.catalog.pull_fit.chat_pull_fit", fit)
    format_rows(_model(), 16 * _GIB)
    assert seen == ["tok"]


def test_a_catalog_size_of_zero_is_an_unknown_size(
    engines: Callable[[dict[str, list[EngineOption]]], None], hub_fit: list[tuple[str, str]]
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _LLAMA_NOW})
    row = format_rows(replace(_model(), size_gb=0.0), 16 * _GIB)[0]
    assert (row.size_bytes, row.fit) == (None, None)


def test_a_catalog_size_under_one_gigabyte_is_kept(
    engines: Callable[[dict[str, list[EngineOption]]], None], hub_fit: list[tuple[str, str]]
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _LLAMA_NOW})
    row = format_rows(replace(_model(), size_gb=0.5), 16 * _GIB)[0]
    assert row.size_bytes == int(0.5 * _GIB)


def _span_styles(content: Content, text: str) -> list[str]:
    plain = str(content)
    return [str(s.style) for s in content.spans if plain[s.start : s.end] == text]


def test_the_default_marker_is_muted() -> None:
    row = FormatRow(ModelFormat.GGUF, True, "llama.cpp", None, None)
    assert _span_styles(format_prompt(row), msg.FORMAT_ROW_DEFAULT) == ["$text-muted"]


@pytest.mark.parametrize(
    ("level", "text", "color"),
    [
        (FitLevel.FITS, msg.FORMAT_FIT_FITS, "$success"),
        (FitLevel.TIGHT, msg.FORMAT_FIT_TIGHT, "$warning"),
        (FitLevel.WONT_RUN, msg.FORMAT_FIT_WONT_RUN, "$error"),
    ],
)
def test_the_fit_word_takes_its_colour(level: FitLevel, text: str, color: str) -> None:
    row = FormatRow(ModelFormat.GGUF, False, "llama.cpp", _GIB, level)
    assert _span_styles(format_prompt(row), text) == [color]


async def test_the_fit_is_worked_out_from_the_memory_the_screen_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        "lilbee.cli.tui.screens.catalog.format_rows",
        lambda model, available: asked.append((model, available)) or [],
    )
    monkeypatch.setattr("lilbee.cli.tui.screens.catalog.call_from_thread", lambda *_a, **_kw: None)
    async with _App().run_test(size=(140, 40)) as pilot:
        screen = CatalogScreen()
        await pilot.app.push_screen(screen)
        await pilot.pause()
        screen._available_memory_bytes = 12345
        model = _model()
        screen._offer_formats(model)
    assert asked == [(model, 12345)]


async def test_the_engine_install_gets_the_running_app(
    monkeypatch: pytest.MonkeyPatch,
    engines: Callable[[dict[str, list[EngineOption]]], None],
    hub_fit: list[tuple[str, str]],
) -> None:
    engines({"gguf": _LLAMA_NOW, "safetensors": _MLX_MISSING})
    async with _App().run_test(size=(140, 40)) as pilot:
        screen, _started, _installs = await _open_picker(pilot, monkeypatch, _model())
        apps: list[object] = []
        monkeypatch.setattr(
            "lilbee.cli.tui.screens.catalog.install_then",
            lambda app, _name, _then: apps.append(app),
        )
        await pilot.press("down", "enter")
        assert await wait_until(pilot, lambda: bool(apps))
        assert apps == [screen.app]
        assert apps[0] is pilot.app
