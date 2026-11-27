"""Each engine has a color mark in every theme, and the Status screen uses it."""

from __future__ import annotations

from itertools import combinations
from unittest.mock import MagicMock

import pytest
from textual.color import Color
from textual.theme import BUILTIN_THEMES
from textual.widgets import Static

from lilbee.app.services import set_services
from lilbee.app.status import ServingStatus
from lilbee.app.themes import DARK_THEMES
from lilbee.cli.tui import engine_mark as marks
from lilbee.cli.tui.screens import status as status_screen
from lilbee.cli.tui.screens.status import StatusScreen
from lilbee.data.store import SourceSummary
from lilbee.modelhub.model_info import ModelArchInfo
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.roles import WorkerRole
from tests._async_wait import wait_for_widget
from tests._lilbee_app_test_host import LilbeeAppHost
from tests.conftest import make_mock_services

# The least distance in RGB between two engine colors, and between an engine color and a
# fit or warning color of the theme.
MIN_APART = 30.0
MIN_FROM_SEMANTIC = 40.0
SEMANTIC = ("success", "warning", "error")


def _theme_variables(theme: str) -> dict[str, str]:
    return BUILTIN_THEMES[theme].to_color_system().generate()


def _apart(first: Color, second: Color) -> float:
    return sum((a - b) ** 2 for a, b in zip(first.rgb, second.rgb, strict=True)) ** 0.5


def test_every_engine_has_a_hue() -> None:
    assert set(marks.ENGINE_HUES) == set(EngineName)


def test_the_variable_name_is_the_engine_id_in_kebab_case() -> None:
    assert marks.variable_name("llama.cpp") == "engine-llama-cpp"
    assert marks.variable_name("ik_llama.cpp") == "engine-ik-llama-cpp"
    assert marks.variable_name("tei") == "engine-tei"


@pytest.mark.parametrize("theme", DARK_THEMES)
def test_a_theme_defines_one_variable_for_each_engine(theme: str) -> None:
    variables = marks.engine_variables(_theme_variables(theme))
    assert set(variables) == {marks.variable_name(engine) for engine in EngineName}


@pytest.mark.parametrize("theme", DARK_THEMES)
def test_each_engine_color_reads_against_the_theme(theme: str) -> None:
    theme_variables = _theme_variables(theme)
    for name, value in marks.engine_variables(theme_variables).items():
        for surface in ("background", "surface"):
            ratio = marks.contrast(Color.parse(value), Color.parse(theme_variables[surface]))
            assert ratio >= marks.MIN_CONTRAST, f"{theme} {name} on {surface}: {ratio:.2f}"


@pytest.mark.parametrize("theme", DARK_THEMES)
def test_no_two_engines_share_a_color_in_a_theme(theme: str) -> None:
    colors = {
        name: Color.parse(value)
        for name, value in marks.engine_variables(_theme_variables(theme)).items()
    }
    for (first, one), (second, two) in combinations(colors.items(), 2):
        assert _apart(one, two) >= MIN_APART, f"{theme}: {first} and {second}"


@pytest.mark.parametrize("theme", DARK_THEMES)
def test_an_engine_color_is_never_a_fit_or_warning_color(theme: str) -> None:
    theme_variables = _theme_variables(theme)
    for name, value in marks.engine_variables(theme_variables).items():
        for key in SEMANTIC:
            gap = _apart(Color.parse(value), Color.parse(theme_variables[key]))
            assert gap >= MIN_FROM_SEMANTIC, f"{theme}: {name} against {key}"


def test_the_mark_takes_the_engine_color_and_the_name_keeps_the_text_color() -> None:
    content = marks.engine_mark("mlx")
    assert content.plain == f"{marks.MARK} mlx"
    colored = [(span.start, span.end, str(span.style)) for span in content.spans]
    assert colored == [(0, 1, "$engine-mlx")]


def test_an_engine_the_table_does_not_know_gets_the_muted_mark() -> None:
    content = marks.engine_mark("future.rs")
    assert content.plain == f"{marks.MARK} future.rs"
    assert [str(span.style) for span in content.spans] == [marks.UNKNOWN_ENGINE_COLOR]


def test_the_app_carries_the_engine_variables_for_the_theme_it_shows() -> None:
    app = LilbeeAppHost()
    for theme in DARK_THEMES:
        app.theme = theme
        shown = app.get_css_variables()
        assert shown["engine-llama-cpp"] == marks.engine_variables(shown)["engine-llama-cpp"]
        assert set(marks.engine_variables(shown)) <= set(shown)


def _serving(engine: str, version: str) -> list[ServingStatus]:
    return [ServingStatus(role=WorkerRole.CHAT, model="m", engine=engine, version=version)]


def test_the_status_engine_row_names_the_engine_with_its_mark(monkeypatch) -> None:
    monkeypatch.setattr(status_screen, "serving_roles", lambda: _serving("tei", "1.9"))
    content = status_screen._build_config_content()
    row = "Chat engine".ljust(status_screen._KV_LABEL_WIDTH) + f"{marks.MARK} tei 1.9"
    assert row in content.plain
    assert "$engine-tei" in {str(span.style) for span in content.spans}


class _StatusApp(LilbeeAppHost):
    CSS = ""

    def on_mount(self) -> None:
        self.push_screen(StatusScreen())


async def test_the_status_screen_shows_the_mark_beside_the_engine(monkeypatch) -> None:
    store = MagicMock()
    store.get_sources.return_value = []
    store.count_sources.return_value = 0
    store.source_summary.return_value = SourceSummary(0, 0, {}, [])
    set_services(make_mock_services(store=store))
    monkeypatch.setattr(status_screen, "held_out_sources", lambda: ([], 0))
    monkeypatch.setattr(status_screen, "serving_roles", lambda: _serving("mlx", "0.3"))
    app = _StatusApp()
    try:
        async with app.run_test(size=(120, 40)) as pilot:
            info = await wait_for_widget(pilot, app, "#config-info", Static)
            assert f"{marks.MARK} mlx 0.3" in str(info.render())
            assert (
                app.theme_variables["engine-mlx"]
                == marks.engine_variables(app.theme_variables)["engine-mlx"]
            )
    finally:
        set_services(None)


def test_the_handler_pill_does_not_use_the_accent() -> None:
    info = ModelArchInfo(chat_arch="llama", embed_arch="bert", active_handler="llama.cpp")
    content = status_screen._build_arch_content(info)
    styles = {str(span.style) for span in content.spans}
    assert not any("$accent" in style for style in styles)
    assert not any("$engine-" in style for style in styles)
