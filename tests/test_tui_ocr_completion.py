"""Path completion in /add keeps working after the OCR options."""

from types import SimpleNamespace

import pytest

from lilbee.cli.tui.screens.chat import ChatScreen
from lilbee.cli.tui.widgets.autocomplete import get_completions, split_ocr_options


@pytest.fixture
def docs_folder(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("x", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_split_ocr_options_takes_finished_pairs_only():
    assert split_ocr_options("--ocr all --ocr-timeout 5 do") == ("--ocr all --ocr-timeout 5 ", "do")
    assert split_ocr_options("--ocr all") == ("", "--ocr all")
    assert split_ocr_options("docs/x") == ("", "docs/x")


@pytest.mark.parametrize(
    "text",
    ["/add do", "/add --ocr all do", "/add --ocr-timeout 5 --ocr off do"],
)
def test_add_offers_the_path_after_options(docs_folder, text):
    assert get_completions(text) == ["docs/"]


def test_add_lists_the_folder_after_options_with_no_path(docs_folder):
    assert "docs/" in get_completions("/add --ocr all ")


def test_add_inside_a_folder_after_options(docs_folder):
    assert get_completions("/add --ocr all docs/g") == ["guide.md"]


def test_other_path_commands_do_not_skip_options(docs_folder):
    assert get_completions("/export --ocr all do") == []


@pytest.mark.parametrize(
    ("typed", "display", "expected"),
    [
        ("/add --ocr all do", "docs/", "/add --ocr all docs/"),
        ("/add --ocr all docs/g", "guide.md", "/add --ocr all docs/guide.md"),
        ("/add do", "docs/", "/add docs/"),
    ],
)
def test_accepting_a_completion_keeps_the_options(typed, display, expected):
    screen = SimpleNamespace(_completion_origin=typed, _chat_input=SimpleNamespace(value=typed))
    assert ChatScreen._completion_value(screen, display) == expected
