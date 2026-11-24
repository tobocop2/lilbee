"""The stdio MCP server fills an empty chat model once at start, as the HTTP server does.

Each test starts the server through ``main`` and calls tools through the tool entry
points, so a tool that sends an empty model name fails the test.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from lilbee import mcp_server as mcp_mod
from lilbee.app import services as svc_mod
from lilbee.core.config import CONFIG_FILE_NAME, cfg
from lilbee.providers.base import ChatResult, FinishReason
from lilbee.wiki.lazy import UnknownStubError
from tests.conftest import install_fake_model, make_mock_services

UNREACHABLE_CHAT = "ollama/llama3:latest"
MISSING_CHAT = "org/Missing-GGUF/missing-Q4_K_M.gguf"
WIKI_CALLS = (
    ("wiki_build", {}),
    ("wiki_update", {}),
    ("wiki_synthesize", {}),
    ("wiki_generate", {"slug": "ford"}),
)
WIKI_TOOLS = tuple(name for name, _ in WIKI_CALLS)


@pytest.fixture(autouse=True)
def wiki_services(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Services as doubles, the wiki on, and a project folder."""
    monkeypatch.setattr(cfg, "wiki", True)
    monkeypatch.setattr(cfg, "data_root", tmp_path / "project")
    cfg.data_root.mkdir(parents=True)
    svc_mod.set_services(make_mock_services())
    yield
    svc_mod.set_services(None)


@pytest.fixture
def installed_chat() -> str:
    return install_fake_model("org/Installed-Chat-GGUF", "installed-chat-Q4_K_M.gguf", task="chat")


def _start_and_call(calls: tuple[tuple[str, dict[str, Any]], ...]) -> dict[str, Any]:
    """Start the stdio server through ``main``, call each tool, and return its payload."""
    payloads: dict[str, Any] = {}

    async def serve(server: Any) -> None:
        for name, args in calls:
            result = await server.call_tool(name, args)
            payloads[name] = result.structured_content

    with (
        mock.patch.object(mcp_mod, "_serve_stdio", serve),
        mock.patch("lilbee.parent_monitor.parse_parent_pid", return_value=None),
    ):
        mcp_mod.main()
    return payloads


def _call_each_wiki_tool(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Call each model-reading wiki tool by its own call; return the chat model each saw.

    The work functions are doubles that record the model, so these tests prove which model
    the tools run with, not what the real provider answers.
    """
    seen: dict[str, str] = {}
    current = {"tool": ""}

    def record(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        seen[current["tool"]] = cfg.chat_model
        return {}

    def record_stub(*_args: Any, **_kwargs: Any) -> Any:
        seen[current["tool"]] = cfg.chat_model
        raise UnknownStubError("no such page")

    monkeypatch.setattr("lilbee.wiki.run_full_build", record)
    monkeypatch.setattr("lilbee.wiki.run_full_synthesize", record)
    monkeypatch.setattr("lilbee.wiki.lazy.generate_stub_page", record_stub)

    async def serve(server: Any) -> None:
        for name, args in WIKI_CALLS:
            current["tool"] = name
            await server.call_tool(name, args)

    with (
        mock.patch.object(mcp_mod, "_serve_stdio", serve),
        mock.patch("lilbee.parent_monitor.parse_parent_pid", return_value=None),
    ):
        mcp_mod.main()
    return seen


def test_an_empty_chat_model_is_filled_before_each_wiki_tool_runs(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "")

    with caplog.at_level(logging.INFO, logger=mcp_mod.log.name):
        seen = _call_each_wiki_tool(monkeypatch)

    assert seen == dict.fromkeys(WIKI_TOOLS, installed_chat)
    notices = [
        r for r in caplog.records if r.name == mcp_mod.log.name and installed_chat in r.getMessage()
    ]
    assert [r.levelno for r in notices] == [logging.INFO]


def test_the_services_build_after_the_chat_model_is_filled(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "")
    built_with: list[str] = []
    build = mcp_mod.get_services

    def record_the_chat_model() -> Any:
        built_with.append(cfg.chat_model)
        return build()

    monkeypatch.setattr(mcp_mod, "get_services", record_the_chat_model)

    _start_and_call(())

    assert built_with[:1] == [installed_chat]


def test_the_fill_is_never_written_to_the_config_file(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "")

    _call_each_wiki_tool(monkeypatch)

    assert cfg.chat_model == installed_chat
    assert not (cfg.data_root / CONFIG_FILE_NAME).exists()


def test_a_later_settings_set_does_not_write_the_filled_chat_model(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "")

    payloads = _start_and_call((("settings_set", {"updates": {"top_k": 7}}),))

    assert payloads["settings_set"]["updated"] == ["top_k"]
    written = (cfg.data_root / CONFIG_FILE_NAME).read_text(encoding="utf-8")
    assert "top_k" in written
    assert "chat_model" not in written
    assert cfg.chat_model == installed_chat


def test_a_set_and_installed_chat_model_is_kept(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    other = install_fake_model("org/Other-Chat-GGUF", "other-chat-Q4_K_M.gguf", task="chat")
    monkeypatch.setattr(cfg, "chat_model", other)

    seen = _call_each_wiki_tool(monkeypatch)

    assert seen == dict.fromkeys(WIKI_TOOLS, other)


def _assert_kept(monkeypatch: pytest.MonkeyPatch, model: str) -> None:
    """Each tool runs with *model* as set, with the model unreachable or uninstalled."""
    monkeypatch.setattr(cfg, "chat_model", model)
    with (
        mock.patch("lilbee.providers.availability.provider_sdk_available", return_value=True),
        mock.patch(
            "lilbee.modelhub.model_manager.validation._local_server_reachable",
            return_value=False,
        ),
    ):
        seen = _call_each_wiki_tool(monkeypatch)

    assert seen == dict.fromkeys(WIKI_TOOLS, model)


def test_a_set_chat_model_that_is_not_reachable_is_kept(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    _assert_kept(monkeypatch, UNREACHABLE_CHAT)


def test_a_set_chat_model_that_is_not_installed_is_kept(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    _assert_kept(monkeypatch, MISSING_CHAT)


def _write_wiki_page_made_by(model: str) -> None:
    page = cfg.data_root / "wiki" / "summaries" / "doc.md"
    page.parent.mkdir(parents=True)
    page.write_text(f"---\ngenerated_by: {model}\n---\n\n# Doc\n", encoding="utf-8")
    svc_mod.get_services().store.get_citations_for_wiki.return_value = []


def _lint_issue_types_for_a_page_made_by(model: str) -> list[str]:
    _write_wiki_page_made_by(model)
    payloads = _start_and_call((("wiki_lint", {"wiki_source": "wiki/summaries/doc.md"}),))
    return [issue["issue_type"] for issue in payloads["wiki_lint"]["issues"]]


def test_wiki_lint_compares_a_page_with_the_filled_chat_model(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "")

    assert "model_changed" not in _lint_issue_types_for_a_page_made_by(installed_chat)


def test_wiki_lint_still_flags_a_page_made_by_another_model(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "")

    assert "model_changed" in _lint_issue_types_for_a_page_made_by("some/other-model")


def test_wiki_status_lints_against_the_filled_chat_model(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "")
    _write_wiki_page_made_by(installed_chat)

    payloads = _start_and_call((("wiki_status", {}),))

    assert payloads["wiki_status"]["lint_warnings"] == 0


def test_wiki_status_counts_a_page_made_by_another_model(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "")
    _write_wiki_page_made_by("some/other-model")

    payloads = _start_and_call((("wiki_status", {}),))

    assert payloads["wiki_status"]["lint_warnings"] == 1


def test_search_expands_the_query_with_the_filled_chat_model(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "")
    monkeypatch.setattr(cfg, "query_expansion_count", 2)
    monkeypatch.setattr(cfg, "expansion_short_query_tokens", 0)
    monkeypatch.setattr(cfg, "expansion_skip_threshold", 0.0)
    seen: list[str] = []

    def chat(*_args: Any, **_kwargs: Any) -> ChatResult:
        seen.append(cfg.chat_model)
        return ChatResult(text="first\nsecond", tool_calls=(), finish_reason=FinishReason.STOP)

    svc_mod.get_services().provider.chat.side_effect = chat

    _start_and_call((("search", {"query": "how does the gearbox shift between ratios"}),))

    assert seen
    assert set(seen) == {installed_chat}


def test_a_failed_fill_stops_the_start_and_leaves_stdout_empty(
    capfd: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    served = []

    async def serve(server: Any) -> None:
        served.append(server)

    # The CLI sends the log to stderr; stdout carries the MCP protocol.
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        with (
            mock.patch.object(mcp_mod, "fill_empty_chat_model", side_effect=OSError("registry")),
            mock.patch.object(mcp_mod, "_serve_stdio", serve),
            mock.patch("lilbee.parent_monitor.parse_parent_pid", return_value=None),
            pytest.raises(OSError, match="registry"),
        ):
            mcp_mod.main()
    finally:
        root.removeHandler(handler)

    out, err = capfd.readouterr()
    assert not served
    assert out == ""
    assert "The services build failed (the chat-model fill or the build)" in err
    [record] = [r for r in caplog.records if "services build failed" in r.getMessage()]
    assert record.levelno == logging.ERROR
    assert record.exc_info
    assert record.exc_info[0] is OSError
