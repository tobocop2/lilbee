"""The stdio MCP server fills an empty chat model once at start, as the HTTP server does.

Each test starts the server through ``main`` and calls the wiki tools through the tool
entry points, so a tool that sends an empty model name fails the test.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from lilbee import mcp_server as mcp_mod
from lilbee.app import services as svc_mod
from lilbee.core.config import CONFIG_FILE_NAME, cfg
from tests.conftest import install_fake_model, make_mock_services

UNREACHABLE_CHAT = "ollama/llama3:latest"
WIKI_TOOLS = ("wiki_build", "wiki_update", "wiki_synthesize", "wiki_generate")


@pytest.fixture(autouse=True)
def wiki_services(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Services as doubles, the wiki on, and one installed chat model."""
    monkeypatch.setattr(cfg, "wiki", True)
    monkeypatch.setattr(cfg, "data_root", tmp_path / "project")
    cfg.data_root.mkdir(parents=True)
    svc_mod.set_services(make_mock_services())
    yield
    svc_mod.set_services(None)


@pytest.fixture
def installed_chat() -> str:
    return install_fake_model("org/Installed-Chat-GGUF", "installed-chat-Q4_K_M.gguf", task="chat")


def _model_seen_by_each_wiki_tool(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Start the stdio server, call every model-reading wiki tool, and record the model."""
    seen: dict[str, str] = {}

    def spy(name: str) -> Callable[..., Any]:
        def record(*_args: Any, **_kwargs: Any) -> Any:
            seen[name] = cfg.chat_model
            return {} if name != "wiki_generate" else None

        return record

    monkeypatch.setattr("lilbee.wiki.run_full_build", spy("wiki_build"))
    monkeypatch.setattr("lilbee.wiki.run_full_synthesize", spy("wiki_synthesize"))
    monkeypatch.setattr("lilbee.wiki.lazy.generate_stub_page", spy("wiki_generate"))

    async def serve(server: Any) -> None:
        for tool, args in (
            ("wiki_build", {}),
            ("wiki_update", {}),
            ("wiki_synthesize", {}),
            ("wiki_generate", {"slug": "ford"}),
        ):
            await server.call_tool(tool, args)

    with (
        mock.patch.object(mcp_mod, "_serve_stdio", serve),
        mock.patch("lilbee.parent_monitor.parse_parent_pid", return_value=None),
    ):
        mcp_mod.main()
    seen["wiki_update"] = seen["wiki_build"]
    return seen


def test_an_empty_chat_model_is_filled_before_the_wiki_tools_run(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "")

    seen = _model_seen_by_each_wiki_tool(monkeypatch)

    assert {seen[tool] for tool in WIKI_TOOLS} == {installed_chat}


def test_the_fill_is_never_written_to_the_config_file(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "")
    path = cfg.data_root / CONFIG_FILE_NAME

    _model_seen_by_each_wiki_tool(monkeypatch)

    assert cfg.chat_model == installed_chat
    assert not path.exists()


def test_a_set_and_installed_chat_model_is_kept(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    other = install_fake_model("org/Other-Chat-GGUF", "other-chat-Q4_K_M.gguf", task="chat")
    monkeypatch.setattr(cfg, "chat_model", other)

    seen = _model_seen_by_each_wiki_tool(monkeypatch)

    assert {seen[tool] for tool in WIKI_TOOLS} == {other}


def test_a_set_chat_model_that_is_not_reachable_is_kept(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    monkeypatch.setattr(cfg, "chat_model", UNREACHABLE_CHAT)

    with (
        mock.patch("lilbee.providers.availability.provider_sdk_available", return_value=True),
        mock.patch(
            "lilbee.modelhub.model_manager.validation._local_server_reachable",
            return_value=False,
        ),
    ):
        seen = _model_seen_by_each_wiki_tool(monkeypatch)

    assert {seen[tool] for tool in WIKI_TOOLS} == {UNREACHABLE_CHAT}


def test_a_set_chat_model_that_is_not_installed_is_kept(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    missing = "org/Missing-GGUF/missing-Q4_K_M.gguf"
    monkeypatch.setattr(cfg, "chat_model", missing)

    seen = _model_seen_by_each_wiki_tool(monkeypatch)

    assert {seen[tool] for tool in WIKI_TOOLS} == {missing}


def _lint_message_for_a_page_made_by(model: str) -> list[str]:
    """Start the stdio server, lint one wiki page made by *model*, return its issue types."""
    page = cfg.data_root / "wiki" / "summaries" / "doc.md"
    page.parent.mkdir(parents=True)
    page.write_text(f"---\ngenerated_by: {model}\n---\n\n# Doc\n")
    svc_mod.get_services().store.get_citations_for_wiki.return_value = []
    result: dict[str, Any] = {}

    async def serve(server: Any) -> None:
        call = await server.call_tool("wiki_lint", {"wiki_source": "wiki/summaries/doc.md"})
        result.update(call.structured_content)

    with (
        mock.patch.object(mcp_mod, "_serve_stdio", serve),
        mock.patch("lilbee.parent_monitor.parse_parent_pid", return_value=None),
    ):
        mcp_mod.main()
    return [issue["issue_type"] for issue in result["issues"]]


def test_wiki_lint_compares_a_page_with_the_filled_chat_model(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "")

    assert "model_changed" not in _lint_message_for_a_page_made_by(installed_chat)


def test_wiki_lint_still_flags_a_page_made_by_another_model(
    monkeypatch: pytest.MonkeyPatch, installed_chat: str
) -> None:
    monkeypatch.setattr(cfg, "chat_model", "")

    assert "model_changed" in _lint_message_for_a_page_made_by("some/other-model")


def test_a_failed_fill_warns_on_stderr_and_leaves_stdout_empty(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    served = []

    async def serve(server: Any) -> None:
        served.append(server)

    # The CLI sends the log to stderr at WARNING; stdout carries the MCP protocol.
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        with (
            mock.patch.object(mcp_mod, "fill_empty_chat_model", side_effect=OSError("registry")),
            mock.patch.object(mcp_mod, "_serve_stdio", serve),
            mock.patch("lilbee.parent_monitor.parse_parent_pid", return_value=None),
        ):
            mcp_mod.main()
    finally:
        root.removeHandler(handler)

    out, err = capfd.readouterr()
    assert served
    assert out == ""
    assert "The chat model could not be filled at start (registry)" in err
