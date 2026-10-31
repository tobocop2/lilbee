"""An embed-role engine change asks first when the index identity says the vectors can differ."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from lilbee.app import engine_service
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.data.store.canary import CanaryState
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.retrieval import canary_probe
from tests.test_engine_service import GGUF_ENGINE, GGUF_REF, OPENVINO_ENGINE

MODEL = "org/embed/embed.gguf"


@pytest.fixture
def with_openvino(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second engine that loads GGUF, so a choice differs from the host default."""
    monkeypatch.setitem(choice.ADAPTERS, EngineName.LLAMA, GGUF_ENGINE)  # type: ignore[arg-type]
    monkeypatch.setitem(choice.ADAPTERS, EngineName.OPENVINO, OPENVINO_ENGINE)  # type: ignore[arg-type]


def _stored(key: str) -> object:
    return settings.load(cfg.data_root).get(key)


class _Store:
    def __init__(self, documents: int, trigger: str | None) -> None:
        self.documents = documents
        self.trigger = trigger

    def source_summary(self, _newest: int) -> SimpleNamespace:
        return SimpleNamespace(document_count=self.documents)

    def canary_state(self) -> CanaryState | None:
        return None if self.trigger is None else CanaryState([[1.0]], self.trigger, True)


@pytest.fixture
def library(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A running library: the embed model resolves to a file and the store holds a baseline."""
    model_file = tmp_path / "embed.gguf"
    model_file.write_bytes(b"x")
    monkeypatch.setattr(canary_probe, "resolve_model_path", lambda *_: model_file)
    monkeypatch.setattr(canary_probe, "engine_build_id", lambda: "build")
    monkeypatch.setattr(cfg, "embedding_model", MODEL)
    monkeypatch.setattr(cfg, "chat_model", GGUF_REF)
    monkeypatch.setattr(cfg, "engine_choices", None)
    monkeypatch.setattr(cfg, "preferred_engine", "auto")
    state = {"store": _Store(3, canary_probe.embedder_trigger(cfg, SimpleNamespace()))}
    services = SimpleNamespace(
        store=state["store"], registry=SimpleNamespace(), provider=mock.Mock()
    )
    monkeypatch.setattr(engine_service, "peek_services", lambda: services)
    return services


def test_an_embed_engine_that_changes_the_trigger_asks_with_the_count(
    library: SimpleNamespace, with_openvino: None
) -> None:
    with pytest.raises(engine_service.EmbedConfirmRequiredError) as caught:
        engine_service.use(MODEL, "openvino")
    assert caught.value.documents == 3
    assert caught.value.library == cfg.data_root.name
    assert _stored("engine_choices") is None
    assert cfg.engine_choices is None


def test_a_confirmed_embed_change_writes(library: SimpleNamespace, with_openvino: None) -> None:
    engine_service.use(MODEL, "openvino", confirmed=True)
    assert json.loads(str(_stored("engine_choices"))) == {MODEL: "openvino"}


def test_a_change_that_keeps_the_trigger_asks_nothing(
    library: SimpleNamespace, with_openvino: None
) -> None:
    engine_service.use(MODEL, "llama.cpp")
    assert json.loads(str(_stored("engine_choices"))) == {MODEL: "llama.cpp"}


def test_a_change_of_a_model_that_is_not_the_embed_model_asks_nothing(
    library: SimpleNamespace, with_openvino: None
) -> None:
    engine_service.use(GGUF_REF, "openvino")
    assert json.loads(str(_stored("engine_choices"))) == {GGUF_REF: "openvino"}


def test_an_empty_library_asks_nothing(library: SimpleNamespace, with_openvino: None) -> None:
    library.store.documents = 0
    engine_service.use(MODEL, "openvino")
    assert json.loads(str(_stored("engine_choices"))) == {MODEL: "openvino"}


def test_a_library_with_no_baseline_asks(library: SimpleNamespace, with_openvino: None) -> None:
    library.store.trigger = None
    with pytest.raises(engine_service.EmbedConfirmRequiredError):
        engine_service.use(MODEL, "llama.cpp")


def test_prefer_asks_when_it_changes_the_embed_engine(
    library: SimpleNamespace, with_openvino: None
) -> None:
    with pytest.raises(engine_service.EmbedConfirmRequiredError):
        engine_service.prefer("openvino")
    assert _stored("preferred_engine") is None
    assert engine_service.prefer("openvino", confirmed=True) == "openvino"


def test_prefer_that_keeps_the_embed_engine_asks_nothing(
    library: SimpleNamespace, with_openvino: None
) -> None:
    assert engine_service.prefer("llama.cpp") == "llama.cpp"


def test_the_trigger_names_a_non_default_embed_engine_only(
    library: SimpleNamespace, with_openvino: None
) -> None:
    default = canary_probe.embedder_trigger(cfg, library.registry)
    cfg_other = cfg.model_copy(update={"engine_choices": json.dumps({MODEL: "openvino"})})
    other = canary_probe.embedder_trigger(cfg_other, library.registry)
    assert "engine=" not in default
    assert other == default + "|engine=openvino"
    assert choice.ADAPTERS[EngineName.LLAMA] is GGUF_ENGINE
    assert choice.ADAPTERS[EngineName.OPENVINO] is OPENVINO_ENGINE


from lilbee import mcp_server  # noqa: E402
from tests.test_engine_surfaces import _isolated, _rest, _run_cli  # noqa: E402, F401

_CHOSEN = {MODEL: "openvino"}


def test_rest_answers_409_with_the_library_and_count_then_writes_when_confirmed(
    library: SimpleNamespace, with_openvino: None
) -> None:
    body = {"model": MODEL, "engine": "openvino"}
    code, answer = _rest("PUT", "/api/engines/roles/embed", body)
    assert code == 409
    assert answer["extra"] == {
        "code": "embed_engine_confirm_required",
        "library": cfg.data_root.name,
        "documents": 3,
    }
    assert "confirm_embed_change" in answer["detail"]
    assert _stored("engine_choices") is None
    code, _ = _rest("PUT", "/api/engines/roles/embed", {**body, "confirm_embed_change": True})
    assert code == 200
    assert json.loads(str(_stored("engine_choices"))) == _CHOSEN


def test_rest_prefer_asks_the_same_question(library: SimpleNamespace, with_openvino: None) -> None:
    code, answer = _rest("PUT", "/api/engines/preferred", {"engine": "openvino"})
    assert (code, answer["extra"]["documents"]) == (409, 3)
    assert _stored("preferred_engine") is None


def test_mcp_returns_its_own_shape_then_writes_when_confirmed(
    library: SimpleNamespace, with_openvino: None
) -> None:
    from lilbee.providers.roles import WorkerRole

    result = mcp_server.engine_use_tool(WorkerRole.EMBED, MODEL, "openvino")
    assert result["code"] == "embed_engine_confirm_required"
    assert (result["library"], result["documents"]) == (cfg.data_root.name, 3)
    assert "confirm_embed_change" in result["error"]
    assert _stored("engine_choices") is None
    done = mcp_server.engine_use_tool(
        WorkerRole.EMBED, MODEL, "openvino", confirm_embed_change=True
    )
    assert done == {"engine": "openvino"}
    asked = mcp_server.engine_prefer_tool("llama.cpp")
    assert asked["code"] == "embed_engine_confirm_required"
    assert mcp_server.engine_prefer_tool("llama.cpp", True) == {"engine": "llama.cpp"}


def test_the_cli_refuses_without_a_terminal_and_names_the_flag(
    library: SimpleNamespace, with_openvino: None
) -> None:
    code, body = _run_cli("use", "embed", MODEL, "openvino")
    assert code == 1
    assert "--yes" in body["error"]
    assert _stored("engine_choices") is None
    code, body = _run_cli("use", "embed", MODEL, "openvino", "--yes")
    assert (code, body["engine"]) == (0, "openvino")
    assert json.loads(str(_stored("engine_choices"))) == _CHOSEN


def test_the_cli_asks_on_a_terminal_and_a_refusal_leaves_the_file_unchanged(
    library: SimpleNamespace, with_openvino: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from lilbee.cli import app as cli_app

    monkeypatch.setattr("lilbee.cli.engine._interactive", lambda: True)
    result = CliRunner().invoke(cli_app, ["engine", "use", "embed", MODEL, "openvino"], input="n\n")
    assert result.exit_code == 1
    assert "The engine is unchanged." in result.output.replace("\n", " ")
    assert _stored("engine_choices") is None


def test_the_slash_command_refuses_and_leaves_the_file_unchanged(
    library: SimpleNamespace, with_openvino: None
) -> None:
    from lilbee.cli.tui.engine_commands import run_engine_command

    out = run_engine_command(f"use embed {MODEL} openvino")
    assert out.ok is False
    assert "3 documents" in out.text
    assert out.text.endswith("Run the command again with --yes.")
    assert _stored("engine_choices") is None


def test_the_slash_command_writes_with_the_yes_word(
    library: SimpleNamespace, with_openvino: None
) -> None:
    from lilbee.cli.tui.engine_commands import run_engine_command

    out = run_engine_command(f"use embed {MODEL} openvino --yes")
    assert out.ok is True
    assert json.loads(str(_stored("engine_choices"))) == {MODEL: "openvino"}


def test_the_slash_command_prefer_writes_with_the_yes_word(
    library: SimpleNamespace, with_openvino: None
) -> None:
    from lilbee.cli.tui.engine_commands import run_engine_command

    assert run_engine_command("prefer openvino").ok is False
    assert _stored("preferred_engine") is None
    assert run_engine_command("prefer --yes openvino").ok is True
    assert _stored("preferred_engine") == "openvino"
