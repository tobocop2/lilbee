"""A one-off command never starts a silent second fleet over a held config.toml write.

A one-off command (ask, add, sync, wiki, search) starts on the chat_model config.toml sets.
While it runs, the owner of the same config root (serve, the stdio MCP server or the TUI)
takes a chat_model write and restarts its fleet on the new model. The one-off holds its
start model, so it cannot bind that fleet. It must not build a private fleet beside it: it
ends with a clear error that names the written key and says to run the command again. This
holds at its first engine start and at its one retry.

A fleet of another config root, and a model this process chose itself (a flag), still
overflow to the private engine dir.

The peer is built with the keyword arguments that the services of a one-off process pass
the provider factory, so the tests do not depend on how the engine is wired.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from lilbee.app import services as services_mod
from lilbee.app.services import follow_config_one_off, get_services, set_services
from lilbee.cli import app
from lilbee.core import settings as persistent_settings
from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet import provider as prov_mod
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.provider import FleetProvider, _launches_by_group
from lilbee.providers.fleet.swap_manager import PlanOrigin, SwapState
from tests._fleet_engines import engines_for_all_roles
from tests.test_fleet_provider import _connection_error, _FakeSwap
from tests.test_fleet_reload_plan import _CHAT_REF, _OTHER_CHAT_REF, real_planner
from tests.test_fleet_settings_bind import _owner, _record

__all__ = ["real_planner"]

runner = CliRunner()


class _FactorySeen(Exception):
    """Stops the services build once the factory's arguments are known."""


def _one_off_provider_kwargs(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """The keyword arguments the services of a one-off process give the fleet provider."""
    set_services(None)
    seen: dict[str, object] = {}

    def capture(_config: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise _FactorySeen

    monkeypatch.setattr("lilbee.providers.factory.create_provider", capture)
    with pytest.raises(_FactorySeen):
        get_services()
    params = inspect.signature(FleetProvider.__init__).parameters
    return {k: v for k, v in seen.items() if k in params and k != "hold_warm"}


def _stamp(states: dict, origin: PlanOrigin) -> None:
    for group, state in list(states.items()):
        states[group] = dataclasses.replace(state, origin=origin)


def _peer_beside_the_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, states: dict, kwargs: dict
) -> tuple[FleetProvider, list[Path]]:
    """A one-off peer: the machine slot is *states* with a live user; the private dir is free."""
    monkeypatch.setattr(prov_mod, "machine_engine_dir", lambda: tmp_path)
    monkeypatch.setattr(prov_mod, "kernel_arbitrates_locks", lambda _d: True)
    monkeypatch.setattr(prov_mod, "live_users_exist", lambda d, **_kw: Path(d) == tmp_path)
    monkeypatch.setattr(
        prov_mod, "_healthy_states", lambda d, **_kw: dict(states) if Path(d) == tmp_path else {}
    )
    monkeypatch.setattr(prov_mod, "reap_stale", lambda _d, **_kw: None)
    monkeypatch.setattr(prov_mod, "engine_record_exists", lambda _d: False)
    monkeypatch.setattr(prov_mod, "_watch_bound_engine", lambda _d: None)
    monkeypatch.setattr(prov_mod, "_can_build_engine", lambda _w, _e: True)
    monkeypatch.setattr(prov_mod, "stop_engine", lambda _d: pytest.fail("stopped the owner fleet"))
    monkeypatch.setattr(prov_mod, "_OWNER_CATCH_UP_WAIT_S", 0.2)

    class _BindingSwap(_FakeSwap):
        def bind(self, _state: SwapState) -> bool:
            self.bound = True
            return True

    monkeypatch.setattr(prov_mod, "SwapManager", lambda _dir, _group: _BindingSwap())
    peer = FleetProvider(**kwargs)
    monkeypatch.setattr(peer, "_hold_membership", lambda _dir: None)
    built: list[Path] = []
    monkeypatch.setattr(peer, "_plan_and_spawn", lambda d: built.append(d) or True)
    return peer, built


def _outcome(run) -> str:
    """The text the one-off ends with: a raised error, else the engine's recorded refusal."""
    try:
        run()
    except Exception as exc:
        return str(exc)
    return ""


def _owner_switched_chat_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, states: dict):
    """The owner writes chat_model to config.toml, then restarts its fleet on it."""
    persistent_settings.save(cfg.data_root, {"chat_model": _OTHER_CHAT_REF})
    monkeypatch.setattr(cfg, "chat_model", _OTHER_CHAT_REF)
    for group, launches in _launches_by_group(
        planning_mod.plan_all_launches(engines=engines_for_all_roles(LLAMA))
    ).items():
        _record(states, group, list(launches))
    _stamp(states, prov_mod._plan_origin())
    # The one-off holds its start model.
    monkeypatch.setattr(cfg, "chat_model", _CHAT_REF)


def test_first_start_after_a_held_chat_model_write_ends_with_a_clear_error(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    persistent_settings.save(cfg.data_root, {"chat_model": _CHAT_REF})
    follow_config_one_off(frozenset())
    kwargs = _one_off_provider_kwargs(monkeypatch)
    _owner_provider, states = _owner(monkeypatch, tmp_path)
    _owner_switched_chat_model(monkeypatch, tmp_path, states)
    peer, built = _peer_beside_the_slot(monkeypatch, tmp_path, states, kwargs)
    message = _outcome(peer._ensure_fleet) or (peer._engine_refusal or "")
    print(f"first-start built={built} message={message!r}")
    assert built == [], "the one-off built a private fleet beside the owner's (first start)"
    assert "chat_model" in message and "again" in message, message


def test_retry_after_a_held_chat_model_write_ends_with_a_clear_error(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    persistent_settings.save(cfg.data_root, {"chat_model": _CHAT_REF})
    follow_config_one_off(frozenset())
    kwargs = _one_off_provider_kwargs(monkeypatch)
    _owner_provider, states = _owner(monkeypatch, tmp_path)
    _stamp(states, prov_mod._plan_origin())
    peer, built = _peer_beside_the_slot(monkeypatch, tmp_path, states, kwargs)
    assert peer._ensure_fleet() is True and built == [], "the peer did not bind at the start"
    _owner_switched_chat_model(monkeypatch, tmp_path, states)
    tries = {"n": 0}

    def call() -> bool:
        tries["n"] += 1
        if tries["n"] == 1:
            raise _connection_error()
        return peer._ensure_fleet()

    message = _outcome(lambda: peer._with_rediscover(call)) or (peer._engine_refusal or "")
    print(f"retry built={built} message={message!r} chat_model={cfg.chat_model}")
    assert built == [], "the one-off built a private fleet beside the owner's (retry)"
    assert "chat_model" in message and "again" in message, message


def test_a_fleet_of_another_config_root_still_overflows(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    persistent_settings.save(cfg.data_root, {"chat_model": _CHAT_REF})
    follow_config_one_off(frozenset())
    kwargs = _one_off_provider_kwargs(monkeypatch)
    monkeypatch.setattr(cfg, "chat_model", _OTHER_CHAT_REF)
    _owner_provider, states = _owner(monkeypatch, tmp_path)
    _stamp(states, PlanOrigin(str(tmp_path / "another-root"), "{}"))
    monkeypatch.setattr(cfg, "chat_model", _CHAT_REF)
    peer, built = _peer_beside_the_slot(monkeypatch, tmp_path, states, kwargs)
    message = _outcome(peer._ensure_fleet)
    assert message == "", message
    assert len(built) == 1 and built[0] != tmp_path, built


def test_a_model_this_command_chose_by_flag_still_overflows(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    """`lilbee ask --model X` beside a serve on the config.toml model: a private fleet, no error."""
    persistent_settings.save(cfg.data_root, {"chat_model": _CHAT_REF})
    follow_config_one_off(frozenset({"chat_model"}))
    kwargs = _one_off_provider_kwargs(monkeypatch)
    _owner_provider, states = _owner(monkeypatch, tmp_path)
    _stamp(states, prov_mod._plan_origin())
    monkeypatch.setattr(cfg, "chat_model", _OTHER_CHAT_REF)
    peer, built = _peer_beside_the_slot(monkeypatch, tmp_path, states, kwargs)
    message = _outcome(peer._ensure_fleet)
    assert message == "", message
    assert len(built) == 1 and built[0] != tmp_path, built


def test_the_error_names_the_written_key_and_its_new_value() -> None:
    message = prov_mod._held_write_message({"chat_model": _OTHER_CHAT_REF})
    assert message == (
        f"chat_model changed to {_OTHER_CHAT_REF} in config.toml while this command ran, "
        "and the running engine now serves it. Run the command again."
    )


def test_a_fleet_planned_on_a_flag_is_not_stamped_with_the_file_view(monkeypatch) -> None:
    """`ask --model X` plans X while config.toml names another model: its fleet is not current."""
    persistent_settings.save(cfg.data_root, {"chat_model": _CHAT_REF})
    monkeypatch.setattr(cfg, "chat_model", _OTHER_CHAT_REF)
    stamped = prov_mod._plan_origin()
    assert stamped != prov_mod._file_origin()
    assert json.loads(stamped.fleet_view or "{}") == {"chat_model": _OTHER_CHAT_REF}


def test_a_long_running_process_reports_no_held_write() -> None:
    """serve, the TUI and the stdio MCP server take or keep a write; none ends them."""
    persistent_settings.save(cfg.data_root, {"chat_model": _CHAT_REF})
    follower = services_mod.follow_config_long_running(frozenset(), keeper="the TUI")
    persistent_settings.save(cfg.data_root, {"chat_model": _OTHER_CHAT_REF})
    assert follower.held_writes() == {}


def test_a_torn_config_toml_reports_no_held_write(monkeypatch) -> None:
    follow_config_one_off(frozenset())
    monkeypatch.setattr(persistent_settings, "read_persisted", lambda _root: (None, None))
    assert services_mod._held_writes_of_the_process_follower() == {}


def test_a_process_with_no_follower_reports_no_held_write(monkeypatch) -> None:
    monkeypatch.setattr(services_mod._state, "follower", None)
    assert services_mod._held_writes_of_the_process_follower() == {}


# --- The one-off CLI surfaces ---------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A data root with one document and config.toml on the start chat model."""
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    snapshot = cfg.model_copy()
    cfg.data_root = tmp_path
    cfg.documents_dir = tmp_path / "documents"
    cfg.documents_dir.mkdir(exist_ok=True)
    cfg.data_dir = tmp_path / "data"
    cfg.data_dir.mkdir(exist_ok=True)
    cfg.lancedb_dir = tmp_path / "data" / "lancedb"
    cfg.worker_pool_eager_start = False
    cfg.wiki = True
    persistent_settings.save(cfg.data_root, {"chat_model": _CHAT_REF})
    cfg.chat_model = _CHAT_REF
    doc = tmp_path / "note.md"
    doc.write_text("# Bees\n\nBees make honey.\n", encoding="utf-8")
    set_services(None)
    yield doc
    services_mod.reset_services()
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


_ONE_OFF_COMMANDS = {
    "ask": ["ask", "what do bees make"],
    "search": ["search", "bees"],
    "add": ["add", "{doc}"],
    "sync": ["sync"],
    "rebuild": ["rebuild"],
    "wiki": ["wiki", "build"],
    "topics": ["topics"],
}


@pytest.mark.parametrize("command", sorted(_ONE_OFF_COMMANDS))
def test_each_one_off_command_hands_the_engine_its_held_writes(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """Each one-off command's engine hears of a chat_model write made during its run."""
    seen: dict[str, object] = {}

    def capture(_config: object, **kwargs: object) -> None:
        seen.update(kwargs)
        raise _FactorySeen

    monkeypatch.setattr("lilbee.providers.factory.create_provider", capture)
    args = [arg.format(doc=cli_env) for arg in _ONE_OFF_COMMANDS[command]]
    runner.invoke(app, args)
    held_writes = seen.get("held_writes")
    assert callable(held_writes), f"{command} gave the engine no held-write check"
    persistent_settings.save(cfg.data_root, {"chat_model": _OTHER_CHAT_REF})
    assert held_writes() == {"chat_model": _OTHER_CHAT_REF}


def test_add_stops_with_the_clear_error_on_a_chat_model_write(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fleet demand covers chat, so `lilbee add` ends on the write and builds nothing."""

    def refusing_provider(_config: object, **kwargs: object) -> MagicMock:
        # The owner writes chat_model once this command's settings are loaded.
        persistent_settings.save(cfg.data_root, {"chat_model": _OTHER_CHAT_REF})
        held_writes = kwargs["held_writes"]

        def refuse(*_a: object, **_kw: object) -> None:
            raise ProviderError(
                prov_mod._held_write_message(held_writes()),  # type: ignore[operator]
                kind=ProviderErrorKind.BAD_REQUEST,
            )

        provider = MagicMock()
        for name in ("embed", "count_tokens", "chat", "vision_ocr", "rerank"):
            getattr(provider, name).side_effect = refuse
        return provider

    monkeypatch.setattr("lilbee.providers.factory.create_provider", refusing_provider)
    monkeypatch.setattr(
        "lilbee.retrieval.embedder.Embedder.validate_model", lambda _self, *_a, **_kw: True
    )
    result = runner.invoke(app, ["add", str(cli_env)])
    print(f"add rc={result.exit_code} output={result.output!r}")
    assert result.exit_code == 1, result.output
    assert "chat_model changed to" in result.output and "again" in result.output, result.output
