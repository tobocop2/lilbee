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


def test_a_fleet_of_another_config_root_overflows_on_a_held_write(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    """The held write is real, but the slot's fleet is not this root's: overflow, no error."""
    persistent_settings.save(cfg.data_root, {"chat_model": _CHAT_REF})
    follow_config_one_off(frozenset())
    kwargs = _one_off_provider_kwargs(monkeypatch)
    persistent_settings.save(cfg.data_root, {"chat_model": _OTHER_CHAT_REF})
    monkeypatch.setattr(cfg, "chat_model", _OTHER_CHAT_REF)
    _owner_provider, states = _owner(monkeypatch, tmp_path)
    other_root = tmp_path / "another-root"
    _stamp(states, PlanOrigin(str(other_root), prov_mod._plan_origin().fleet_view))
    monkeypatch.setattr(cfg, "chat_model", _CHAT_REF)
    assert services_mod._held_writes_of_the_process_follower() == {"chat_model": _OTHER_CHAT_REF}
    peer, built = _peer_beside_the_slot(monkeypatch, tmp_path, states, kwargs)
    message = _outcome(peer._ensure_fleet) or (peer._engine_refusal or "")
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
        f"config.toml changed chat_model to {_OTHER_CHAT_REF} while this command ran. "
        "Run the command again."
    )


def test_the_error_names_each_written_key_with_its_own_value() -> None:
    message = prov_mod._held_write_message({"chat_model": _OTHER_CHAT_REF, "chunk_size": 256})
    assert message == (
        f"config.toml changed chat_model to {_OTHER_CHAT_REF}, chunk_size to 256 while this "
        "command ran. Run the command again."
    )


_THIRD_CHAT_REF = "Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q8_0.gguf"


def test_a_slot_fleet_on_a_flagged_model_after_the_write_gets_a_true_error(
    real_planner, monkeypatch, tmp_path: Path
) -> None:
    """The slot runs `ask --model X`, planned after the write: the error claims nothing it serves.

    The fleet carries the file's view, so the one-off stops. The error must not say the
    engine serves the written model, because it serves X.
    """
    persistent_settings.save(cfg.data_root, {"chat_model": _CHAT_REF})
    follow_config_one_off(frozenset())
    kwargs = _one_off_provider_kwargs(monkeypatch)
    persistent_settings.save(cfg.data_root, {"chat_model": _THIRD_CHAT_REF})
    monkeypatch.setattr(cfg, "chat_model", _OTHER_CHAT_REF)  # the other command's flag
    _owner_provider, states = _owner(monkeypatch, tmp_path)
    _stamp(states, prov_mod._plan_origin())
    monkeypatch.setattr(cfg, "chat_model", _CHAT_REF)  # this one-off holds its start model
    peer, built = _peer_beside_the_slot(monkeypatch, tmp_path, states, kwargs)
    message = _outcome(peer._ensure_fleet) or (peer._engine_refusal or "")
    assert built == []
    assert message == (
        f"config.toml changed chat_model to {_THIRD_CHAT_REF} while this command ran. "
        "Run the command again."
    )


def _one_off_follower_on_a(follower_flags: frozenset[str] = frozenset()):
    """A one-off follower on chat_model A in config.toml, as follow_config_one_off builds it."""
    persistent_settings.save(cfg.data_root, {"chat_model": _CHAT_REF})
    follow_config_one_off(follower_flags)
    return services_mod._state.follower


def test_a_process_on_an_override_with_no_write_reports_no_held_write(monkeypatch) -> None:
    """An environment override makes the process run on B while the file says A; no write."""
    follower = _one_off_follower_on_a()
    monkeypatch.setattr(cfg, "chat_model", _OTHER_CHAT_REF)
    assert follower.held_writes() == {}


def test_a_write_of_the_value_the_process_runs_on_is_no_held_write(monkeypatch) -> None:
    """The process runs on B through an override; the file changes from A to B."""
    follower = _one_off_follower_on_a()
    monkeypatch.setattr(cfg, "chat_model", _OTHER_CHAT_REF)
    persistent_settings.save(cfg.data_root, {"chat_model": _OTHER_CHAT_REF})
    assert follower.held_writes() == {}


def test_a_held_write_of_a_new_value_is_reported(monkeypatch) -> None:
    follower = _one_off_follower_on_a()
    monkeypatch.setattr(cfg, "chat_model", _CHAT_REF)
    persistent_settings.save(cfg.data_root, {"chat_model": _OTHER_CHAT_REF})
    assert follower.held_writes() == {"chat_model": _OTHER_CHAT_REF}


def test_the_follower_reports_no_write_of_a_key_a_flag_set(monkeypatch) -> None:
    from lilbee.app.config_follow import ONE_OFF_HOLD, ConfigFileFollower

    persistent_settings.save(cfg.data_root, {"chat_model": _CHAT_REF})
    follower = ConfigFileFollower(
        ONE_OFF_HOLD | {"chat_model"},
        long_running=False,
        keeper="the command",
        flags=frozenset({"chat_model"}),
    )
    monkeypatch.setattr(cfg, "chat_model", _THIRD_CHAT_REF)  # the flag's model
    persistent_settings.save(cfg.data_root, {"chat_model": _OTHER_CHAT_REF})
    assert follower.held_writes() == {}


def test_the_services_report_no_write_of_a_key_a_flag_set(monkeypatch) -> None:
    _one_off_follower_on_a(frozenset({"chat_model"}))
    monkeypatch.setattr(cfg, "chat_model", _THIRD_CHAT_REF)  # the flag's model
    persistent_settings.save(cfg.data_root, {"chat_model": _OTHER_CHAT_REF})
    assert services_mod._held_writes_of_the_process_follower() == {}


def test_the_factory_hands_the_held_writes_to_the_local_engine() -> None:
    from lilbee.core.config.enums import LlmProvider
    from lilbee.providers.factory import create_provider
    from lilbee.providers.routing_provider import RoutingProvider

    def held_writes() -> dict[str, object]:
        return {}

    config = cfg.model_copy(update={"llm_provider": LlmProvider.AUTO})
    provider = create_provider(config, held_writes=held_writes)
    assert isinstance(provider, RoutingProvider)
    assert provider._held_writes is held_writes


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
def test_each_one_off_command_reports_a_held_chat_model_write(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """Each one-off command holds chat_model, so its engine hears of a write during its run.

    The services hand the engine this same question (see the first-start test above).
    """
    monkeypatch.setattr(services_mod._state, "follower", None)
    monkeypatch.setattr("lilbee.providers.factory.create_provider", _refuse_to_build)
    args = [arg.format(doc=cli_env) for arg in _ONE_OFF_COMMANDS[command]]
    runner.invoke(app, args)
    follower = services_mod._state.follower
    assert follower is not None and not follower.long_running, command
    persistent_settings.save(cfg.data_root, {"chat_model": _OTHER_CHAT_REF})
    assert services_mod._held_writes_of_the_process_follower() == {"chat_model": _OTHER_CHAT_REF}


def _refuse_to_build(_config: object, **_kwargs: object) -> None:
    raise _FactorySeen


_ENGINE_CALLS = (
    "embed",
    "count_tokens",
    "embed_token_cap",
    "chat",
    "chat_with_tools",
    "vision_ocr",
    "rerank",
)


def _engine_refuses_on_the_write(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every engine call ends with the held-write error, as the fleet does; returns the calls."""
    calls: list[str] = []

    def refusing_provider(_config: object, **kwargs: object) -> MagicMock:
        # The owner writes chat_model once this command's settings are loaded.
        persistent_settings.save(cfg.data_root, {"chat_model": _OTHER_CHAT_REF})
        held_writes = kwargs["held_writes"]

        def refusing(name: str):
            def refuse(*_a: object, **_kw: object) -> None:
                calls.append(name)
                raise ProviderError(
                    prov_mod._held_write_message(held_writes()),  # type: ignore[operator]
                    kind=ProviderErrorKind.BAD_REQUEST,
                )

            return refuse

        provider = MagicMock()
        for name in _ENGINE_CALLS:
            getattr(provider, name).side_effect = refusing(name)
        provider.max_concurrent_chats.return_value = 1
        provider.supports_rerank.return_value = False
        for name in ("vision_slot_capacity", "served_chat_ctx", "served_chat_slots"):
            getattr(provider, name).return_value = None
        return provider

    monkeypatch.setattr("lilbee.providers.factory.create_provider", refusing_provider)
    monkeypatch.setattr(
        "lilbee.retrieval.embedder.Embedder.validate_model", lambda _self, *_a, **_kw: True
    )
    return calls


def test_add_stops_with_the_clear_error_on_a_chat_model_write(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fleet demand covers chat, so `lilbee add` ends on the write and builds nothing."""
    _engine_refuses_on_the_write(monkeypatch)
    result = runner.invoke(app, ["add", str(cli_env)])
    print(f"add rc={result.exit_code} output={result.output!r}")
    assert result.exit_code == 1, result.output
    assert "changed chat_model to" in result.output and "again" in result.output, result.output


# The commands besides add that reach the engine on this one-document library. ask reaches
# it only with an installed chat model (the real-model end to end covers it), topics never
# starts an engine, and a wiki build logs a failed model call per source and goes on.
_ENGINE_COMMANDS = ("rebuild", "search", "sync")


@pytest.mark.parametrize("command", _ENGINE_COMMANDS)
def test_each_one_off_command_ends_with_the_clear_error(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    """A command that reaches the engine prints the error and exits 1, with no traceback."""
    calls = _engine_refuses_on_the_write(monkeypatch)
    args = [arg.format(doc=cli_env) for arg in _ONE_OFF_COMMANDS[command]]
    result = runner.invoke(app, args)
    assert calls, f"{command} never reached the engine"
    assert result.exit_code == 1, result.output
    assert "changed chat_model to" in result.output and "again" in result.output, result.output
