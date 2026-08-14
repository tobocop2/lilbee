"""A launched agent gets a chat window large enough for its baseline prompt.

Agent clients (opencode, hermes) open with a big system prompt + tool schemas
and reserve output tokens, so the RAM-derived default chat context (24576 below
128 GiB hosts) overflows on the first turn. ``run_launcher`` raises the
served-context target to an agent floor before the server spawns, and warns when
it must reuse a running server whose window it cannot raise; these tests pin that.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import typer

from lilbee.core.config import cfg


def test_agent_floor_raises_a_small_configured_target() -> None:
    from lilbee.app.agent_configs.window import AGENT_CHAT_CTX_FLOOR, agent_chat_ctx_target

    assert agent_chat_ctx_target(24576) == AGENT_CHAT_CTX_FLOOR


def test_agent_floor_never_lowers_a_larger_configured_target() -> None:
    from lilbee.app.agent_configs.window import agent_chat_ctx_target

    assert agent_chat_ctx_target(131072) == 131072


def test_agent_floor_is_at_least_the_hermes_minimum() -> None:
    # hermes refuses to start under this window, so the generic agent floor must
    # clear it or a hermes launch is dead on arrival.
    from lilbee.app.agent_configs.window import AGENT_CHAT_CTX_FLOOR
    from lilbee.cli.launchers.hermes import _HERMES_MIN_CTX

    assert AGENT_CHAT_CTX_FLOOR >= _HERMES_MIN_CTX


def _run_launcher_capturing(monkeypatch, *, served_ctx: int | None = None) -> dict:
    """Drive run_launcher with every side effect stubbed, capturing the spawn state.

    ``served_ctx`` is what the (reused) server reports as its live chat window;
    the stub keeps the reused-window probe off the network.
    """
    from lilbee.cli.launchers import launcher as launcher_mod

    captured: dict = {}

    def fake_ensure(*, env_overrides=None):
        captured["env_overrides"] = env_overrides
        captured["cfg_target_at_spawn"] = cfg.chat_n_ctx_target
        return ("tok", 8080), None

    monkeypatch.setattr(launcher_mod, "ensure_server_running", fake_ensure)
    monkeypatch.setattr(launcher_mod, "served_chat_ctx", lambda _p: served_ctx)
    monkeypatch.setattr(launcher_mod, "installed_chat_model_refs", lambda: [])
    monkeypatch.setattr(launcher_mod, "wait_for_chat_warm", lambda _p: True)
    monkeypatch.setattr(launcher_mod.subprocess, "run", lambda *_a, **_k: MagicMock(returncode=0))

    fake = MagicMock()
    fake.name = "opencode"
    fake.find_binary.return_value = "/usr/bin/opencode"
    fake.prepare.return_value = ([], {})
    with pytest.raises(typer.Exit):
        launcher_mod.run_launcher(fake)
    return captured


def test_run_launcher_raises_the_target_to_the_agent_floor(monkeypatch) -> None:
    from lilbee.app.agent_configs.window import AGENT_CHAT_CTX_FLOOR

    monkeypatch.setattr(cfg, "chat_n_ctx_target", 24576)
    captured = _run_launcher_capturing(monkeypatch)

    # In-process (the launcher's own warm wait + window warning) sees the floor,
    assert captured["cfg_target_at_spawn"] == AGENT_CHAT_CTX_FLOOR
    # and the spawned `lilbee serve` child is told to serve it via LILBEE_ env.
    assert captured["env_overrides"]["LILBEE_CHAT_N_CTX_TARGET"] == str(AGENT_CHAT_CTX_FLOOR)


def test_run_launcher_preserves_a_larger_configured_target(monkeypatch) -> None:
    monkeypatch.setattr(cfg, "chat_n_ctx_target", 131072)
    captured = _run_launcher_capturing(monkeypatch)

    assert captured["cfg_target_at_spawn"] == 131072
    assert captured["env_overrides"]["LILBEE_CHAT_N_CTX_TARGET"] == "131072"


def test_run_launcher_warns_when_a_reused_server_window_is_below_the_floor(
    monkeypatch, capsys
) -> None:
    # The env override reaches only a freshly spawned serve; a reused server keeps
    # the window it booted with, so the launch says so and names the remedy.
    monkeypatch.setattr(cfg, "chat_n_ctx_target", 24576)
    _run_launcher_capturing(monkeypatch, served_ctx=24576)

    err = capsys.readouterr().err
    assert "24,576-token" in err
    assert "re-run this command" in err


def test_run_launcher_stays_quiet_when_a_reused_server_window_holds_an_agent(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(cfg, "chat_n_ctx_target", 24576)
    _run_launcher_capturing(monkeypatch, served_ctx=65536)

    assert "re-run this command" not in capsys.readouterr().err


def test_run_launcher_stays_quiet_when_a_reused_server_window_is_unknown(
    monkeypatch, capsys
) -> None:
    # A cold chat engine reports no window; there is nothing to check yet.
    monkeypatch.setattr(cfg, "chat_n_ctx_target", 24576)
    _run_launcher_capturing(monkeypatch, served_ctx=None)

    assert "re-run this command" not in capsys.readouterr().err


def test_spawn_server_applies_env_overrides_over_inherited_env(monkeypatch) -> None:
    """spawn_server merges its overrides onto the inherited environment."""
    from lilbee.cli.launchers import server as server_mod

    monkeypatch.setenv("LILBEE_LAUNCHER_SERVE_QUIET", "1")  # route output to DEVNULL
    captured: dict = {}

    def fake_popen(cmd, *, stdout, stderr, env):
        captured["env"] = env
        return MagicMock()

    monkeypatch.setattr(server_mod, "spawn_bound_child", fake_popen)
    server_mod.spawn_server(8080, env_overrides={"LILBEE_CHAT_N_CTX_TARGET": "65536"})

    assert captured["env"]["LILBEE_CHAT_N_CTX_TARGET"] == "65536"
    # Inherited vars survive the merge (not replaced by the overrides dict).
    assert captured["env"]["LILBEE_LAUNCHER_SERVE_QUIET"] == "1"


def test_spawn_server_without_overrides_still_arms_the_parent_watcher(monkeypatch) -> None:
    """Even with nothing to override, serve must learn the pid it should outlive."""
    import os

    from lilbee.cli.launchers import server as server_mod
    from lilbee.parent_monitor import PARENT_PID_ENV

    monkeypatch.setenv("LILBEE_LAUNCHER_SERVE_QUIET", "1")
    captured: dict = {}

    def fake_popen(cmd, *, stdout, stderr, env):
        captured["env"] = env
        return MagicMock()

    monkeypatch.setattr(server_mod, "spawn_bound_child", fake_popen)
    server_mod.spawn_server(8080)

    assert captured["env"][PARENT_PID_ENV] == str(os.getpid())
    # The rest of the environment still comes through the merge.
    assert captured["env"]["LILBEE_LAUNCHER_SERVE_QUIET"] == "1"
