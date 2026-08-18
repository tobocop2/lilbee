"""The chat window a launcher advertises to the client it starts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lilbee.core.config import cfg
from lilbee.providers.roles import WorkerRole


def _health(monkeypatch, body: dict) -> None:
    from lilbee.cli.launchers import server as launch_mod

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    # /api/health needs the token like every other route, and the probe reads
    # it from server.json, which does not exist under test.
    monkeypatch.setattr(launch_mod, "_session_token", lambda: "t")
    monkeypatch.setattr(launch_mod.httpx, "get", lambda url, **_kw: resp)


@pytest.mark.no_warm_default
def test_client_chat_ctx_prefers_the_window_the_server_reports(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    _health(monkeypatch, {"chat_ready": True, "chat_ctx": 40960})
    monkeypatch.setattr(launch_mod, "planned_chat_ctx", lambda: 65536)
    assert launch_mod.client_chat_ctx(8765) == 40960


@pytest.mark.no_warm_default
def test_client_chat_ctx_falls_back_to_planned_window_when_engine_is_cold(monkeypatch):
    # The chat role builds lazily, so a launcher that hands off before the engine
    # is up would otherwise write a config with no context window at all.
    from lilbee.cli.launchers import server as launch_mod

    _health(monkeypatch, {"chat_ready": False, "chat_ctx": None})
    monkeypatch.setattr(launch_mod, "planned_chat_ctx", lambda: 65536)
    assert launch_mod.client_chat_ctx(8765) == 65536


@pytest.mark.no_warm_default
def test_client_chat_ctx_is_none_when_no_local_window_can_be_planned(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    _health(monkeypatch, {"chat_ready": False})
    monkeypatch.setattr(launch_mod, "planned_chat_ctx", lambda: None)
    assert launch_mod.client_chat_ctx(8765) is None


@pytest.mark.no_warm_default
def test_client_chat_ctx_warns_when_the_window_is_below_the_configured_target(monkeypatch, capsys):
    from lilbee.cli.launchers import server as launch_mod

    monkeypatch.setattr(cfg, "chat_n_ctx_target", 65536)
    _health(monkeypatch, {"chat_ready": True, "chat_ctx": 32768})
    assert launch_mod.client_chat_ctx(8765) == 32768
    err = capsys.readouterr().err
    assert "32,768" in err
    assert "65,536" in err


@pytest.mark.no_warm_default
def test_client_chat_ctx_is_quiet_when_the_target_is_met(monkeypatch, capsys):
    from lilbee.cli.launchers import server as launch_mod

    monkeypatch.setattr(cfg, "chat_n_ctx_target", 32768)
    _health(monkeypatch, {"chat_ready": True, "chat_ctx": 32768})
    assert launch_mod.client_chat_ctx(8765) == 32768
    assert capsys.readouterr().err == ""


@pytest.mark.no_warm_default
def test_planned_chat_ctx_is_none_for_a_remote_chat_model(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    monkeypatch.setattr(cfg, "chat_model", "ollama/qwen3:4b")
    assert launch_mod.planned_chat_ctx() is None


@pytest.mark.no_warm_default
def test_planned_chat_ctx_honors_an_explicit_num_ctx_pin(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod

    monkeypatch.setattr(cfg, "chat_model", "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf")
    monkeypatch.setattr(cfg, "num_ctx", 12288)
    assert launch_mod.planned_chat_ctx() == 12288


@pytest.mark.no_warm_default
def test_hermes_warns_when_the_window_is_below_its_hard_minimum(capsys):
    from lilbee.cli.launchers.hermes import warn_if_below_hermes_minimum

    warn_if_below_hermes_minimum(32768)
    err = capsys.readouterr().err
    assert "32,768" in err
    assert "64,000" in err


@pytest.mark.no_warm_default
def test_hermes_is_quiet_when_the_window_clears_its_minimum(capsys):
    from lilbee.cli.launchers.hermes import warn_if_below_hermes_minimum

    warn_if_below_hermes_minimum(65536)
    assert capsys.readouterr().err == ""


@pytest.mark.no_warm_default
def test_hermes_is_quiet_when_the_window_is_unknown(capsys):
    from lilbee.cli.launchers.hermes import warn_if_below_hermes_minimum

    warn_if_below_hermes_minimum(None)
    assert capsys.readouterr().err == ""


@pytest.mark.no_warm_default
def test_planned_chat_ctx_sizes_a_local_model_the_way_the_fleet_does(monkeypatch, tmp_path):
    # Exercises the header-math fallback: the file is not a parsable GGUF, so the
    # estimator cannot answer and resolve_chat_ctx does the arithmetic itself.
    from lilbee.cli.launchers import server as launch_mod

    gguf = tmp_path / "chat.gguf"
    gguf.write_bytes(b"0" * 1024)
    meta = {
        "architecture": "qwen3",
        "block_count": "36",
        "head_count": "32",
        "head_count_kv": "8",
        "key_length": "128",
        "value_length": "128",
        "context_length": "40960",
    }
    monkeypatch.setattr(cfg, "chat_model", "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "num_ctx_max", None)
    monkeypatch.setattr(cfg, "chat_n_ctx_target", 65536)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _ref: gguf)
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: meta)

    # A 40,960-token trained window caps the 65,536 target.
    assert launch_mod.planned_chat_ctx() == 40960


@pytest.mark.no_warm_default
def test_planned_chat_ctx_sizes_against_the_gpu_the_fleet_will_use(monkeypatch, tmp_path):
    # The advertised window has to be the one the fleet will serve. Sized against
    # host memory instead, an 8 GiB card is told to expect the whole trained
    # window and the client trims history to a window it never gets.
    import os

    from lilbee.cli.launchers import server as launch_mod
    from lilbee.core.config.enums import KvCacheType
    from lilbee.providers.fleet import planning as planning_mod
    from lilbee.providers.fleet.devices import FleetDevice

    gguf = tmp_path / "chat.gguf"
    gguf.touch()
    os.truncate(gguf, 2 * 1024**3)
    meta = {
        "architecture": "qwen3",
        "block_count": "36",
        "head_count": "32",
        "head_count_kv": "8",
        "key_length": "128",
        "value_length": "128",
        "context_length": "40960",
    }
    monkeypatch.setattr(cfg, "chat_model", "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "num_ctx_max", None)
    monkeypatch.setattr(cfg, "chat_n_ctx_target", 65536)
    monkeypatch.setattr(cfg, "gpu_memory_fraction", 0.75)
    monkeypatch.setattr(cfg, "kv_cache_type", KvCacheType.F16)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _ref: gguf)
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: meta)
    monkeypatch.setattr(planning_mod, "resolve_llama_server", lambda: Path("/bin/srv"))
    monkeypatch.setattr(
        planning_mod._read_device_cache,
        "get",
        lambda _b: [FleetDevice("CUDA", 0, "gpu", 8 * 1024**3, 8 * 1024**3)],
    )

    # 6 GiB budget, less 2 GiB of weights and their 10% buffer, over 147,456 bytes
    # of KV per token, quantized down: well under the 40,960 the model was trained for.
    assert launch_mod.planned_chat_ctx() == 27648


@pytest.mark.no_warm_default
def test_planned_chat_ctx_is_none_when_the_model_file_is_missing(monkeypatch):
    from lilbee.cli.launchers import server as launch_mod
    from lilbee.providers.base import ProviderError

    def _missing(_ref):
        raise ProviderError("not installed")

    monkeypatch.setattr(cfg, "chat_model", "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", _missing)

    assert launch_mod.planned_chat_ctx() is None


@pytest.mark.no_warm_default
def test_both_grant_surfaces_report_the_window_the_estimator_fits(monkeypatch, tmp_path):
    # The launcher advertises a window the fleet has to serve, so the two must
    # come from one answer. Both read resolve_chat_ctx; this pins that they agree
    # on the estimator's number rather than one of them keeping header math.
    from lilbee.cli.launchers import server as launch_mod
    from lilbee.providers.fleet import planning as planning_mod

    gguf = tmp_path / "chat.gguf"
    gguf.write_bytes(b"0" * 1024)
    meta = {"architecture": "qwen3", "block_count": "36", "context_length": "262144"}
    fitted = 249856

    monkeypatch.setattr(cfg, "chat_model", "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
    monkeypatch.setattr(cfg, "num_ctx", None)
    monkeypatch.setattr(cfg, "num_ctx_max", None)
    monkeypatch.setattr(cfg, "chat_n_ctx_target", 262144)
    monkeypatch.setattr("lilbee.providers.engine_params.resolve_model_path", lambda _ref: gguf)
    monkeypatch.setattr("lilbee.providers.gguf_meta.read_gguf_metadata", lambda _p: meta)
    monkeypatch.setattr(planning_mod, "fit_chat_ctx", lambda *_a, **_k: fitted)

    assert launch_mod.planned_chat_ctx() == fitted
    assert planning_mod._role_ctx(WorkerRole.CHAT, gguf, meta) == fitted
