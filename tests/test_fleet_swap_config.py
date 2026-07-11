"""Tests for the llama-swap config generator."""

from __future__ import annotations

import json

import pytest

from lilbee.providers.fleet import swap_config as swap_config_mod
from lilbee.providers.fleet.launch import InstanceLaunch
from lilbee.providers.fleet.swap_config import build_swap_config
from lilbee.providers.roles import WorkerRole


def _mid(role: WorkerRole, replica: int = 0) -> str:
    """The llama-swap model id for a role's replica (matches InstanceLaunch.model_id)."""
    return f"{role.value}-{replica}"


def _launch(
    role: WorkerRole, argv: list[str], env: dict[str, str] | None = None, replica: int = 0
) -> InstanceLaunch:
    return InstanceLaunch(
        role=role,
        replica=replica,
        argv=argv,
        env_overrides=env or {},
        model=f"{role.value}-model",
    )


_BASE_PORT = 5900


def _config(launches: list[InstanceLaunch]) -> dict:
    ports = {launch.model_id: _BASE_PORT + i for i, launch in enumerate(launches)}
    return json.loads(build_swap_config(launches, ports))


def test_instance_launch_rerank_mode_defaults_none() -> None:
    assert _launch(WorkerRole.RERANK, ["llama-server"]).rerank_mode is None


class TestBuildSwapConfig:
    def test_emits_valid_json_with_top_level_keys(self) -> None:
        cfg = _config([_launch(WorkerRole.CHAT, ["/bin/llama-server", "--model", "/m/c.gguf"])])
        assert "startPort" not in cfg  # members carry explicit ports, never a fixed range
        assert cfg["healthCheckTimeout"] >= 1
        assert "logLevel" in cfg

    def test_one_model_per_launch_keyed_by_model_id(self) -> None:
        cfg = _config(
            [
                _launch(WorkerRole.CHAT, ["/bin/llama-server", "--jinja"]),
                _launch(WorkerRole.EMBED, ["/bin/llama-server", "--embeddings"]),
            ]
        )
        assert set(cfg["models"]) == {_mid(WorkerRole.CHAT), _mid(WorkerRole.EMBED)}

    def test_replicas_get_distinct_model_entries_all_co_resident(self) -> None:
        # Each embed replica is its own llama-swap model id, all held in the group.
        cfg = _config(
            [
                _launch(WorkerRole.CHAT, ["/bin/llama-server"]),
                _launch(WorkerRole.EMBED, ["/bin/llama-server", "--embeddings"], replica=0),
                _launch(WorkerRole.EMBED, ["/bin/llama-server", "--embeddings"], replica=1),
            ]
        )
        assert set(cfg["models"]) == {
            _mid(WorkerRole.CHAT),
            _mid(WorkerRole.EMBED, 0),
            _mid(WorkerRole.EMBED, 1),
        }
        (group,) = cfg["groups"].values()
        assert _mid(WorkerRole.EMBED, 0) in group["members"]
        assert _mid(WorkerRole.EMBED, 1) in group["members"]

    def test_group_holds_all_roles_co_resident(self) -> None:
        cfg = _config(
            [
                _launch(WorkerRole.CHAT, ["/bin/llama-server"]),
                _launch(WorkerRole.EMBED, ["/bin/llama-server"]),
                _launch(WorkerRole.RERANK, ["/bin/llama-server"]),
            ]
        )
        (group,) = cfg["groups"].values()
        assert group["swap"] is False  # never evict a member to load another
        assert group["persistent"] is True
        assert set(group["members"]) == {
            _mid(WorkerRole.CHAT),
            _mid(WorkerRole.EMBED),
            _mid(WorkerRole.RERANK),
        }

    def test_co_tenant_group_evicts_between_its_members(self) -> None:
        # swap=True is what makes llama-swap unload chat to load vision, and back.
        cfg = json.loads(
            build_swap_config(
                [
                    _launch(WorkerRole.CHAT, ["/bin/llama-server"]),
                    _launch(WorkerRole.VISION, ["/bin/llama-server"]),
                ],
                {_mid(WorkerRole.CHAT): 1, _mid(WorkerRole.VISION): 2},
                swap=True,
            )
        )
        (group,) = cfg["groups"].values()
        assert group["swap"] is True
        assert set(group["members"]) == {_mid(WorkerRole.CHAT), _mid(WorkerRole.VISION)}

    def test_command_carries_explicit_port_and_role_argv(self) -> None:
        cfg = _config([_launch(WorkerRole.CHAT, ["/bin/llama-server", "--jinja"])])
        cmd = cfg["models"][_mid(WorkerRole.CHAT)]["cmd"]
        assert "--jinja" in cmd
        assert cmd.endswith(f"--port {_BASE_PORT}")
        assert cfg["models"][_mid(WorkerRole.CHAT)]["proxy"] == f"http://127.0.0.1:{_BASE_PORT}"

    def test_each_member_gets_its_own_port(self) -> None:
        cfg = _config(
            [
                _launch(WorkerRole.CHAT, ["/bin/llama-server"]),
                _launch(WorkerRole.EMBED, ["/bin/llama-server", "--embeddings"]),
            ]
        )
        chat = cfg["models"][_mid(WorkerRole.CHAT)]
        embed = cfg["models"][_mid(WorkerRole.EMBED)]
        assert chat["cmd"].endswith(f"--port {_BASE_PORT}")
        assert embed["cmd"].endswith(f"--port {_BASE_PORT + 1}")
        assert embed["proxy"] == f"http://127.0.0.1:{_BASE_PORT + 1}"

    def test_embed_command_keeps_embeddings_flag(self) -> None:
        cfg = _config([_launch(WorkerRole.EMBED, ["/bin/llama-server", "--embeddings"])])
        assert "--embeddings" in cfg["models"][_mid(WorkerRole.EMBED)]["cmd"]

    def test_spaced_model_path_is_quoted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(swap_config_mod.sys, "platform", "linux")
        argv = ["/bin/llama-server", "--model", "/Application Support/lilbee/m.gguf"]
        cfg = _config([_launch(WorkerRole.CHAT, argv)])
        cmd = cfg["models"][_mid(WorkerRole.CHAT)]["cmd"]
        assert "'/Application Support/lilbee/m.gguf'" in cmd  # shell-quoted, survives the space

    def test_windows_uses_ms_quoting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # llama-swap splits cmd with MS rules on Windows; POSIX single quotes
        # would stay literal in the paths and the server spawn fails.
        monkeypatch.setattr(swap_config_mod.sys, "platform", "win32")
        argv = ["C:\\llama\\llama-server.exe", "--model", "C:\\Program Files\\lilbee\\m.gguf"]
        cmd = _config([_launch(WorkerRole.CHAT, argv)])["models"][_mid(WorkerRole.CHAT)]["cmd"]
        assert "'" not in cmd
        assert '"C:\\Program Files\\lilbee\\m.gguf"' in cmd
        assert cmd.startswith("C:\\llama\\llama-server.exe")

    def test_env_overrides_become_env_list_when_present(self) -> None:
        cfg = _config(
            [_launch(WorkerRole.CHAT, ["/bin/llama-server"], env={"CUDA_VISIBLE_DEVICES": "0"})]
        )
        assert cfg["models"][_mid(WorkerRole.CHAT)]["env"] == ["CUDA_VISIBLE_DEVICES=0"]

    def test_env_key_absent_when_no_overrides(self) -> None:
        cfg = _config([_launch(WorkerRole.CHAT, ["/bin/llama-server"])])
        assert "env" not in cfg["models"][_mid(WorkerRole.CHAT)]

    def test_member_never_times_out(self) -> None:
        cfg = _config([_launch(WorkerRole.CHAT, ["/bin/llama-server"])])
        assert cfg["models"][_mid(WorkerRole.CHAT)]["ttl"] == 0


class TestHealthCheckTimeoutScaling:
    def test_small_model_gets_the_floor(self) -> None:
        launch = _launch(WorkerRole.CHAT, ["/bin/llama-server"])
        launch.weights_bytes = 4 * 1024**3
        cfg = _config([launch])
        assert cfg["healthCheckTimeout"] == swap_config_mod._HEALTH_CHECK_TIMEOUT_FLOOR_S

    def test_giant_model_scales_the_timeout_past_the_floor(self) -> None:
        # 300 GB at the conservative 150 MB/s disk rate needs ~2048s, not 600s;
        # llama-swap would otherwise kill the server mid-load.
        launch = _launch(WorkerRole.CHAT, ["/bin/llama-server"])
        launch.weights_bytes = 300 * 1024**3
        cfg = _config([launch])
        expected = (300 * 1024**3) // swap_config_mod._COLD_LOAD_BYTES_PER_S
        assert cfg["healthCheckTimeout"] == expected
        assert expected > swap_config_mod._HEALTH_CHECK_TIMEOUT_FLOOR_S

    def test_heaviest_member_sets_the_proxy_global_timeout(self) -> None:
        small = _launch(WorkerRole.EMBED, ["/bin/llama-server"])
        small.weights_bytes = 1 * 1024**3
        giant = _launch(WorkerRole.CHAT, ["/bin/llama-server"])
        giant.weights_bytes = 300 * 1024**3
        cfg = _config([small, giant])
        expected = (300 * 1024**3) // swap_config_mod._COLD_LOAD_BYTES_PER_S
        assert cfg["healthCheckTimeout"] == expected

    def test_cold_load_timeout_floors_small_weights(self) -> None:
        # The shared per-member helper; the provider's client timeout derives from it.
        from lilbee.providers.fleet.swap_config import cold_load_timeout_s

        assert cold_load_timeout_s(0) == swap_config_mod._HEALTH_CHECK_TIMEOUT_FLOOR_S
        assert cold_load_timeout_s(4 * 1024**3) == swap_config_mod._HEALTH_CHECK_TIMEOUT_FLOOR_S

    def test_cold_load_timeout_scales_giant_weights(self) -> None:
        from lilbee.providers.fleet.swap_config import cold_load_timeout_s

        weights = 300 * 1024**3
        assert cold_load_timeout_s(weights) == weights // swap_config_mod._COLD_LOAD_BYTES_PER_S
