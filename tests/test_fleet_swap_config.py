"""Tests for the llama-swap config generator."""

from __future__ import annotations

import json
from pathlib import Path

from lilbee.providers.fleet.fleet import InstanceLaunch
from lilbee.providers.fleet.swap_config import build_swap_config
from lilbee.providers.roles import WorkerRole


def _launch(role: WorkerRole, argv: list[str], env: dict[str, str] | None = None) -> InstanceLaunch:
    return InstanceLaunch(
        role=role,
        argv=argv,
        env_overrides=env or {},
        model=f"{role.value}-model",
        port_file=Path(f"/data/{role.value}.port"),
    )


def _config(launches: list[InstanceLaunch]) -> dict:
    return json.loads(build_swap_config(launches))


class TestBuildSwapConfig:
    def test_emits_valid_json_with_top_level_keys(self) -> None:
        cfg = _config([_launch(WorkerRole.CHAT, ["/bin/llama-server", "--model", "/m/c.gguf"])])
        assert cfg["startPort"] == 5800
        assert cfg["healthCheckTimeout"] >= 1
        assert "logLevel" in cfg

    def test_one_model_per_role_keyed_by_role_name(self) -> None:
        cfg = _config(
            [
                _launch(WorkerRole.CHAT, ["/bin/llama-server", "--jinja"]),
                _launch(WorkerRole.EMBED, ["/bin/llama-server", "--embeddings"]),
            ]
        )
        assert set(cfg["models"]) == {WorkerRole.CHAT.value, WorkerRole.EMBED.value}

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
            WorkerRole.CHAT.value,
            WorkerRole.EMBED.value,
            WorkerRole.RERANK.value,
        }

    def test_command_carries_port_macro_and_role_argv(self) -> None:
        cfg = _config([_launch(WorkerRole.CHAT, ["/bin/llama-server", "--jinja"])])
        cmd = cfg["models"][WorkerRole.CHAT.value]["cmd"]
        assert "--jinja" in cmd
        assert cmd.endswith("--port ${PORT}")
        assert cfg["models"][WorkerRole.CHAT.value]["proxy"] == "http://localhost:${PORT}"

    def test_embed_command_keeps_embeddings_flag(self) -> None:
        cfg = _config([_launch(WorkerRole.EMBED, ["/bin/llama-server", "--embeddings"])])
        assert "--embeddings" in cfg["models"][WorkerRole.EMBED.value]["cmd"]

    def test_spaced_model_path_is_quoted(self) -> None:
        argv = ["/bin/llama-server", "--model", "/Application Support/lilbee/m.gguf"]
        cfg = _config([_launch(WorkerRole.CHAT, argv)])
        cmd = cfg["models"][WorkerRole.CHAT.value]["cmd"]
        assert "'/Application Support/lilbee/m.gguf'" in cmd  # shell-quoted, survives the space

    def test_env_overrides_become_env_list_when_present(self) -> None:
        cfg = _config(
            [_launch(WorkerRole.CHAT, ["/bin/llama-server"], env={"CUDA_VISIBLE_DEVICES": "0"})]
        )
        assert cfg["models"][WorkerRole.CHAT.value]["env"] == ["CUDA_VISIBLE_DEVICES=0"]

    def test_env_key_absent_when_no_overrides(self) -> None:
        cfg = _config([_launch(WorkerRole.CHAT, ["/bin/llama-server"])])
        assert "env" not in cfg["models"][WorkerRole.CHAT.value]

    def test_member_never_times_out(self) -> None:
        cfg = _config([_launch(WorkerRole.CHAT, ["/bin/llama-server"])])
        assert cfg["models"][WorkerRole.CHAT.value]["ttl"] == 0
