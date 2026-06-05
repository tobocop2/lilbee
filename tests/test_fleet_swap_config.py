"""Tests for the llama-swap config generator."""

from __future__ import annotations

import json
from pathlib import Path

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

    def test_command_carries_port_macro_and_role_argv(self) -> None:
        cfg = _config([_launch(WorkerRole.CHAT, ["/bin/llama-server", "--jinja"])])
        cmd = cfg["models"][_mid(WorkerRole.CHAT)]["cmd"]
        assert "--jinja" in cmd
        assert cmd.endswith("--port ${PORT}")
        assert cfg["models"][_mid(WorkerRole.CHAT)]["proxy"] == "http://localhost:${PORT}"

    def test_embed_command_keeps_embeddings_flag(self) -> None:
        cfg = _config([_launch(WorkerRole.EMBED, ["/bin/llama-server", "--embeddings"])])
        assert "--embeddings" in cfg["models"][_mid(WorkerRole.EMBED)]["cmd"]

    def test_spaced_model_path_is_quoted(self) -> None:
        argv = ["/bin/llama-server", "--model", "/Application Support/lilbee/m.gguf"]
        cfg = _config([_launch(WorkerRole.CHAT, argv)])
        cmd = cfg["models"][_mid(WorkerRole.CHAT)]["cmd"]
        assert "'/Application Support/lilbee/m.gguf'" in cmd  # shell-quoted, survives the space

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
