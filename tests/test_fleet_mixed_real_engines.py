"""Plan a real MlxAdapter chat beside a sized stand-in for llama.cpp on one unified Apple pool."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.providers import engine_params
from lilbee.providers.fleet import host_caps, planning
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName, MemoryEstimate, MemoryKind
from lilbee.providers.fleet.engines.mlx import MLX
from lilbee.providers.fleet.model_memory import UNKNOWN_SIZE_RESERVE_BYTES
from lilbee.providers.fleet.swap_config import build_swap_config
from lilbee.providers.roles import WorkerRole
from tests._fleet_engines import GIB, engine_pair, one_machine
from tests._stated_sizes import state_size

pytestmark = pytest.mark.usefixtures("stated_sizes")

_LOG = "lilbee.providers.fleet.planning"
CHAT, EMBED = WorkerRole.CHAT, WorkerRole.EMBED


def _mlx_folder(root: Path, gib: float) -> Path:
    """A model folder whose one safetensors file holds *gib* GiB of weights (sparse on disk)."""
    folder = root / "mlx-chat"
    folder.mkdir()
    state_size(folder / "model.safetensors", int(gib * GIB))
    (folder / "config.json").write_text(
        json.dumps(
            {
                "num_hidden_layers": 4,
                "num_key_value_heads": 2,
                "hidden_size": 256,
                "num_attention_heads": 4,
                "max_position_embeddings": 4096,
            }
        ),
        encoding="utf-8",
    )
    return folder


def _host(monkeypatch, tmp_path, *, pool_gib, chat_gib, embed_gib):
    device = one_machine(monkeypatch, tmp_path, unified=True, total=pool_gib * GIB)
    folder = _mlx_folder(tmp_path, chat_gib) if chat_gib else None
    embed_file = tmp_path / "embed.gguf"
    embed_file.write_bytes(b"x" * 4096)
    monkeypatch.setattr(
        engine_params,
        "resolve_model_path",
        lambda ref: folder if str(ref).endswith("#mlx") else embed_file,
    )
    monkeypatch.setattr(cfg, "chat_model", "org/chat#mlx")
    mlx_bin = tmp_path / "mlx-lm-server"
    mlx_bin.write_bytes(b"m")
    monkeypatch.setattr(type(MLX), "binary", lambda _self: mlx_bin)
    _chat_unused, llama = engine_pair(tmp_path, device, 1, embed_gib)
    return {CHAT: MLX, EMBED: llama}, llama


def _warnings(caplog) -> str:
    return " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)


def test_a_real_mlx_chat_and_a_llama_embed_that_fit_the_pool_are_both_served(
    monkeypatch, tmp_path, caplog
) -> None:
    engines, _ = _host(monkeypatch, tmp_path, pool_gib=24, chat_gib=8, embed_gib=2)
    with caplog.at_level(logging.WARNING, logger=_LOG):
        plan = planning.plan_all_launches(engines=engines)
    assert {launch.role for launch in plan.launches} == {CHAT, EMBED}
    assert "does not fit" not in _warnings(caplog)


def test_a_real_mlx_chat_past_the_pool_is_refused_and_names_the_mlx_engine(
    monkeypatch, tmp_path, caplog
) -> None:
    engines, _ = _host(monkeypatch, tmp_path, pool_gib=16, chat_gib=20, embed_gib=2)
    with caplog.at_level(logging.WARNING, logger=_LOG):
        plan = planning.plan_all_launches(engines=engines)
    assert {launch.role for launch in plan.launches} == {EMBED}
    assert "(mlx)" in _warnings(caplog)


def test_the_mlx_and_llama_charges_add_up_against_one_pool(monkeypatch, tmp_path, caplog) -> None:
    # 9 GiB of chat weights and 3 GiB of embed each fit a 10 GiB pool alone; together they do not.
    engines, _ = _host(monkeypatch, tmp_path, pool_gib=10, chat_gib=9, embed_gib=3)
    with caplog.at_level(logging.WARNING, logger=_LOG):
        plan = planning.plan_all_launches(engines=engines)
    assert {launch.role for launch in plan.launches} == {EMBED}
    assert "does not fit" in _warnings(caplog)


@pytest.mark.parametrize(("pool_gib", "served"), [(14, {CHAT, EMBED}), (9, {EMBED})])
def test_an_mlx_folder_with_no_readable_weights_is_charged_the_stated_reserve(
    monkeypatch, tmp_path, caplog, pool_gib, served
) -> None:
    assert UNKNOWN_SIZE_RESERVE_BYTES == 8 * GIB
    engines, _ = _host(monkeypatch, tmp_path, pool_gib=pool_gib, chat_gib=0, embed_gib=2)
    empty = tmp_path / "empty-folder"
    empty.mkdir()
    embed_file = tmp_path / "embed.gguf"
    monkeypatch.setattr(
        engine_params,
        "resolve_model_path",
        lambda ref: empty if str(ref).endswith("#mlx") else embed_file,
    )
    with caplog.at_level(logging.WARNING, logger=_LOG):
        plan = planning.plan_all_launches(engines=engines)
    assert {launch.role for launch in plan.launches} == served
    assert "is unknown, so 8.0 GiB is reserved" in _warnings(caplog)


def test_the_mlx_chat_and_the_llama_embed_share_one_swap_config(monkeypatch, tmp_path) -> None:
    engines, llama = _host(monkeypatch, tmp_path, pool_gib=24, chat_gib=8, embed_gib=2)
    plan = planning.plan_all_launches(engines=engines)
    ports = {launch.model_id: 5900 + i for i, launch in enumerate(plan.launches)}
    config = json.loads(build_swap_config(list(plan.launches), ports, swap=True))
    command_of = {launch.role: config["models"][launch.model_id]["cmd"] for launch in plan.launches}
    assert set(command_of) == {CHAT, EMBED}
    assert str(llama.binary()) in command_of[EMBED]
    assert str(llama.binary()) not in command_of[CHAT]
    assert str(MLX.binary()) in command_of[CHAT]
    assert str(MLX.binary()) not in command_of[EMBED]


def test_the_engine_choice_refuses_an_mlx_chat_on_a_linux_host() -> None:
    linux = host_caps.collect(
        system="Linux", machine="x86_64", cpu_info={}, nvidia=None, mac_ver=""
    )
    installed = {EngineName.MLX: {ModelFormat.MLX}, EngineName.LLAMA: {ModelFormat.GGUF}}
    picked = choice.choose_engine(
        fmt=ModelFormat.MLX,
        host=lambda: linux,
        installed=installed,
        preferred=EngineName.MLX,
        override=None,
        role=CHAT,
    )
    assert picked.name is None
    assert "cannot run on this host" in (picked.notice or "")


def test_the_mlx_estimate_is_read_through_the_adapter_not_the_planner(
    monkeypatch, tmp_path, caplog
) -> None:
    engines, _ = _host(monkeypatch, tmp_path, pool_gib=10, chat_gib=1, embed_gib=2)
    seen: list[object] = []

    def charge(self, request):
        seen.append(request)
        return MemoryEstimate(
            kind=MemoryKind.SIZED,
            device_bytes=40 * GIB,
            host_bytes=0,
            shared_bytes=40 * GIB,
            per_device=(40 * GIB,),
            per_device_shared=(40 * GIB,),
        )

    monkeypatch.setattr(type(MLX), "estimate_memory", charge)
    with caplog.at_level(logging.WARNING, logger=_LOG):
        plan = planning.plan_all_launches(engines=engines)
    assert seen
    assert {launch.role for launch in plan.launches} == {EMBED}
    assert "(mlx)" in _warnings(caplog)
