"""Tests for multi-GPU role specs and the llama-server argv builder."""

from __future__ import annotations

from pathlib import Path

from lilbee.providers.multi_gpu.adapters import ROLE_SPECS, build_server_argv
from lilbee.providers.worker.transport import WorkerRole


def test_every_worker_role_has_a_spec() -> None:
    assert set(ROLE_SPECS) == set(WorkerRole)


def test_chat_and_embed_are_server_capable_rerank_vision_are_not() -> None:
    assert ROLE_SPECS[WorkerRole.CHAT].server_capable
    assert ROLE_SPECS[WorkerRole.EMBED].server_capable
    assert not ROLE_SPECS[WorkerRole.RERANK].server_capable
    assert not ROLE_SPECS[WorkerRole.VISION].server_capable


def test_embed_spec_carries_embeddings_flag() -> None:
    assert "--embeddings" in ROLE_SPECS[WorkerRole.EMBED].extra_args
    assert ROLE_SPECS[WorkerRole.EMBED].endpoint_path == "/v1/embeddings"


def test_build_argv_single_device_has_no_tensor_split() -> None:
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
        port=42700,
        devices=(0,),
        n_gpu_layers=-1,
        slots=4,
        ctx_per_slot=4096,
    )
    assert "--tensor-split" not in argv
    # str(Path) renders with the platform separator (backslash on Windows).
    assert argv[:3] == [str(Path("/bin/llama-server")), "--model", str(Path("/models/chat.gguf"))]
    assert "--cont-batching" in argv
    # total ctx = per-slot * slots
    assert argv[argv.index("--ctx-size") + 1] == str(4096 * 4)
    assert argv[argv.index("--parallel") + 1] == "4"


def test_build_argv_multi_device_adds_tensor_split() -> None:
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
        port=42700,
        devices=(0, 1),
        n_gpu_layers=-1,
        slots=2,
        ctx_per_slot=4096,
    )
    assert argv[argv.index("--tensor-split") + 1] == "1,1"


def test_build_argv_appends_role_extra_args() -> None:
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.EMBED],
        model_path=Path("/models/embed.gguf"),
        port=42701,
        devices=(0,),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=2048,
    )
    assert "--embeddings" in argv
