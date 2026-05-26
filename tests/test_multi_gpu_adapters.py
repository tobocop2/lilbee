"""Tests for multi-GPU role specs and the llama-server argv builder."""

from __future__ import annotations

from pathlib import Path

from lilbee.providers.multi_gpu.adapters import ROLE_SPECS, build_server_argv
from lilbee.providers.roles import WorkerRole


def test_every_worker_role_has_a_spec() -> None:
    assert set(ROLE_SPECS) == set(WorkerRole)


def test_all_roles_are_server_capable() -> None:
    assert all(spec.server_capable for spec in ROLE_SPECS.values())


def test_embed_spec_carries_embeddings_flag() -> None:
    assert "--embeddings" in ROLE_SPECS[WorkerRole.EMBED].extra_args
    assert ROLE_SPECS[WorkerRole.EMBED].endpoint_path == "/v1/embeddings"


def test_rerank_spec_uses_rank_pooling_embeddings_not_rerank_endpoint() -> None:
    spec = ROLE_SPECS[WorkerRole.RERANK]
    # Rank-pooling embeddings primitive (avoids /v1/rerank's template dependency).
    assert spec.endpoint_path == "/v1/embeddings"
    assert spec.extra_args == ("--embeddings", "--pooling", "rank")
    assert "--reranking" not in spec.extra_args


def test_build_argv_adds_mmproj_for_vision() -> None:
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.VISION],
        model_path=Path("/models/vision.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=8192,
        mmproj=Path("/models/mmproj.gguf"),
    )
    assert argv[argv.index("--mmproj") + 1] == str(Path("/models/mmproj.gguf"))


def test_build_argv_single_device_has_no_tensor_split() -> None:
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
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
        devices=(0, 1),
        n_gpu_layers=-1,
        slots=2,
        ctx_per_slot=4096,
    )
    assert argv[argv.index("--tensor-split") + 1] == "1,1"


def test_build_argv_uses_proportional_tensor_split() -> None:
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
        devices=(0, 1),
        n_gpu_layers=-1,
        slots=2,
        ctx_per_slot=4096,
        tensor_split=(21, 14),
    )
    assert argv[argv.index("--tensor-split") + 1] == "21,14"


def test_build_argv_appends_role_extra_args() -> None:
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.EMBED],
        model_path=Path("/models/embed.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=2048,
    )
    assert "--embeddings" in argv


def test_build_argv_flash_attn_value() -> None:
    # llama.cpp v0.3.20 (vendored f49e9178) takes --flash-attn [on|off|auto].
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=4,
        ctx_per_slot=4096,
        flash_attn="on",
    )
    assert argv[argv.index("--flash-attn") + 1] == "on"


def test_build_argv_cache_type_sets_k_and_v() -> None:
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=4,
        ctx_per_slot=4096,
        cache_type="q8_0",
    )
    assert argv[argv.index("--cache-type-k") + 1] == "q8_0"
    assert argv[argv.index("--cache-type-v") + 1] == "q8_0"


def test_build_argv_batch_size_raises_both_batch_and_ubatch() -> None:
    # Embeddings: the server forces n_batch = n_ubatch and defaults n_ubatch to
    # 512, so a full-context embed needs --ubatch-size raised, not just --batch-size.
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.EMBED],
        model_path=Path("/models/embed.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=8192,
        batch_size=8192,
    )
    assert argv[argv.index("--batch-size") + 1] == "8192"
    assert argv[argv.index("--ubatch-size") + 1] == "8192"


def test_build_argv_threads_sets_threads_and_threads_batch() -> None:
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.VISION],
        model_path=Path("/models/vision.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=4096,
        threads=12,
    )
    assert argv[argv.index("--threads") + 1] == "12"
    assert argv[argv.index("--threads-batch") + 1] == "12"


def test_build_argv_omits_optional_flags_by_default() -> None:
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=4,
        ctx_per_slot=4096,
    )
    for flag in ("--flash-attn", "--cache-type-k", "--cache-type-v", "--batch-size", "--threads"):
        assert flag not in argv
