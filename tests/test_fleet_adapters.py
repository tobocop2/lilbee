"""Tests for multi-GPU role specs and the llama-server argv builder."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lilbee.core.config.enums import RerankerType
from lilbee.providers.fleet.adapters import (
    LLM_RERANK_SPEC,
    ROLE_SPECS,
    build_server_argv,
    embed_spec,
    expert_offload_patterns,
    rerank_spec,
    resolve_rerank_mode,
)
from lilbee.providers.roles import RerankMode, WorkerRole


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


def test_chat_server_spec_enables_jinja() -> None:
    from lilbee.providers.fleet.adapters import ROLE_SPECS
    from lilbee.providers.roles import WorkerRole

    # --jinja is what makes the chat server render the model template and parse
    # native tool calls; tool-calling depends on it being present.
    assert "--jinja" in ROLE_SPECS[WorkerRole.CHAT].extra_args


def test_chat_server_spec_extracts_reasoning_server_side() -> None:
    from lilbee.providers.fleet.adapters import ROLE_SPECS
    from lilbee.providers.roles import WorkerRole

    # The server parses every model's native reasoning dialect (<think>, gpt-oss
    # harmony) into reasoning_content; 'none' would leak raw dialect tokens into
    # answers. The chat client re-inlines the extracted reasoning as <think>.
    extra_args = ROLE_SPECS[WorkerRole.CHAT].extra_args
    idx = extra_args.index("--reasoning-format")
    assert extra_args[idx + 1] == "deepseek"


def test_chat_server_spec_disables_assistant_prefill() -> None:
    from lilbee.providers.fleet.adapters import ROLE_SPECS
    from lilbee.providers.roles import WorkerRole

    # With prefill on, the server treats a trailing assistant message as text to
    # continue, and rejects two of them outright. Agents send both shapes, so the
    # chat server must read a trailing assistant message as a finished turn.
    assert "--no-prefill-assistant" in ROLE_SPECS[WorkerRole.CHAT].extra_args


def test_build_argv_chat_launches_without_assistant_prefill() -> None:
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=4096,
    )
    assert "--no-prefill-assistant" in argv


@pytest.mark.parametrize(
    ("reranker_type", "arch", "expected"),
    [
        (RerankerType.AUTO, "qwen3", RerankMode.LLM),
        (RerankerType.AUTO, "qwen2", RerankMode.LLM),
        (RerankerType.AUTO, "llama", RerankMode.LLM),
        (RerankerType.AUTO, "bert", RerankMode.CROSS_ENCODER),
        (RerankerType.AUTO, "xlm-roberta", RerankMode.CROSS_ENCODER),
        (RerankerType.AUTO, "nomic-bert", RerankMode.CROSS_ENCODER),
        (RerankerType.AUTO, None, RerankMode.CROSS_ENCODER),
        (RerankerType.AUTO, "totally-unknown-arch", RerankMode.CROSS_ENCODER),
        (RerankerType.CROSS_ENCODER, "qwen3", RerankMode.CROSS_ENCODER),
        (RerankerType.LLM, "bert", RerankMode.LLM),
    ],
)
def test_resolve_rerank_mode(reranker_type, arch, expected) -> None:
    assert resolve_rerank_mode(reranker_type, arch) is expected


def test_rerank_spec_selects_by_mode() -> None:
    assert rerank_spec(RerankMode.CROSS_ENCODER) is ROLE_SPECS[WorkerRole.RERANK]
    assert rerank_spec(RerankMode.LLM) is LLM_RERANK_SPEC


def test_llm_rerank_spec_is_generative_chat() -> None:
    assert LLM_RERANK_SPEC.role is WorkerRole.RERANK
    assert LLM_RERANK_SPEC.endpoint_path == "/v1/chat/completions"
    assert LLM_RERANK_SPEC.extra_args == ("--jinja",)


def test_embed_spec_encoder_arch_uses_plain_spec() -> None:
    # No declared pooling + a non-decoder arch keeps the GGUF's own mean/cls (no flag).
    assert embed_spec({"architecture": "bert"}) is ROLE_SPECS[WorkerRole.EMBED]


def test_embed_spec_without_metadata_is_plain() -> None:
    assert embed_spec(None) is ROLE_SPECS[WorkerRole.EMBED]


def test_embed_spec_decoder_arch_forces_last_token_pooling() -> None:
    spec = embed_spec({"architecture": "qwen3"})
    assert spec.extra_args == ("--embeddings", "--pooling", "last")


@pytest.mark.parametrize(
    ("pooling_type", "expected"),
    [("1", "mean"), ("2", "cls"), ("3", "last"), ("4", "rank")],
)
def test_embed_spec_honors_declared_pooling_type(pooling_type, expected) -> None:
    # A declared GGUF pooling_type wins, even over a decoder arch's default.
    meta = {"architecture": "qwen3", "pooling_type": pooling_type}
    assert embed_spec(meta).extra_args == ("--embeddings", "--pooling", expected)


def test_embed_spec_declared_none_falls_through_to_arch() -> None:
    # pooling_type 0 (NONE) is the unset default: a decoder arch still gets last,
    assert embed_spec({"architecture": "qwen3", "pooling_type": "0"}).extra_args == (
        "--embeddings",
        "--pooling",
        "last",
    )
    # and an encoder arch keeps the plain spec.
    assert embed_spec({"architecture": "bert", "pooling_type": "0"}) is ROLE_SPECS[WorkerRole.EMBED]
    assert LLM_RERANK_SPEC.server_capable is True


def test_build_argv_no_mmap_appends_the_flag() -> None:
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=4096,
        no_mmap=True,
    )
    assert "--no-mmap" in argv


def test_build_argv_defaults_to_mmap() -> None:
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=4096,
    )
    assert "--no-mmap" not in argv


def _chat_argv(**kwargs) -> list[str]:
    """A chat command line with the offload knobs under test applied."""
    return build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=4,
        ctx_per_slot=4096,
        **kwargs,
    )


def test_build_argv_has_no_expert_offload_by_default() -> None:
    argv = _chat_argv()
    assert "--cpu-moe" not in argv
    assert "--n-cpu-moe" not in argv


def test_build_argv_offloads_every_expert_when_asked() -> None:
    assert "--cpu-moe" in _chat_argv(cpu_moe=True)


def test_build_argv_offloads_a_layer_count_when_asked() -> None:
    argv = _chat_argv(n_cpu_moe=24)
    assert argv[argv.index("--n-cpu-moe") + 1] == "24"


def test_build_argv_layer_count_wins_over_offload_everything() -> None:
    # The pair would offload the same tensors twice.
    argv = _chat_argv(cpu_moe=True, n_cpu_moe=8)
    assert "--cpu-moe" not in argv
    assert argv[argv.index("--n-cpu-moe") + 1] == "8"


def test_expert_offload_patterns_empty_when_not_configured() -> None:
    assert expert_offload_patterns(cpu_moe=False, n_cpu_moe=None) == ()


def test_expert_offload_patterns_blanket_for_offload_everything() -> None:
    # One pattern covering every block, matching llama.cpp's --cpu-moe.
    patterns = expert_offload_patterns(cpu_moe=True, n_cpu_moe=None)
    assert patterns == (r"\.ffn_(up|down|gate|gate_up)_(ch|)exps",)


def test_expert_offload_patterns_are_per_block_for_a_layer_count() -> None:
    # llama.cpp expands --n-cpu-moe N into one pattern per block below N.
    patterns = expert_offload_patterns(cpu_moe=False, n_cpu_moe=3)
    assert patterns == (
        r"blk\.0\.ffn_(up|down|gate|gate_up)_(ch|)exps",
        r"blk\.1\.ffn_(up|down|gate|gate_up)_(ch|)exps",
        r"blk\.2\.ffn_(up|down|gate|gate_up)_(ch|)exps",
    )


def test_expert_offload_patterns_layer_count_wins() -> None:
    assert expert_offload_patterns(cpu_moe=True, n_cpu_moe=1) == (
        r"blk\.0\.ffn_(up|down|gate|gate_up)_(ch|)exps",
    )


def test_expert_offload_patterns_compile_as_regexes() -> None:
    # gguf-parser compiles these with Go's RE2 and llama.cpp with std::regex;
    # a pattern that only parses in Python would break both.
    for pattern in expert_offload_patterns(cpu_moe=False, n_cpu_moe=2):
        re.compile(pattern)
