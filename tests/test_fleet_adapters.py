"""Tests for multi-GPU role specs and the llama-server argv builder."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lilbee.core.config.enums import RerankerType
from lilbee.providers.fleet import adapters
from lilbee.providers.fleet import binary as binary_mod
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


def test_build_argv_multi_device_without_a_ratio_leaves_the_split_to_the_engine() -> None:
    # A multi-device plan that carries no ratio is the tight placement, which
    # exists to let the engine keep what fits and spill the rest. The engine
    # aborts its fit pass when tensor_split is user-set, so inventing an even
    # split here would force every layer onto the cards and OOM the load.
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
        devices=(0, 1),
        n_gpu_layers=-1,
        slots=2,
        ctx_per_slot=4096,
    )
    assert "--tensor-split" not in argv


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


def test_build_argv_leaves_the_thread_count_alone_on_an_unconstrained_host(monkeypatch) -> None:
    """llama-server counts physical math cores and skips efficiency cores itself.

    Where nothing caps this process, that default is better informed than any
    count lilbee could compute.
    """
    monkeypatch.setattr(adapters, "engine_thread_count", lambda: None)
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.VISION],
        model_path=Path("/models/vision.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=4096,
    )
    assert "--threads" not in argv
    assert "--threads-batch" not in argv


def test_build_argv_caps_threads_at_the_cpu_budget_when_one_applies(monkeypatch) -> None:
    monkeypatch.setattr(adapters, "engine_thread_count", lambda: 8)
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=4096,
    )
    assert argv[argv.index("--threads") + 1] == "8"
    # llama-server's batch threads default to --threads, so one flag covers both.
    assert "--threads-batch" not in argv


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
    for flag in ("--flash-attn", "--cache-type-k", "--cache-type-v", "--batch-size"):
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


_HELP_NO_MMAP = "--mmap, --no-mmap                       whether to memory-map model\n"
_HELP_LOAD_MODE = "-lm,   --load-mode MODE                 model loading mode (default: auto)\n"
_HELP_PROSE = "-ngl,  --n-gpu-layers N                  slower than --no-mmap; see --load-mode\n"


def _argv_with_help(monkeypatch, help_text: str | None, *, no_mmap: bool = True) -> list[str]:
    """A chat argv built against a binary whose ``--help`` prints *help_text* (None: cannot run)."""

    def _run(*_a, **_k):
        if help_text is None:
            raise OSError("no such file")
        return SimpleNamespace(stdout=help_text, stderr="")

    monkeypatch.setattr(binary_mod.subprocess, "run", _run)
    return build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=4096,
        no_mmap=no_mmap,
    )


def _mmap_tokens(argv: list[str]) -> list[str]:
    return [t for t in argv if t in ("--no-mmap", "--load-mode", "none")]


@pytest.mark.parametrize(
    ("help_text", "expected"),
    [
        (_HELP_LOAD_MODE, ["--load-mode", "none"]),
        (_HELP_NO_MMAP, ["--no-mmap"]),
        (_HELP_NO_MMAP + _HELP_LOAD_MODE, ["--load-mode", "none"]),
        (_HELP_PROSE, []),
        ("", []),
        (None, []),
    ],
    ids=["load-mode", "no-mmap", "both-prefers-load-mode", "prose-only", "empty", "cannot-run"],
)
def test_build_argv_no_mmap_sends_the_flag_the_binary_lists(
    monkeypatch, help_text: str | None, expected: list[str]
) -> None:
    assert _mmap_tokens(_argv_with_help(monkeypatch, help_text)) == expected


def test_build_argv_no_mmap_reads_help_on_stderr(monkeypatch) -> None:
    monkeypatch.setattr(
        binary_mod.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(stdout="", stderr=_HELP_LOAD_MODE),
    )
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
    assert _mmap_tokens(argv) == ["--load-mode", "none"]


def test_build_argv_defaults_to_mmap(monkeypatch) -> None:
    argv = _argv_with_help(monkeypatch, _HELP_NO_MMAP + _HELP_LOAD_MODE, no_mmap=False)
    assert _mmap_tokens(argv) == []


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


def test_build_argv_sends_one_fit_target_per_device_in_mib() -> None:
    argv = _chat_argv(fit_target_mib=(2458, 3482))
    assert argv[argv.index("--fit-target") + 1] == "2458,3482"


def test_build_argv_leaves_the_fit_target_to_the_engine_by_default() -> None:
    assert "--fit-target" not in _chat_argv()


def test_build_argv_shares_one_kv_pool_capped_at_the_window_per_slot() -> None:
    # An explicit --parallel turns unified KV off unless it is asked for, and the
    # per-slot cap keeps one slot from taking the whole pool.
    argv = _chat_argv(kv_unified_per_slot=4096)
    assert "--kv-unified" in argv
    assert argv[argv.index("--kv-unified-per-slot") + 1] == "4096"
    assert argv[argv.index("--parallel") + 1] == "4"
    assert argv[argv.index("--ctx-size") + 1] == "16384"


def test_build_argv_keeps_per_slot_kv_by_default() -> None:
    argv = _chat_argv()
    assert "--kv-unified" not in argv
    assert "--kv-unified-per-slot" not in argv


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


def test_build_argv_emits_the_device_names_it_is_given() -> None:
    """Vulkan and SYCL pin by name, in the space --list-devices printed them."""
    argv = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/models/chat.gguf"),
        devices=(0, 2),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=4096,
        device_names=("Vulkan0", "Vulkan2"),
    )
    assert argv[argv.index("--device") + 1] == "Vulkan0,Vulkan2"


def _help_listing(flags, *, indent: str = "") -> str:
    return "\n".join(f"{indent}{flag} X    what it does" for flag in flags) + "\n"


def _every_optional_flag_argv(
    monkeypatch, help_text: str | None, *, cpu_moe_only: bool = False
) -> list[str]:
    """A chat argv that carries every optional flag, built against a binary printing *help_text*."""

    def _run(*_a, **_k):
        if help_text is None:
            raise OSError("no such file")
        return SimpleNamespace(stdout=help_text, stderr="")

    binary_mod.engine_help.cache_clear()
    monkeypatch.setattr(binary_mod.subprocess, "run", _run)
    monkeypatch.setattr(adapters, "engine_thread_count", lambda: 4)
    spec = replace(
        ROLE_SPECS[WorkerRole.CHAT],
        extra_args=(*ROLE_SPECS[WorkerRole.CHAT].extra_args, "--pooling", "rank"),
    )
    return build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=spec,
        model_path=Path("/models/chat.gguf"),
        devices=(0, 1),
        device_names=("CUDA0", "CUDA1"),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=4096,
        tensor_split=(1, 1),
        mmproj=Path("/models/mmproj.gguf"),
        flash_attn="on",
        cache_type="q8_0",
        batch_size=512,
        cpu_moe=cpu_moe_only,
        n_cpu_moe=None if cpu_moe_only else 3,
        kv_unified_per_slot=4096,
    )


_OPTIONAL = sorted(adapters._OPTIONAL_FLAGS)
_ALWAYS_WRITTEN = [
    "--device",
    "--tensor-split",
    "--embeddings",
    "--fit-target",
    "--kv-unified-per-slot",
]


# Literal argv, written by hand from the engine's flag grammar: each flag and its values
# leave together. They do not read the width from the flag table under test.
def _argv(text: str) -> list[str]:
    return text.split()


_FULL_ARGV = _argv(
    "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 --n-gpu-layers "
    "-1 --parallel 1 --cont-batching --ctx-size 4096 --threads 4 --flash-attn on "
    "--cache-type-k q8_0 --cache-type-v q8_0 --batch-size 512 --ubatch-size 512 "
    "--mmproj /models/mmproj.gguf --device CUDA0,CUDA1 --tensor-split 1,1 "
    "--n-cpu-moe 3 --kv-unified --kv-unified-per-slot 4096 --jinja "
    "--reasoning-format deepseek --no-prefill-assistant --pooling rank"
)
_FULL_ARGV_CPU_MOE = _argv(
    "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 --n-gpu-layers "
    "-1 --parallel 1 --cont-batching --ctx-size 4096 --threads 4 --flash-attn on "
    "--cache-type-k q8_0 --cache-type-v q8_0 --batch-size 512 --ubatch-size 512 "
    "--mmproj /models/mmproj.gguf --device CUDA0,CUDA1 --tensor-split 1,1 "
    "--cpu-moe --kv-unified --kv-unified-per-slot 4096 --jinja --reasoning-format "
    "deepseek --no-prefill-assistant --pooling rank"
)
_ARGV_WITHOUT = {
    "--flash-attn": _argv(
        "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 "
        "--n-gpu-layers -1 --parallel 1 --cont-batching --ctx-size 4096 --threads "
        "4 --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 512 --ubatch-size "
        "512 --mmproj /models/mmproj.gguf --device CUDA0,CUDA1 --tensor-split 1,1 "
        "--n-cpu-moe 3 --kv-unified --kv-unified-per-slot 4096 --jinja "
        "--reasoning-format deepseek --no-prefill-assistant --pooling rank"
    ),
    "--cache-type-k": _argv(
        "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 "
        "--n-gpu-layers -1 --parallel 1 --cont-batching --ctx-size 4096 --threads "
        "4 --flash-attn on --cache-type-v q8_0 --batch-size 512 --ubatch-size 512 "
        "--mmproj /models/mmproj.gguf --device CUDA0,CUDA1 --tensor-split 1,1 "
        "--n-cpu-moe 3 --kv-unified --kv-unified-per-slot 4096 --jinja "
        "--reasoning-format deepseek --no-prefill-assistant --pooling rank"
    ),
    "--cache-type-v": _argv(
        "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 "
        "--n-gpu-layers -1 --parallel 1 --cont-batching --ctx-size 4096 --threads "
        "4 --flash-attn on --cache-type-k q8_0 --batch-size 512 --ubatch-size 512 "
        "--mmproj /models/mmproj.gguf --device CUDA0,CUDA1 --tensor-split 1,1 "
        "--n-cpu-moe 3 --kv-unified --kv-unified-per-slot 4096 --jinja "
        "--reasoning-format deepseek --no-prefill-assistant --pooling rank"
    ),
    "--batch-size": _argv(
        "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 "
        "--n-gpu-layers -1 --parallel 1 --cont-batching --ctx-size 4096 --threads "
        "4 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --ubatch-size "
        "512 --mmproj /models/mmproj.gguf --device CUDA0,CUDA1 --tensor-split 1,1 "
        "--n-cpu-moe 3 --kv-unified --kv-unified-per-slot 4096 --jinja "
        "--reasoning-format deepseek --no-prefill-assistant --pooling rank"
    ),
    "--ubatch-size": _argv(
        "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 "
        "--n-gpu-layers -1 --parallel 1 --cont-batching --ctx-size 4096 --threads "
        "4 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size "
        "512 --mmproj /models/mmproj.gguf --device CUDA0,CUDA1 --tensor-split 1,1 "
        "--n-cpu-moe 3 --kv-unified --kv-unified-per-slot 4096 --jinja "
        "--reasoning-format deepseek --no-prefill-assistant --pooling rank"
    ),
    "--mmproj": _argv(
        "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 "
        "--n-gpu-layers -1 --parallel 1 --cont-batching --ctx-size 4096 --threads "
        "4 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size "
        "512 --ubatch-size 512 --device CUDA0,CUDA1 --tensor-split 1,1 "
        "--n-cpu-moe 3 --kv-unified --kv-unified-per-slot 4096 --jinja "
        "--reasoning-format deepseek --no-prefill-assistant --pooling rank"
    ),
    "--n-cpu-moe": _argv(
        "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 "
        "--n-gpu-layers -1 --parallel 1 --cont-batching --ctx-size 4096 --threads "
        "4 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size "
        "512 --ubatch-size 512 --mmproj /models/mmproj.gguf --device CUDA0,CUDA1 "
        "--tensor-split 1,1 --kv-unified --kv-unified-per-slot 4096 --jinja "
        "--reasoning-format deepseek --no-prefill-assistant --pooling rank"
    ),
    "--cpu-moe": _argv(
        "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 "
        "--n-gpu-layers -1 --parallel 1 --cont-batching --ctx-size 4096 --threads "
        "4 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size "
        "512 --ubatch-size 512 --mmproj /models/mmproj.gguf --device CUDA0,CUDA1 "
        "--tensor-split 1,1 --kv-unified --kv-unified-per-slot 4096 --jinja "
        "--reasoning-format deepseek --no-prefill-assistant --pooling rank"
    ),
    "--threads": _argv(
        "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 "
        "--n-gpu-layers -1 --parallel 1 --cont-batching --ctx-size 4096 "
        "--flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size 512 "
        "--ubatch-size 512 --mmproj /models/mmproj.gguf --device CUDA0,CUDA1 "
        "--tensor-split 1,1 --n-cpu-moe 3 --kv-unified --kv-unified-per-slot 4096 "
        "--jinja --reasoning-format deepseek --no-prefill-assistant --pooling "
        "rank"
    ),
    "--jinja": _argv(
        "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 "
        "--n-gpu-layers -1 --parallel 1 --cont-batching --ctx-size 4096 --threads "
        "4 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size "
        "512 --ubatch-size 512 --mmproj /models/mmproj.gguf --device CUDA0,CUDA1 "
        "--tensor-split 1,1 --n-cpu-moe 3 --kv-unified --kv-unified-per-slot 4096 "
        "--reasoning-format deepseek --no-prefill-assistant --pooling rank"
    ),
    "--reasoning-format": _argv(
        "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 "
        "--n-gpu-layers -1 --parallel 1 --cont-batching --ctx-size 4096 --threads "
        "4 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size "
        "512 --ubatch-size 512 --mmproj /models/mmproj.gguf --device CUDA0,CUDA1 "
        "--tensor-split 1,1 --n-cpu-moe 3 --kv-unified --kv-unified-per-slot 4096 "
        "--jinja --no-prefill-assistant --pooling rank"
    ),
    "--no-prefill-assistant": _argv(
        "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 "
        "--n-gpu-layers -1 --parallel 1 --cont-batching --ctx-size 4096 --threads "
        "4 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size "
        "512 --ubatch-size 512 --mmproj /models/mmproj.gguf --device CUDA0,CUDA1 "
        "--tensor-split 1,1 --n-cpu-moe 3 --kv-unified --kv-unified-per-slot 4096 "
        "--jinja --reasoning-format deepseek --pooling rank"
    ),
    "--pooling": _argv(
        "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 "
        "--n-gpu-layers -1 --parallel 1 --cont-batching --ctx-size 4096 --threads "
        "4 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size "
        "512 --ubatch-size 512 --mmproj /models/mmproj.gguf --device CUDA0,CUDA1 "
        "--tensor-split 1,1 --n-cpu-moe 3 --kv-unified --kv-unified-per-slot 4096 "
        "--jinja --reasoning-format deepseek --no-prefill-assistant"
    ),
    "--kv-unified": _argv(
        "/bin/llama-server --model /models/chat.gguf --host 127.0.0.1 "
        "--n-gpu-layers -1 --parallel 1 --cont-batching --ctx-size 4096 --threads "
        "4 --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --batch-size "
        "512 --ubatch-size 512 --mmproj /models/mmproj.gguf --device CUDA0,CUDA1 "
        "--tensor-split 1,1 --n-cpu-moe 3 --kv-unified-per-slot 4096 --jinja "
        "--reasoning-format deepseek --no-prefill-assistant --pooling rank"
    ),
}


def test_the_optional_flag_table_is_not_empty() -> None:
    assert len(_OPTIONAL) == 14


def test_the_literal_argv_table_names_every_optional_flag() -> None:
    assert sorted(_ARGV_WITHOUT) == _OPTIONAL


def test_a_build_that_lists_every_flag_gets_the_literal_full_argv(monkeypatch) -> None:
    listed = _help_listing(_OPTIONAL + _ALWAYS_WRITTEN)
    assert _every_optional_flag_argv(monkeypatch, listed) == _FULL_ARGV
    cpu_moe = _every_optional_flag_argv(monkeypatch, listed, cpu_moe_only=True)
    assert cpu_moe == _FULL_ARGV_CPU_MOE


@pytest.mark.parametrize("flag", _OPTIONAL)
def test_a_flag_the_build_does_not_list_is_dropped_with_its_value(
    monkeypatch, caplog, flag: str
) -> None:
    cpu_moe_only = flag == "--cpu-moe"
    listed = [f for f in _OPTIONAL + _ALWAYS_WRITTEN if f != flag]
    with caplog.at_level("WARNING"):
        argv = _every_optional_flag_argv(
            monkeypatch, _help_listing(listed), cpu_moe_only=cpu_moe_only
        )
    assert argv == _ARGV_WITHOUT[flag]
    lines = [r.getMessage() for r in caplog.records if f"list {flag};" in r.getMessage()]
    assert len(lines) == 1
    assert adapters._OPTIONAL_FLAGS[flag].effect in lines[0]


def test_a_build_that_lists_every_flag_gets_the_full_argv(monkeypatch, caplog) -> None:
    full = _every_optional_flag_argv(monkeypatch, None)
    with caplog.at_level("WARNING"):
        argv = _every_optional_flag_argv(monkeypatch, _help_listing(_OPTIONAL + _ALWAYS_WRITTEN))
    assert argv == full
    assert not [r for r in caplog.records if "does not list" in r.getMessage()]


def test_option_lines_indented_as_in_ik_layout_count_as_listed(monkeypatch) -> None:
    full = _every_optional_flag_argv(monkeypatch, None)
    help_text = _help_listing(_OPTIONAL[:7], indent="  ") + _help_listing(
        _OPTIONAL[7:], indent="         "
    )
    assert _every_optional_flag_argv(monkeypatch, help_text) == full


def test_a_flag_quoted_in_a_description_does_not_count_as_listed(monkeypatch) -> None:
    help_text = _help_listing([f for f in _OPTIONAL if f != "--jinja"])
    help_text += "-x, --other X                 see --jinja for details\n"
    assert "--jinja" not in _every_optional_flag_argv(monkeypatch, help_text)


def test_placement_and_embedding_flags_are_written_whatever_the_help_says(monkeypatch) -> None:
    argv = _every_optional_flag_argv(monkeypatch, _help_listing(["--model"]))
    assert "--device" in argv
    assert "--tensor-split" in argv
    embed = build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.EMBED],
        model_path=Path("/models/embed.gguf"),
        devices=(0,),
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=512,
    )
    assert "--embeddings" in embed


def test_a_build_with_no_help_text_keeps_every_flag_and_warns_once(monkeypatch, caplog) -> None:
    full = _every_optional_flag_argv(monkeypatch, None)
    with caplog.at_level("WARNING"):
        first = _every_optional_flag_argv(monkeypatch, "")
        second = _every_optional_flag_argv(monkeypatch, "")
    assert first == second == full
    blind = [r for r in caplog.records if "printed no --help text" in r.getMessage()]
    assert len(blind) == 1


def test_a_dropped_flag_is_logged_once_per_binary(monkeypatch, caplog) -> None:
    help_text = _help_listing([f for f in _OPTIONAL if f != "--jinja"])
    with caplog.at_level("WARNING"):
        _every_optional_flag_argv(monkeypatch, help_text)
        _every_optional_flag_argv(monkeypatch, help_text)
    lines = [r for r in caplog.records if "list --jinja;" in r.getMessage()]
    assert len(lines) == 1
