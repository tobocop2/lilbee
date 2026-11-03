"""The model set whose llama-swap config the golden records; it imports nothing from the engines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lilbee.providers.roles import RerankMode, WorkerRole

BINARY = Path("/engine/llama-server")
LOG_DIR = Path("/engine/logs")
BASE_PORT = 5900
TTL_SECONDS = 300
THREADS = 6


@dataclass(frozen=True)
class Member:
    """One instance of the model set: its role, argv inputs and launch fields."""

    role: WorkerRole
    argv: dict[str, Any]
    env: dict[str, str]
    weights_bytes: int
    replica: int = 0
    rerank_mode: RerankMode | None = None
    meta: dict[str, str] | None = None


def members() -> list[Member]:
    """A chat on two cards, two embed replicas, a rerank and a vision model."""
    chat = {
        "model_path": Path("/m/chat.gguf"),
        "devices": (0, 1),
        "device_names": ("CUDA0", "CUDA1"),
        "n_gpu_layers": 99,
        "slots": 2,
        "ctx_per_slot": 8192,
        "tensor_split": (3, 1),
        "flash_attn": "on",
        "cache_type": "q8_0",
        "batch_size": 512,
        "no_mmap": True,
        "n_cpu_moe": 4,
        "memory_endpoint": True,
        "fit_target_mib": (1024, 1024),
        "kv_unified_per_slot": 4096,
    }
    embed = {
        "model_path": Path("/m/embed.gguf"),
        "devices": (),
        "n_gpu_layers": 0,
        "slots": 1,
        "ctx_per_slot": 512,
    }
    rerank = {
        "model_path": Path("/m/rerank.gguf"),
        "devices": (0,),
        "device_names": ("CUDA0",),
        "n_gpu_layers": 99,
        "slots": 1,
        "ctx_per_slot": 2048,
        "cpu_moe": True,
    }
    vision = {
        "model_path": Path("/m/vision.gguf"),
        "devices": (1,),
        "device_names": ("CUDA1",),
        "n_gpu_layers": 99,
        "slots": 1,
        "ctx_per_slot": 4096,
        "mmproj": Path("/m/mmproj.gguf"),
    }
    gib = 1 << 30
    return [
        Member(WorkerRole.CHAT, chat, {"CUDA_VISIBLE_DEVICES": "0,1"}, 9 * gib),
        Member(
            WorkerRole.EMBED,
            embed,
            {},
            1 * gib,
            meta={"architecture": "bert", "pooling_type": "mean"},
        ),
        Member(
            WorkerRole.EMBED,
            embed,
            {},
            1 * gib,
            replica=1,
            meta={"architecture": "bert", "pooling_type": "mean"},
        ),
        Member(
            WorkerRole.RERANK,
            rerank,
            {"CUDA_VISIBLE_DEVICES": "0"},
            2 * gib,
            rerank_mode=RerankMode.CROSS_ENCODER,
        ),
        Member(WorkerRole.VISION, vision, {"CUDA_VISIBLE_DEVICES": "1"}, 4 * gib),
    ]


# The keyword arguments of each config written for the model set.
CONFIGS: dict[str, dict[str, Any]] = {
    "default": {},
    "swap-ttl-logs": {"swap": True, "ttl_seconds": TTL_SECONDS, "engine_log_dir": LOG_DIR},
}
