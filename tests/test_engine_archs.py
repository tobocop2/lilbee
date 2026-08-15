"""The generated engine arch list stays in step with the engine pin.

``engine_archs.py`` is generated from the llama.cpp commit that the pinned
repo's ref resolves to. Nothing at runtime re-derives it, so these tests are
what stops a bumped engine from leaving a stale support claim behind. All of
them are offline: the generator reaches the network, the checks do not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lilbee._generated.engine_archs import (
    ENGINE_LLAMA_CPP_REF,
    LLAMA_CPP_COMMIT,
    SUPPORTED_ARCHS,
)
from lilbee.catalog.compat import classify
from lilbee.catalog.types import ModelCompat

_ENGINE_ENV = Path(__file__).resolve().parents[1] / "engine-versions.env"


def _pinned_engine_ref() -> str:
    for line in _ENGINE_ENV.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "ENGINE_LLAMA_CPP_REF":
            return value.strip()
    raise AssertionError(f"ENGINE_LLAMA_CPP_REF missing from {_ENGINE_ENV}")


def test_generated_list_matches_the_engine_pin() -> None:
    # The whole point of the generated file: bump the engine without rerunning
    # `make engine-archs` and this fails rather than shipping a stale claim.
    assert _pinned_engine_ref() == ENGINE_LLAMA_CPP_REF


def test_records_the_llama_cpp_commit_it_was_read_from() -> None:
    assert len(LLAMA_CPP_COMMIT) == 40
    assert all(c in "0123456789abcdef" for c in LLAMA_CPP_COMMIT)


def test_supported_archs_is_populated() -> None:
    # llama.cpp served well over 100 architectures at the current pin; a parse that
    # silently matched almost nothing would otherwise look like a valid empty set.
    assert len(SUPPORTED_ARCHS) > 100


def test_unknown_sentinel_is_not_a_supported_arch() -> None:
    # LLM_ARCH_UNKNOWN's name is "(unknown)", which means "unrecognised", not a
    # model anything can load. It must never make a GGUF look pullable.
    assert "(unknown)" not in SUPPORTED_ARCHS
    assert classify("(unknown)") is ModelCompat.UNSUPPORTED


@pytest.mark.parametrize("arch", ["llama", "qwen3", "gemma3", "phi3"])
def test_mainstream_architectures_are_supported(arch: str) -> None:
    assert classify(arch) is ModelCompat.SUPPORTED


@pytest.mark.parametrize(
    "arch",
    [
        "cohere2moe",
        "deepseek2-ocr",
        "deepseek32",
        "eagle3",
        "gemma4",
        "gemma4-assistant",
        "hunyuan_vl",
        "mellum",
        "mistral4",
        "talkie",
    ],
)
def test_architectures_the_engine_serves_ahead_of_the_gguf_package(arch: str) -> None:
    # Why the list moved off the gguf package: the pinned engine loads all of these,
    # but gguf enumerated only gemma4 (hand-allowlisted) and refused the rest at pull
    # time. The gguf package tracks its own release cadence, so it lags the engine by
    # a different amount at every version; reading the engine's table removes the lag.
    assert classify(arch) is ModelCompat.SUPPORTED


def test_readme_arch_block_matches_the_generated_list() -> None:
    # The README's collapsible list is generated from the same table; a pin
    # bump that skips `make engine-archs` must fail here, not ship a stale
    # support claim to the front page.
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    start = readme.index("<!-- supported-archs:start -->")
    end = readme.index("<!-- supported-archs:end -->")
    block = readme[start:end]
    listed = set(re.findall(r"`([^`]+)`", block))
    assert listed == SUPPORTED_ARCHS
    assert f"All {len(SUPPORTED_ARCHS)} supported model architectures" in block


# general.architecture probed from each tested repo's GGUF header (the repos in
# docs/tested-models.md). Update alongside that document.
_TESTED_REPO_ARCHS = {
    "LiquidAI/LFM2-1.2B-GGUF": "lfm2",
    "Qwen/Qwen3-4B-GGUF": "qwen3",
    "Qwen/Qwen3-Embedding-0.6B-GGUF": "qwen3",
    "Qwen/Qwen3-Embedding-8B-GGUF": "qwen3",
    "Qwen/Qwen3-VL-4B-Instruct-GGUF": "qwen3vl",
    "bartowski/OLMoE-1B-7B-0924-Instruct-GGUF": "olmoe",
    "cjpais/llava-1.6-mistral-7b-gguf": "llama",
    "ggml-org/InternVL3-2B-Instruct-GGUF": "qwen2",
    "ggml-org/Qwen2.5-VL-3B-Instruct-GGUF": "qwen2vl",
    "ggml-org/SmolVLM2-2.2B-Instruct-GGUF": "llama",
    "ggml-org/dots.ocr-GGUF": "qwen2",
    "ggml-org/gemma-3-4b-it-GGUF": "gemma3",
    "ggml-org/gemma-4-12B-it-GGUF": "gemma4",
    "gpustack/bge-m3-GGUF": "bert",
    "gpustack/bge-reranker-v2-m3-GGUF": "bert",
    "hugging-quants/Llama-3.2-3B-Instruct-Q4_K_M-GGUF": "llama",
    "mradermacher/DeepSeek-V2-Lite-Chat-GGUF": "deepseek2",
    "mradermacher/Qwen3-Reranker-0.6B-GGUF": "qwen3",
    "noctrex/LightOnOCR-2-1B-GGUF": "qwen3",
    "nomic-ai/nomic-embed-text-v1.5-GGUF": "nomic-bert",
    "openbmb/MiniCPM-V-2_6-gguf": "qwen2",
    "second-state/All-MiniLM-L6-v2-Embedding-GGUF": "bert",
}


def test_every_tested_family_arch_stays_supported() -> None:
    # An engine bump that drops or renames one of these architectures strands
    # a hardware-tested family; that must fail the suite, not ship quietly.
    unsupported = {r: a for r, a in _TESTED_REPO_ARCHS.items() if a not in SUPPORTED_ARCHS}
    assert not unsupported


def test_tested_repo_map_matches_the_tested_models_doc() -> None:
    # The map above is verification data for docs/tested-models.md; a repo
    # swapped there without updating the map (or vice versa) is drift.
    doc = (Path(__file__).resolve().parents[1] / "docs/tested-models.md").read_text(
        encoding="utf-8"
    )
    missing_in_doc = [r for r in _TESTED_REPO_ARCHS if r not in doc]
    assert not missing_in_doc
