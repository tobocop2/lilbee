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
    listed = set(re.findall(r"`([^`]+)`(?: ·|\n)", block))
    assert listed == SUPPORTED_ARCHS
    assert f"All {len(SUPPORTED_ARCHS)} supported model architectures" in block
