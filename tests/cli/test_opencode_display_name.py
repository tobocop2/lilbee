"""Tests for the opencode model-picker label (a clean, human-friendly name).

The label is the model entry's ``name`` field, which opencode shows in its
``/models`` picker. The entry key stays the full ref so routing to lilbee's
``/v1`` endpoint still resolves; only the displayed name is cleaned.
"""

from __future__ import annotations

import pytest

from lilbee.cli.agent_configs.opencode import opencode_config


def _entry_name(ref: str) -> str:
    from lilbee.catalog import agent_model_id

    cfg = opencode_config(base_url="http://127.0.0.1:8080", api_key="k", model_refs=[ref])
    models = cfg["provider"]["lilbee"]["models"]
    # The entry key is the clean agent id (the routing id); its "name" is the label.
    key = agent_model_id(ref)
    assert key in models
    return models[key]["name"]


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        # Subdir-quant giant ref (the reel models): a four-segment
        # <org>/<repo>/<quant-dir>/<file>.gguf path must not leak whole into the
        # picker; it renders as the clean repo label.
        (
            "unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF/UD-Q4_K_XL/"
            "Qwen3-235B-A22B-Instruct-2507-UD-Q4_K_XL-00001-of-00003.gguf",
            "Qwen3 235B A22B",
        ),
        # Plain three-segment native refs.
        ("Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf", "Qwen3 4B"),
        ("Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf", "Qwen3 0.6B"),
        (
            "bartowski/Mistral-7B-Instruct-v0.3-GGUF/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            "Mistral 7B v0.3",
        ),
    ],
)
def test_native_ref_renders_clean_repo_label(ref: str, expected: str) -> None:
    """A native GGUF ref (three or four segments) shows its clean repo label."""
    assert _entry_name(ref) == expected


def test_provider_prefixed_ref_uses_short_name() -> None:
    """Ollama/OpenAI refs already have a short canonical form: the part after the prefix."""
    assert _entry_name("ollama/qwen3:8b") == "qwen3:8b"
    assert _entry_name("openai/gpt-4o") == "gpt-4o"
