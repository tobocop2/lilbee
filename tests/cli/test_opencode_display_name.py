"""Tests for ``display_name`` (opencode model-picker label)."""

from __future__ import annotations

import pytest

from lilbee.cli.agent_configs.opencode import display_name


@pytest.mark.parametrize(
    "ref, expected",
    [
        # Native GGUF refs from lilbee's featured catalog.
        ("Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf", "Qwen3-0.6B Q8_0"),
        ("Qwen/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf", "Qwen3-4B Q4_K_M"),
        (
            "bartowski/Mistral-7B-Instruct-v0.3-GGUF/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
            "Mistral-7B-Instruct-v0.3 Q4_K_M",
        ),
        (
            "unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_K_M.gguf",
            "gemma-4-E2B-it Q4_K_M",
        ),
        # Dot separator (some bartowski uploads use ``.Q*`` instead of ``-Q*``).
        (
            "bartowski/SmolLM-135M-Instruct-GGUF/SmolLM-135M-Instruct.Q8_0.gguf",
            "SmolLM-135M-Instruct Q8_0",
        ),
        # IQuant pattern (importance-aware quantisation).
        (
            "unsloth/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-IQ4_XS.gguf",
            "Llama-3.3-70B-Instruct IQ4_XS",
        ),
        # Float fallbacks.
        (
            "org/repo-GGUF/repo-F16.gguf",
            "repo F16",
        ),
        (
            "org/repo-GGUF/repo-BF16.gguf",
            "repo BF16",
        ),
    ],
)
def test_native_gguf_ref_renders_as_model_quant(ref: str, expected: str) -> None:
    """Every shipped native-GGUF shape produces a ``<model> <quant>`` label."""
    assert display_name(ref) == expected


def test_non_native_ref_is_returned_unchanged() -> None:
    """SDK provider refs (Ollama, OpenAI, etc.) pass through verbatim.

    Those refs already have a short canonical form like ``openai/gpt-4o``
    and don't carry the ``.gguf`` filename trailer that triggers the
    extractor.
    """
    assert display_name("openai/gpt-4o") == "openai/gpt-4o"
    assert display_name("ollama/qwen3:8b") == "ollama/qwen3:8b"


def test_filename_with_no_recognised_quant_falls_back_to_stem() -> None:
    """A native-shape ref whose filename has no known quant suffix returns
    the bare stem (filename minus ``.gguf``)."""
    assert display_name("org/repo-GGUF/model-without-quant.gguf") == "model-without-quant"


def test_malformed_ref_is_returned_unchanged() -> None:
    """Refs missing the ``<org>/<repo>/<filename>.gguf`` shape pass through."""
    assert display_name("not-a-ref") == "not-a-ref"
    assert display_name("only/two-parts") == "only/two-parts"
    assert display_name("") == ""
