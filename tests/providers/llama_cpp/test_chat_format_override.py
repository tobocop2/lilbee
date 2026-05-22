"""Coverage for the GGUF-name -> llama-cpp chat_format override map."""

from __future__ import annotations

import pytest

from lilbee.providers.llama_cpp.chat_format_override import resolve_chat_format_override


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        # Hermes-3 family across quantizers and case.
        ({"name": "Hermes 3 Llama 3.1 8B"}, "chatml-function-calling"),
        ({"name": "hermes-3-llama-3.1-8b"}, "chatml-function-calling"),
        ({"general.name": "Hermes_3_Llama_3.1_70B"}, "chatml-function-calling"),
        # Functionary v1 + v2 GGUFs.
        ({"name": "functionary-v1"}, "functionary-v1"),
        ({"name": "Functionary V2 Small"}, "functionary-v2"),
        # No override: tool-template already intact upstream.
        ({"name": "Qwen3-4B-Instruct"}, None),
        ({"name": "gemma-4-E2B-it"}, None),
        ({"name": "Meta Llama 3.1 8B Instruct"}, None),
        # No metadata at all.
        ({}, None),
        (None, None),
    ],
)
def test_resolve_known_models(metadata: dict | None, expected: str | None) -> None:
    """Override only fires for names we have explicit, documented entries for."""
    assert resolve_chat_format_override(metadata) == expected


def test_resolve_returns_none_for_non_string_name() -> None:
    """A malformed GGUF whose name field isn't a string must not raise."""
    assert resolve_chat_format_override({"name": 42}) is None
    assert resolve_chat_format_override({"name": None}) is None


def test_smollm3_override_targets_chatml_function_calling() -> None:
    from lilbee.providers.llama_cpp.chat_format_override import (
        resolve_chat_format_override,
        resolve_override_family,
    )
    from lilbee.providers.worker.response_parser.families import TemplateFamily

    meta = {"name": "SmolLM3 3B"}
    assert resolve_chat_format_override(meta) == "chatml-function-calling"
    assert resolve_override_family(meta) is TemplateFamily.SMOLLM


def test_functionary_v3_override_matches_by_repo_path() -> None:
    """Functionary v3.x fine-tunes inherit Llama-3.1's general.name; the override
    has to consult the ref path to recognise them.
    """
    from lilbee.providers.llama_cpp.chat_format_override import (
        resolve_chat_format_override,
        resolve_override_family,
    )
    from lilbee.providers.worker.response_parser.families import TemplateFamily

    meta = {"name": "Meta Llama 3.1 8B Instruct"}
    ref = "meetkai/functionary-small-v3.2-GGUF/functionary-small-v3.2.Q8_0.gguf"
    assert resolve_chat_format_override(meta, ref=ref) == "functionary-v2"
    assert resolve_override_family(meta, ref=ref) is TemplateFamily.FUNCTIONARY_V3


def test_functionary_override_does_not_fire_for_plain_llama() -> None:
    """A non-Functionary Llama-3.1 GGUF must not get the Functionary preset."""
    from lilbee.providers.llama_cpp.chat_format_override import (
        resolve_chat_format_override,
    )

    meta = {"name": "Meta Llama 3.1 8B Instruct"}
    ref = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    assert resolve_chat_format_override(meta, ref=ref) is None
