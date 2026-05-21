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
