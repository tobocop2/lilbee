"""Closed set of ``llama_cpp`` chat-format preset names lilbee may select."""

from __future__ import annotations

from enum import StrEnum


class LlamaCppChatFormatPreset(StrEnum):
    """Chat-format presets registered in ``llama_cpp.llama_chat_format``.

    Values are the exact handler keys ``LlamaChatCompletionHandlerRegistry``
    looks up. The package's :func:`assert_presets_resolve_in_upstream` test
    keeps this enum honest against the installed ``llama-cpp-python`` build.
    """

    CHATML_FUNCTION_CALLING = "chatml-function-calling"
    FUNCTIONARY_V1 = "functionary-v1"
    FUNCTIONARY_V2 = "functionary-v2"
