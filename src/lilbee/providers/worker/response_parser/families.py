"""Model-family detection from a GGUF chat template."""

from __future__ import annotations

from enum import StrEnum

_QWEN3_CODER_FUNCTION_MARKER = "<function="
_QWEN3_CODER_PARAMETER_MARKER = "<parameter="
_GEMMA4_QUOTE_MARKER = '<|"|>'
_MISTRAL_TOOL_CALLS_MARKER = "[TOOL_CALLS]"
_QWEN3_TOOL_CALL_OPEN = "<tool_call>"
_QWEN3_TOOL_CALL_CLOSE = "</tool_call>"


class ModelFamily(StrEnum):
    """Chat-template family used to pick a response schema."""

    QWEN3 = "qwen3"
    QWEN3_CODER = "qwen3_coder"
    MISTRAL = "mistral"
    GEMMA4 = "gemma4"
    UNKNOWN = "unknown"


def detect_family(chat_template: str | None) -> ModelFamily:
    """Classify *chat_template* into a known family by its distinctive markers.

    Checks the most-specific markers first so a model that combines several
    conventions (Qwen3-Coder uses ``<tool_call>`` but ALSO ``<function=``)
    matches the more specific family.
    """
    if not chat_template:
        return ModelFamily.UNKNOWN
    if (
        _QWEN3_CODER_FUNCTION_MARKER in chat_template
        and _QWEN3_CODER_PARAMETER_MARKER in chat_template
    ):
        return ModelFamily.QWEN3_CODER
    if _GEMMA4_QUOTE_MARKER in chat_template:
        return ModelFamily.GEMMA4
    if _MISTRAL_TOOL_CALLS_MARKER in chat_template:
        return ModelFamily.MISTRAL
    if _QWEN3_TOOL_CALL_OPEN in chat_template and _QWEN3_TOOL_CALL_CLOSE in chat_template:
        return ModelFamily.QWEN3
    return ModelFamily.UNKNOWN
