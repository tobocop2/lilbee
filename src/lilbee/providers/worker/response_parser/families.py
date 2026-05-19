"""Model-family detection from GGUF chat template + architecture metadata."""

from __future__ import annotations

from enum import StrEnum

_QWEN3_CODER_FUNCTION_MARKER = "<function="
_QWEN3_CODER_PARAMETER_MARKER = "<parameter="
_GEMMA4_QUOTE_MARKER = '<|"|>'
_MISTRAL_TOOL_CALLS_MARKER = "[TOOL_CALLS]"
_QWEN3_TOOL_CALL_OPEN = "<tool_call>"
_QWEN3_TOOL_CALL_CLOSE = "</tool_call>"
_COHERE_START_ACTION = "<|START_ACTION|>"
_GPT_OSS_CHANNEL = "<|channel|>"
_GPT_OSS_CALL = "<|call|>"
_ERNIE_BOS = "<|begin_of_sentence|>"
_ERNIE_EOS = "<|end_of_sentence|>"
_HERMES_MARKER = "You are a function calling AI model"
_DEEPSEEK_V31_TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
_GRANITE_ROLE_MARKER = "<|start_of_role|>"
_PHI4_TOOL_OPEN = "<|tool|>"
_PHI4_TOOL_CLOSE = "<|/tool|>"
_FUNCTIONARY_V3_ALL = ">>>all"
_LLAMA3_PYTHON_TAG = "<|python_tag|>"
_GLM_ARG_KEY = "<arg_key>"
_GLM_ARG_VALUE = "<arg_value>"
_GLM47_NO_NEWLINE_MARKER = "<tool_call>{function-name}<arg_key>"
_KIMI_K2_SECTION_BEGIN = "<|tool_calls_section_begin|>"
_KIMI_K2_ARG_BEGIN = "<|tool_call_argument_begin|>"
_INTERNLM2_ACTION_PLUGIN = "<|action_start|><|plugin|>"
_OLMO3_FUNCTION_CALLS_OPEN = "<function_calls>"
_LFM2_TOOL_CALL_START = "<|tool_call_start|>"
_LFM2_TOOL_LIST_START = "<|tool_list_start|>"

# GGUF `general.architecture` values that map to a known family. Used as a
# fallback after chat-template-marker detection, for families whose chat
# templates use generic ChatML markers (SmolLM3 shares <tool_call> with
# Qwen3; InternLM2's chat_template is minimal so the action-block markers
# only appear in model output, not in the template).
_ARCHITECTURE_TO_FAMILY: dict[str, str] = {
    "smollm3": "smollm",
    "internlm2": "internlm2",
    "internlm": "internlm2",
}


class ModelFamily(StrEnum):
    """Chat-template family used to pick a response schema."""

    QWEN3 = "qwen3"
    QWEN3_CODER = "qwen3_coder"
    MISTRAL = "mistral"
    GEMMA4 = "gemma4"
    COHERE = "cohere"
    ERNIE = "ernie"
    GPT_OSS = "gpt_oss"
    SMOLLM = "smollm"
    HERMES = "hermes"
    DEEPSEEK_V31 = "deepseek_v31"
    GRANITE = "granite"
    PHI4MINI = "phi4mini"
    FUNCTIONARY_V3 = "functionary_v3"
    LLAMA3 = "llama3"
    GLM46 = "glm46"
    GLM47 = "glm47"
    KIMI_K2 = "kimi_k2"
    INTERNLM2 = "internlm2"
    OLMO3 = "olmo3"
    LFM2 = "lfm2"
    UNKNOWN = "unknown"


# Ordered most-specific-first: families whose markers are a superset of another
# family's (Qwen3-Coder uses both `<tool_call>` and `<function=`) must come
# before the more-general family. Each entry needs ALL of its markers present.
_FAMILY_DETECTORS: tuple[tuple[ModelFamily, tuple[str, ...]], ...] = (
    (ModelFamily.COHERE, (_COHERE_START_ACTION,)),
    (ModelFamily.GPT_OSS, (_GPT_OSS_CHANNEL, _GPT_OSS_CALL)),
    (ModelFamily.ERNIE, (_ERNIE_BOS, _ERNIE_EOS)),
    (ModelFamily.DEEPSEEK_V31, (_DEEPSEEK_V31_TOOL_CALLS_BEGIN,)),
    (ModelFamily.GRANITE, (_GRANITE_ROLE_MARKER,)),
    (ModelFamily.PHI4MINI, (_PHI4_TOOL_OPEN, _PHI4_TOOL_CLOSE)),
    (ModelFamily.FUNCTIONARY_V3, (_FUNCTIONARY_V3_ALL,)),
    (ModelFamily.HERMES, (_HERMES_MARKER,)),
    (ModelFamily.LLAMA3, (_LLAMA3_PYTHON_TAG,)),
    (ModelFamily.KIMI_K2, (_KIMI_K2_SECTION_BEGIN, _KIMI_K2_ARG_BEGIN)),
    (ModelFamily.OLMO3, (_OLMO3_FUNCTION_CALLS_OPEN,)),
    (ModelFamily.LFM2, (_LFM2_TOOL_LIST_START,)),
    # GLM47 is GLM46 minus the newline after function-name; the system-prompt
    # scaffolding makes the no-newline form a unique substring.
    (ModelFamily.GLM47, (_GLM47_NO_NEWLINE_MARKER,)),
    (ModelFamily.GLM46, (_GLM_ARG_KEY, _GLM_ARG_VALUE)),
    (ModelFamily.QWEN3_CODER, (_QWEN3_CODER_FUNCTION_MARKER, _QWEN3_CODER_PARAMETER_MARKER)),
    (ModelFamily.GEMMA4, (_GEMMA4_QUOTE_MARKER,)),
    (ModelFamily.MISTRAL, (_MISTRAL_TOOL_CALLS_MARKER,)),
    (ModelFamily.QWEN3, (_QWEN3_TOOL_CALL_OPEN, _QWEN3_TOOL_CALL_CLOSE)),
)


def detect_family(chat_template: str | None, *, architecture: str | None = None) -> ModelFamily:
    """Classify a model by chat-template markers with an optional GGUF
    architecture fallback for families whose templates share markers.
    """
    if not chat_template:
        return _detect_from_architecture_only(architecture)
    for family, markers in _FAMILY_DETECTORS:
        if all(marker in chat_template for marker in markers):
            return _apply_architecture_refinement(family, architecture)
    return _detect_from_architecture_only(architecture)


def _detect_from_architecture_only(architecture: str | None) -> ModelFamily:
    """Classify via GGUF architecture when the chat template has no marker."""
    arch_family = _ARCHITECTURE_TO_FAMILY.get((architecture or "").lower())
    if arch_family is None:
        return ModelFamily.UNKNOWN
    return ModelFamily(arch_family)


def _apply_architecture_refinement(family: ModelFamily, architecture: str | None) -> ModelFamily:
    """Override family classification when GGUF architecture disambiguates."""
    if family is not ModelFamily.QWEN3:
        return family
    arch_family = _ARCHITECTURE_TO_FAMILY.get((architecture or "").lower())
    if arch_family == ModelFamily.SMOLLM.value:
        return ModelFamily.SMOLLM
    return family
