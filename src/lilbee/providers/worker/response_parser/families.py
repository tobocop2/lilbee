"""Model-family detection from GGUF chat template + architecture metadata."""

from __future__ import annotations

from dataclasses import dataclass
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
_OLMO3_FUNCTION_CALLS_OPEN = "<function_calls>"
_LFM2_TOOL_LIST_START = "<|tool_list_start|>"


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


# GGUF `general.architecture` -> family fallback. Used when a chat template has
# no marker hits (InternLM2's template is minimal) or shares generic ChatML
# markers with another family (SmolLM3 emits the same ``<tool_call>`` blocks
# as Qwen3 but has its own response schema).
_ARCHITECTURE_TO_FAMILY: dict[str, ModelFamily] = {
    "smollm3": ModelFamily.SMOLLM,
    "internlm2": ModelFamily.INTERNLM2,
    "internlm": ModelFamily.INTERNLM2,
}

# Families whose chat-template detection should be refined by architecture
# (e.g. Qwen3-template + smollm3 architecture is SmolLM, not Qwen3).
_ARCHITECTURE_REFINEMENTS: dict[ModelFamily, frozenset[ModelFamily]] = {
    ModelFamily.QWEN3: frozenset({ModelFamily.SMOLLM}),
}


@dataclass(frozen=True)
class _FamilyDetector:
    """One family + the ALL-must-be-present marker substrings that identify it."""

    family: ModelFamily
    markers: tuple[str, ...]


# Ordered most-specific-first: a family whose markers are a superset of another
# family's (e.g. Qwen3-Coder uses both ``<tool_call>`` and ``<function=``) must
# come before the more-general family so it matches first.
_FAMILY_DETECTORS: tuple[_FamilyDetector, ...] = (
    _FamilyDetector(ModelFamily.COHERE, (_COHERE_START_ACTION,)),
    _FamilyDetector(ModelFamily.GPT_OSS, (_GPT_OSS_CHANNEL, _GPT_OSS_CALL)),
    _FamilyDetector(ModelFamily.ERNIE, (_ERNIE_BOS, _ERNIE_EOS)),
    _FamilyDetector(ModelFamily.DEEPSEEK_V31, (_DEEPSEEK_V31_TOOL_CALLS_BEGIN,)),
    _FamilyDetector(ModelFamily.GRANITE, (_GRANITE_ROLE_MARKER,)),
    _FamilyDetector(ModelFamily.PHI4MINI, (_PHI4_TOOL_OPEN, _PHI4_TOOL_CLOSE)),
    _FamilyDetector(ModelFamily.FUNCTIONARY_V3, (_FUNCTIONARY_V3_ALL,)),
    _FamilyDetector(ModelFamily.HERMES, (_HERMES_MARKER,)),
    _FamilyDetector(ModelFamily.LLAMA3, (_LLAMA3_PYTHON_TAG,)),
    _FamilyDetector(ModelFamily.KIMI_K2, (_KIMI_K2_SECTION_BEGIN, _KIMI_K2_ARG_BEGIN)),
    _FamilyDetector(ModelFamily.OLMO3, (_OLMO3_FUNCTION_CALLS_OPEN,)),
    _FamilyDetector(ModelFamily.LFM2, (_LFM2_TOOL_LIST_START,)),
    # GLM47 is GLM46 minus the newline after the function name; the system-prompt
    # scaffolding makes the no-newline form a unique substring.
    _FamilyDetector(ModelFamily.GLM47, (_GLM47_NO_NEWLINE_MARKER,)),
    _FamilyDetector(ModelFamily.GLM46, (_GLM_ARG_KEY, _GLM_ARG_VALUE)),
    _FamilyDetector(
        ModelFamily.QWEN3_CODER,
        (_QWEN3_CODER_FUNCTION_MARKER, _QWEN3_CODER_PARAMETER_MARKER),
    ),
    _FamilyDetector(ModelFamily.GEMMA4, (_GEMMA4_QUOTE_MARKER,)),
    _FamilyDetector(ModelFamily.MISTRAL, (_MISTRAL_TOOL_CALLS_MARKER,)),
    _FamilyDetector(ModelFamily.QWEN3, (_QWEN3_TOOL_CALL_OPEN, _QWEN3_TOOL_CALL_CLOSE)),
)


def detect_family(chat_template: str | None, *, architecture: str | None = None) -> ModelFamily:
    """Classify a model by chat-template markers with a GGUF-architecture fallback."""
    if chat_template:
        for detector in _FAMILY_DETECTORS:
            if all(marker in chat_template for marker in detector.markers):
                return _apply_architecture_refinement(detector.family, architecture)
    return _detect_from_architecture_only(architecture)


def _detect_from_architecture_only(architecture: str | None) -> ModelFamily:
    """Classify via GGUF architecture when the chat template has no marker hit."""
    return _ARCHITECTURE_TO_FAMILY.get((architecture or "").lower(), ModelFamily.UNKNOWN)


def _apply_architecture_refinement(family: ModelFamily, architecture: str | None) -> ModelFamily:
    """Override classification when architecture disambiguates a marker collision."""
    allowed = _ARCHITECTURE_REFINEMENTS.get(family)
    if allowed is None:
        return family
    arch_family = _ARCHITECTURE_TO_FAMILY.get((architecture or "").lower())
    if arch_family is not None and arch_family in allowed:
        return arch_family
    return family
