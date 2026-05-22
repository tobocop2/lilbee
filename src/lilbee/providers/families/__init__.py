"""Per-family model compatibility profiles.

Each module under this package declares one ``PROFILE = FamilyProfile(...)``
data literal that consolidates everything lilbee needs to know to load,
prompt, and extract from one chat-template family: matching patterns, the
llama-cpp chat_format preset to use (if any), an HF tokenizer download
target (if required), the streaming policy, and the output-format hint
the response parser uses to decide whether bare-JSON fallback applies.

The single source of truth is :data:`ALL_PROFILES`, an explicit tuple
that names the order profiles are consulted at detection time. The
order is most-specific-first: ref-pattern matches win over template-
marker matches win over architecture-fallback matches.
"""

from __future__ import annotations

from lilbee.providers.families import (
    cohere,
    deepseek_v31,
    ernie,
    functionary_v3,
    gemma4,
    glm46,
    glm47,
    gpt_oss,
    granite,
    hermes,
    internlm2,
    kimi_k2,
    lfm2,
    llama3,
    mistral,
    olmo3,
    phi4mini,
    qwen3,
    qwen3_coder,
    smollm,
)
from lilbee.providers.families.profile import (
    FamilyProfile,
    OutputFormat,
    StreamingPolicy,
)
from lilbee.providers.families.registry import detect, registry

ALL_PROFILES: tuple[FamilyProfile, ...] = (
    qwen3_coder.PROFILE,
    cohere.PROFILE,
    gpt_oss.PROFILE,
    ernie.PROFILE,
    deepseek_v31.PROFILE,
    granite.PROFILE,
    phi4mini.PROFILE,
    functionary_v3.PROFILE,
    hermes.PROFILE,
    smollm.PROFILE,
    llama3.PROFILE,
    kimi_k2.PROFILE,
    olmo3.PROFILE,
    lfm2.PROFILE,
    glm47.PROFILE,
    glm46.PROFILE,
    gemma4.PROFILE,
    mistral.PROFILE,
    qwen3.PROFILE,
    internlm2.PROFILE,
)

__all__ = [
    "ALL_PROFILES",
    "FamilyProfile",
    "OutputFormat",
    "StreamingPolicy",
    "detect",
    "registry",
]
