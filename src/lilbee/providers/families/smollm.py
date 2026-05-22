"""SmolLM3 family profile."""

from __future__ import annotations

import re

from lilbee.providers.families.profile import FamilyProfile, OutputFormat, StreamingPolicy
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.SMOLLM,
    architectures=("smollm3",),
    name_patterns=(re.compile(r"smollm[\s\-_]?3", re.IGNORECASE),),
    chat_format_override="chatml-function-calling",
    streaming_policy=StreamingPolicy.DOWNGRADE_AUTO_TOOL_CHOICE,
    output_format=OutputFormat.CHATML_TOOL_CALL,
    reason=(
        "SmolLM3 embeds a chat template with HuggingFace's {% generation %} tag; "
        "the chatml-function-calling preset renders the prompt with ChatML markers "
        "and emits <tool_call> wrappers."
    ),
)
