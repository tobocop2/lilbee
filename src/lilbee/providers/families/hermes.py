"""Hermes-3 family profile."""

from __future__ import annotations

import re

from lilbee.providers.families.profile import FamilyProfile, OutputFormat, StreamingPolicy
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.HERMES,
    template_markers=("You are a function calling AI model",),
    name_patterns=(re.compile(r"hermes[\s\-_]?3", re.IGNORECASE),),
    chat_format_override="chatml-function-calling",
    streaming_policy=StreamingPolicy.DOWNGRADE_AUTO_TOOL_CHOICE,
    output_format=OutputFormat.CHATML_TOOL_CALL,
    reason=(
        "Community Hermes-3 quants embed bare ChatML with no tool blocks; "
        "chatml-function-calling injects them and emits <tool_call> markers."
    ),
)
