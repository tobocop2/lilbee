"""Hermes-3: community ChatML quants prompted via the ``chatml-function-calling`` preset.

Community Hermes-3 GGUFs embed bare ChatML with no tool blocks. The
``chatml-function-calling`` preset injects them at prompt time and the model
emits ``<tool_call>{json}</tool_call>`` wrappers.
"""

from __future__ import annotations

import re

from lilbee.providers.chat_format import LlamaCppChatFormatPreset
from lilbee.providers.families.profile import FamilyProfile, OutputFormat, StreamingPolicy
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.HERMES,
    template_markers=("You are a function calling AI model",),
    name_patterns=(re.compile(r"hermes[\s\-_]?3", re.IGNORECASE),),
    chat_format_override=LlamaCppChatFormatPreset.CHATML_FUNCTION_CALLING,
    streaming_policy=StreamingPolicy.DOWNGRADE_AUTO_TOOL_CHOICE,
    output_format=OutputFormat.CHATML_TOOL_CALL,
)
