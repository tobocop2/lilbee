"""SmolLM3: embedded template uses HuggingFace's ``{% generation %}`` tag.

llama-cpp-python's bundled Jinja cannot compile the tag so lilbee bypasses the
embedded template and prompts via the ``chatml-function-calling`` preset; the
model emits ``<tool_call>{json}</tool_call>`` wrappers.
"""

from __future__ import annotations

import re

from lilbee.providers.chat_format import LlamaCppChatFormatPreset
from lilbee.providers.families.profile import FamilyProfile, OutputFormat, StreamingPolicy
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.SMOLLM,
    architectures=("smollm3",),
    name_patterns=(re.compile(r"smollm[\s\-_]?3", re.IGNORECASE),),
    chat_format_override=LlamaCppChatFormatPreset.CHATML_FUNCTION_CALLING,
    streaming_policy=StreamingPolicy.DOWNGRADE_AUTO_TOOL_CHOICE,
    output_format=OutputFormat.CHATML_TOOL_CALL,
)
