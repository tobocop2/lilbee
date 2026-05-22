"""MeetKai Functionary v3: ``>>>name\\n{json}`` wire shape via the ``functionary-v2`` preset.

Functionary v3 GGUFs inherit Llama-3.1's ``general.name``, so identity matches
the HF repo reference rather than name metadata. The ``functionary-v2`` preset
needs the HF tokenizer to emit its tool-call special tokens.
"""

from __future__ import annotations

import re

from lilbee.providers.chat_format import LlamaCppChatFormatPreset
from lilbee.providers.families.profile import FamilyProfile, OutputFormat, StreamingPolicy
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.FUNCTIONARY_V3,
    template_markers=(">>>all",),
    ref_patterns=(re.compile(r"functionary[^/]*v3", re.IGNORECASE),),
    chat_format_override=LlamaCppChatFormatPreset.FUNCTIONARY_V2,
    hf_tokenizer_repo="meetkai/functionary-small-v3.2",
    streaming_policy=StreamingPolicy.DOWNGRADE_AUTO_TOOL_CHOICE,
    output_format=OutputFormat.NATIVE,
)
