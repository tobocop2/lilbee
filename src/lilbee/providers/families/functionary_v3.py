"""MeetKai Functionary v3: ``>>>name\\n{json}`` wire shape rendered from the HF template.

Functionary v3 GGUFs inherit Llama-3.1's ``general.name``, so identity matches
the HF repo reference rather than name metadata. The GGUF ships a stripped chat
template that never emits the tool definitions, and the llama-cpp ``functionary-v2``
preset reads a tokenizer attribute transformers 5.x dropped. So render the prompt
from the upstream HF tokenizer's own tool-aware template and let the
``functionary_v3`` schema parse the ``>>>name`` calls back out of the text.
"""

from __future__ import annotations

import re

from lilbee.providers.families.profile import FamilyProfile, OutputFormat, StreamingPolicy
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.FUNCTIONARY_V3,
    template_markers=(">>>all",),
    ref_patterns=(re.compile(r"functionary[^/]*v3", re.IGNORECASE),),
    hf_tokenizer_repo="meetkai/functionary-small-v3.2",
    render_with_hf_template=True,
    streaming_policy=StreamingPolicy.DOWNGRADE_AUTO_TOOL_CHOICE,
    output_format=OutputFormat.NATIVE,
)
