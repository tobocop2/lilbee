"""AI2 OLMo-3: ``<function_calls>`` wrapper with ``name(key=value)`` syntax.

The GGUF's embedded template doesn't teach the ``<function_calls>`` convention,
so the model improvises a bracketed pseudo-call (``[runs name(...)]``) that the
schema can't read. Render from the upstream HF tokenizer's tool-aware template
instead so the model emits the real ``<function_calls>`` block.
"""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.OLMO3,
    template_markers=("<function_calls>",),
    hf_tokenizer_repo="allenai/Olmo-3-7B-Instruct",
    render_with_hf_template=True,
    stringify_hf_tool_args=True,  # preserve historical render shape for this family
    output_format=OutputFormat.NATIVE,
)
