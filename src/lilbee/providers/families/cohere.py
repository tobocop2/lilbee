"""Cohere Command-R: native ``<|START_ACTION|>...<|END_ACTION|>`` JSON wrapper.

The Command-R7B GGUF ships no chat template, so the prompt can't be rendered
with the tool schema (llama-cpp hits ``NoneType.render``). Render from the
upstream HF tokenizer's tool-aware template instead and let the ``cohere``
schema parse the action block out of the text.
"""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.COHERE,
    template_markers=("<|START_ACTION|>",),
    hf_tokenizer_repo="CohereLabs/c4ai-command-r7b-12-2024",
    render_with_hf_template=True,
    output_format=OutputFormat.NATIVE,
)
