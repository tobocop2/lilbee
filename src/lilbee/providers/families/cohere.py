"""Cohere Command-R: native ``<|START_ACTION|>...<|END_ACTION|>`` JSON wrapper.

The Command-R7B GGUF ships no chat template, so the prompt can't be rendered
with the tool schema (llama-cpp hits ``NoneType.render``). Render from the
upstream HF tokenizer's tool-aware template instead and let the ``cohere``
schema parse the action block out of the text.

The GGUF also under-declares context as 8192 even though the model trains to
~128K (``max_position_embeddings`` 132096, rope_theta 50000), which left
opencode's prompt over budget. Correct the trained-context ceiling so the n_ctx
picker can reach the chat target.
"""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

# Command-R7B's real trained context; the GGUF metadata says 8192.
_COMMAND_R7B_CONTEXT = 131072

PROFILE = FamilyProfile(
    family=TemplateFamily.COHERE,
    template_markers=("<|START_ACTION|>",),
    hf_tokenizer_repo="CohereLabs/c4ai-command-r7b-12-2024",
    render_with_hf_template=True,
    context_length_override=_COMMAND_R7B_CONTEXT,
    output_format=OutputFormat.NATIVE,
)
