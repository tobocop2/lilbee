"""GLM-4.5 / 4.6: ``<tool_call>NAME`` + ``<arg_key>K</arg_key><arg_value>V</arg_value>``.

The GLM chat template renders tool args with ``tojson(ensure_ascii=False)``, which
the bundled llama.cpp Jinja2 formatter's ``tojson`` filter rejects (``do_tojson()
got an unexpected keyword argument 'ensure_ascii'``). Render from the upstream HF
tokenizer's template instead, whose Jinja environment supports the keyword; that
template emits the same ``<arg_key>``/``<arg_value>`` tool format this schema reads.
"""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.GLM46,
    template_markers=("<arg_key>", "<arg_value>"),
    hf_tokenizer_repo="zai-org/GLM-4.5-Air",
    render_with_hf_template=True,
    output_format=OutputFormat.NATIVE,
)
