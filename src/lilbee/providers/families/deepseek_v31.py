"""DeepSeek-V3.1: native ``tool_call_begin / tool_sep / tool_call_end`` with full-width pipes.

The pipe characters in ``template_markers`` are full-width (``｜``) on purpose;
substituting ASCII ``|`` breaks the match. ruff allowed-confusables permits
these in the lilbee repo.
"""

from __future__ import annotations

from lilbee.providers.families.profile import FamilyProfile, OutputFormat
from lilbee.providers.worker.response_parser.families import TemplateFamily

PROFILE = FamilyProfile(
    family=TemplateFamily.DEEPSEEK_V31,
    template_markers=("<｜tool▁calls▁begin｜>",),
    output_format=OutputFormat.NATIVE,
)
