"""Chat-template overrides for models whose GGUF-embedded template can't drive tools.

Some GGUFs ship a chat template that renders no tool-calling Jinja (so the fleet's
``supports_tools`` probe rejects them) or one that trips llama.cpp's minja engine.
This registry maps a lowercase family token (as it appears in the model ref or GGUF
path) to a packaged override template; :func:`resolve_chat_template` returns its path,
and the fleet appends ``--chat-template-file`` so llama-server renders the override.

Mirrors :data:`lilbee.vision._NATIVE_OCR_PROMPTS` / :func:`lilbee.vision.resolve_ocr_prompt`.
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATE_DIR = Path(__file__).parent / "chat_template_files"

# Lowercase family token (as in the model ref or GGUF path) -> override template
# filename. Unlisted models fall back to their GGUF-embedded template (None).
_CHAT_TEMPLATE_OVERRIDES: tuple[tuple[str, str], ...] = (
    ("glm-4-9b", "glm4.jinja"),
    ("hermes", "llama3-tools.jinja"),
    ("llama-3.1", "llama3-tools.jinja"),
    ("internlm2", "internlm2.jinja"),
    ("olmo3", "olmo3.jinja"),
    ("olmo-3", "olmo3.jinja"),
)


def resolve_chat_template(model_ref: str) -> Path | None:
    """Return *model_ref*'s override chat-template path, or None if it has none."""
    needle = model_ref.lower()
    for family, filename in _CHAT_TEMPLATE_OVERRIDES:
        if family in needle:
            return _TEMPLATE_DIR / filename
    return None
