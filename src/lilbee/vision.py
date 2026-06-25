"""Vision-model OCR helpers: prompt resolution and OpenAI-compatible image messages.

PDF rasterisation and the page loop now live inside xberg (the registered
lilbee-vision OCR backend); this module only builds the single-image request the
provider's ``vision_ocr`` sends to the vision server.
"""

OCR_PROMPT = (
    "Extract ALL text from this page as clean markdown. "
    "Preserve table structure using markdown table syntax. "
    "Include all rows, columns, headers, and page text exactly as shown."
)

# Lowercase family token (as in the model ref or GGUF path) -> the model's
# documented OCR prompt. Unlisted models fall back to OCR_PROMPT.
_NATIVE_OCR_PROMPTS: tuple[tuple[str, str], ...] = (
    ("deepseek-ocr", "<|grounding|>Convert the document to markdown."),
    ("glm-ocr", "OCR"),
)


def resolve_ocr_prompt(model_ref: str) -> str:
    """Return *model_ref*'s native OCR prompt, or the generic one if it has none."""
    needle = model_ref.lower()
    for family, prompt in _NATIVE_OCR_PROMPTS:
        if family in needle:
            return prompt
    return OCR_PROMPT


def _png_to_data_url(png_bytes: bytes) -> str:
    """Convert raw PNG bytes to a base64 data URL for OpenAI-compatible messages."""
    import base64

    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_vision_messages(prompt: str, png_bytes: bytes) -> list[dict]:
    """Build OpenAI-compatible messages with image content for vision models.

    Uses the multipart content format expected by llama.cpp's mtmd pipeline.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _png_to_data_url(png_bytes)}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
