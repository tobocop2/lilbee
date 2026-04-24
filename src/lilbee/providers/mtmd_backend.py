"""Vision OCR backend that drives llama.cpp's mtmd subsystem.

``llama_cpp.llama_chat_format.Llava15ChatHandler`` already carries the full
mtmd pipeline: it reads the mmproj, calls ``mtmd_tokenize`` + ``mtmd_helper_
eval_chunk_single``, and then hands off to the standard completion sampler.
The only reason earlier lilbee code picked a specific subclass
(ObsidianChatHandler, Qwen25VLChatHandler, ...) was to get the right chat
template. Those subclasses hardcode ``CHAT_FORMAT`` strings which drift
from the GGUF's own embedded template and cause silent failures:

- LightOnOCR-2-1B's GGUF ends turns on ``<|im_end|>`` but
  ``ObsidianChatHandler`` emits ``###``. Sampling never sees the real EOT
  and runs to the token cap.
- ``Qwen25VLChatHandler`` injects a ``"You are a helpful assistant."``
  system turn that pulls the model out of OCR mode and hallucinates.

The fix is to load the chat template from the main GGUF itself (every
recent vision GGUF ships one under ``tokenizer.chat_template``), rewrite
the model's image-placeholder token into the URL-substitution pattern
the upstream handler expects, and otherwise reuse the upstream mtmd
pipeline unchanged.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from gguf import GGUFReader

log = logging.getLogger(__name__)


# Image-placeholder tokens we know ship inside GGUF chat templates.
# ``Llava15ChatHandler.__call__`` replaces ``{{ content.image_url.url }}``
# emissions with mtmd's default media marker; rewriting these tokens into
# that Jinja expression makes the handler's pipeline work unchanged.
_GGUF_IMAGE_TOKENS: tuple[str, ...] = (
    "<|image_pad|>",
    "<image>",
    "<IMAGE>",
    "<__media__>",
    "<__image__>",
)
_IMAGE_URL_JINJA = "{{ content.image_url.url }}"

_TOKENIZER_CHAT_TEMPLATE_KEY = "tokenizer.chat_template"


def read_chat_template(model_path: Path) -> str | None:
    """Return the Jinja chat template embedded in a GGUF model, or None."""
    try:
        reader = GGUFReader(str(model_path))
        field = reader.get_field(_TOKENIZER_CHAT_TEMPLATE_KEY)
    except Exception:
        log.debug("Failed to read chat template from %s", model_path, exc_info=True)
        return None
    if field is None:
        return None
    return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")


def adapt_gguf_template_for_mtmd(template: str) -> str:
    """Rewrite image-placeholder tokens into the URL-substitution Jinja expression.

    ``Llava15ChatHandler.__call__`` renders the template, then replaces
    every image URL in the rendered text with ``mtmd_default_marker()``.
    Many GGUF chat templates skip the URL and emit a fixed token instead
    (``<|image_pad|>`` on Qwen-family vision models, ``<image>`` on
    Llama/Llava, ``<__media__>`` on newer mtmd builds). Swapping those
    for ``{{ content.image_url.url }}`` restores the URL contract so the
    replacement loop finds a marker at the same position the template
    meant to reserve for the image.
    """
    for token in _GGUF_IMAGE_TOKENS:
        if token in template:
            template = template.replace(token, _IMAGE_URL_JINJA)
    return template


def build_vision_chat_handler(model_path: Path, mmproj_path: Path) -> Any:
    """Build a chat handler that uses the main GGUF's embedded chat template.

    Returns an instance of a ``Llava15ChatHandler`` subclass with
    ``CHAT_FORMAT`` replaced by the GGUF template (image tokens rewritten)
    and ``DEFAULT_SYSTEM_MESSAGE`` set to ``None`` so we never splice a
    stray ``"You are a helpful assistant."`` into OCR prompts. If the
    GGUF has no embedded template, falls back to the upstream
    ``Llava15ChatHandler`` defaults.
    """
    from llama_cpp.llama_chat_format import Llava15ChatHandler

    class _GgufTemplateChatHandler(Llava15ChatHandler):
        DEFAULT_SYSTEM_MESSAGE = None

    handler_cls: type[Llava15ChatHandler] = _GgufTemplateChatHandler

    template = read_chat_template(model_path)
    if template is not None:
        handler_cls.CHAT_FORMAT = adapt_gguf_template_for_mtmd(template)
        log.info(
            "Vision chat handler: using GGUF-embedded template (%d bytes) from %s",
            len(template),
            model_path.name,
        )
    else:
        log.info(
            "Vision chat handler: no GGUF template in %s, falling back to Llava15 default",
            model_path.name,
        )

    return handler_cls(str(mmproj_path), verbose=False)


def load_vision_llama(model_path: Path, mmproj_path: Path | None = None) -> Any:
    """Load a vision-capable ``Llama`` tied to a GGUF-templated chat handler.

    Replaces ``llama_cpp_provider.load_vision_llama``. Every vision model
    goes through the same code path, driven by the main GGUF's chat
    template and mmproj file. No per-model handler class, no projector
    type lookup table.
    """
    from llama_cpp import Llama

    from lilbee.config import cfg
    from lilbee.providers.llama_cpp_provider import _suppress_stderr, find_mmproj_for_model

    if mmproj_path is None:
        mmproj_path = find_mmproj_for_model(model_path)

    chat_handler = build_vision_chat_handler(model_path, mmproj_path)

    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "chat_handler": chat_handler,
        "verbose": False,
        "n_gpu_layers": -1,
        "n_ctx": cfg.num_ctx if cfg.num_ctx is not None else 0,
    }

    llama = _suppress_stderr(Llama, **kwargs)
    metadata = getattr(llama, "metadata", {}) or {}
    n_ctx_fn = getattr(llama, "n_ctx", None)
    n_ctx = n_ctx_fn() if callable(n_ctx_fn) else "?"
    log.info(
        "Vision model loaded: model=%s mmproj=%s n_ctx=%s arch=%s",
        model_path.name,
        mmproj_path.name,
        n_ctx,
        metadata.get("general.architecture", "?"),
    )
    return llama
