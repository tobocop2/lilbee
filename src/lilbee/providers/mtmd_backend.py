"""Vision OCR loader that drives llama.cpp's mtmd pipeline with the GGUF's
own chat template, so there's no projector-type-to-handler lookup table.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from gguf import GGUFReader

from lilbee.core.config import cfg
from lilbee.providers.llama_cpp.abort_signal import abort_callback
from lilbee.providers.llama_cpp.gguf_meta import (
    find_mmproj_for_model,
    read_gguf_metadata,
    train_ctx_from_meta,
)
from lilbee.providers.llama_cpp.log_dispatch import (
    import_llama_cpp,
    install_llama_log_handler,
    suppress_native_stderr,
)

log = logging.getLogger(__name__)


# Image-placeholder tokens seen in GGUF chat templates. The upstream
# mtmd pipeline substitutes image URLs with mtmd's media marker, so
# these get rewritten to {{ content.image_url.url }} before rendering.
# Case matters: GGUF templates are machine-emitted and stable, so a
# case-insensitive replace would risk corrupting unrelated Jinja
# identifiers.
_GGUF_IMAGE_TOKENS: tuple[str, ...] = (
    "<|image_pad|>",
    "<image>",
    "<IMAGE>",
    "<__media__>",
    "<__image__>",
)
_IMAGE_URL_JINJA = "{{ content.image_url.url }}"

_TOKENIZER_CHAT_TEMPLATE_KEY = "tokenizer.chat_template"

_VISION_FALLBACK_N_CTX = 4096
"""n_ctx for a vision load when the GGUF has no ``context_length`` in metadata.

Most vision GGUFs report their training context (typical values: 4096, 8192,
32768); this covers the rare missing/unreadable-metadata case so the loader
still gets a sensible explicit n_ctx.
"""


def read_chat_template(model_path: Path) -> str | None:
    """Return the Jinja chat template embedded in a GGUF model, or None."""
    try:
        reader = GGUFReader(str(model_path))
        field = reader.get_field(_TOKENIZER_CHAT_TEMPLATE_KEY)
    except (OSError, ValueError, IndexError, KeyError):
        log.debug("Failed to read chat template from %s", model_path, exc_info=True)
        return None
    if field is None:
        return None
    return bytes(field.parts[field.data[0]]).decode("utf-8", errors="replace")


def adapt_gguf_template_for_mtmd(template: str) -> str:
    """Rewrite known image-placeholder tokens to ``{{ content.image_url.url }}``."""
    for token in _GGUF_IMAGE_TOKENS:
        if token in template:
            template = template.replace(token, _IMAGE_URL_JINJA)
    return template


def build_vision_chat_handler(model_path: Path, mmproj_path: Path) -> Any:
    """Return the mtmd chat handler configured with the GGUF's embedded template.

    ``DEFAULT_SYSTEM_MESSAGE`` is set to ``None`` so no stray system turn
    is injected. Falls back to the upstream default template when the
    GGUF has no ``tokenizer.chat_template``.
    """
    # Surface the libvulkan-missing hint before submodule import, since
    # importing llama_cpp.llama_chat_format triggers the parent package's
    # native loader as a side effect.
    import_llama_cpp()
    from llama_cpp.llama_chat_format import Llava15ChatHandler

    # Defined per call so each loaded model binds its own ``CHAT_FORMAT``
    # (set below) to a fresh class; hoisting this to module scope would
    # make the first loaded model's template leak into every subsequent
    # one.
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
            "Vision chat handler: no GGUF-embedded chat template for %s; using upstream default",
            model_path.name,
        )

    return handler_cls(str(mmproj_path), verbose=False)


def load_vision_llama(
    model_path: Path,
    mmproj_path: Path | None = None,
    *,
    abort_callback_override: Any = None,
) -> Any:
    """Load a vision-capable ``Llama`` using the GGUF-templated chat handler.

    ``abort_callback_override`` lets pool workers bind a callback that
    reads the worker's shared ``mp.Value`` abort flag.
    """
    Llama = import_llama_cpp().Llama  # noqa: N806 # heavy native lib; keep import lazy

    install_llama_log_handler()
    if mmproj_path is None:
        mmproj_path = find_mmproj_for_model(model_path)

    chat_handler = build_vision_chat_handler(model_path, mmproj_path)

    import os

    # llama-cpp-python defaults n_threads to ~cpu_count()//2 which leaves the
    # GPU starved on prompt-eval work (image projection through the vision
    # adapter is CPU-bound on the encode side even with all layers on GPU).
    # Ollama runs full-core. Match that here for perf parity.
    n_threads = os.cpu_count() or 4
    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "chat_handler": chat_handler,
        "verbose": False,
        "n_gpu_layers": -1,
        "n_ctx": _resolve_vision_n_ctx(model_path),
        "n_threads": n_threads,
        "n_threads_batch": n_threads,
    }
    if cfg.main_gpu is not None:
        kwargs["main_gpu"] = cfg.main_gpu
    if abort_callback_override is not None:
        kwargs["abort_callback"] = abort_callback_override
    else:
        kwargs.setdefault("abort_callback", abort_callback)

    llama = suppress_native_stderr(Llama, **kwargs)
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


def _resolve_vision_n_ctx(model_path: Path) -> int:
    """Pick n_ctx for a vision load using the model's training context.

    Reads ``<arch>.context_length`` from the GGUF metadata and uses it
    directly. The chat-tuned ``cfg.num_ctx`` is not propagated: a vision pass
    packs image-token embeddings plus the prompt (often hundreds to a few
    thousand tokens per page), and clamping to a small chat ctx truncates OCR
    output. An explicit value (rather than 0) keeps the OOM-retry path
    working since ``_halve_ctx_for_retry`` cannot bisect from 0.
    """
    try:
        meta = read_gguf_metadata(model_path)
    except Exception:
        log.debug("read_gguf_metadata failed for vision %s", model_path, exc_info=True)
        meta = None
    return train_ctx_from_meta(meta, fallback=_VISION_FALLBACK_N_CTX, model_path=model_path)
