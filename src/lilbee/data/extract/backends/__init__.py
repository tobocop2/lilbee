"""Plugin backends that bind lilbee's providers into xberg's global registries.

:mod:`.registry` owns the binding table, the ``sync_*`` entry points and the
on-demand ``bind_backend``; the :mod:`.embedding`, :mod:`.tokenizer` and
:mod:`.vision_ocr` modules each declare one binding and self-register it there
at import.
"""

from __future__ import annotations

from .registry import (
    BackendKind,
    XbergBinding,
    bind_backend,
    register_binding,
    sync_xberg_backend,
    sync_xberg_backends,
)

__all__ = [
    "BackendKind",
    "XbergBinding",
    "bind_backend",
    "register_binding",
    "sync_xberg_backend",
    "sync_xberg_backends",
]
