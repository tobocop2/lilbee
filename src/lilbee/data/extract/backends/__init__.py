"""Plugin backends that bind lilbee's providers into xberg's global registries.

:mod:`.registry` owns the binding table and the ``sync_*`` entry points; the
:mod:`.embedding`, :mod:`.tokenizer` and :mod:`.vision_ocr` modules each declare
one binding and self-register it there at import.
"""

from __future__ import annotations

from .registry import (
    BackendKind,
    XbergBinding,
    register_binding,
    sync_xberg_backend,
    sync_xberg_backends,
)

__all__ = [
    "BackendKind",
    "XbergBinding",
    "register_binding",
    "sync_xberg_backend",
    "sync_xberg_backends",
]
