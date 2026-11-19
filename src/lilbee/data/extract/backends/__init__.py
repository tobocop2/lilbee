"""Plugin backends that bind lilbee's providers into xberg's global registries.

:mod:`.registry` owns the binding table, the ``sync_*`` entry points and the
on-demand ``bind_backend``; the :mod:`.vision_ocr` module declares one binding and
self-registers it there at import. :mod:`.tokenizer` and :mod:`.embedding` register
one backend per provider on the chunker's first use.
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
