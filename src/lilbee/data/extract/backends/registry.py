"""Registry that binds lilbee's providers into xberg's process-global backends.

xberg exposes three pluggable backends -- OCR, embedding and tokenizer -- each
with its own ``list_/register_/unregister_`` trio. lilbee registers one of each so
scanned-page OCR, semantic-chunk boundary detection and token-budgeted chunk
sizing route through the fleet instead of xberg's bundled defaults.

Each backend module declares one :class:`XbergBinding` and self-registers it here
at import, so the backend, its gate and its factory live next to the class they
wire rather than in the services container. :func:`sync_xberg_backends` (re)binds
them all; :func:`sync_xberg_backend` rebinds one after a setting change. The bind
is a locked unregister-then-register so a concurrent rebuild cannot collide on
xberg's "already registered", and it re-binds every time because each backend
captures a provider callable that a rebuild (``reset_services``) invalidates.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lilbee.core.config.model import Config
    from lilbee.providers.base import LLMProvider


class BackendKind(Enum):
    """Which xberg registry a binding targets (selects the register trio)."""

    OCR = auto()
    EMBEDDING = auto()
    TOKENIZER = auto()


@dataclass(frozen=True)
class XbergBinding:
    """How to bind one lilbee backend into xberg. Declared by each backend module.

    ``enabled`` gates registration on live cfg (embedding is always on; OCR and
    the tokenizer are opt-in). ``make`` builds the backend from the current
    provider, capturing the provider callable the binding routes through.
    """

    kind: BackendKind
    name: str
    enabled: Callable[[Config], bool]
    make: Callable[[LLMProvider, Config], Any]


# Keyed by kind: lilbee registers exactly one backend per xberg registry, and the
# per-registry names are not unique across kinds (embedding and tokenizer both use
# "lilbee"), so the kind is the only collision-free key.
_BINDINGS: dict[BackendKind, XbergBinding] = {}
_bind_lock = threading.Lock()


def register_binding(binding: XbergBinding) -> None:
    """Add a backend binding to the registry (called at backend-module import)."""
    _BINDINGS[binding.kind] = binding


def _registry_fns(kind: BackendKind) -> tuple[Any, Any, Any]:
    """xberg's (list, register, unregister) functions for *kind*, imported lazily.

    Kept here so the backend modules never import xberg at module scope -- it is a
    heavy dependency and importing it eagerly would slow every CLI startup.
    """
    import xberg

    return {
        BackendKind.OCR: (
            xberg.list_ocr_backends,
            xberg.register_ocr_backend,
            xberg.unregister_ocr_backend,
        ),
        BackendKind.EMBEDDING: (
            xberg.list_embedding_backends,
            xberg.register_embedding_backend,
            xberg.unregister_embedding_backend,
        ),
        BackendKind.TOKENIZER: (
            xberg.list_tokenizer_backends,
            xberg.register_tokenizer_backend,
            xberg.unregister_tokenizer_backend,
        ),
    }[kind]


def _sync(binding: XbergBinding, provider: LLMProvider, cfg: Config) -> None:
    list_fn, register_fn, unregister_fn = _registry_fns(binding.kind)
    with _bind_lock:
        present = binding.name in list_fn()
        if binding.enabled(cfg):
            # Re-register so the backend always binds to the current provider.
            if present:
                unregister_fn(binding.name)
            register_fn(binding.make(provider, cfg))
        elif present:
            unregister_fn(binding.name)


def _load_bindings() -> None:
    """Import the backend modules so their ``register_binding`` calls have run."""
    from . import embedding, tokenizer, vision_ocr  # noqa: F401


def sync_xberg_backends(provider: LLMProvider) -> None:
    """(Re)bind every registered backend to *provider*."""
    from lilbee.core.config import cfg

    _load_bindings()
    for binding in _BINDINGS.values():
        _sync(binding, provider, cfg)


def sync_xberg_backend(kind: BackendKind, provider: LLMProvider) -> None:
    """(Re)bind one backend by kind, after the setting that gates it changes."""
    from lilbee.core.config import cfg

    _load_bindings()
    _sync(_BINDINGS[kind], provider, cfg)
