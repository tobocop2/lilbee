"""Binds lilbee's providers into xberg's process-global OCR/embedding/tokenizer backends.

Each backend module declares an :class:`XbergBinding` and self-registers it here at
import. A bind is a locked unregister-then-register: a rebuild invalidates the
captured provider, and the lock avoids racing xberg's "already registered".
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
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

    ``enabled`` gates registration at services init on live cfg (embedding is
    always on; OCR and the tokenizer are opt-in). ``make`` builds the backend from
    the current provider, capturing the provider callable the binding routes through.
    """

    kind: BackendKind
    name: str
    enabled: Callable[[Config], bool]
    make: Callable[[LLMProvider, Config], Any]


def _registry_fns(kind: BackendKind) -> tuple[Any, Any, Any]:
    """xberg's (list, register, unregister) functions for *kind*, imported lazily
    so backend modules never import the heavy xberg package at module scope.
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


@dataclass
class _BackendRegistry:
    """The binding table, the bind lock, and the provider each kind is bound to."""

    # Keyed by kind, not name: embedding and tokenizer both register as "lilbee".
    bindings: dict[BackendKind, XbergBinding] = field(default_factory=dict)
    bound: dict[BackendKind, LLMProvider] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def bind(self, kind: BackendKind, provider: LLMProvider, cfg: Config) -> None:
        """Register *kind* for *provider*; a no-op when xberg already holds that binding."""
        binding = self.bindings[kind]
        list_fn, register_fn, unregister_fn = _registry_fns(kind)
        with self.lock:
            present = binding.name in list_fn()
            if present and self.bound.get(kind) is provider:
                return
            if present:
                unregister_fn(binding.name)
            register_fn(binding.make(provider, cfg))
            self.bound[kind] = provider

    def unbind(self, kind: BackendKind) -> None:
        """Unregister *kind* from xberg when present."""
        binding = self.bindings[kind]
        list_fn, _register_fn, unregister_fn = _registry_fns(kind)
        with self.lock:
            if binding.name in list_fn():
                unregister_fn(binding.name)
            self.bound.pop(kind, None)

    def sync(self, kind: BackendKind, provider: LLMProvider, cfg: Config) -> None:
        """Bind or unbind *kind* as its ``enabled`` gate says under *cfg*."""
        if self.bindings[kind].enabled(cfg):
            self.bind(kind, provider, cfg)
        else:
            self.unbind(kind)


_registry = _BackendRegistry()


def register_binding(binding: XbergBinding) -> None:
    """Add a backend binding to the registry (called at backend-module import)."""
    _registry.bindings[binding.kind] = binding


def _load_bindings() -> None:
    """Import the backend modules so their ``register_binding`` calls have run."""
    from . import embedding, tokenizer, vision_ocr  # noqa: F401


def sync_xberg_backends(provider: LLMProvider) -> None:
    """(Re)bind every registered backend to *provider*."""
    from lilbee.core.config import cfg

    _load_bindings()
    for kind in _registry.bindings:
        _registry.sync(kind, provider, cfg)


def sync_xberg_backend(kind: BackendKind, provider: LLMProvider) -> None:
    """(Re)bind one backend by kind, after the setting that gates it changes."""
    from lilbee.core.config import cfg

    _load_bindings()
    _registry.sync(kind, provider, cfg)


def bind_backend(kind: BackendKind, provider: LLMProvider) -> None:
    """Bind one backend to *provider* on demand, whatever its ``enabled`` gate says."""
    from lilbee.core.config import cfg

    _load_bindings()
    _registry.bind(kind, provider, cfg)
