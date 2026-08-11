"""Per-role model readiness: can each configured model role serve right now."""

from __future__ import annotations

import logging

from lilbee.core.config import cfg
from lilbee.providers.model_ref import parse_model_ref

log = logging.getLogger(__name__)


def is_fresh_install() -> bool:
    """True when this lilbee has no data directory yet, so nothing has run here."""
    if cfg.lancedb_dir.is_dir():
        return False
    log.debug("fresh install: lancedb_dir missing (%s)", cfg.lancedb_dir)
    return True


def chat_ready() -> bool:
    """True when the chat ref resolves to something usable now."""
    return _role_ready(cfg.chat_model)


def embedding_ready() -> bool:
    """True when the embedding ref resolves to something usable now."""
    return _role_ready(cfg.embedding_model)


def _role_ready(model: str) -> bool:
    """Whether *model* can serve. Empty means unconfigured: not ready, not an error.

    Remote-prefixed refs (ollama/lm_studio/API) are validated against current
    state instead of probed on disk: an ``ollama/`` ref whose litellm extra is
    missing or whose server is down is unusable. Does disk reads and, for
    local-server refs, an HTTP probe, so callers run it off the UI thread.
    """
    from lilbee.modelhub.model_manager import ValidationResult, validate_persisted_model
    from lilbee.providers.base import ProviderError
    from lilbee.providers.engine_params import resolve_model_path

    if not model:
        return False
    if parse_model_ref(model).is_remote:
        if validate_persisted_model(model) != ValidationResult.OK:
            log.debug("role_ready: remote model %r not usable", model)
            return False
        return True
    try:
        resolve_model_path(model)
    except (ProviderError, KeyError, ValueError) as exc:
        log.debug("role_ready: model %r unresolved: %s", model, exc)
        return False
    return True


def models_ready() -> bool:
    """True when both the chat and embedding refs resolve to something usable now."""
    return chat_ready() and embedding_ready()
