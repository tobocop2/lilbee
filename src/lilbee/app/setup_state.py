"""The two questions the TUI asks before it hands anyone a screen.

``models_ready`` is what chat depends on; ``is_fresh_install`` is why a brand
new lilbee still meets the setup wizard. They are kept apart because folding
the data-dir check into the chat gate would lock chat behind an ingest that
cannot be started from anywhere else. ``LilbeeApp.settle_setup_state`` composes
them.
"""

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


def models_ready() -> bool:
    """True when the chat and embedding refs both resolve to something usable now.

    This is the question chat depends on: without both, there is no engine to
    answer a prompt. Remote-prefixed refs (ollama/lm_studio/API) are validated
    against current state instead of probed on disk: an ``ollama/`` ref whose
    litellm extra is missing or whose server is down is unusable and must route
    the user to setup, not be assumed live.

    Does disk reads and, for local-server refs, an HTTP probe, so callers run it
    off the UI thread.
    """
    from lilbee.modelhub.model_manager import ValidationResult, validate_persisted_model
    from lilbee.providers.base import ProviderError
    from lilbee.providers.engine_params import resolve_model_path

    for label, model in (("chat", cfg.chat_model), ("embedding", cfg.embedding_model)):
        if parse_model_ref(model).is_remote:
            if validate_persisted_model(model) != ValidationResult.OK:
                log.debug("models_ready: remote %s model %r not usable", label, model)
                return False
            continue
        try:
            resolve_model_path(model)
        except (ProviderError, KeyError, ValueError) as exc:
            log.debug("models_ready: %s model %r unresolved: %s", label, model, exc)
            return False
    return True
