"""Whether the first-run setup wizard has to run before anything can load."""

from __future__ import annotations

import logging

from lilbee.core.config import cfg
from lilbee.providers.model_ref import parse_model_ref

log = logging.getLogger(__name__)


def needs_setup() -> bool:
    """True when the setup wizard should run: fresh data dir or unresolved models.

    Remote-prefixed refs (ollama/lm_studio/API) are validated against current
    state instead of probed on disk: an ``ollama/`` ref whose litellm extra is
    missing or whose server is down is unusable and must route the user to
    setup, not be assumed live.
    """
    if not cfg.lancedb_dir.is_dir():
        log.debug("needs_setup: lancedb_dir missing (%s)", cfg.lancedb_dir)
        return True
    from lilbee.modelhub.model_manager import ValidationResult, validate_persisted_model
    from lilbee.providers.base import ProviderError
    from lilbee.providers.engine_params import resolve_model_path

    for label, model in (("chat", cfg.chat_model), ("embedding", cfg.embedding_model)):
        if parse_model_ref(model).is_remote:
            if validate_persisted_model(model) != ValidationResult.OK:
                log.debug("needs_setup: remote %s model %r not usable", label, model)
                return True
            continue
        try:
            resolve_model_path(model)
        except (ProviderError, KeyError, ValueError) as exc:
            log.debug("needs_setup: %s model %r unresolved: %s", label, model, exc)
            return True
    return False
