"""Catalog/role validation helpers and the :func:`ConfigField` wrapper."""

import os
import sys
from typing import Any

from pydantic import Field

from lilbee.providers.model_ref import PROVIDER_PREFIXES


def ConfigField(
    *args: Any,
    writable: bool = False,
    reindex: bool = False,
    write_only: bool = False,
    public: bool = True,
    **kwargs: Any,
) -> Any:
    """Wrap pydantic ``Field`` and attach metadata via ``json_schema_extra``."""
    extra: dict[str, bool] = {}
    if writable:
        extra["writable"] = True
    if reindex:
        extra["reindex"] = True
    if write_only:
        extra["write_only"] = True
    if not public:
        extra["public"] = False
    if extra:
        kwargs["json_schema_extra"] = extra
    return Field(*args, **kwargs)


# Test-only bypass. Both the env var and pytest must be present so a
# leaked env var cannot disable validation in production.
_SKIP_MODEL_TASK_VALIDATION_ENV = "LILBEE_SKIP_MODEL_TASK_VALIDATION"


def _model_task_validation_bypassed() -> bool:
    if not os.environ.get(_SKIP_MODEL_TASK_VALIDATION_ENV):
        return False
    return sys.modules.get("pytest") is not None


_MODEL_FIELD_TO_TASK: dict[str, str] = {
    "chat_model": "chat",
    "embedding_model": "embedding",
    "vision_model": "vision",
    "reranker_model": "rerank",
}


def _find_model_catalog_entry(ref: str) -> Any:
    # circular import: catalog imports cfg.
    from lilbee.catalog import find_catalog_entry

    return find_catalog_entry(ref)


def _enforce_role_match(ref: str, entry: Any, field_name: str) -> None:
    from lilbee.models import ModelTask

    want = ModelTask(_MODEL_FIELD_TO_TASK[field_name])
    if entry.task == want:
        return
    from lilbee.server.handlers import format_task_mismatch

    raise ValueError(format_task_mismatch(ref, ModelTask(entry.task), want))


def _skips_catalog_check(ref: str, *, allow_bypass: bool) -> bool:
    """Whether *ref* skips the catalog check."""
    if not ref or not ref.strip():
        return True
    if allow_bypass and _model_task_validation_bypassed():
        return True
    return ref.split("/", 1)[0] in PROVIDER_PREFIXES


def validate_model_task_assignment(field_name: str, ref: str, *, allow_bypass: bool = True) -> str:
    """Check *ref* is a catalog entry whose task matches *field_name*; return the canonical ref."""
    if _skips_catalog_check(ref, allow_bypass=allow_bypass):
        return ref
    entry = _find_model_catalog_entry(ref)
    if entry is None:
        raise ValueError(
            f"Model '{ref}' is not in the featured catalog. "
            "Pick a featured model for this role, or install one via "
            "POST /api/models/pull with a known catalog ref."
        )
    _enforce_role_match(ref, entry, field_name)
    # Keep a full ``<repo>/<file>.gguf`` so resolve_model_path lands on
    # the exact installed quant; fall back to the catalog ref otherwise.
    if ref.endswith(".gguf") and ref.count("/") >= 2:
        return ref
    canonical: str = entry.ref
    return canonical
