"""Catalog/role validation helpers and the :func:`ConfigField` wrapper."""

import os
import sys
from typing import TYPE_CHECKING, Any

from pydantic import Field

from lilbee.providers.model_ref import PROVIDER_PREFIXES

if TYPE_CHECKING:
    from lilbee.catalog.types import ModelTask


def ConfigField(  # noqa: N802  pydantic Field wrapper; matches Field's PascalCase
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


class TaskMismatchError(ValueError):
    """A role slot was assigned a model whose catalog task does not match.

    Carries the structured fields so each surface (HTTP, CLI, TUI, MCP)
    can format its own user-facing message. The default ``str()`` form is
    surface-neutral so it is safe to surface unmodified.
    """

    def __init__(self, ref: str, entry_task: "ModelTask", expected_task: "ModelTask") -> None:
        self.ref = ref
        self.entry_task = entry_task
        self.expected_task = expected_task
        super().__init__(f"Model '{ref}' is a {entry_task} model, not {expected_task}.")


# A native GGUF ref of the form ``<owner>/<repo>/<file>.gguf`` has at least
# two ``/`` separators; one-slash refs are bare repo IDs.
_NATIVE_GGUF_REF_MIN_SLASHES = 2


def _find_model_catalog_entry(ref: str) -> Any:
    # circular import: catalog imports cfg.
    from lilbee.catalog import find_catalog_entry

    return find_catalog_entry(ref)


def _resolve_installed_task(ref: str) -> Any:
    """Return the ``ModelTask`` for an installed non-featured *ref*, or ``None``.

    Featured is a discovery overlay, not an admission check: any model the
    user has actually pulled is a valid role assignment, even if its hf_repo
    is not in ``FEATURED_ALL``. The picker shows installed models from the
    registry, so the validator has to accept the same set or the picker is a
    trap. ``reclassify_by_name`` mirrors the picker's bucketing so a ref
    whose name reads as "reranker" / "vision" lands in the same role here.
    """
    from lilbee.catalog.types import ModelTask
    from lilbee.core.config import cfg
    from lilbee.modelhub.model_manager.discovery import reclassify_by_name
    from lilbee.modelhub.registry import ModelRegistry

    try:
        manifest = ModelRegistry(cfg.models_dir).get_manifest(ref)
    except (OSError, ValueError):
        return None
    if manifest is None:
        return None
    return ModelTask(reclassify_by_name(ref, manifest.task))


def _skips_catalog_check(ref: str, *, allow_bypass: bool) -> bool:
    """Whether *ref* skips the catalog check."""
    if not ref or not ref.strip():
        return True
    if allow_bypass and _model_task_validation_bypassed():
        return True
    return ref.split("/", 1)[0] in PROVIDER_PREFIXES


def validate_model_task_assignment(field_name: str, ref: str, *, allow_bypass: bool = True) -> str:
    """Check *ref* is assignable to *field_name*; return the canonical ref.

    Accepts featured catalog refs and installed non-featured refs (any model
    the user has pulled). Raises ``TaskMismatchError`` on role mismatch and
    ``ValueError`` when the model is neither featured nor installed.
    """
    from lilbee.catalog.types import ModelTask

    if _skips_catalog_check(ref, allow_bypass=allow_bypass):
        return ref
    want = ModelTask(_MODEL_FIELD_TO_TASK[field_name])

    entry = _find_model_catalog_entry(ref)
    if entry is not None:
        if entry.task != want:
            raise TaskMismatchError(ref, ModelTask(entry.task), want)
        # Keep a full ``<repo>/<file>.gguf`` so resolve_model_path lands on
        # the exact installed quant; fall back to the catalog ref otherwise.
        if ref.endswith(".gguf") and ref.count("/") >= _NATIVE_GGUF_REF_MIN_SLASHES:
            return ref
        canonical: str = entry.ref
        return canonical

    installed_task = _resolve_installed_task(ref)
    if installed_task is None:
        raise ValueError(
            f"Model '{ref}' is not installed. "
            "Install it with 'lilbee model pull <ref>' "
            "(or POST /api/models/pull) before assigning it to a role."
        )
    if installed_task != want:
        raise TaskMismatchError(ref, installed_task, want)
    return ref
