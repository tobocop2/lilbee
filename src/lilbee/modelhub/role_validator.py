"""Role-slot assignment validation for the four model config fields."""

import os
import sys
from typing import Any

from lilbee.catalog import find_catalog_entry
from lilbee.catalog.refs import is_bare_hf_repo
from lilbee.catalog.types import ModelTask
from lilbee.core.config import cfg
from lilbee.modelhub.model_manager.discovery import reclassify_by_name
from lilbee.modelhub.registry import ModelRegistry
from lilbee.providers.model_ref import PROVIDER_PREFIXES

# Test-only bypass. Both the env var and pytest must be present so a
# leaked env var cannot disable validation in production.
_SKIP_MODEL_TASK_VALIDATION_ENV = "LILBEE_SKIP_MODEL_TASK_VALIDATION"

_MODEL_FIELD_TO_TASK: dict[str, str] = {
    "chat_model": "chat",
    "embedding_model": "embedding",
    "vision_model": "vision",
    "reranker_model": "rerank",
}

# A native GGUF ref of the form ``<owner>/<repo>/<file>.gguf`` has at least
# two ``/`` separators; one-slash refs are bare repo IDs.
_NATIVE_GGUF_REF_MIN_SLASHES = 2


class TaskMismatchError(ValueError):
    """A role slot was assigned a model whose catalog task does not match.

    Carries the structured fields so each surface (HTTP, CLI, TUI, MCP)
    can format its own user-facing message. The default ``str()`` form is
    surface-neutral so it is safe to surface unmodified.
    """

    def __init__(self, ref: str, entry_task: ModelTask, expected_task: ModelTask) -> None:
        self.ref = ref
        self.entry_task = entry_task
        self.expected_task = expected_task
        super().__init__(f"Model '{ref}' is a {entry_task} model, not {expected_task}.")


def _model_task_validation_bypassed() -> bool:
    if not os.environ.get(_SKIP_MODEL_TASK_VALIDATION_ENV):
        return False
    return sys.modules.get("pytest") is not None


def _resolve_installed_task(registry: ModelRegistry, ref: str) -> ModelTask | None:
    """Return the manifest's ``ModelTask`` for *ref*, name-reclassified, or ``None``."""
    manifest = registry.get_manifest(ref)
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


def _canonical_featured_ref(ref: str, entry: Any, want: ModelTask) -> str:
    """Role-check a featured entry and pick the canonical ref to persist."""
    if entry.task != want:
        raise TaskMismatchError(ref, ModelTask(entry.task), want)
    # Keep a full ``<repo>/<file>.gguf`` so resolve_model_path lands on
    # the exact installed quant; fall back to the catalog ref otherwise.
    if ref.endswith(".gguf") and ref.count("/") >= _NATIVE_GGUF_REF_MIN_SLASHES:
        return ref
    canonical: str = entry.ref
    return canonical


def _validate_installed_ref(ref: str, want: ModelTask) -> str:
    """Role-check a non-featured ref by consulting the installed registry.

    A bare ``<org>/<repo>`` ref canonicalizes to its installed quant's full
    ref so the persisted value always names the exact GGUF file.
    """
    registry = ModelRegistry(cfg.models_dir)
    if is_bare_hf_repo(ref):
        ref = registry.installed_ref_for_repo(ref) or ref
    installed_task = _resolve_installed_task(registry, ref)
    if installed_task is None:
        raise ValueError(
            f"Model '{ref}' is not installed. "
            "Install it with 'lilbee model pull <ref>' "
            "(or POST /api/models/pull) before assigning it to a role."
        )
    if installed_task != want:
        raise TaskMismatchError(ref, installed_task, want)
    return ref


def validate_model_task_assignment(field_name: str, ref: str, *, allow_bypass: bool = True) -> str:
    """Check *ref* is assignable to *field_name*; return the canonical ref.

    Accepts featured catalog refs and installed non-featured refs (any model
    the user has pulled). Raises ``TaskMismatchError`` on role mismatch and
    ``ValueError`` when the model is neither featured nor installed.
    """
    if _skips_catalog_check(ref, allow_bypass=allow_bypass):
        return ref
    want = ModelTask(_MODEL_FIELD_TO_TASK[field_name])
    entry: Any = find_catalog_entry(ref)
    if entry is not None:
        return _canonical_featured_ref(ref, entry, want)
    return _validate_installed_ref(ref, want)
