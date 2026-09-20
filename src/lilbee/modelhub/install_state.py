"""The single definition of where a native model ref loads from."""

from __future__ import annotations

from enum import StrEnum

from lilbee.modelhub.registry import ModelRegistry
from lilbee.providers.model_ref import is_loose_model_file


class InstallState(StrEnum):
    """Where a native model ref loads from."""

    REGISTERED = "registered"
    LOOSE_FILE = "loose_file"
    MISSING = "missing"


def install_state(ref: str, registry: ModelRegistry) -> InstallState:
    """Where *ref* loads from: a registry manifest, a loose GGUF file, or nowhere.

    A loose file loads in the TUI, the CLI and the fleet, but no model listing
    shows it and the HTTP chat route cannot resolve it.
    """
    if registry.is_installed(ref):
        return InstallState.REGISTERED
    if is_loose_model_file(ref):
        return InstallState.LOOSE_FILE
    return InstallState.MISSING
