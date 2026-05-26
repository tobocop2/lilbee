"""Resolve the ``llama-server`` binary for the local engine fleet."""

from __future__ import annotations

import shutil
from pathlib import Path

from lilbee.providers.base import ProviderError

_BINARY_NAME = "llama-server"
_INSTALL_HINT = (
    "Reinstall lilbee to get the bundled engine, or set LILBEE_LLAMA_SERVER_PATH "
    "to a llama-server binary (a llama.cpp release download or `brew install llama.cpp`)."
)


def resolve_llama_server_binary() -> Path:
    """Resolve the executable: bundled wheel -> configured path -> PATH -> error.

    Never downloads anything; the binary arrives via the bundled
    ``lilbee-llama-server`` wheel or bring-your-own.
    """
    bundled = _bundled_binary()
    if bundled is not None:
        return bundled

    from lilbee.core.config import cfg

    if cfg.llama_server_path:
        configured = Path(cfg.llama_server_path)
        if not configured.is_file():
            raise ProviderError(f"LILBEE_LLAMA_SERVER_PATH is not a file: {configured}")
        return configured

    found = shutil.which(_BINARY_NAME)
    if found is not None:
        return Path(found)

    raise ProviderError(f"{_BINARY_NAME} binary not found. {_INSTALL_HINT}")


def _bundled_binary() -> Path | None:
    """Binary from the ``lilbee-llama-server`` wheel, or ``None`` if not installed."""
    try:
        import lilbee_llama_server
    except ImportError:
        return None
    path = Path(lilbee_llama_server.get_binary_path())  # pragma: no cover - wheel absent in tests
    return path if path.is_file() else None  # pragma: no cover


def llama_server_runtime_env() -> dict[str, str]:
    """Extra environment for the spawned llama-server. Empty on every platform.

    The bundled wheel ships its own ggml/llama/mtmd next to the binary with a baked
    rpath (``@loader_path`` on macOS, ``$ORIGIN`` on Linux), and a bring-your-own
    binary carries its own libraries, so the fleet never has to inject a library
    search path. Kept as the single hook for any future per-spawn environment.
    """
    return {}
