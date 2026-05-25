"""Resolve the ``llama-server`` binary for the managed multi-GPU fleet."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from lilbee.providers.base import ProviderError

_BINARY_NAME = "llama-server"
_INSTALL_HINT = (
    "Install with `pip install 'lilbee[multi-gpu]'`, or set LILBEE_LLAMA_SERVER_PATH "
    "to a llama-server binary (a llama.cpp release download or `brew install llama.cpp`)."
)


def resolve_llama_server_binary() -> Path:
    """Resolve the executable: bundled wheel -> configured path -> PATH -> error.

    Never downloads anything; the binary arrives only via the explicit
    ``lilbee[multi-gpu]`` install or bring-your-own.
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
    """Env additions so the *bundled* llama-server finds its shared backend libs.

    The bundled binary is dynamically linked and ships without ggml/llama/mtmd;
    those come from the version-matched ``llama-cpp-python`` lilbee already
    bundles, so we append that package's ``lib`` dir to the platform's library
    search path. Returns ``{}`` for a bring-your-own binary (it carries its own
    libs) or when the lib dir can't be located.
    """
    if _bundled_binary() is None:
        return {}
    lib_dir = _llama_cpp_lib_dir()
    if lib_dir is None:
        return {}
    var = _lib_path_var()
    existing = os.environ.get(var, "")
    value = f"{existing}{os.pathsep}{lib_dir}" if existing else str(lib_dir)
    return {var: value}


def _llama_cpp_lib_dir() -> Path | None:
    """The ``lib`` dir of the installed llama-cpp-python, where its shared libs live."""
    try:
        import llama_cpp
    except ImportError:  # pragma: no cover - llama_cpp is a core dep, always present
        return None
    lib = Path(llama_cpp.__file__).parent / "lib"
    return lib if lib.is_dir() else None


def _lib_path_var() -> str:
    """The dynamic-library search-path env var for the current platform."""
    if sys.platform == "darwin":
        return "DYLD_LIBRARY_PATH"
    if sys.platform == "win32":
        return "PATH"
    return "LD_LIBRARY_PATH"
