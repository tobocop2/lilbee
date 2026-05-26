"""Resolve the ``llama-server`` binary for the managed multi-GPU fleet."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from lilbee.providers.base import ProviderError

_BINARY_NAME = "llama-server"
_INSTALL_HINT = (
    "Reinstall lilbee to get the bundled engine, or set LILBEE_LLAMA_SERVER_PATH "
    "to a llama-server binary (a llama.cpp release download or `brew install llama.cpp`)."
)
# Library-name stems that mark a self-contained bundle: when the engine wheel
# ships ggml/llama next to the binary (with a baked rpath), no external lib dir
# is needed.
_COLOCATED_LIB_STEMS = ("libllama", "libggml", "llama")


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

    A self-contained engine wheel ships ggml/llama next to the binary with a baked
    rpath and needs nothing here. A bundle that ships only the binary borrows the
    version-matched libs from the ``llama-cpp-python`` package by appending its
    ``lib`` dir to the platform library search path. Returns ``{}`` for a
    bring-your-own binary, a self-contained bundle, or when no borrow dir is found.
    """
    bundled = _bundled_binary()
    if bundled is None or _has_colocated_libs(bundled):
        return {}
    lib_dir = _llama_cpp_lib_dir()
    if lib_dir is None:
        return {}
    var = _lib_path_var()
    existing = os.environ.get(var, "")
    value = f"{existing}{os.pathsep}{lib_dir}" if existing else str(lib_dir)
    return {var: value}


def _has_colocated_libs(binary: Path) -> bool:
    """True when shared backend libs sit next to *binary* (a self-contained bundle)."""
    parent = binary.parent
    return any(
        child.name.startswith(_COLOCATED_LIB_STEMS) and child.suffix in (".so", ".dylib", ".dll")
        for child in parent.glob("*")
        if child.is_file()
    )


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
