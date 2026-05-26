"""Bundles the compiled llama.cpp ``llama-server`` binary for ``lilbee[multi-gpu]``.

CI builds the platform/backend-specific binary into ``bin/`` (see
``tools/wheel-build/build_llama_server.sh``) and ships it as a per-platform wheel.
``lilbee.providers.multi_gpu.binary`` imports this and calls ``get_binary_path()``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BIN_DIR = Path(__file__).parent / "bin"


def get_binary_path() -> Path:
    """Absolute path to the bundled ``llama-server`` executable."""
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    return _BIN_DIR / name
