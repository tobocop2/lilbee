"""Bundles lilbee's engine binaries: llama-server, llama-swap, gguf-parser.

CI fills ``bin/`` per platform (see ``tools/wheel-build/``) and ships this as a
single per-platform wheel. ``lilbee.providers.fleet.binary`` resolves each tool.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BIN_DIR = Path(__file__).parent / "bin"
_EXE = ".exe" if sys.platform == "win32" else ""


def get_llama_server_path() -> Path:
    """Absolute path to the bundled ``llama-server`` executable."""
    return _BIN_DIR / f"llama-server{_EXE}"


def get_llama_swap_path() -> Path:
    """Absolute path to the bundled ``llama-swap`` executable."""
    return _BIN_DIR / f"llama-swap{_EXE}"


def get_gguf_parser_path() -> Path:
    """Absolute path to the bundled ``gguf-parser`` executable."""
    return _BIN_DIR / f"gguf-parser{_EXE}"
