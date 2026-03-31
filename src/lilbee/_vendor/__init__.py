"""Vendored dependencies for lilbee platform wheels.

When lilbee is installed from a platform-specific wheel, llama_cpp is
vendored here (including compiled shared libraries). Call
``ensure_vendor_importable()`` before importing llama_cpp to add this
directory to sys.path so the vendored copy is found.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_DIR = Path(__file__).resolve().parent


def ensure_vendor_importable() -> None:
    """Add the vendor directory to sys.path if llama_cpp is vendored there."""
    vendored_llama = _VENDOR_DIR / "llama_cpp"
    if not vendored_llama.exists():
        return
    vendor_str = str(_VENDOR_DIR)
    if vendor_str not in sys.path:
        sys.path.insert(0, vendor_str)
