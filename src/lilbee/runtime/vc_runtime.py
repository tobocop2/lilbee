"""What lilbee tells a Windows user whose machine lacks the Visual C++ runtime.

A light module with no lilbee imports: the launcher reads it at start, before the
fleet package is loaded.
"""

from __future__ import annotations

import re

VC_REDIST_URL = "https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist"

# Windows status STATUS_DLL_NOT_FOUND: the loader stopped the process before main.
STATUS_DLL_NOT_FOUND = 0xC0000135

# How a supervisor words that exit code: Go prints a code of 65536 or more in hex
# ("exit status 0xc0000135"); other tools print the decimal. The exact llama-swap
# line is not yet recorded on Windows, so only the status is matched.
_STATUS_PATTERN = re.compile(rf"\b(?:0x{STATUS_DLL_NOT_FOUND:x}|{STATUS_DLL_NOT_FOUND})\b", re.I)


def dll_not_found_status(tail: str) -> bool:
    """Whether an engine's output reports the Windows DLL-not-found exit status."""
    return _STATUS_PATTERN.search(tail) is not None
