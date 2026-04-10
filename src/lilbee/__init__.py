"""lilbee — Local RAG knowledge base."""

from __future__ import annotations

import os

# Disable huggingface_hub's xet transfer layer before any HF submodule loads.
# huggingface_hub.constants reads HF_HUB_DISABLE_XET at import time, so this
# must run before the first `import huggingface_hub` anywhere in the process.
# Workaround for HF issue #4058: xet-core reports progress in 3-4 coarse jumps
# instead of continuously, making download bars appear stuck on large files.
# Forcing the HTTP path restores smooth per-chunk tqdm updates. Users can still
# opt back into xet by setting HF_HUB_DISABLE_XET=0 in their environment.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lilbee.api import Lilbee

__all__ = ["Lilbee"]


def __getattr__(name: str) -> object:
    if name == "Lilbee":
        from lilbee.api import Lilbee

        return Lilbee
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
