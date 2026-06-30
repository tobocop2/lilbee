"""Apple Metal GPU utilization stub.

Util and temperature require either IOReport (private macOS framework, no public
headers, ctypes ABI shifts across OS versions) or powermetrics (needs sudo).
Neither is safe to ship without empirical validation on real Apple Silicon.

VRAM comes from the structural probe (llama-server --list-devices) and is not
repeated here. The orchestrator in gpu_stats.py keeps structural VRAM when a
backend returns free_bytes=0, total_bytes=0.

# TODO(apple-ioreport): implement via ctypes IOReport when validated on-device.
"""

from __future__ import annotations

from lilbee.providers.fleet.gpu_backends.base import UtilSample

# The backend string as emitted by llama-server --list-devices on Apple Silicon:
# "MTL0: Apple M1 Pro (21845 MiB, 21844 MiB free)" -> prefix "MTL".
BACKEND_KEY = "MTL"


class AppleBackend:
    """Apple Metal util stub; returns {} until IOReport path is validated."""

    def sample(self, indices: frozenset[int]) -> dict[int, UtilSample]:
        _ = indices  # stub: no tool available without sudo or private framework
        return {}
