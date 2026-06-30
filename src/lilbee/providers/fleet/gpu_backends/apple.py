"""Apple Metal GPU utilization stub."""

from __future__ import annotations

from lilbee.providers.fleet.gpu_backends.base import UtilSample

# The backend string as emitted by llama-server --list-devices on Apple Silicon:
# "MTL0: Apple M1 Pro (21845 MiB, 21844 MiB free)" -> prefix "MTL".
# Build-dependent variants (e.g. "Metal") are also registered in __init__.py.
BACKEND_KEY = "MTL"


# TODO(apple-ioreport): implement via ctypes IOReport when validated on-device.
# powermetrics needs sudo; IOReport is a private macOS framework whose ctypes ABI
# shifts across OS versions. Neither is safe to ship without empirical validation.
# VRAM comes from the structural probe (llama-server --list-devices); the
# orchestrator keeps it when this backend returns free_bytes=0, total_bytes=0.
class AppleBackend:
    """Apple Metal util stub; returns {} until IOReport path is validated."""

    def sample(self, indices: frozenset[int]) -> dict[int, UtilSample]:
        _ = indices  # stub: no tool available without sudo or private framework
        return {}
