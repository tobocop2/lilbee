"""Per-vendor GPU utilization backends.

Each vendor module exposes a class implementing the UtilBackend Protocol. Adding
a new vendor requires one new file and one line in _REGISTRY below; gpu_stats.py
stays untouched.

resolve_backend(device_backend) returns the backend for a given llama-server
backend string, or None when no backend covers that vendor.
"""

from __future__ import annotations

from lilbee.providers.fleet.gpu_backends.amd import AmdBackend
from lilbee.providers.fleet.gpu_backends.apple import BACKEND_KEY as _APPLE_KEY
from lilbee.providers.fleet.gpu_backends.apple import AppleBackend
from lilbee.providers.fleet.gpu_backends.base import UtilBackend, UtilSample
from lilbee.providers.fleet.gpu_backends.intel import IntelBackend
from lilbee.providers.fleet.gpu_backends.nvidia import NvidiaBackend

# Maps the backend string that llama-server --list-devices emits to the backend
# instance. One entry per backend string (HIP and ROCm share an instance).
_apple = AppleBackend()

_REGISTRY: dict[str, UtilBackend] = {
    "CUDA": NvidiaBackend(),
    "ROCm": AmdBackend(),
    "HIP": AmdBackend(),
    "SYCL": IntelBackend(),
    # Apple Metal: register both strings seen in the wild (build-dependent).
    _APPLE_KEY: _apple,
    "Metal": _apple,
}


def resolve_backend(device_backend: str) -> UtilBackend | None:
    """Return the UtilBackend for device_backend, or None if unregistered."""
    return _REGISTRY.get(device_backend)


__all__ = [
    "UtilBackend",
    "UtilSample",
    "resolve_backend",
]
