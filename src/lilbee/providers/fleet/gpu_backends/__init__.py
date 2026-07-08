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


_VULKAN = "Vulkan"
# Vulkan is vendor-agnostic, and a consumer GPU is often only exposed to the
# engine via Vulkan. Map a Vulkan device to a vendor's util backend by the vendor
# named in its device string so its utilization still reads.
_VENDOR_KEYS: tuple[tuple[str, str], ...] = (
    ("intel", "SYCL"),
    ("nvidia", "CUDA"),
    ("radeon", "ROCm"),
    ("amd", "ROCm"),
)


def util_backend_name(backend: str, name: str) -> str:
    """Registry key for a device's util backend.

    A recognized inference backend already implies the vendor. Vulkan does not, so
    a Vulkan device is mapped by the vendor named in *name*; anything unrecognized
    is returned unchanged (and resolves to no backend, i.e. structural fallback).
    """
    if backend in _REGISTRY:
        return backend
    if backend == _VULKAN:
        lowered = name.lower()
        for hint, key in _VENDOR_KEYS:
            if hint in lowered:
                return key
    return backend


__all__ = [
    "UtilBackend",
    "UtilSample",
    "resolve_backend",
    "util_backend_name",
]
