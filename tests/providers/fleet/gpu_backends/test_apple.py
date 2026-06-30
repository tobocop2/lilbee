"""Tests for the Apple Metal utilization stub."""

from __future__ import annotations

from lilbee.providers.fleet import gpu_backends
from lilbee.providers.fleet.gpu_backends import apple as apple_mod
from lilbee.providers.fleet.gpu_backends.apple import BACKEND_KEY, AppleBackend

_INDICES = frozenset({0})


def test_sample_returns_empty() -> None:
    assert AppleBackend().sample(_INDICES) == {}


def test_stub_does_not_import_subprocess() -> None:
    """The Metal stub must not depend on subprocess — no CLI is available."""
    assert not hasattr(apple_mod, "subprocess"), (
        "apple.py must not import subprocess; the stub has no CLI to run"
    )


def test_sample_accepts_any_indices() -> None:
    assert AppleBackend().sample(frozenset()) == {}
    assert AppleBackend().sample(frozenset({0, 1, 2})) == {}


def test_backend_key_is_mtl() -> None:
    assert BACKEND_KEY == "MTL"


def test_registry_maps_mtl_to_apple_backend() -> None:
    assert isinstance(gpu_backends.resolve_backend("MTL"), AppleBackend)


def test_registry_returns_none_for_unknown() -> None:
    assert gpu_backends.resolve_backend("Vulkan") is None
    assert gpu_backends.resolve_backend("BLAS") is None
    assert gpu_backends.resolve_backend("") is None
