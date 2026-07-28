"""Which backend wins when two rank the same.

CUDA, ROCm and HIP all sit at the same rank, so the tie is decided on memory.
Summing raw totals lets a shared-heap carveout outbid a discrete card, and the
carveout is host RAM that the host budget is already counting.
"""

from __future__ import annotations

from lilbee.providers.fleet.devices import FleetDevice, _backend_preference

_GB = 1024**3


def _dev(backend: str, idx: int, total: int, *, unified: bool) -> FleetDevice:
    return FleetDevice(backend, idx, f"{backend} dev {idx}", total, total, unified=unified)


def test_a_discrete_card_outranks_a_larger_shared_heap() -> None:
    discrete = ("CUDA", [_dev("CUDA", 0, 16 * _GB, unified=False)])
    carveout = ("ROCm", [_dev("ROCm", 0, 32 * _GB, unified=True)])
    assert max([discrete, carveout], key=_backend_preference) == discrete


def test_two_discrete_backends_still_compare_on_size() -> None:
    small = ("CUDA", [_dev("CUDA", 0, 16 * _GB, unified=False)])
    large = ("ROCm", [_dev("ROCm", 0, 24 * _GB, unified=False)])
    assert max([small, large], key=_backend_preference) == large


def test_two_shared_heaps_still_compare_on_size() -> None:
    small = ("CUDA", [_dev("CUDA", 0, 16 * _GB, unified=True)])
    large = ("ROCm", [_dev("ROCm", 0, 24 * _GB, unified=True)])
    assert max([small, large], key=_backend_preference) == large
