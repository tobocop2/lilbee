"""Tests for ``lilbee.runtime.hardware.compute_fit``."""

from __future__ import annotations

from lilbee.runtime.hardware import FitLevel, compute_fit

_GB = 1024**3


def test_fits_when_headroom_at_least_one_gb() -> None:
    chip = compute_fit(model_size_bytes=4 * _GB, available_bytes=8 * _GB)
    assert chip.level is FitLevel.FITS
    assert chip.headroom_gb == 4.0


def test_tight_when_headroom_under_one_gb() -> None:
    chip = compute_fit(model_size_bytes=int(7.5 * _GB), available_bytes=8 * _GB)
    assert chip.level is FitLevel.TIGHT
    assert 0 <= chip.headroom_gb < 1


def test_tight_at_exact_zero_headroom() -> None:
    chip = compute_fit(model_size_bytes=8 * _GB, available_bytes=8 * _GB)
    assert chip.level is FitLevel.TIGHT
    assert chip.headroom_gb == 0.0


def test_wont_run_when_oversized() -> None:
    chip = compute_fit(model_size_bytes=10 * _GB, available_bytes=8 * _GB)
    assert chip.level is FitLevel.WONT_RUN
    assert chip.headroom_gb == -2.0


def test_fits_at_exact_one_gb_boundary() -> None:
    chip = compute_fit(model_size_bytes=7 * _GB, available_bytes=8 * _GB)
    assert chip.level is FitLevel.FITS
    assert chip.headroom_gb == 1.0


def test_chip_is_immutable() -> None:
    import dataclasses

    chip = compute_fit(model_size_bytes=4 * _GB, available_bytes=8 * _GB)
    try:
        chip.level = FitLevel.WONT_RUN  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("FitChip should be frozen")
