"""Unit tests for the multi-GPU snapshot parsing and span assertion."""

from __future__ import annotations

from gpu_snapshot import GpuSnapshot, parse_nvidia_smi

_TWO_H200_GIANT = "0, 119000, 143771, 87\n1, 111000, 143771, 91\n"
_SINGLE_CARD = "0, 41000, 81920, 64\n1, 350, 81920, 0\n2, 350, 81920, 0\n"


def _snap(csv: str, min_devices: int = 2, min_used_mb: int = 2000) -> GpuSnapshot:
    return GpuSnapshot(
        label="t",
        devices=parse_nvidia_smi(csv),
        min_devices=min_devices,
        min_used_mb=min_used_mb,
    )


def test_giant_spans_two_cards() -> None:
    snap = _snap(_TWO_H200_GIANT)
    assert snap.spans_multi_gpu
    assert len(snap.loaded_devices) == 2
    assert snap.total_used_mb == 230000


def test_single_card_load_is_not_multi_gpu() -> None:
    snap = _snap(_SINGLE_CARD)
    assert not snap.spans_multi_gpu
    assert len(snap.loaded_devices) == 1


def test_idle_cards_excluded_by_threshold() -> None:
    # Two cards just above idle but below the weights threshold do not count.
    snap = _snap("0, 800, 81920, 0\n1, 900, 81920, 0\n", min_used_mb=2000)
    assert not snap.spans_multi_gpu


def test_malformed_rows_skipped() -> None:
    snap = _snap("garbage\n0, 119000, 143771, 87\n\n1, 111000, 143771, 91\n")
    assert len(snap.devices) == 2
    assert snap.spans_multi_gpu
