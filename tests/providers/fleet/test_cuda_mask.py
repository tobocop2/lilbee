"""One reader of CUDA_VISIBLE_DEVICES: the card listing and the model cache keep the same cards.

The measured runtime behaviour (driver 580 on a one-card GTX 1070 Ti, libcuda cuInit and
cuDeviceGetCount) is in the case ids below: a zero-padded index and a UUID prefix name a
card, the prefix ``GPU-`` alone and a lower-case ``gpu-`` prefix name none, and a repeated
index leaves no device visible.
"""

from __future__ import annotations

import pytest

from lilbee.providers.fleet import devices
from lilbee.providers.fleet.gpu_backends import nvidia as nvidia_mod
from lilbee.providers.fleet.gpu_backends.nvidia import CardTotal, NvidiaCard
from lilbee.providers.fleet.gpu_env import cuda_visible_cards
from lilbee.providers.model_cache import _try_nvidia_memory

_GIB = 1024**3
_UUIDS = ("GPU-aaaa1111", "GPU-bbbb2222", "GPU-bbcc3333")
_TOTALS = (1 * _GIB, 2 * _GIB, 4 * _GIB)

# mask -> printed indexes of the cards the runtime keeps, in its order.
_MASKS = {
    "unset-keeps-all": (None, [0, 1, 2]),
    "plain-index": ("1", [1]),
    "padded-index": ("02", [2]),
    "padded-zero": ("00,01", [0, 1]),
    "whitespace-around-entries": (" 1 , 0 ", [1, 0]),
    "full-uuid": ("GPU-aaaa1111", [0]),
    "unique-prefix": ("GPU-aaaa", [0]),
    "prefix-then-index": ("GPU-bbbb,0", [1, 0]),
    "ambiguous-prefix": ("GPU-bb", []),
    "ambiguous-prefix-after-a-card": ("0,GPU-bb,1", [0]),
    "bare-gpu-prefix": ("GPU-", []),
    "lower-case-prefix": ("gpu-aaaa", []),
    "mig-entry": ("MIG-aaaa1111", []),
    "unknown-index-stops": ("0,9,1", [0]),
    "unknown-uuid-stops": ("2,GPU-zzzz,1", [2]),
    "empty-mask": ("", []),
    "empty-entry-stops": ("0,,1", [0]),
    "repeated-index": ("1,1", []),
    "repeated-through-prefix": ("GPU-bbbb,1", []),
    "non-ascii-digit": ("²", []),
    "negative-index": ("-1", []),
    "repeat-after-a-stop-is-not-reached": ("1,9,1", [1]),
}


def _set_mask(monkeypatch: pytest.MonkeyPatch, mask: str | None) -> None:
    if mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)


def _totals() -> list[CardTotal]:
    return [
        CardTotal(i, uuid, total)
        for i, (uuid, total) in enumerate(zip(_UUIDS, _TOTALS, strict=True))
    ]


def _smi_cards() -> list[NvidiaCard]:
    return [
        NvidiaCard(i, uuid, f"card{i}", total, total)
        for i, (uuid, total) in enumerate(zip(_UUIDS, _TOTALS, strict=True))
    ]


@pytest.mark.parametrize(("mask", "indexes"), list(_MASKS.values()), ids=list(_MASKS))
def test_the_reader_keeps_the_cards_the_runtime_keeps(
    mask: str | None, indexes: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_mask(monkeypatch, mask)
    assert [card.index for card in cuda_visible_cards(_totals())] == indexes


@pytest.mark.parametrize(("mask", "indexes"), list(_MASKS.values()), ids=list(_MASKS))
def test_the_card_listing_and_the_model_cache_agree_on_every_mask(
    mask: str | None, indexes: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each card has its own power-of-two total, so a sum names the set of cards kept."""
    _set_mask(monkeypatch, mask)
    monkeypatch.setattr(nvidia_mod, "cards", _smi_cards)
    monkeypatch.setattr("lilbee.providers.model_cache._nvidia_device_totals", _totals)
    listed = [d.total_bytes for d in devices.visible_nvidia_cards()]
    assert listed == [_TOTALS[i] for i in indexes]
    expected = sum(_TOTALS[i] for i in indexes) or None
    assert _try_nvidia_memory(sum) == expected


def test_a_mask_names_the_printed_index_in_the_card_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cards that print indexes 1 and 2 sit at list positions 0 and 1."""
    cards = [
        NvidiaCard(1, "GPU-aaaa1111", "card1", _GIB, _GIB),
        NvidiaCard(2, "GPU-bbbb2222", "card2", 2 * _GIB, 2 * _GIB),
    ]
    monkeypatch.setattr(nvidia_mod, "cards", lambda: cards)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert [d.name for d in devices.visible_nvidia_cards()] == ["card2"]


def test_an_unset_mask_sizes_the_budget_from_the_cards_that_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cards = [CardTotal(0, "GPU-aaaa1111", None), CardTotal(1, "GPU-bbbb2222", _GIB)]
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr("lilbee.providers.model_cache._nvidia_device_totals", lambda: cards)
    assert _try_nvidia_memory(sum) == _GIB


def test_a_mask_naming_an_unread_card_leaves_the_budget_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cards = [CardTotal(0, "GPU-aaaa1111", None), CardTotal(1, "GPU-bbbb2222", _GIB)]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr("lilbee.providers.model_cache._nvidia_device_totals", lambda: cards)
    assert _try_nvidia_memory(sum) is None


def test_an_unread_card_stays_in_a_mask_that_names_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reader keeps a card whose total does not read; the caller decides what that means."""
    cards = [CardTotal(0, "GPU-aaaa1111", None), CardTotal(1, "GPU-bbbb2222", _GIB)]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert cuda_visible_cards(cards) == cards[:1]
