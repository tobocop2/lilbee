"""AMD exposes three numeric visibility variables, not two."""

from __future__ import annotations

import pytest

from lilbee.providers.fleet import devices as devices_mod

_AMD_VARS = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _AMD_VARS:
        monkeypatch.delenv(var, raising=False)


class TestTheOrdinalVariableIsRespected:
    """GPU_DEVICE_ORDINAL is a first-class ROCm mask and was invisible here.

    Writing HIP_VISIBLE_DEVICES on top of an ordinal mask both overrides it and
    re-exposes hidden cards, because the indices lilbee enumerated are relative to
    the already-filtered list the ordinal produced.
    """

    def test_an_existing_ordinal_mask_is_the_one_written(self, monkeypatch) -> None:
        monkeypatch.setenv("GPU_DEVICE_ORDINAL", "1")
        assert devices_mod.amd_visible_var() == "GPU_DEVICE_ORDINAL"

    def test_hip_still_wins_when_it_is_the_one_set(self, monkeypatch) -> None:
        # Matches the runtime's own precedence: HIP first, then the ordinal.
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("GPU_DEVICE_ORDINAL", "1")
        assert devices_mod.amd_visible_var() == "HIP_VISIBLE_DEVICES"

    def test_rocr_still_wins_when_it_alone_restricts(self, monkeypatch) -> None:
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
        assert devices_mod.amd_visible_var() == "ROCR_VISIBLE_DEVICES"

    def test_hip_is_the_default_when_nothing_restricts(self) -> None:
        assert devices_mod.amd_visible_var() == "HIP_VISIBLE_DEVICES"

    def test_an_empty_ordinal_does_not_claim_precedence(self, monkeypatch) -> None:
        # Empty means "no devices", not "this is the variable in use".
        monkeypatch.setenv("GPU_DEVICE_ORDINAL", "")
        assert devices_mod.amd_visible_var() == "HIP_VISIBLE_DEVICES"
