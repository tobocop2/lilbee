"""The AMD visibility variables lilbee writes are the two the runtime reads."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import lilbee
from lilbee.core.config import cfg
from lilbee.providers.fleet import devices as devices_mod
from lilbee.providers.fleet.devices import FleetDevice, visible_env
from lilbee.providers.fleet.gpu_env import gpu_environ, shard_visible_devices

_ORDINAL = "GPU_DEVICE_ORDINAL"
_AMD_VARS = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", _ORDINAL)


def _rocm_card(index: int) -> tuple[FleetDevice, ...]:
    return (FleetDevice("ROCm", index, "card", 8 << 30, 8 << 30),)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in _AMD_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cfg, "gpu_devices", "")


class TestTheOrdinalIsNoMask:
    """The ROCm runtime ignores the ordinal, so lilbee never reads or writes it."""

    def test_an_inherited_ordinal_is_not_the_variable_written(self, monkeypatch) -> None:
        monkeypatch.setenv(_ORDINAL, "1")
        assert devices_mod.amd_visible_var(os.environ) == "HIP_VISIBLE_DEVICES"

    def test_hip_is_written_when_only_the_ordinal_is_set(self, monkeypatch) -> None:
        monkeypatch.setenv(_ORDINAL, "1")
        assert visible_env(_rocm_card(0)) == {"HIP_VISIBLE_DEVICES": "0"}

    def test_rocr_is_written_when_it_alone_restricts_beside_an_ordinal(self, monkeypatch) -> None:
        monkeypatch.setenv(_ORDINAL, "1")
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3")
        assert visible_env(_rocm_card(1)) == {"ROCR_VISIBLE_DEVICES": "3"}

    def test_visible_env_composes_on_hip_not_on_the_ordinal(self, monkeypatch) -> None:
        monkeypatch.setenv(_ORDINAL, "7,8,9")
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "4,5")
        assert visible_env(_rocm_card(1)) == {"HIP_VISIBLE_DEVICES": "5"}

    def test_gpu_devices_pin_never_lands_on_the_ordinal(self, monkeypatch) -> None:
        monkeypatch.setattr(cfg, "gpu_devices", "2,3")
        monkeypatch.setenv(_ORDINAL, "7")
        env = gpu_environ()
        assert env["HIP_VISIBLE_DEVICES"] == "2,3"
        assert env[_ORDINAL] == "7"

    def test_shard_visible_devices_has_no_ordinal_key(self, monkeypatch) -> None:
        monkeypatch.setenv(_ORDINAL, "7")
        assert set(shard_visible_devices(1)) == {
            "GGML_VK_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
            "HIP_VISIBLE_DEVICES",
        }

    def test_no_source_names_the_ordinal(self) -> None:
        root = Path(lilbee.__file__).parent
        texts = {path: path.read_text(encoding="utf-8") for path in root.rglob("*.py")}
        assert texts
        assert any("HIP_VISIBLE_DEVICES" in text for text in texts.values())
        assert [str(p) for p, text in texts.items() if _ORDINAL in text] == []


class TestAnEmptyOrdinalIsIgnoredToo:
    def test_hip_is_the_default_when_nothing_restricts(self) -> None:
        assert devices_mod.amd_visible_var(os.environ) == "HIP_VISIBLE_DEVICES"

    def test_rocr_still_wins_when_it_alone_restricts(self, monkeypatch) -> None:
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
        assert devices_mod.amd_visible_var(os.environ) == "ROCR_VISIBLE_DEVICES"

    def test_hip_wins_over_rocr(self, monkeypatch) -> None:
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
        assert devices_mod.amd_visible_var(os.environ) == "HIP_VISIBLE_DEVICES"

    def test_an_empty_hip_does_not_claim_precedence(self, monkeypatch) -> None:
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "")
        monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1")
        assert devices_mod.amd_visible_var(os.environ) == "ROCR_VISIBLE_DEVICES"
