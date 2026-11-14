"""A preset's memory reserve lowers the admission budget, never the sizing budget."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.app.presets import add_preset
from lilbee.core import settings
from lilbee.core.config import cfg
from lilbee.providers.fleet import planning as planning_mod
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.roles import EngineBackend
from lilbee.providers.server_registry import PRESETS_KEY, is_loopback_url, loopback_reserve_bytes

GB = 1024**3
TOTAL = 64 * GB
FREE = TOTAL - 2 * GB - 8 * GB  # 2 GiB for the OS, 8 GiB held by the preset server
SIZING = 53687091200
ADMISSION = 64424509440
ADMISSION_WITH_RESERVE = 55834574848
LOOPBACK = "http://127.0.0.1:8000/v1"


@pytest.fixture(autouse=True)
def _host(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "lilbee.providers.fleet.engines.llama.resolve_llama_server", lambda: Path("/bin/srv")
    )
    monkeypatch.setattr(
        planning_mod,
        "_read_devices",
        lambda _b, engine: planning_mod.DeviceReading([], EngineBackend.UNKNOWN),
    )
    monkeypatch.setattr("lilbee.providers.model_cache.total_system_memory", lambda: TOTAL)
    monkeypatch.setattr("lilbee.providers.model_cache.free_system_memory", lambda: FREE)
    monkeypatch.setattr(cfg, "system_memory_reserve_gb", 4.0)
    monkeypatch.setattr(cfg, "data_root", tmp_path)
    monkeypatch.setattr(cfg, "server_presets", None)
    planning_mod.clear_plan_probe()
    yield
    planning_mod.clear_plan_probe()


def _budgets() -> tuple[int | None, int | None]:
    return (
        planning_mod._unified_memory_budget([], LLAMA),
        planning_mod._unified_admission_budget([], LLAMA),
    )


def _add(name: str, url: str, reserve: float | None) -> None:
    add_preset(name, url, roles=["chat"], memory_reserve_gb=reserve)


class TestAdmissionBudget:
    def test_no_preset_keeps_the_base_budgets(self):
        assert _budgets() == (SIZING, ADMISSION)

    def test_a_preset_without_a_reserve_keeps_the_base_budgets(self):
        _add("omlx", LOOPBACK, None)
        assert _budgets() == (SIZING, ADMISSION)

    def test_a_loopback_reserve_lowers_admission_and_leaves_sizing(self):
        _add("omlx", LOOPBACK, 8)
        assert _budgets() == (SIZING, ADMISSION_WITH_RESERVE)

    def test_the_reserves_of_two_loopback_presets_add_up(self):
        _add("a", LOOPBACK, 5)
        _add("b", "http://localhost:9000/v1", 3)
        assert _budgets() == (SIZING, ADMISSION_WITH_RESERVE)

    def test_a_remote_preset_reserve_is_ignored(self):
        _add("far", "http://93.184.216.34:8000/v1", 8)
        assert _budgets() == (SIZING, ADMISSION)

    def test_a_private_lan_preset_reserve_is_ignored(self):
        _add("lan", "http://192.168.1.20:8000/v1", 8)
        assert _budgets() == (SIZING, ADMISSION)

    def test_a_reserve_started_after_the_snapshot_still_applies(self):
        planning_mod.capture_plan_probe(engine=LLAMA)
        _add("omlx", LOOPBACK, 8)
        assert _budgets() == (SIZING, ADMISSION_WITH_RESERVE)

    def test_the_budget_stops_at_zero(self):
        _add("omlx", LOOPBACK, 64)
        _add("b", "http://localhost:9000/v1", 64)
        _add("c", "http://[::1]:9001/v1", 64)
        assert _budgets()[1] == 0

    def test_a_dedicated_gpu_host_has_no_unified_budget(self, monkeypatch):
        from lilbee.providers.fleet.devices import FleetDevice

        _add("omlx", LOOPBACK, 8)
        card = FleetDevice("CUDA", 0, "gpu", 24 * GB, 15 * GB)
        assert planning_mod._unified_admission_budget([card], LLAMA) is None

    def test_an_unreadable_store_reserves_nothing(self):
        settings.set_value(cfg.data_root, PRESETS_KEY, "not json")
        assert _budgets() == (SIZING, ADMISSION)

    def test_a_row_with_a_bad_reserve_reserves_nothing(self):
        row = '[{"name": "x", "url": "http://127.0.0.1:1/v1", "roles": ["chat"], '
        settings.set_value(cfg.data_root, PRESETS_KEY, row + '"memory_reserve_gb": -3}]')
        assert loopback_reserve_bytes() == 0


class TestLoopbackReserve:
    def test_it_sums_gib_as_bytes(self):
        _add("a", LOOPBACK, 1.5)
        assert loopback_reserve_bytes() == int(1.5 * GB)

    @pytest.mark.parametrize(
        ("url", "loopback"),
        [
            ("http://127.0.0.1:8000/v1", True),
            ("http://127.9.9.9/v1", True),
            ("http://localhost:8000/v1", True),
            ("http://LOCALHOST/v1", True),
            ("http://[::1]:8000/v1", True),
            ("http://192.168.1.20:8000/v1", False),
            ("http://10.0.0.5/v1", False),
            ("http://example.test/v1", False),
            ("http://127.1:8000/v1", True),
            ("http://localhost.:8000/v1", True),
            ("http://0.0.0.0:8000/v1", True),
            ("http://2130706433/v1", True),
            ("http://[::]:8000/v1", True),
            ("http://1.2.3.4.5/v1", False),
            ("http://127.0.0.1 x/v1", False),
            ("http://127.0.0.1\x00x/v1", False),
            ("not a url", False),
            ("http://[::1/v1", False),
        ],
    )
    def test_which_hosts_count(self, url, loopback):
        assert is_loopback_url(url) is loopback
