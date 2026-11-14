"""A refusal states the cards the host holds, counted once for the plan."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import pytest

from lilbee.core.config import cfg
from lilbee.providers.fleet import gpu_hardware, planning
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.gpu_backends import nvidia as nvidia_mod
from lilbee.providers.roles import WorkerRole
from tests._fleet_engines import CardEngine, Role, engines_for_all_roles, model_file, prepare_host
from tests._nvidia_smi import prints

GIB = 1 << 30
C, E = WorkerRole.CHAT, WorkerRole.EMBED
_ROLES = (Role(C, 4), Role(E, 2))
_VULKAN = "Vulkan"
_NV_NAME = "NVIDIA GeForce RTX 3090"
_AMD_NAME = "AMD Radeon RX 7900"
_INTEL_NAME = "Intel Arc A770"
_NVIDIA_ID = 0x10DE
_AMD_ID = 0x1002
_INTEL_ID = 0x8086


def _card(index: int, name: str, backend: str = _VULKAN) -> FleetDevice:
    return FleetDevice(backend, index, name, 8 * GIB, 8 * GIB)


def _driver_cards(count: int) -> list[nvidia_mod.NvidiaCard]:
    return [nvidia_mod.NvidiaCard(i, f"GPU-{i}", "n", 8 * GIB, 8 * GIB) for i in range(count)]


class _Host:
    """The host's two tables, with a count of how often each was read."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.driver_answers: list[int] = []
        self.tree = Counter[int]()
        self.driver_reads = 0
        self.tree_reads = 0
        monkeypatch.setattr(planning, "nvidia_cards", self._driver)
        monkeypatch.setattr(gpu_hardware, "installed_gpu_counts", self._tree)

    def _driver(self) -> list[nvidia_mod.NvidiaCard]:
        """The next scripted card count; the driver fails once the script is spent."""
        self.driver_reads += 1
        if not self.driver_answers:
            return []
        return _driver_cards(self.driver_answers.pop(0))

    def _tree(self) -> Counter[int]:
        self.tree_reads += 1
        return self.tree


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch) -> _Host:
    return _Host(monkeypatch)


class TestHostCardsTotal:
    @pytest.mark.parametrize(
        ("visible", "driver", "tree", "total"),
        [
            pytest.param([_NV_NAME], [2], {}, 2, id="mask-hides-an-nvidia-card"),
            pytest.param([_NV_NAME], [], {_NVIDIA_ID: 2}, 2, id="driver-fails-tree-counts-nvidia"),
            pytest.param(
                [_NV_NAME, _AMD_NAME],
                [],
                {_NVIDIA_ID: 2, _AMD_ID: 1},
                3,
                id="hidden-nvidia-beside-visible-amd",
            ),
            pytest.param([_NV_NAME], [1], {_AMD_ID: 1}, 2, id="hidden-amd-counts"),
            pytest.param([_AMD_NAME], [], {}, 1, id="no-table-keeps-the-visible-count"),
            pytest.param(
                [_NV_NAME, _NV_NAME, _NV_NAME], [2], {_NVIDIA_ID: 2}, 3, id="visible-above-tables"
            ),
            pytest.param([_AMD_NAME], [3], {_AMD_ID: 1, _INTEL_ID: 1}, 5, id="three-vendors"),
            pytest.param([], [2], {}, 2, id="no-visible-card"),
            pytest.param(["Tesla T4"], [1], {_NVIDIA_ID: 1}, 1, id="unnamed-card-fills-a-slot"),
            pytest.param(["Tesla T4", "Tesla T4"], [1], {}, 2, id="unnamed-cards-above-tables"),
            pytest.param([_INTEL_NAME], [], {_INTEL_ID: 2}, 2, id="hidden-intel"),
        ],
    )
    def test_count_by_vendor(
        self,
        host: _Host,
        visible: list[str],
        driver: list[int],
        tree: dict[int, int],
        total: int,
    ) -> None:
        host.driver_answers = list(driver)
        host.tree = Counter(tree)
        devices = [_card(i, name) for i, name in enumerate(visible)]
        assert planning.HostCards(devices).total == total

    @pytest.mark.parametrize(
        ("backend", "vendor_tree"),
        [("CUDA", _NVIDIA_ID), ("ROCm", _AMD_ID), ("HIP", _AMD_ID), ("SYCL", _INTEL_ID)],
    )
    def test_the_backend_names_the_vendor_of_a_visible_card(
        self, host: _Host, backend: str, vendor_tree: int
    ) -> None:
        host.tree = Counter({vendor_tree: 2})
        assert planning.HostCards([_card(0, "gpu0", backend)]).total == 2

    def test_a_sycl_card_is_an_intel_card_beside_a_hidden_nvidia_one(self, host: _Host) -> None:
        host.driver_answers = [1]
        assert planning.HostCards([_card(0, "gpu0", "SYCL")]).total == 2

    def test_the_gate_count_is_the_drivers_own(self, host: _Host) -> None:
        host.driver_answers = [3]
        host.tree = Counter({_NVIDIA_ID: 5})
        assert planning.HostCards([]).nvidia_driver == 3

    @pytest.mark.parametrize(
        "memory", ["[N/A], [N/A]", "8192, [N/A]"], ids=["total-unread", "free-unread"]
    )
    def test_a_card_whose_memory_does_not_read_leaves_the_gate_count_at_zero(
        self, monkeypatch: pytest.MonkeyPatch, memory: str
    ) -> None:
        """The driver's card list is unknown, so the gate falls back as for no answer."""
        csv = f"0, GPU-a, 8192, 8000, {_NV_NAME}\n1, GPU-b, {memory}, {_NV_NAME}\n"
        monkeypatch.setattr(nvidia_mod, "run_tool", prints(lambda *_a, **_k: csv))
        assert planning.HostCards([]).nvidia_driver == 0

    def test_the_gate_counts_every_card_of_a_readable_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv = f"0, GPU-a, 8192, 8000, {_NV_NAME}\n1, GPU-b, 8192, 8000, {_NV_NAME}\n"
        monkeypatch.setattr(nvidia_mod, "run_tool", prints(lambda *_a, **_k: csv))
        assert planning.HostCards([]).nvidia_driver == 2

    def test_each_table_is_read_once_for_the_value(self, host: _Host) -> None:
        host.driver_answers = [2]
        on_host = planning.HostCards([_card(0, _NV_NAME)])
        assert (on_host.nvidia_driver, on_host.total, on_host.total, on_host.nvidia_driver) == (
            2,
            2,
            2,
            2,
        )
        assert (host.driver_reads, host.tree_reads) == (1, 1)

    def test_a_value_nobody_asks_reads_nothing(self, host: _Host) -> None:
        planning.HostCards([_card(0, _NV_NAME)])
        assert (host.driver_reads, host.tree_reads) == (0, 0)


def _refusal(model: str, gpus: int, engine: str) -> str:
    return (
        f"{model} will not be served: this host has {gpus} GPUs and {engine} cannot be "
        "told which one to use. Use another engine for this role."
    )


def _two_engines(tmp_path: Path, devices: list[FleetDevice]) -> dict[WorkerRole, CardEngine]:
    """Chat on an engine that takes device names; embed on one that does not."""
    chat = CardEngine("engine-a", tmp_path / "a", _ROLES, devices, takes_device_names=True)
    embed = CardEngine("engine-b", tmp_path / "b", _ROLES, devices)
    return {C: chat, E: embed}


class TestMixedPlanWarning:
    """The warning for a role no pin reaches, through plan_all_launches."""

    @staticmethod
    def _text(monkeypatch, tmp_path, caplog, devices: list[FleetDevice]) -> str:
        prepare_host(monkeypatch, tmp_path, _ROLES, 64 * GIB)
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
            plan = planning.plan_all_launches(engines=_two_engines(tmp_path, devices))
        assert {launch.role for launch in plan.launches} == {C}
        return caplog.text

    def test_a_driver_that_fails_after_the_gate_still_counts_the_hidden_card(
        self, host: _Host, monkeypatch, tmp_path, caplog
    ) -> None:
        host.driver_answers = [2]
        text = self._text(monkeypatch, tmp_path, caplog, [_card(0, _NV_NAME)])
        model = f"embed model org/repo/{model_file(E)} (engine-b)"
        assert _refusal(model, 2, "engine-b") in text
        assert (host.driver_reads, host.tree_reads) == (1, 1)

    def test_a_card_of_another_vendor_the_driver_cannot_see_counts(
        self, host: _Host, monkeypatch, tmp_path, caplog
    ) -> None:
        host.tree = Counter({_NVIDIA_ID: 2, _AMD_ID: 1})
        devices = [_card(0, _NV_NAME), _card(1, _AMD_NAME)]
        text = self._text(monkeypatch, tmp_path, caplog, devices)
        model = f"embed model org/repo/{model_file(E)} (engine-b)"
        assert _refusal(model, 3, "engine-b") in text

    def test_a_host_with_no_readable_table_states_the_visible_count(
        self, host: _Host, monkeypatch, tmp_path, caplog
    ) -> None:
        devices = [_card(0, _NV_NAME), _card(1, _NV_NAME)]
        text = self._text(monkeypatch, tmp_path, caplog, devices)
        model = f"embed model org/repo/{model_file(E)} (engine-b)"
        assert _refusal(model, 2, "engine-b") in text

    def test_a_hidden_card_of_another_vendor_does_not_open_the_gate(
        self, host: _Host, monkeypatch, tmp_path, caplog
    ) -> None:
        host.driver_answers = [1]
        host.tree = Counter({_NVIDIA_ID: 1, _AMD_ID: 1})
        prepare_host(monkeypatch, tmp_path, _ROLES, 64 * GIB)
        with caplog.at_level(logging.WARNING, logger="lilbee.providers.fleet.planning"):
            plan = planning.plan_all_launches(
                engines=_two_engines(tmp_path, [_card(0, _NV_NAME, "CUDA")])
            )
        assert "cannot be told which one" not in caplog.text
        assert {launch.role for launch in plan.launches} == {C, E}
        assert host.tree_reads == 0


class TestPinRefusal:
    """The refusal for a pin to a card the engine does not list, through plan_all_launches."""

    SPEC = '{"chat": {"devices": [0]}, "embed": {"target": "gpu", "devices": [1]}}'

    def _refusals(self, monkeypatch, tmp_path, devices: list[FleetDevice]) -> str:
        prepare_host(monkeypatch, tmp_path, _ROLES, 64 * GIB)
        monkeypatch.setattr(cfg, "placement", self.SPEC)
        engine = CardEngine("engine-a", tmp_path / "a", _ROLES, devices)
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        return plan.refused_roles[E]

    @staticmethod
    def _text(found: str) -> str:
        return (
            f"embed: GPU 1 is not on this machine ({found}). "
            "Set the embed devices to a GPU that exists, or set its target to cpu."
        )

    def test_a_card_a_mask_hides_is_counted_on_the_host(
        self, host: _Host, monkeypatch, tmp_path
    ) -> None:
        host.driver_answers = [2]
        refusal = self._refusals(monkeypatch, tmp_path, [_card(0, _NV_NAME, "CUDA")])
        assert refusal == self._text("1 found, 2 on this host")

    def test_a_hidden_card_of_another_vendor_is_counted(
        self, host: _Host, monkeypatch, tmp_path
    ) -> None:
        host.driver_answers = [1]
        host.tree = Counter({_NVIDIA_ID: 1, _AMD_ID: 1})
        refusal = self._refusals(monkeypatch, tmp_path, [_card(0, _NV_NAME, "CUDA")])
        assert refusal == self._text("1 found, 2 on this host")

    def test_a_host_with_no_readable_table_keeps_the_visible_count(
        self, host: _Host, monkeypatch, tmp_path
    ) -> None:
        refusal = self._refusals(monkeypatch, tmp_path, [_card(0, _AMD_NAME)])
        assert refusal == self._text("1 found")

    def test_a_pin_to_a_card_that_exists_reads_no_table(
        self, host: _Host, monkeypatch, tmp_path
    ) -> None:
        prepare_host(monkeypatch, tmp_path, _ROLES, 64 * GIB)
        monkeypatch.setattr(cfg, "placement", self.SPEC)
        devices = [_card(0, _NV_NAME, "CUDA"), _card(1, _NV_NAME, "CUDA")]
        engine = CardEngine("engine-a", tmp_path / "a", _ROLES, devices)
        plan = planning.plan_all_launches(engines=engines_for_all_roles(engine))
        assert plan.refused_roles == {}
        assert (host.driver_reads, host.tree_reads) == (0, 0)

    def test_an_unmasked_host_keeps_the_visible_count(
        self, host: _Host, monkeypatch, tmp_path
    ) -> None:
        host.driver_answers = [1]
        refusal = self._refusals(monkeypatch, tmp_path, [_card(0, _NV_NAME, "CUDA")])
        assert refusal == self._text("1 found")
