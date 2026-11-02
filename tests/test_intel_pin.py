"""A listed integrated GPU can be pinned, and a hidden one is named instead of miscounted.

The host is a hybrid laptop: an NVIDIA card and an Intel integrated GPU. The default
launch hides the Intel driver, so the engine lists the NVIDIA card alone. With
``VK_LOADER_DRIVERS_DISABLE=nomatch*`` the engine lists both.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lilbee.app import placement as app_placement
from lilbee.app.engine_choice import BookReading
from lilbee.core.config import cfg
from lilbee.providers.fleet import devices as fleet_devices
from lilbee.providers.fleet import gpu_select, planning
from lilbee.providers.fleet.devices import FleetDevice
from lilbee.providers.fleet.gpu_select import IcdChoice, PCIVendorID, VulkanIcdEnvVar
from lilbee.providers.fleet.placement_spec import PlacementError, PlacementSpec
from lilbee.providers.fleet.replicas import gpu_device_count
from lilbee.providers.roles import EngineBackend, WorkerRole
from lilbee.server.models import PlacementResponse
from tests._fleet_engines import CardEngine, Role, engines_for_all_roles, prepare_host

GIB = 1 << 30
CHAT_ONLY = (Role(WorkerRole.CHAT, 1),)
ROLES = (*CHAT_ONLY, Role(WorkerRole.EMBED, 0.4))
NVIDIA = FleetDevice("Vulkan", 0, "NVIDIA GeForce GTX 1650 Ti", 4 * GIB, 4 * GIB)
INTEL = FleetDevice("Vulkan", 1, "Intel(R) UHD Graphics (CML GT2)", 8 * GIB, 8 * GIB, unified=True)
OVERRIDE_SENTENCE = (
    "lilbee hides the Intel GPU from the engine because an NVIDIA GPU is present. "
    "To use the Intel GPU, set VK_LOADER_DRIVERS_DISABLE=nomatch* and start lilbee again."
)
CHAT_ON_INTEL = '{"chat": {"devices": [1]}}'
CHAT_ON_INTEL_EMBED_AUTO = '{"chat": {"devices": [1]}, "embed": {"target": "auto"}}'
MIXED = '{"chat": {"devices": [0]}, "embed": {"devices": [1]}}'


def _host(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    devices: list[FleetDevice],
    *,
    backend: EngineBackend = EngineBackend.VULKAN,
    roles: tuple[Role, ...] = CHAT_ONLY,
) -> CardEngine:
    prepare_host(monkeypatch, tmp_path, roles, 64 * GIB)
    engine = CardEngine("engine-a", tmp_path / "engine-server", roles, devices, backend=backend)
    reading = BookReading(adapters=engines_for_all_roles(engine), notices={})
    monkeypatch.setattr(app_placement, "_planning_engines", lambda: reading)
    return engine


def _two_vendor_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both drivers installed and both chips present, with no loader variable set."""
    monkeypatch.setattr(gpu_select, "_platform_supports_icd_pin", lambda: True)
    monkeypatch.setattr(
        gpu_select, "_vulkan_vendors_present", lambda: {PCIVendorID.NVIDIA, PCIVendorID.INTEL}
    )
    monkeypatch.setattr(gpu_select, "_vendors_with_hardware", set)
    monkeypatch.setattr(cfg, "gpu_devices", None)
    for var in VulkanIcdEnvVar:
        monkeypatch.delenv(var, raising=False)


def _where(spec: str, engine: CardEngine) -> dict[str, tuple[int, ...]]:
    resolved = planning.resolve_placement_plan(
        PlacementSpec.from_json(spec), engines=engines_for_all_roles(engine)
    )
    return {p.role.value: p.devices for p in resolved.instances}


def _refusal(spec: str, engine: CardEngine) -> str:
    with pytest.raises(PlacementError) as caught:
        planning.resolve_placement_plan(
            PlacementSpec.from_json(spec), engines=engines_for_all_roles(engine)
        )
    return str(caught.value)


class TestAListedIntegratedGpuCanBePinned:
    def test_a_pin_names_the_listed_integrated_gpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, [NVIDIA, INTEL])
        assert _where(CHAT_ON_INTEL, engine) == {"chat": (1,)}

    def test_a_mixed_plan_puts_embed_on_the_integrated_gpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, [NVIDIA, INTEL], roles=ROLES)
        assert _where(MIXED, engine) == {"chat": (0,), "embed": (1,)}

    def test_a_role_with_the_auto_target_stays_off_the_named_integrated_gpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, [NVIDIA, INTEL], roles=ROLES)
        spec = '{"chat": {"devices": [1]}, "embed": {"target": "auto"}}'
        assert _where(spec, engine) == {"chat": (1,), "embed": (0,)}

    def test_the_auto_plan_stays_off_the_integrated_gpu(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, [NVIDIA, INTEL], roles=ROLES)
        resolved = planning.resolve_placement_plan(None, engines=engines_for_all_roles(engine))
        used = {index for plan in resolved.instances for index in plan.devices}
        assert used == {0}

    def test_auto_replicas_count_the_dedicated_card_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, [NVIDIA, INTEL])
        gpu_device_count.cache_clear()
        assert gpu_device_count(engine) == 1

    def test_a_host_of_integrated_gpus_alone_keeps_them_all(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        second = FleetDevice("Vulkan", 1, "Intel(R) Arc", 8 * GIB, 8 * GIB, unified=True)
        first = FleetDevice("Vulkan", 0, "Intel(R) UHD", 8 * GIB, 8 * GIB, unified=True)
        engine = _host(monkeypatch, tmp_path, [first, second])
        gpu_device_count.cache_clear()
        assert gpu_device_count(engine) == 2

    def test_a_launch_with_a_pin_on_the_integrated_gpu_plans_its_capacity_with_the_pinned_card(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        engine = _host(monkeypatch, tmp_path, [NVIDIA, INTEL], roles=ROLES)
        monkeypatch.setattr(cfg, "placement", CHAT_ON_INTEL_EMBED_AUTO)
        plan = planning.plan_launches(
            None,
            {NVIDIA.index: NVIDIA, INTEL.index: INTEL},
            [NVIDIA, INTEL],
            engines=engines_for_all_roles(engine),
        )
        assert {launch.role for launch in plan.launches} == {WorkerRole.CHAT, WorkerRole.EMBED}
        assert plan.refused_roles == {}


class TestAPinPastTheListSaysWhatTheListHolds:
    def test_one_listed_device(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        cuda = FleetDevice("CUDA", 0, "NVIDIA GeForce GTX 1650 Ti", 4 * GIB, 4 * GIB)
        engine = _host(monkeypatch, tmp_path, [cuda], backend=EngineBackend.CUDA)
        assert _refusal('{"chat": {"devices": [3]}}', engine) == (
            "chat is pinned to device 3, but this machine lists only device 0."
        )

    def test_two_listed_devices(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        engine = _host(monkeypatch, tmp_path, [NVIDIA, INTEL])
        monkeypatch.setattr(gpu_select, "vulkan_icd_choice", lambda: None)
        assert _refusal('{"chat": {"devices": [9]}}', engine) == (
            "chat is pinned to device 9, but this machine lists only devices 0 and 1."
        )

    def test_three_listed_devices(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        third = FleetDevice("Vulkan", 2, "AMD", 8 * GIB, 8 * GIB)
        engine = _host(monkeypatch, tmp_path, [NVIDIA, INTEL, third])
        monkeypatch.setattr(gpu_select, "vulkan_icd_choice", lambda: None)
        assert _refusal('{"chat": {"devices": [9]}}', engine).startswith(
            "chat is pinned to device 9, but this machine lists only devices 0, 1 and 2."
        )

    def test_no_listed_device(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        engine = _host(monkeypatch, tmp_path, [], backend=EngineBackend.CPU)
        assert _refusal('{"chat": {"devices": [0]}}', engine) == (
            "chat is pinned to device 0, but this machine lists no devices."
        )


class TestAHiddenDeviceIsNamed:
    def test_a_pin_to_the_hidden_device_names_it_and_the_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _two_vendor_launch(monkeypatch)
        engine = _host(monkeypatch, tmp_path, [NVIDIA])
        assert _refusal(CHAT_ON_INTEL, engine) == (
            "chat is pinned to device 1, but this machine lists only device 0. " + OVERRIDE_SENTENCE
        )

    def test_a_mixed_plan_to_the_hidden_device_gets_the_same_text(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _two_vendor_launch(monkeypatch)
        engine = _host(monkeypatch, tmp_path, [NVIDIA], roles=ROLES)
        assert _refusal(MIXED, engine) == (
            "embed is pinned to device 1, but this machine lists only device 0. "
            + OVERRIDE_SENTENCE
        )

    def test_a_saved_spec_on_the_hidden_device_gets_the_same_sentence(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _two_vendor_launch(monkeypatch)
        engine = _host(monkeypatch, tmp_path, [NVIDIA])
        monkeypatch.setattr(cfg, "placement", '{"chat": {"devices": [1], "target": "gpu"}}')
        plan = planning.plan_launches(
            None, {NVIDIA.index: NVIDIA}, [NVIDIA], engines=engines_for_all_roles(engine)
        )
        assert plan.refused_roles == {
            WorkerRole.CHAT: (
                "chat: GPU 1 is not on this machine (1 found). "
                + OVERRIDE_SENTENCE
                + " Set the chat devices to a GPU that exists, or set its target to cpu."
            )
        }

    def test_the_fallback_log_for_a_saved_pin_has_no_doubled_punctuation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _two_vendor_launch(monkeypatch)
        engine = _host(monkeypatch, tmp_path, [NVIDIA])
        monkeypatch.setattr(cfg, "placement", CHAT_ON_INTEL)
        with caplog.at_level(logging.WARNING):
            planning.plan_launches(
                None, {NVIDIA.index: NVIDIA}, [NVIDIA], engines=engines_for_all_roles(engine)
            )
        assert "start lilbee again); using automatic placement instead." in caplog.text

    def test_a_build_the_loader_cannot_change_gets_no_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _two_vendor_launch(monkeypatch)
        cuda = FleetDevice("CUDA", 0, "NVIDIA GeForce GTX 1650 Ti", 4 * GIB, 4 * GIB)
        engine = _host(monkeypatch, tmp_path, [cuda], backend=EngineBackend.CUDA)
        assert _refusal(CHAT_ON_INTEL, engine) == (
            "chat is pinned to device 1, but this machine lists only device 0."
        )

    def test_the_view_and_the_response_carry_the_hidden_vendor(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _two_vendor_launch(monkeypatch)
        _host(monkeypatch, tmp_path, [NVIDIA])
        view = app_placement.get_placement()
        assert [gpu.index for gpu in view.gpus] == [0]
        assert view.hidden_gpus == ("Intel",)
        assert view.hidden_note == (
            "Intel GPU not used: lilbee runs the engine on the NVIDIA GPU. "
            "To use the Intel GPU, set VK_LOADER_DRIVERS_DISABLE=nomatch* and start lilbee again."
        )
        assert PlacementResponse.from_view(view).model_dump(mode="json")["hidden_gpus"] == ["Intel"]

    def test_nothing_is_hidden_when_the_user_set_the_loader_variable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _two_vendor_launch(monkeypatch)
        monkeypatch.setenv("VK_LOADER_DRIVERS_DISABLE", "nomatch*")
        _host(monkeypatch, tmp_path, [NVIDIA, INTEL])
        view = app_placement.get_placement()
        assert [gpu.index for gpu in view.gpus] == [0, 1]
        assert view.hidden_gpus == ()
        assert view.hidden_note is None
        assert PlacementResponse.from_view(view).hidden_gpus == []

    def test_a_cuda_build_reports_nothing_hidden(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _two_vendor_launch(monkeypatch)
        cuda = FleetDevice("CUDA", 0, "NVIDIA GeForce GTX 1650 Ti", 4 * GIB, 4 * GIB)
        _host(monkeypatch, tmp_path, [cuda], backend=EngineBackend.CUDA)
        assert app_placement.get_placement().hidden_gpus == ()


class TestTheDriverChoice:
    def test_hardware_less_vendors_are_not_reported_hidden(self) -> None:
        choice = IcdChoice(
            kept=PCIVendorID.NVIDIA,
            installed=frozenset(PCIVendorID),
            present=frozenset({PCIVendorID.NVIDIA, PCIVendorID.INTEL}),
        )
        assert choice.hidden == (PCIVendorID.INTEL,)
        assert choice.disable == "amdvlk*,amd_icd*,amd-vulkan*,radeon*,intel*,igvk*"

    def test_two_hidden_vendors_are_named_together(self) -> None:
        choice = IcdChoice(
            kept=PCIVendorID.NVIDIA,
            installed=frozenset(PCIVendorID),
            present=frozenset(PCIVendorID),
        )
        assert choice.hidden_sentence() == (
            "lilbee hides the AMD and Intel GPUs from the engine because an NVIDIA GPU is "
            "present. To use them, set VK_LOADER_DRIVERS_DISABLE=nomatch* and start lilbee again."
        )
        assert choice.note().startswith("AMD and Intel GPUs not used: lilbee runs the engine on")

    def test_no_choice_when_the_user_set_a_loader_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _two_vendor_launch(monkeypatch)
        assert gpu_select.vulkan_icd_choice() is not None
        monkeypatch.setenv("VK_DRIVER_FILES", "/somewhere")
        assert gpu_select.vulkan_icd_choice() is None
        assert gpu_select.disable_conflicting_vulkan_icds() is None

    def test_the_launch_disables_what_the_choice_hides(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _two_vendor_launch(monkeypatch)
        assert gpu_select.disable_conflicting_vulkan_icds() == "intel*,igvk*"


class TestAProbeStoppedByASignal:
    def test_the_warning_does_not_blame_the_flag(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(fleet_devices, "_run_list_devices", lambda _binary, _t: ("", -15))
        with caplog.at_level(logging.WARNING, logger=fleet_devices.log.name):
            fleet_devices.probe_devices(Path("llama-server"))
        text = " ".join(record.getMessage() for record in caplog.records)
        assert "stopped by signal 15" in text
        assert "does not appear to support" not in text

    def test_an_engine_without_the_flag_still_says_so(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(fleet_devices, "_run_list_devices", lambda _binary, _t: ("usage", 1))
        with caplog.at_level(logging.WARNING, logger=fleet_devices.log.name):
            fleet_devices.probe_devices(Path("llama-server"))
        assert "does not appear to support the flag" in caplog.text
