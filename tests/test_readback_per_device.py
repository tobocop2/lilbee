"""The readback has to check the dimension the planner actually decides in."""

from __future__ import annotations

import logging

import pytest

from lilbee.providers.fleet.readback import MIB, check_launch, engine_log_path
from lilbee.providers.roles import WorkerRole

_LOGGER = "lilbee.providers.fleet.readback"


def _split_log(card0_mib: float, card1_mib: float) -> str:
    """A two-card load reporting the given per-card totals."""
    return (
        "srv    load_model: loading model 'm.gguf'\n"
        f"load_tensors: CUDA0 model buffer size = {card0_mib:8.2f} MiB\n"
        f"load_tensors: CUDA1 model buffer size = {card1_mib:8.2f} MiB\n"
        "srv    load_model: initializing slots, n_slots = 1\n"
    )


class TestASkewedSplitIsCaught:
    """Totals agree exactly whatever the distribution, so a plan of 50/50 that
    lands 80/20 passed silently, and card 0 is the one that OOMs."""

    def test_a_skewed_split_with_the_right_total_is_reported(self, tmp_path, caplog) -> None:
        engine_log_path(tmp_path, "chat-0").write_text(_split_log(8000.0, 2000.0))
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            warned = check_launch(
                tmp_path,
                "chat-0",
                WorkerRole.CHAT,
                "org/m.gguf",
                5000 * MIB * 2,
                est_by_device={"CUDA0": 5000 * MIB, "CUDA1": 5000 * MIB},
            )
        assert warned is True
        assert "CUDA0" in caplog.text

    def test_a_split_that_landed_as_planned_stays_quiet(self, tmp_path, caplog) -> None:
        engine_log_path(tmp_path, "chat-0").write_text(_split_log(5000.0, 5000.0))
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            warned = check_launch(
                tmp_path,
                "chat-0",
                WorkerRole.CHAT,
                "org/m.gguf",
                5000 * MIB * 2,
                est_by_device={"CUDA0": 5000 * MIB, "CUDA1": 5000 * MIB},
            )
        assert warned is False
        assert caplog.text == ""

    def test_a_role_that_landed_on_an_unplanned_card_is_reported(self, tmp_path, caplog) -> None:
        # Planned for CUDA0, the engine put it on CUDA1 entirely.
        engine_log_path(tmp_path, "chat-0").write_text(_split_log(0.0, 10000.0))
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            warned = check_launch(
                tmp_path,
                "chat-0",
                WorkerRole.CHAT,
                "org/m.gguf",
                10000 * MIB,
                est_by_device={"CUDA0": 10000 * MIB},
            )
        assert warned is True
        assert "CUDA1" in caplog.text

    def test_without_a_per_device_estimate_the_total_is_still_checked(
        self, tmp_path, caplog
    ) -> None:
        # A model the estimator could only size as one number keeps the old check.
        engine_log_path(tmp_path, "chat-0").write_text(_split_log(8000.0, 8000.0))
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            warned = check_launch(tmp_path, "chat-0", WorkerRole.CHAT, "m", 4000 * MIB)
        assert warned is True


class TestTheDeviceNameIsTheJoin:
    """CUDA0 / MTL0 is what the engine prints and what --device and
    --tensor-split take, so it joins the two sides without the index-space
    ambiguity that from_loader exists to mark."""

    @pytest.mark.parametrize(
        ("backend", "index", "name"),
        [("CUDA", 0, "CUDA0"), ("MTL", 0, "MTL0"), ("Vulkan", 1, "Vulkan1")],
    )
    def test_a_fleet_device_names_itself_the_way_the_engine_does(
        self, backend: str, index: int, name: str
    ) -> None:
        from lilbee.providers.fleet.devices import FleetDevice
        from lilbee.providers.fleet.readback import device_label

        assert device_label(FleetDevice(backend, index, "gpu", 1, 1)) == name


class TestTheChargeIsSplitTheWayTheLaunchIs:
    """What each card was planned for has to match how the instance launches."""

    @staticmethod
    def _cards(count: int):
        from lilbee.providers.fleet.devices import FleetDevice

        return tuple(FleetDevice("CUDA", i, f"gpu{i}", 1, 1) for i in range(count))

    def test_one_card_carries_the_whole_charge(self) -> None:
        from lilbee.providers.fleet.planning import _charge_by_device

        assert _charge_by_device(self._cards(1), (), 900) == {"CUDA0": 900}

    def test_a_split_carries_it_in_the_launch_proportions(self) -> None:
        from lilbee.providers.fleet.planning import _charge_by_device

        assert _charge_by_device(self._cards(2), (3, 1), 800) == {"CUDA0": 600, "CUDA1": 200}

    def test_a_ratio_that_does_not_match_the_cards_falls_back_to_even(self) -> None:
        from lilbee.providers.fleet.planning import _charge_by_device

        assert _charge_by_device(self._cards(2), (1, 1, 1), 800) == {"CUDA0": 400, "CUDA1": 400}

    def test_an_unsized_model_carries_nothing(self) -> None:
        from lilbee.providers.fleet.planning import _charge_by_device

        assert _charge_by_device(self._cards(2), (1, 1), 0) == {}
