"""The two sides of the self-check have to count the same bytes.

The estimate adds the vision projector's weights. llama.cpp allocates them in
clip's loader, which prints a size but not the "buffer size = N MiB" shape the
readback reads, so the report is short by exactly that and the check warned on
every correctly sized vision load.
"""

from __future__ import annotations

import logging

from lilbee.providers.fleet.readback import _without_unreported, check_launch
from lilbee.providers.roles import WorkerRole

_MIB = 1024 * 1024


def test_the_projector_comes_off_the_busiest_card() -> None:
    est = {"CUDA0": 900 * _MIB, "CUDA1": 400 * _MIB}
    assert _without_unreported(est, 300 * _MIB) == {"CUDA0": 600 * _MIB, "CUDA1": 400 * _MIB}


def test_nothing_unreported_leaves_the_estimate_alone() -> None:
    est = {"CUDA0": 900 * _MIB}
    assert _without_unreported(est, 0) is est


def test_a_correctly_sized_vision_load_does_not_warn(tmp_path, caplog) -> None:
    log = tmp_path / "vision-0.log"
    log.write_text(
        "0.00 I load_model:   initializing, n_slots = 1\n"
        "load_tensors:        CUDA0 model buffer size =   600.00 MiB\n"
    )
    with caplog.at_level(logging.WARNING):
        check_launch(
            tmp_path,
            "vision-0",
            WorkerRole.VISION,
            "org/vlm",
            estimated_bytes=900 * _MIB,
            est_by_device={"CUDA0": 900 * _MIB},
            unreported_bytes=300 * _MIB,
        )
    assert "CUDA0" not in caplog.text, caplog.text


class TestARealVisionLoad:
    """Captured on a GTX 1070 Ti, Vulkan engine build 9665, SmolVLM-256M with a
    98.96 MiB projector sidecar.

    Until this capture, the whole projector correction rested on reading
    llama.cpp's clip loader. The load confirms the premise: the projector's
    weights get no "buffer size" line at all, while clip's compute buffer does.
    """

    @staticmethod
    def _log() -> str:
        from pathlib import Path

        return (Path(__file__).parent / "fixtures" / "engine-load-vision-vulkan.log").read_text()

    def test_clips_compute_buffer_is_reported_under_its_own_prefix(self) -> None:
        # reserve_compute_meta is a prefix no other fixture carries, and it parses
        # because the pattern matches the shape rather than a list of known kinds.
        from lilbee.providers.fleet.readback import parse_device_buffers

        assert "reserve_compute_meta" in self._log()
        assert set(parse_device_buffers(self._log())) == {"CPU", "Vulkan0", "Vulkan_Host"}

    def test_the_projector_weights_appear_in_no_buffer_line(self) -> None:
        # The engine states them only as prose ("worst-case memory usage of
        # mmproj"), which the pattern correctly refuses. That absence is the
        # entire reason the estimate has to be adjusted before comparing.
        from lilbee.providers.fleet.readback import device_footprint

        assert "worst-case memory usage of mmproj" in self._log()
        # 136.47 model + 45.00 KV + 13.26 compute + 21.00 clip compute.
        assert round(device_footprint(self._log()) / 1024**2, 2) == 215.73

    def test_the_correction_silences_a_correctly_sized_load(self, tmp_path, caplog) -> None:
        import logging
        import shutil
        from pathlib import Path

        from lilbee.providers.fleet.readback import check_launch, device_footprint

        src = Path(__file__).parent / "fixtures" / "engine-load-vision-vulkan.log"
        shutil.copy(src, tmp_path / "engine-vision.log")
        actual = device_footprint(self._log())
        mmproj = int(98.96 * 1024**2)
        planned = actual + mmproj  # what the planner charges, projector included

        with caplog.at_level(logging.WARNING):
            check_launch(
                tmp_path, "vision", WorkerRole.VISION, "m", planned, {"Vulkan0": planned}, mmproj
            )
        assert caplog.text == "", caplog.text

        # And without the correction the same load is reported as a shortfall.
        with caplog.at_level(logging.WARNING):
            check_launch(tmp_path, "vision", WorkerRole.VISION, "m", planned, {"Vulkan0": planned})
        assert "did not land where it was planned" in caplog.text
