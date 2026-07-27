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
