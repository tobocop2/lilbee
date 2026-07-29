"""A tight group has to let the engine decide the split.

The tight-role warning promises the engine will keep what fits on the GPU and
spill the rest to system memory. That promise rests on the engine's own fit
pass, and the pinned build aborts that pass when tensor_split is user-set
(common/fit.cpp: "model_params::tensor_split already set by user, abort"),
after which a negative n_gpu_layers means every layer. Emitting an invented
even split therefore turns the promise into a load-time OOM.
"""

from __future__ import annotations

from pathlib import Path

from lilbee.providers.fleet.adapters import ROLE_SPECS, build_server_argv
from lilbee.providers.roles import WorkerRole


def _argv(devices: tuple[int, ...], tensor_split: tuple[int, ...]) -> list[str]:
    return build_server_argv(
        binary=Path("/bin/llama-server"),
        spec=ROLE_SPECS[WorkerRole.CHAT],
        model_path=Path("/m/model.gguf"),
        devices=devices,
        n_gpu_layers=-1,
        slots=1,
        ctx_per_slot=4096,
        tensor_split=tensor_split,
    )


def test_a_tight_group_emits_no_tensor_split() -> None:
    # Multiple cards and no chosen ratio is the tight placement: the planner
    # could not size it, so the engine has to.
    assert "--tensor-split" not in _argv((0, 1), ())


def test_a_planned_split_still_emits_its_ratio() -> None:
    argv = _argv((0, 1), (24, 12))
    assert argv[argv.index("--tensor-split") + 1] == "24,12"


def test_a_single_card_never_emits_one() -> None:
    assert "--tensor-split" not in _argv((0,), ())
