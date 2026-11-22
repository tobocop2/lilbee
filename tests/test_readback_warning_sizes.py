"""Readback warnings keep small embedding models readable in MiB."""

from pathlib import Path

from lilbee.core.health_warnings import WarningCode
from lilbee.providers.fleet.readback import MIB, check_launch, engine_log_path
from lilbee.providers.roles import WorkerRole

_MODEL_ID = "embed-0"
_MODEL = "org/embed-48m.gguf"
_PLANNED_BYTES = 24 * MIB


def _write_48_mib_engine_log(log_dir: Path) -> None:
    engine_log_path(log_dir, _MODEL_ID).write_text(
        "load_tensors: CUDA0 model buffer size = 48.00 MiB\n"
        "srv    load_model: initializing slots, n_slots = 1\n",
        encoding="utf-8",
    )


def test_total_readback_warning_formats_small_embed_sizes_in_mib(tmp_path: Path) -> None:
    _write_48_mib_engine_log(tmp_path)

    warning = check_launch(tmp_path, _MODEL_ID, WorkerRole.EMBED, _MODEL, _PLANNED_BYTES)

    assert warning is not None
    assert warning.code is WarningCode.PLACEMENT_DIVERGED
    assert "allocated 48 MiB" in warning.message
    assert "planned for 24 MiB" in warning.message


def test_per_card_readback_warning_formats_small_embed_sizes_in_mib(tmp_path: Path) -> None:
    _write_48_mib_engine_log(tmp_path)

    warning = check_launch(
        tmp_path,
        _MODEL_ID,
        WorkerRole.EMBED,
        _MODEL,
        _PLANNED_BYTES,
        est_by_device={"CUDA0": _PLANNED_BYTES},
    )

    assert warning is not None
    assert warning.code is WarningCode.PLACEMENT_DIVERGED
    assert "CUDA0 holds 48 MiB" in warning.message
    assert "planned for 24 MiB" in warning.message
