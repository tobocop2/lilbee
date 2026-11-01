"""What a model folder is charged, and a pin on one card with no per-card breakdown."""

from __future__ import annotations

from pathlib import Path

import pytest

from lilbee.core.config.enums import KvCacheType
from lilbee.providers import engine_params
from lilbee.providers.fleet import model_memory, planning
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.fleet.engines.tei import TEI
from lilbee.providers.fleet.memory_types import MemoryEstimate, MemoryKind, MemoryRequest
from lilbee.providers.fleet.placement import _charge_devices, placement_from_spec
from lilbee.providers.fleet.placement_spec import (
    PlacementError,
    PlacementSpec,
    RolePlacement,
    TargetRefusedError,
)
from lilbee.providers.roles import RoleTarget, WorkerRole

GIB = 1024**3
_CARD = 24 * GIB
_REF = "org/repo/model"


def _sparse(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)


def _folder(root: Path, *shards: int) -> Path:
    """A flat model folder of *shards*, with the small files a pull also brings."""
    folder = root / "model"
    folder.mkdir()
    for index, size in enumerate(shards):
        _sparse(folder / f"model-{index:05d}.safetensors", size)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "tokenizer.json").write_text("{}", encoding="utf-8")
    return folder


def _snapshot(root: Path, size: int) -> Path:
    """A Hugging Face snapshot: the folder holds symlinks to a blob elsewhere."""
    blob = root / "blobs" / "abc"
    _sparse(blob, size)
    folder = root / "snapshots" / "rev"
    folder.mkdir(parents=True)
    (folder / "model.safetensors").symlink_to(blob)
    return folder


def _request(path: Path) -> MemoryRequest:
    return MemoryRequest(
        model_path=path,
        ctx=4096,
        slots=1,
        gpu_layers=-1,
        flash_attn=False,
        kv_cache_type=KvCacheType.F16,
    )


class _StatedEngine(type(MISTRAL)):
    """The mistral.rs adapter with a stated estimate in place of its own."""

    def __init__(self, estimate: MemoryEstimate) -> None:
        self.estimate = estimate

    def estimate_memory(self, request: MemoryRequest) -> MemoryEstimate:
        return self.estimate


def _pin(engine, role: WorkerRole, path: Path, devices: tuple[int, ...], monkeypatch, alone=False):
    """Place *role* through the planner's own estimator and the manual placement path."""
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda _ref: path)
    peak = planning._peak_estimator({role: "ref"}, {role: engine})
    roles = {role: RolePlacement(devices=devices)}
    if alone:
        roles[WorkerRole.RERANK] = RolePlacement(devices=(), target=RoleTarget.CPU)
    return placement_from_spec(
        PlacementSpec(roles=roles), (role,), dict.fromkeys(range(2), _CARD), estimate_peak=peak
    )


class TestTheFolderTotal:
    def test_the_adapter_sums_the_shards_of_a_folder(self, tmp_path: Path) -> None:
        folder = _folder(tmp_path, 4 * GIB, 3 * GIB)
        estimate = MISTRAL.estimate_memory(_request(folder))
        assert estimate.device_bytes == 7 * GIB
        assert estimate.shared_bytes == 7 * GIB
        assert estimate.host_bytes == 0

    def test_the_adapter_follows_the_symlinks_of_a_snapshot(self, tmp_path: Path) -> None:
        assert MISTRAL.estimate_memory(_request(_snapshot(tmp_path, 3 * GIB))).device_bytes == (
            3 * GIB
        )

    def test_an_empty_folder_is_charged_the_reserve(self, tmp_path: Path) -> None:
        estimate = MISTRAL.estimate_memory(_request(_folder(tmp_path)))
        assert estimate.device_bytes == model_memory.UNKNOWN_SIZE_RESERVE_BYTES

    def test_a_single_file_is_charged_its_size(self, tmp_path: Path) -> None:
        model = tmp_path / "m.gguf"
        _sparse(model, 3 * GIB)
        assert MISTRAL.estimate_memory(_request(model)).device_bytes == 3 * GIB

    def test_a_missing_file_is_charged_the_reserve(self, tmp_path: Path) -> None:
        estimate = MISTRAL.estimate_memory(_request(tmp_path / "gone.gguf"))
        assert estimate.device_bytes == model_memory.UNKNOWN_SIZE_RESERVE_BYTES


class TestTheFolderWeights:
    def test_weights_sum_the_shards_and_skip_the_small_files(self, tmp_path: Path) -> None:
        assert planning._weights_bytes(_folder(tmp_path, GIB, 2 * GIB)) == 3 * GIB

    def test_weights_follow_the_symlinks_of_a_snapshot(self, tmp_path: Path) -> None:
        assert planning._weights_bytes(_snapshot(tmp_path, GIB)) == GIB

    def test_an_empty_folder_has_no_weights(self, tmp_path: Path) -> None:
        assert planning._weights_bytes(_folder(tmp_path)) == 0

    def test_a_folder_named_like_a_shard_is_still_a_folder(self, tmp_path: Path) -> None:
        folder = tmp_path / "m-00001-of-00002.gguf"
        _sparse(folder / "model.safetensors", GIB)
        assert planning._weights_bytes(folder) == GIB

    def test_an_empty_folder_reads_as_not_installed(self, tmp_path: Path, monkeypatch) -> None:
        folder = _folder(tmp_path)
        monkeypatch.setattr(engine_params, "resolve_model_path", lambda _ref: folder)
        assert planning._role_weights_bytes(WorkerRole.CHAT, _REF) == 0
        assert not planning.role_model_placeable(WorkerRole.CHAT, _REF, _CARD, engine=MISTRAL)


class TestThePeakEstimator:
    def _peak(self, tmp_path: Path, monkeypatch, estimate: MemoryEstimate, ratio: tuple[int, ...]):
        monkeypatch.setattr(engine_params, "resolve_model_path", lambda _ref: tmp_path)
        role = WorkerRole.CHAT
        peak = planning._peak_estimator({role: "ref"}, {role: _StatedEngine(estimate)})
        return peak(role, ratio)

    def test_one_card_with_no_breakdown_takes_the_device_bytes(self, tmp_path, monkeypatch) -> None:
        estimate = MemoryEstimate(
            kind=MemoryKind.SIZED, device_bytes=5 * GIB, host_bytes=7 * GIB, shared_bytes=11 * GIB
        )
        assert self._peak(tmp_path, monkeypatch, estimate, (1,)) == (5 * GIB,)

    def test_two_cards_with_no_breakdown_get_no_shares(self, tmp_path, monkeypatch) -> None:
        estimate = MemoryEstimate(
            kind=MemoryKind.SIZED, device_bytes=5 * GIB, host_bytes=7 * GIB, shared_bytes=11 * GIB
        )
        assert self._peak(tmp_path, monkeypatch, estimate, (1, 1)) == ()

    def test_a_breakdown_keeps_its_own_figure(self, tmp_path, monkeypatch) -> None:
        estimate = MemoryEstimate(
            kind=MemoryKind.SIZED,
            device_bytes=5 * GIB,
            host_bytes=7 * GIB,
            shared_bytes=11 * GIB,
            per_device=(3 * GIB,),
            per_device_shared=(13 * GIB,),
        )
        assert self._peak(tmp_path, monkeypatch, estimate, (1,)) == (3 * GIB,)

    def test_a_total_of_zero_gets_no_share(self, tmp_path, monkeypatch) -> None:
        estimate = MemoryEstimate(
            kind=MemoryKind.SIZED, device_bytes=0, host_bytes=7 * GIB, shared_bytes=11 * GIB
        )
        assert self._peak(tmp_path, monkeypatch, estimate, (1,)) == ()


@pytest.mark.parametrize("alone", [False, True], ids=["whole-spec", "role-alone"])
class TestAOneCardPin:
    def test_a_folder_that_fits_is_placed(self, tmp_path, monkeypatch, alone) -> None:
        folder = _folder(tmp_path, 8 * GIB)
        placed = _pin(MISTRAL, WorkerRole.CHAT, folder, (0,), monkeypatch, alone)
        assert placed.instances[0].devices == (0,)

    def test_a_folder_too_large_is_refused_with_its_size(
        self, tmp_path, monkeypatch, alone
    ) -> None:
        folder = _folder(tmp_path, 30 * GIB)
        with pytest.raises(PlacementError, match=r"chat pinned to device 0 needs 30\.0 GiB"):
            _pin(MISTRAL, WorkerRole.CHAT, folder, (0,), monkeypatch, alone)

    def test_tei_is_placed(self, tmp_path, monkeypatch, alone) -> None:
        folder = _folder(tmp_path, GIB)
        placed = _pin(TEI, WorkerRole.EMBED, folder, (0,), monkeypatch, alone)
        assert placed.instances[0].devices == (0,)

    def test_no_figure_at_all_is_refused(self, tmp_path, monkeypatch, alone) -> None:
        engine = _StatedEngine(
            MemoryEstimate(kind=MemoryKind.SIZED, device_bytes=0, host_bytes=1, shared_bytes=1)
        )
        with pytest.raises(PlacementError, match="memory estimate covers 0"):
            _pin(engine, WorkerRole.CHAT, tmp_path, (0,), monkeypatch, alone)


@pytest.mark.parametrize("alone", [False, True], ids=["whole-spec", "role-alone"])
def test_a_pin_on_two_cards_with_no_breakdown_says_to_pin_one(tmp_path, monkeypatch, alone) -> None:
    folder = _folder(tmp_path, GIB)
    expected = (
        r"^chat has no per-card memory figure, so it cannot be pinned to 2 cards\. "
        r"Pin one card, or set its target to auto\.$"
    )
    with pytest.raises(PlacementError, match=expected):
        _pin(MISTRAL, WorkerRole.CHAT, folder, (0, 1), monkeypatch, alone)


def test_a_breakdown_of_the_wrong_length_keeps_the_count_text() -> None:
    with pytest.raises(PlacementError, match=r"2 device\(s\) but its memory estimate covers 1"):
        _charge_devices(
            WorkerRole.CHAT,
            (0, 1),
            (GIB,),
            dict.fromkeys(range(2), float(_CARD)),
            dict.fromkeys(range(2), _CARD),
            refuse_alone=False,
        )


def test_the_refusal_of_a_two_card_pin_names_the_role_alone(tmp_path, monkeypatch) -> None:
    with pytest.raises(TargetRefusedError) as refused:
        _pin(MISTRAL, WorkerRole.CHAT, _folder(tmp_path, GIB), (0, 1), monkeypatch, alone=True)
    assert set(refused.value.reasons) == {WorkerRole.CHAT}


def test_the_role_weights_of_a_folder_are_its_shards(tmp_path, monkeypatch) -> None:
    folder = _folder(tmp_path, 3 * GIB)
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda _ref: folder)
    assert planning._role_weights_bytes(WorkerRole.CHAT, "ref") == 3 * GIB


def test_a_folder_larger_than_the_hardware_is_not_placeable(tmp_path, monkeypatch) -> None:
    folder = _folder(tmp_path, 40 * GIB)
    monkeypatch.setattr(engine_params, "resolve_model_path", lambda _ref: folder)
    monkeypatch.setattr(planning.model_cache, "total_system_memory", lambda: 8 * GIB)
    monkeypatch.setattr(planning, "total_system_memory", lambda: 8 * GIB)
    assert not planning.role_model_placeable(WorkerRole.CHAT, _REF, _CARD, engine=MISTRAL)
