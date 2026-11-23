"""The memory charge of an NPU model: the footprint its ticket records, or the reserve."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilbee.catalog import query, xdna
from lilbee.catalog.download import download_model
from lilbee.catalog.query import resolve_pull_target
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType
from lilbee.providers import engine_params, model_cache
from lilbee.providers.fleet import model_memory as mm
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.engines.lemonade import LEMONADE
from lilbee.providers.fleet.engines.lemonade_models import parse_models
from lilbee.providers.fleet.memory_types import MemoryRequest
from lilbee.providers.roles import WorkerRole

GIB = 1024**3
TAG = "qwen3:0.6b"
FOOTPRINT_GB = 0.66
FOOTPRINT_BYTES = round(FOOTPRINT_GB * GIB)


def _request(ticket: Path) -> MemoryRequest:
    return MemoryRequest(
        model_path=ticket,
        ctx=2048,
        slots=1,
        gpu_layers=0,
        flash_attn=False,
        kv_cache_type=KvCacheType.F16,
    )


def _listed(footprint: float | None) -> dict[str, object]:
    entry: dict[str, object] = {
        "name": "Qwen3-0.6B-NPU2",
        "size": 600000000,
        "default_context_length": 32768,
        "details": {"parameter_size": "0.6B"},
    }
    if footprint is not None:
        entry["footprint"] = footprint
    return {"models": {"qwen3": {"0.6b": entry}}}


def _old_ticket(models_dir: Path) -> Path:
    """A ticket as a release before the footprint wrote it: the tag and the task only."""
    path = xdna.ticket_path(models_dir, TAG)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"checkpoint": TAG, "task": "chat"}), encoding="utf-8")
    return path


class TestTheTicket:
    def test_a_footprint_is_recorded_in_bytes(self, tmp_path: Path) -> None:
        ticket = xdna.write_ticket(tmp_path, TAG, ModelTask.CHAT, FOOTPRINT_GB)
        assert xdna.read_footprint(ticket) == FOOTPRINT_BYTES

    def test_no_footprint_records_none(self, tmp_path: Path) -> None:
        ticket = xdna.write_ticket(tmp_path, TAG, ModelTask.CHAT)
        assert xdna.read_footprint(ticket) is None
        assert "footprint_bytes" not in json.loads(ticket.read_text(encoding="utf-8"))

    def test_an_old_ticket_has_no_footprint_and_still_reads(self, tmp_path: Path) -> None:
        ticket = _old_ticket(tmp_path)
        assert xdna.read_footprint(ticket) is None
        assert xdna.read_checkpoint(ticket) == TAG
        assert xdna.read_task(ticket) is ModelTask.CHAT

    @pytest.mark.parametrize("value", [0, -1, "708669604", True, 1.5, None])
    def test_a_value_that_is_not_a_byte_count_is_no_footprint(
        self, tmp_path: Path, value: object
    ) -> None:
        ticket = _old_ticket(tmp_path)
        fields = {"checkpoint": TAG, "task": "chat", "footprint_bytes": value}
        ticket.write_text(json.dumps(fields), encoding="utf-8")
        assert xdna.read_footprint(ticket) is None

    def test_an_absent_ticket_has_no_footprint(self, tmp_path: Path) -> None:
        assert xdna.read_footprint(tmp_path / "absent.json") is None


class TestTheListedRow:
    def test_the_row_carries_the_published_footprint(self) -> None:
        (row,) = parse_models(_listed(FOOTPRINT_GB))
        assert row.footprint_gb == FOOTPRINT_GB

    def test_a_row_with_no_footprint_carries_none(self) -> None:
        (row,) = parse_models(_listed(None))
        assert row.footprint_gb == 0.0


class TestThePull:
    def test_a_pull_by_reference_of_a_listed_model_records_its_footprint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        rows = parse_models(_listed(FOOTPRINT_GB))
        monkeypatch.setattr(query, "_engine_rows", lambda offers: rows)
        entry = resolve_pull_target(f"{TAG}#xdna")
        assert entry is not None
        assert xdna.read_footprint(download_model(entry, None)) == FOOTPRINT_BYTES

    def test_a_pull_of_a_model_that_is_not_listed_records_no_footprint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        rows = parse_models(_listed(FOOTPRINT_GB))
        monkeypatch.setattr(query, "_engine_rows", lambda offers: rows)
        entry = resolve_pull_target("zed:1b#xdna")
        assert entry is not None
        assert (entry.hf_repo, entry.format) == ("zed:1b", ModelFormat.XDNA)
        assert xdna.read_footprint(download_model(entry, None)) is None


class TestTheCharge:
    def test_an_embedding_model_is_charged_its_footprint_and_the_encoder_allowance(
        self, tmp_path: Path
    ) -> None:
        ticket = xdna.write_ticket(tmp_path, TAG, ModelTask.EMBEDDING, FOOTPRINT_GB)
        est = mm.estimate_memory(ModelFormat.XDNA, _request(ticket))
        charge = FOOTPRINT_BYTES + mm.ENCODER_RUNTIME_BYTES
        assert (est.size_known, est.shared_bytes, est.device_bytes) == (True, charge, charge)

    def test_the_engine_charges_an_embedding_model_the_same(self, tmp_path: Path) -> None:
        ticket = xdna.write_ticket(tmp_path, TAG, ModelTask.EMBEDDING, FOOTPRINT_GB)
        est = LEMONADE.estimate_memory(_request(ticket))
        assert (est.size_known, est.shared_bytes) == (
            True,
            FOOTPRINT_BYTES + mm.ENCODER_RUNTIME_BYTES,
        )

    def test_a_chat_model_with_a_footprint_takes_the_reserve(self, tmp_path: Path) -> None:
        ticket = xdna.write_ticket(tmp_path, TAG, ModelTask.CHAT, FOOTPRINT_GB)
        est = mm.estimate_memory(ModelFormat.XDNA, _request(ticket))
        assert (est.size_known, est.shared_bytes, est.host_bytes) == (
            False,
            mm.UNKNOWN_SIZE_RESERVE_BYTES,
            mm.UNKNOWN_SIZE_RESERVE_BYTES,
        )

    def test_a_chat_footprint_above_the_reserve_is_the_charge(self, tmp_path: Path) -> None:
        ticket = xdna.write_ticket(tmp_path, TAG, ModelTask.CHAT, 10.5)
        est = mm.estimate_memory(ModelFormat.XDNA, _request(ticket))
        assert (est.size_known, est.shared_bytes) == (False, round(10.5 * GIB))

    def test_a_ticket_with_no_task_takes_the_reserve(self, tmp_path: Path) -> None:
        ticket = xdna.write_ticket(tmp_path, TAG, ModelTask.EMBEDDING, FOOTPRINT_GB)
        fields = {"checkpoint": TAG, "footprint_bytes": FOOTPRINT_BYTES}
        ticket.write_text(json.dumps(fields), encoding="utf-8")
        est = mm.estimate_memory(ModelFormat.XDNA, _request(ticket))
        assert (est.size_known, est.shared_bytes) == (False, mm.UNKNOWN_SIZE_RESERVE_BYTES)

    @pytest.mark.parametrize("task", [ModelTask.CHAT, ModelTask.EMBEDDING])
    def test_no_footprint_takes_the_reserve(self, tmp_path: Path, task: ModelTask) -> None:
        ticket = xdna.write_ticket(tmp_path, TAG, task)
        est = mm.estimate_memory(ModelFormat.XDNA, _request(ticket))
        assert (est.size_known, est.shared_bytes, est.host_bytes) == (
            False,
            mm.UNKNOWN_SIZE_RESERVE_BYTES,
            mm.UNKNOWN_SIZE_RESERVE_BYTES,
        )

    def test_an_old_ticket_takes_the_reserve(self, tmp_path: Path) -> None:
        est = mm.estimate_memory(ModelFormat.XDNA, _request(_old_ticket(tmp_path)))
        assert (est.size_known, est.shared_bytes) == (False, mm.UNKNOWN_SIZE_RESERVE_BYTES)


class TestTheRefusal:
    def _plan(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ticket: Path,
        ram: int,
        role: WorkerRole = WorkerRole.CHAT,
    ) -> planning.PlacementPlan:
        monkeypatch.setattr(cfg, "chat_model", "org/npu-model")
        monkeypatch.setattr(cfg, "embedding_model", "org/npu-model")
        monkeypatch.setattr(engine_params, "resolve_model_path", lambda ref, registry=None: ticket)
        monkeypatch.setattr(type(LEMONADE), "binary", lambda self: Path("/nonexistent/lemond"))
        monkeypatch.setattr(model_cache, "total_system_memory", lambda: ram)
        monkeypatch.setattr(planning, "total_system_memory", lambda: ram)
        return planning.resolve_placement_plan(None, engines={role: LEMONADE})

    def test_a_chat_model_is_refused_with_its_unknown_context_and_the_reserve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ticket = xdna.write_ticket(tmp_path, TAG, ModelTask.CHAT, FOOTPRINT_GB)
        reason = self._plan(monkeypatch, ticket, 6 * GIB).unplaceable_reasons[WorkerRole.CHAT]
        assert reason.endswith(
            "does not fit available memory and will not be served. The size of the chat "
            "model with its context is unknown, so 8.0 GiB is reserved for it. Free up memory."
        )
        assert "smaller model" not in reason

    def test_an_unsized_embedding_model_is_refused_with_the_reserve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ticket = xdna.write_ticket(tmp_path, TAG, ModelTask.EMBEDDING)
        plan = self._plan(monkeypatch, ticket, 6 * GIB, WorkerRole.EMBED)
        reason = plan.unplaceable_reasons[WorkerRole.EMBED]
        assert reason.endswith(
            "does not fit available memory and will not be served. The size of the embed "
            "model is unknown, so 8.0 GiB is reserved for it. Free up memory."
        )

    def test_an_embedding_model_with_a_footprint_places_where_the_reserve_does_not_fit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ticket = xdna.write_ticket(tmp_path, TAG, ModelTask.EMBEDDING, FOOTPRINT_GB)
        plan = self._plan(monkeypatch, ticket, 6 * GIB, WorkerRole.EMBED)
        assert plan.unplaceable_reasons == {}
        assert plan.size_unknown_roles == ()
        assert plan.charged_bytes[WorkerRole.EMBED] == FOOTPRINT_BYTES + mm.ENCODER_RUNTIME_BYTES
