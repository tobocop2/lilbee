"""The XDNA model format: a ref names an NPU model by tag, and the engine downloads it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilbee.catalog import xdna
from lilbee.catalog.download import download_model
from lilbee.catalog.formats import formats_from_tags, select_files
from lilbee.catalog.query import resolve_pull_target
from lilbee.catalog.refs import ModelRef, model_format, parse_model_ref
from lilbee.catalog.types import ModelFormat
from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType
from lilbee.modelhub.registry import ModelRegistry
from lilbee.providers.fleet import model_memory as mm
from lilbee.providers.fleet.memory_types import MemoryRequest
from lilbee.providers.model_ref import routes_to_native_model


class TestRef:
    def test_tag_ref_reads_xdna(self) -> None:
        parsed = parse_model_ref("qwen3:0.6b#xdna")
        assert parsed == ModelRef("qwen3:0.6b", "", ModelFormat.XDNA)
        assert parsed.canonical == "qwen3:0.6b#xdna"

    def test_model_format_of_a_tag_ref(self) -> None:
        assert model_format("embed-gemma:300m#xdna") is ModelFormat.XDNA

    @pytest.mark.parametrize(
        "bad", ["qwen3#xdna", "org/repo#xdna", ":0.6b#xdna", "Qwen3:0.6B#xdna"]
    )
    def test_a_ref_that_is_not_a_tag_is_refused(self, bad: str) -> None:
        with pytest.raises(ValueError, match="<name>:<tag>#xdna"):
            parse_model_ref(bad)

    def test_the_format_error_lists_xdna(self) -> None:
        with pytest.raises(ValueError, match="xdna"):
            parse_model_ref("qwen3:0.6b#nope")

    def test_a_tag_ref_routes_to_a_native_model(self) -> None:
        assert routes_to_native_model("qwen3:0.6b#xdna")

    def test_pull_target_is_an_entry_in_xdna(self) -> None:
        entry = resolve_pull_target("qwen3:0.6b#xdna")
        assert entry is not None
        assert (entry.hf_repo, entry.format) == ("qwen3:0.6b", ModelFormat.XDNA)
        assert entry.pull_ref == "qwen3:0.6b#xdna"

    def test_a_malformed_tag_has_no_pull_target(self) -> None:
        assert resolve_pull_target("qwen3#xdna") is None


class TestHubSide:
    def test_no_hub_tag_names_xdna(self) -> None:
        assert ModelFormat.XDNA not in formats_from_tags(["onnx", "ryzenai-hybrid", "xdna"])

    def test_a_repo_has_no_files_to_select(self) -> None:
        assert select_files(ModelFormat.XDNA, ["model.q4nx", "config.json"]) == []


class TestTicket:
    def test_the_ticket_names_the_tag(self, tmp_path: Path) -> None:
        path = xdna.write_ticket(tmp_path, "qwen3:0.6b")
        assert path == tmp_path / "xdna" / "qwen3@0.6b.json"
        assert json.loads(path.read_text(encoding="utf-8")) == {"checkpoint": "qwen3:0.6b"}
        assert xdna.read_checkpoint(path) == "qwen3:0.6b"

    def test_a_ticket_that_names_no_tag_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        path.write_text(json.dumps({"checkpoint": 3}), encoding="utf-8")
        with pytest.raises(ValueError, match="no NPU model tag"):
            xdna.read_checkpoint(path)

    def test_a_missing_ticket_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no NPU model tag"):
            xdna.read_checkpoint(tmp_path / "absent.json")


class TestPullAndResolve:
    def test_pull_writes_the_ticket_and_fetches_no_weights(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = resolve_pull_target("qwen3:0.6b#xdna")
        assert entry is not None
        path = download_model(entry, None)
        assert path == tmp_path / "xdna" / "qwen3@0.6b.json"
        assert xdna.read_checkpoint(path) == "qwen3:0.6b"

    def test_an_installed_tag_resolves_to_its_ticket(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        assert not registry.is_installed("qwen3:0.6b#xdna")
        ticket = xdna.write_ticket(tmp_path, "qwen3:0.6b")
        assert registry.resolve("qwen3:0.6b#xdna") == ticket
        assert registry.is_installed("qwen3:0.6b#xdna")

    def test_a_missing_ticket_is_not_installed(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError, match="not installed"):
            ModelRegistry(tmp_path).resolve("qwen3:0.6b#xdna")


class TestMemory:
    def test_the_estimate_is_the_unknown_size_reserve(self, tmp_path: Path) -> None:
        request = MemoryRequest(
            model_path=tmp_path / "t.json",
            ctx=2048,
            slots=1,
            gpu_layers=0,
            flash_attn=False,
            kv_cache_type=KvCacheType.F16,
        )
        est = mm.estimate_memory(ModelFormat.XDNA, request)
        assert est.device_bytes == mm.UNKNOWN_SIZE_RESERVE_BYTES
