"""The XDNA model format: a ref names an NPU model by tag, and the engine downloads it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilbee.app.models import remove_model_data, show_model_data
from lilbee.catalog import xdna
from lilbee.catalog.download import download_model
from lilbee.catalog.formats import formats_from_tags, select_files
from lilbee.catalog.query import resolve_pull_target
from lilbee.catalog.refs import ModelRef, model_format, parse_model_ref
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.cli.commands.setup import _installed_model_path
from lilbee.core.config import cfg
from lilbee.core.config.enums import KvCacheType
from lilbee.modelhub.model_manager import ModelNotFoundError
from lilbee.modelhub.model_manager.validation import _first_installed_local_ref
from lilbee.modelhub.registry import ModelRegistry
from lilbee.modelhub.role_validator import validate_model_task_assignment
from lilbee.providers.fleet import model_memory as mm
from lilbee.providers.fleet import planning
from lilbee.providers.fleet.engines import choice
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.lemonade import LEMONADE
from lilbee.providers.fleet.memory_types import MemoryRequest
from lilbee.providers.model_ref import routes_to_native_model
from lilbee.providers.roles import WorkerRole


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
        path = xdna.write_ticket(tmp_path, "qwen3:0.6b", ModelTask.CHAT)
        assert path == tmp_path / "xdna" / "qwen3@0.6b.json"
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "checkpoint": "qwen3:0.6b",
            "task": "chat",
        }
        assert xdna.read_checkpoint(path) == "qwen3:0.6b"
        assert xdna.read_task(path) == ModelTask.CHAT

    @pytest.mark.parametrize("body", [{"checkpoint": "qwen3:0.6b"}, {"task": "bogus"}, []])
    def test_a_ticket_that_names_no_task_is_refused(self, tmp_path: Path, body: object) -> None:
        path = tmp_path / "t.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        with pytest.raises(ValueError, match="no model task"):
            xdna.read_task(path)

    def test_no_ticket_dir_lists_no_ticket(self, tmp_path: Path) -> None:
        assert xdna.ticket_paths(tmp_path) == []
        written = xdna.write_ticket(tmp_path, "qwen3:0.6b", ModelTask.CHAT)
        assert xdna.ticket_paths(tmp_path) == [written]

    def test_a_ticket_that_names_no_tag_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        path.write_text(json.dumps({"checkpoint": 3}), encoding="utf-8")
        with pytest.raises(ValueError, match="no NPU model tag"):
            xdna.read_checkpoint(path)

    @pytest.mark.parametrize("tag", ["../x:1", "qwen3", "qwen3:0.6b --ctx-len 9", ""])
    def test_a_ticket_with_a_malformed_tag_is_refused(self, tmp_path: Path, tag: str) -> None:
        path = tmp_path / "t.json"
        path.write_text(json.dumps({"checkpoint": tag}), encoding="utf-8")
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
        ticket = xdna.write_ticket(tmp_path, "qwen3:0.6b", ModelTask.CHAT)
        assert registry.resolve("qwen3:0.6b#xdna") == ticket
        assert registry.is_installed("qwen3:0.6b#xdna")

    def test_a_missing_ticket_is_not_installed(self, tmp_path: Path) -> None:
        with pytest.raises(KeyError, match="not installed"):
            ModelRegistry(tmp_path).resolve("qwen3:0.6b#xdna")

    def test_removing_a_tag_deletes_its_ticket_once(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        ticket = xdna.write_ticket(tmp_path, "qwen3:0.6b", ModelTask.CHAT)
        assert registry.remove("qwen3:0.6b#xdna") is True
        assert not ticket.exists()
        assert not registry.is_installed("qwen3:0.6b#xdna")
        assert registry.remove("qwen3:0.6b#xdna") is False


class TestInstalledList:
    REF = "qwen3:0.6b#xdna"

    def _pull(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        entry = resolve_pull_target(self.REF)
        assert entry is not None
        download_model(entry, None)

    def test_a_pulled_tag_is_listed_and_assignable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._pull(tmp_path, monkeypatch)
        assert [m.ref for m in ModelRegistry(tmp_path).list_installed()] == [self.REF]
        assert (
            validate_model_task_assignment("chat_model", self.REF, allow_bypass=False) == self.REF
        )

    def test_the_listed_entry_carries_the_tag_and_the_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._pull(tmp_path, monkeypatch)
        (manifest,) = ModelRegistry(tmp_path).list_installed()
        assert (manifest.hf_repo, manifest.format, manifest.task) == (
            "qwen3:0.6b",
            ModelFormat.XDNA,
            ModelTask.CHAT,
        )
        assert manifest.disk_size_bytes == 0
        assert manifest.downloaded_at

    def test_the_task_of_the_ticket_decides_the_role(self, tmp_path: Path) -> None:
        xdna.write_ticket(tmp_path, "qwen3:0.6b", ModelTask.EMBEDDING)
        ref = "qwen3:0.6b#xdna"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(cfg, "models_dir", tmp_path)
            assert validate_model_task_assignment("embedding_model", ref, allow_bypass=False) == ref
            with pytest.raises(ValueError, match="embedding"):
                validate_model_task_assignment("chat_model", ref, allow_bypass=False)

    def test_a_missing_ticket_is_not_listed_and_not_assignable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        assert ModelRegistry(tmp_path).list_installed() == []
        assert ModelRegistry(tmp_path).get_manifest(self.REF) is None
        with pytest.raises(ValueError, match="not installed"):
            validate_model_task_assignment("chat_model", self.REF, allow_bypass=False)

    def _old_ticket(self, tmp_path: Path) -> None:
        path = xdna.ticket_path(tmp_path, "qwen3:0.6b")
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"checkpoint": "qwen3:0.6b"}), encoding="utf-8")

    def test_a_ticket_without_a_task_is_not_installed(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._old_ticket(tmp_path)
        registry = ModelRegistry(tmp_path)
        assert registry.get_manifest(self.REF) is None
        assert not registry.is_installed(self.REF)
        assert registry.has_unreadable_ticket(self.REF)
        assert "Pull the model again" not in caplog.text
        assert registry.list_installed() == []
        assert caplog.text.count("Pull the model again") == 1

    def test_a_ticket_without_a_task_tells_the_user_to_pull_again(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cfg, "models_dir", tmp_path)
        self._old_ticket(tmp_path)
        with pytest.raises(ModelNotFoundError, match="Pull it again"):
            show_model_data(self.REF)

    @pytest.mark.parametrize("ref", ["qwen3:0.6b#xdna", "org/repo/m.gguf", "bad#nope"])
    def test_no_unreadable_ticket_for_a_ref_without_one(self, tmp_path: Path, ref: str) -> None:
        assert not ModelRegistry(tmp_path).has_unreadable_ticket(ref)

    def test_the_self_check_says_it_skips_a_ticket(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._pull(tmp_path, monkeypatch)
        assert _installed_model_path(ModelTask.CHAT, self.REF) is None
        assert (
            f"Skipping {self.REF}: the self-check loads GGUF files only." in capsys.readouterr().out
        )

    def test_the_self_check_never_takes_a_ticket_for_a_gguf(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._pull(tmp_path, monkeypatch)
        assert _installed_model_path(ModelTask.CHAT, self.REF) is None

    def test_a_substitute_model_needs_an_engine_that_loads_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._pull(tmp_path, monkeypatch)
        monkeypatch.setattr(choice, "installed_engines", lambda: {})
        assert _first_installed_local_ref(ModelTask.CHAT) is None
        monkeypatch.setattr(
            choice,
            "installed_engines",
            lambda: {EngineName.LEMONADE: frozenset({ModelFormat.XDNA})},
        )
        assert _first_installed_local_ref(ModelTask.CHAT) == self.REF

    def test_removing_a_listed_tag_reports_no_freed_bytes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._pull(tmp_path, monkeypatch)
        result = remove_model_data(self.REF)
        assert (result.deleted, result.freed_gb) == (True, 0.0)
        assert ModelRegistry(tmp_path).list_installed() == []

    def test_a_removed_tag_leaves_the_list(self, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path)
        xdna.write_ticket(tmp_path, "qwen3:0.6b", ModelTask.CHAT)
        assert len(registry.list_installed()) == 1
        assert registry.remove(self.REF) is True
        assert registry.list_installed() == []


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

    def test_the_estimate_states_the_size_is_unknown_and_charges_host_memory(
        self, tmp_path: Path
    ) -> None:
        request = MemoryRequest(
            model_path=tmp_path / "t.json",
            ctx=2048,
            slots=1,
            gpu_layers=0,
            flash_attn=False,
            kv_cache_type=KvCacheType.F16,
        )
        est = mm.estimate_memory(ModelFormat.XDNA, request)
        assert (est.size_known, est.host_bytes) == (False, mm.UNKNOWN_SIZE_RESERVE_BYTES)

    def test_the_planner_charges_a_lemonade_model_the_reserve_on_card_and_host(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "model"
        folder.mkdir()
        monkeypatch.setattr(planning.engine_params, "resolve_model_path", lambda ref: folder)
        monkeypatch.setattr(LEMONADE, "model_meta", lambda path: None)
        charged = planning._estimate_role(WorkerRole.CHAT, "ref", engine=LEMONADE, slots=1)
        assert (charged.size_known, charged.est_vram_bytes, charged.est_ram_bytes) == (
            False,
            mm.UNKNOWN_SIZE_RESERVE_BYTES,
            mm.UNKNOWN_SIZE_RESERVE_BYTES,
        )
