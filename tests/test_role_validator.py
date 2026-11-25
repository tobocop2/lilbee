"""Role-slot validation against the current picks and the installed registry."""

from __future__ import annotations

import pytest

from conftest import PICKS_CHAT, PICKS_EMBEDDING, install_fake_model
from lilbee.app.services import get_services
from lilbee.modelhub.role_validator import TaskMismatchError, validate_model_task_assignment


class TestPickRefs:
    def test_bare_pick_repo_canonicalizes_to_the_picks_ref(self) -> None:
        """A bare repo that is a current pick resolves without being installed."""
        pick = PICKS_CHAT[0]
        result = validate_model_task_assignment(
            "chat_model",
            pick.hf_repo,
            allow_bypass=False,
            hf_client_for=lambda: get_services().hf_client,
        )
        assert result == pick.ref

    def test_native_gguf_ref_is_kept_verbatim(self) -> None:
        """A full ref names an exact quant, so it survives canonicalization."""
        pick = PICKS_CHAT[0]
        full = f"{pick.hf_repo}/some-quant-Q4_K_M.gguf"
        assert (
            validate_model_task_assignment(
                "chat_model",
                full,
                allow_bypass=False,
                hf_client_for=lambda: get_services().hf_client,
            )
            == full
        )

    def test_pick_in_the_wrong_slot_reports_a_task_mismatch(self) -> None:
        """The pick carries its task, so this is a mismatch, not 'not installed'."""
        pick = PICKS_EMBEDDING[0]
        with pytest.raises(TaskMismatchError):
            validate_model_task_assignment(
                "chat_model",
                pick.hf_repo,
                allow_bypass=False,
                hf_client_for=lambda: get_services().hf_client,
            )


class TestNonPickRefs:
    def test_uninstalled_non_pick_is_rejected(self) -> None:
        """Nothing can vouch for a repo that is neither a pick nor installed."""
        with pytest.raises(ValueError, match="not installed"):
            validate_model_task_assignment(
                "chat_model",
                "nobody/Not-A-Pick-GGUF",
                allow_bypass=False,
                hf_client_for=lambda: get_services().hf_client,
            )

    def test_installed_non_pick_is_accepted_for_its_task(self) -> None:
        ref = install_fake_model("other/Chat-Model-GGUF", "chat-Q4_K_M.gguf", "chat")
        result = validate_model_task_assignment(
            "chat_model", ref, allow_bypass=False, hf_client_for=lambda: get_services().hf_client
        )
        assert result == ref

    def test_installed_non_pick_in_the_wrong_slot_is_rejected(self) -> None:
        ref = install_fake_model("other/Chat-Model-GGUF", "chat-Q4_K_M.gguf", "chat")
        with pytest.raises(TaskMismatchError):
            validate_model_task_assignment(
                "embedding_model",
                ref,
                allow_bypass=False,
                hf_client_for=lambda: get_services().hf_client,
            )

    def test_installed_bare_repo_resolves_to_its_quant(self) -> None:
        ref = install_fake_model("other/Chat-Model-GGUF", "chat-Q4_K_M.gguf", "chat")
        result = validate_model_task_assignment(
            "chat_model",
            "other/Chat-Model-GGUF",
            allow_bypass=False,
            hf_client_for=lambda: get_services().hf_client,
        )
        assert result == ref


class TestNoNetworkOnTheWritePath:
    def test_an_installed_ref_is_validated_without_resolving_picks(self, monkeypatch) -> None:
        """The role write boundary runs on the TUI main thread; it must not fetch."""
        from lilbee.catalog import picks as picks_mod

        ref = install_fake_model("other/Chat-Model-GGUF", "chat-Q4_K_M.gguf", "chat")

        def boom(**_) -> tuple:
            raise AssertionError("resolved picks on the role write path")

        monkeypatch.setattr(picks_mod, "_resolve_picks", boom)
        picks_mod.reset_picks()
        assert (
            validate_model_task_assignment(
                "chat_model",
                ref,
                allow_bypass=False,
                hf_client_for=lambda: get_services().hf_client,
            )
            == ref
        )


_VLM_REPO = "other/Chat-Vlm-GGUF"
_VLM_FILE = "chat-vlm-Q4_K_M.gguf"
_REV = "0123456789abcdef0123456789abcdef01234567"


def _seed_cache_file(repo: str, filename: str, content: bytes) -> None:
    """Lay down a HuggingFace cache entry (blob, snapshot symlink, refs) under cfg.models_dir."""
    import hashlib

    from lilbee.core.config import cfg
    from lilbee.modelhub.registry import repo_to_dir

    cache = cfg.models_dir / f"models--{repo_to_dir(repo)}"
    blob = cache / "blobs" / hashlib.sha256(content).hexdigest()
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(content)
    link = cache / "snapshots" / _REV / filename
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(blob)
    (cache / "refs").mkdir(parents=True, exist_ok=True)
    (cache / "refs" / "main").write_text(_REV, encoding="utf-8")


@pytest.fixture
def recovered_vlm() -> str:
    """A chat model recovered from the cache, with a projector beside it and no manifest."""
    from lilbee.catalog.refs import format_native_gguf_ref
    from lilbee.core.config import cfg
    from lilbee.modelhub.registry import ModelRegistry

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    _seed_cache_file(_VLM_REPO, _VLM_FILE, b"GGUF\x00\x00")
    _seed_cache_file(_VLM_REPO, "mmproj-f16.gguf", b"GGUF\x01")
    ref = format_native_gguf_ref(_VLM_REPO, _VLM_FILE)
    ModelRegistry(cfg.models_dir).resolve(ref)  # recovery writes the manifest
    return ref


class TestRecoveredModelLabelByName:
    def test_a_recovered_vision_named_model_with_a_projector_is_vision(self) -> None:
        from lilbee.catalog.refs import format_native_gguf_ref
        from lilbee.catalog.types import ModelTask
        from lilbee.core.config import cfg
        from lilbee.modelhub.registry import ModelRegistry

        repo, filename = "ggml-org/SmolVLM-256M-Instruct-GGUF", "SmolVLM-256M-Instruct-Q8_0.gguf"
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        _seed_cache_file(repo, filename, b"GGUF\x00\x00")
        _seed_cache_file(repo, "mmproj-SmolVLM-256M-Instruct-f16.gguf", b"GGUF\x01")
        ref = format_native_gguf_ref(repo, filename)
        registry = ModelRegistry(cfg.models_dir)
        registry.resolve(ref)  # recovery writes the manifest
        manifest = registry.get_manifest(ref)
        assert manifest is not None
        assert manifest.task == ModelTask.VISION
        assert (
            validate_model_task_assignment(
                "vision_model",
                ref,
                allow_bypass=False,
                hf_client_for=lambda: get_services().hf_client,
            )
            == ref
        )


class TestUnknownTaskRole:
    def test_assigning_a_model_with_an_unknown_task_names_the_model(self) -> None:
        ref = install_fake_model("other/Plain-Chat-GGUF", "plain-Q4_K_M.gguf", "bogus")
        with pytest.raises(ValueError, match="unknown task 'bogus'") as caught:
            validate_model_task_assignment(
                "chat_model",
                ref,
                allow_bypass=False,
                hf_client_for=lambda: get_services().hf_client,
            )
        assert ref in str(caught.value)
        assert not isinstance(caught.value, TaskMismatchError)


class TestRecoveredModelWithProjector:
    def test_recovery_keeps_the_chat_label(self, recovered_vlm: str) -> None:
        from lilbee.core.config import cfg
        from lilbee.modelhub.registry import ModelRegistry

        manifest = ModelRegistry(cfg.models_dir).get_manifest(recovered_vlm)
        assert manifest is not None
        assert manifest.task == "chat"

    def test_stays_in_the_chat_picker_and_is_offered_for_vision(self, recovered_vlm: str) -> None:
        from lilbee.catalog.types import ModelTask
        from lilbee.cli.tui.widgets.model_bar import _collect_native_models

        buckets: dict = {task: [] for task in ModelTask}
        _collect_native_models(buckets, set())

        assert [o.ref for o in buckets[ModelTask.CHAT]] == [recovered_vlm]
        assert [o.ref for o in buckets[ModelTask.VISION]] == [recovered_vlm]

    def test_vision_role_accepts_it(self, recovered_vlm: str) -> None:
        result = validate_model_task_assignment(
            "vision_model",
            recovered_vlm,
            allow_bypass=False,
            hf_client_for=lambda: get_services().hf_client,
        )
        assert result == recovered_vlm

    def test_vision_route_check_accepts_it(self, recovered_vlm: str, monkeypatch) -> None:
        """PUT /api/models/vision validates through this check; it answered 422 before."""
        from lilbee.catalog.types import ModelTask
        from lilbee.server.handlers import models as handlers

        monkeypatch.setattr(handlers, "_require_model_available", lambda model: model)
        assert handlers._require_model_for_task(recovered_vlm, ModelTask.VISION) == recovered_vlm

    def test_a_chat_model_without_a_projector_is_still_rejected_for_vision(self) -> None:
        ref = install_fake_model("other/Plain-Chat-GGUF", "plain-Q4_K_M.gguf", "chat")
        with pytest.raises(TaskMismatchError):
            validate_model_task_assignment(
                "vision_model",
                ref,
                allow_bypass=False,
                hf_client_for=lambda: get_services().hf_client,
            )


def _services_unavailable() -> None:
    raise RuntimeError("services cannot be built")


class TestRoleWriteNeedsNoServicesUnlessRemote:
    """A role write reaches the HF client only for a ref that needs a remote lookup."""

    def test_an_installed_ref_never_asks_for_the_client(self) -> None:
        ref = install_fake_model("other/Chat-Model-GGUF", "chat-Q4_K_M.gguf", "chat")
        result = validate_model_task_assignment(
            "chat_model", ref, allow_bypass=False, hf_client_for=_services_unavailable
        )
        assert result == ref

    def test_a_bypass_ref_never_asks_for_the_client(self, monkeypatch) -> None:
        monkeypatch.setenv("LILBEE_SKIP_MODEL_TASK_VALIDATION", "1")
        result = validate_model_task_assignment(
            "chat_model", "nobody/Not-A-Pick-GGUF", hf_client_for=_services_unavailable
        )
        assert result == "nobody/Not-A-Pick-GGUF"

    def test_a_remote_lookup_asks_for_the_client(self) -> None:
        with pytest.raises(RuntimeError, match="services cannot be built"):
            validate_model_task_assignment(
                "chat_model",
                "nobody/Not-A-Pick-GGUF",
                allow_bypass=False,
                hf_client_for=_services_unavailable,
            )

    def test_the_settings_write_of_an_installed_ref_builds_no_services(self, monkeypatch) -> None:
        from lilbee.app import services as services_module
        from lilbee.app.settings import _coerce_value

        ref = install_fake_model("other/Chat-Model-GGUF", "chat-Q4_K_M.gguf", "chat")
        monkeypatch.setattr(services_module, "get_services", _services_unavailable)
        assert _coerce_value("chat_model", ref) == ref
