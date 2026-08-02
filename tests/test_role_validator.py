"""Role-slot validation against the current picks and the installed registry."""

from __future__ import annotations

import pytest

from conftest import PICKS_CHAT, PICKS_EMBEDDING, install_fake_model
from lilbee.modelhub.role_validator import TaskMismatchError, validate_model_task_assignment


class TestPickRefs:
    def test_bare_pick_repo_canonicalizes_to_the_picks_ref(self) -> None:
        """A bare repo that is a current pick resolves without being installed."""
        pick = PICKS_CHAT[0]
        result = validate_model_task_assignment("chat_model", pick.hf_repo, allow_bypass=False)
        assert result == pick.ref

    def test_native_gguf_ref_is_kept_verbatim(self) -> None:
        """A full ref names an exact quant, so it survives canonicalization."""
        pick = PICKS_CHAT[0]
        full = f"{pick.hf_repo}/some-quant-Q4_K_M.gguf"
        assert validate_model_task_assignment("chat_model", full, allow_bypass=False) == full

    def test_pick_in_the_wrong_slot_reports_a_task_mismatch(self) -> None:
        """The pick carries its task, so this is a mismatch, not 'not installed'."""
        pick = PICKS_EMBEDDING[0]
        with pytest.raises(TaskMismatchError):
            validate_model_task_assignment("chat_model", pick.hf_repo, allow_bypass=False)


class TestNonPickRefs:
    def test_uninstalled_non_pick_is_rejected(self) -> None:
        """Nothing can vouch for a repo that is neither a pick nor installed."""
        with pytest.raises(ValueError, match="not installed"):
            validate_model_task_assignment(
                "chat_model", "nobody/Not-A-Pick-GGUF", allow_bypass=False
            )

    def test_installed_non_pick_is_accepted_for_its_task(self) -> None:
        ref = install_fake_model("other/Chat-Model-GGUF", "chat-Q4_K_M.gguf", "chat")
        result = validate_model_task_assignment("chat_model", ref, allow_bypass=False)
        assert result == ref

    def test_installed_non_pick_in_the_wrong_slot_is_rejected(self) -> None:
        ref = install_fake_model("other/Chat-Model-GGUF", "chat-Q4_K_M.gguf", "chat")
        with pytest.raises(TaskMismatchError):
            validate_model_task_assignment("embedding_model", ref, allow_bypass=False)

    def test_installed_bare_repo_resolves_to_its_quant(self) -> None:
        ref = install_fake_model("other/Chat-Model-GGUF", "chat-Q4_K_M.gguf", "chat")
        result = validate_model_task_assignment(
            "chat_model", "other/Chat-Model-GGUF", allow_bypass=False
        )
        assert result == ref


class TestNoNetworkOnTheWritePath:
    def test_an_installed_ref_is_validated_without_resolving_picks(self, monkeypatch) -> None:
        """The role write boundary runs on the TUI main thread; it must not fetch."""
        from lilbee.catalog import picks as picks_mod

        ref = install_fake_model("other/Chat-Model-GGUF", "chat-Q4_K_M.gguf", "chat")

        def boom() -> tuple:
            raise AssertionError("resolved picks on the role write path")

        monkeypatch.setattr(picks_mod, "_resolve_picks", boom)
        picks_mod.reset_picks()
        assert validate_model_task_assignment("chat_model", ref, allow_bypass=False) == ref
