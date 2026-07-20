"""Secret files must be owner-only from the moment they exist, not chmod'd afterwards."""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from lilbee.core.security import write_private_text

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits only")


@pytest.fixture()
def permissive_umask():
    """Run the body under umask 0 so any non-atomic write lands world-readable."""
    previous = os.umask(0)
    yield
    os.umask(previous)


def _mode(path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


class TestWritePrivateText:
    def test_creates_file_owner_only_under_permissive_umask(self, tmp_path, permissive_umask):
        target = tmp_path / "secret.txt"
        write_private_text(target, "s3cret")
        assert target.read_text(encoding="utf-8") == "s3cret"
        assert _mode(target) == 0o600

    def test_creates_missing_parents(self, tmp_path):
        target = tmp_path / "nested" / "deeper" / "secret.txt"
        write_private_text(target, "s3cret")
        assert target.read_text(encoding="utf-8") == "s3cret"

    def test_replaces_existing_file_and_narrows_its_mode(self, tmp_path, permissive_umask):
        target = tmp_path / "secret.txt"
        target.write_text("old", encoding="utf-8")
        target.chmod(0o644)
        write_private_text(target, "new")
        assert target.read_text(encoding="utf-8") == "new"
        assert _mode(target) == 0o600

    def test_leaves_no_temp_file_behind_when_the_write_fails(self, tmp_path, monkeypatch):
        def boom(*_args, **_kwargs):
            raise RuntimeError("boom")

        target = tmp_path / "secret.txt"
        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(RuntimeError):
            write_private_text(target, "s3cret")
        assert list(tmp_path.iterdir()) == []


class TestSessionTokenPermissions:
    """server.json holds the bearer token: it must never be readable by other users."""

    @pytest.fixture()
    def fresh_manager(self):
        from lilbee.server.auth import SessionManager, server_json_path

        path = server_json_path()
        path.unlink(missing_ok=True)
        yield SessionManager()
        path.unlink(missing_ok=True)

    def test_generated_token_file_is_never_world_readable(
        self, fresh_manager, permissive_umask, monkeypatch
    ):
        """With the after-the-fact chmod neutered, the file must still be 0600.

        Pins that the descriptor is created restricted rather than widened by
        the umask and narrowed a moment later, which is the TOCTOU window.
        """
        from lilbee.server import auth

        monkeypatch.setattr(auth.Path, "chmod", lambda *_a, **_k: None)
        token = fresh_manager.load_or_generate()
        path = auth.server_json_path()
        assert json.loads(path.read_text(encoding="utf-8"))["token"] == token
        assert _mode(path) == 0o600

    def test_unchmoddable_token_file_warns_instead_of_failing_startup(
        self, fresh_manager, monkeypatch, caplog
    ):
        """A server.json owned by another user must not take the server down."""
        from lilbee.server import auth

        first = fresh_manager.load_or_generate()

        def refuse(*_args, **_kwargs):
            raise PermissionError("not owner")

        monkeypatch.setattr(auth.Path, "chmod", refuse)
        fresh_manager.token = None
        with caplog.at_level("WARNING"):
            assert fresh_manager.load_or_generate() == first
        assert "Could not restrict permissions" in caplog.text

    def test_reused_token_file_is_hardened_on_load(self, fresh_manager):
        """A server.json that arrived world-readable gets narrowed, not left as-is."""
        from lilbee.server.auth import server_json_path

        first = fresh_manager.load_or_generate()
        path = server_json_path()
        path.chmod(0o644)

        fresh_manager.token = None
        second = fresh_manager.load_or_generate()

        assert second == first
        assert _mode(path) == 0o600


class TestPersistedSettingsPermissions:
    """config.toml can hold provider API keys and gets the same treatment."""

    def test_saved_config_is_never_world_readable(self, tmp_path, permissive_umask, monkeypatch):
        from lilbee.core import settings

        monkeypatch.setattr(settings.Path, "chmod", lambda *_a, **_k: None)
        settings.save(tmp_path, {"api_key": "sk-secret"})
        path = tmp_path / "config.toml"
        assert _mode(path) == 0o600
