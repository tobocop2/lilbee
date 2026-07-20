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


@pytest.fixture()
def fresh_manager():
    """Yield a new SessionManager with no server.json on either side of the test."""
    from lilbee.server.auth import SessionManager, server_json_path

    path = server_json_path()
    path.unlink(missing_ok=True)
    yield SessionManager()
    path.unlink(missing_ok=True)


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

    def test_generated_token_file_is_never_world_readable(
        self, fresh_manager, permissive_umask, monkeypatch
    ):
        """With the after-the-fact chmod neutered, the file must still be 0600.

        Pins that the descriptor is created restricted rather than widened by
        the umask and narrowed a moment later, which is the TOCTOU window.
        """
        from lilbee.core import security
        from lilbee.server import auth

        monkeypatch.setattr(security.Path, "chmod", lambda *_a, **_k: None)
        token = fresh_manager.load_or_generate()
        path = auth.server_json_path()
        assert json.loads(path.read_text(encoding="utf-8"))["token"] == token
        assert _mode(path) == 0o600

    def test_unchmoddable_token_file_warns_instead_of_failing_startup(
        self, fresh_manager, monkeypatch, caplog
    ):
        """A server.json owned by another user must not take the server down."""
        from lilbee.core import security
        from lilbee.server import auth

        first = fresh_manager.load_or_generate()
        # Widen it first: hardening skips the chmod when the mode is already
        # right, so without this the refusal below is never reached and the
        # test passes without exercising anything.
        auth.server_json_path().chmod(0o644)

        def refuse(*_args, **_kwargs):
            raise PermissionError("not owner")

        monkeypatch.setattr(security.Path, "chmod", refuse)
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
        from lilbee.core import security, settings

        monkeypatch.setattr(security.Path, "chmod", lambda *_a, **_k: None)
        settings.save(tmp_path, {"api_key": "sk-secret"})
        path = tmp_path / "config.toml"
        assert _mode(path) == 0o600


class TestPersistedTokenIsTotal:
    """Every corruption mode must mint a fresh token, never take the server down."""

    def test_non_utf8_server_json_regenerates_instead_of_crashing(self, fresh_manager):
        """UnicodeDecodeError subclasses ValueError, not OSError, so it escaped
        both except clauses and killed the app lifespan on a truncated write."""
        from lilbee.server.auth import server_json_path

        path = server_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{"token": "\xff\xfe not utf-8"}')

        token = fresh_manager.load_or_generate()

        assert isinstance(token, str)
        assert json.loads(path.read_text(encoding="utf-8"))["token"] == token

    def test_a_token_shorter_than_the_generator_mints_is_rejected(self, fresh_manager):
        """The length floor is a character count, so it must match the encoded
        length, not the raw entropy byte count it used to be compared against."""
        from lilbee.server.auth import server_json_path

        path = server_json_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # 32 chars: passed the old byte-count floor, but well short of the ~43
        # characters secrets.token_urlsafe(32) actually produces.
        path.write_text(json.dumps({"token": "a" * 32}), encoding="utf-8")

        token = fresh_manager.load_or_generate()

        assert token != "a" * 32
        assert len(token) > 32


class TestValidatePathWithinAnchorsToRoot:
    """A relative path resolved against the process CWD, not the root it is
    being checked into, so a bare name was judged against wherever the process
    happened to be started."""

    def test_a_relative_path_resolves_under_root(self, tmp_path):
        from lilbee.core.security import validate_path_within

        assert validate_path_within("notes.txt", tmp_path) == (tmp_path / "notes.txt").resolve()

    def test_a_relative_traversal_is_still_blocked(self, tmp_path):
        from lilbee.core.security import PathTraversalError, validate_path_within

        with pytest.raises(PathTraversalError):
            validate_path_within("../escape.txt", tmp_path)

    def test_an_absolute_path_is_unchanged(self, tmp_path):
        from lilbee.core.security import validate_path_within

        target = tmp_path / "a.txt"
        assert validate_path_within(str(target), tmp_path) == target.resolve()


class TestPersistedSettingsAreHardenedOnLoad:
    """config.toml holds provider API keys and is read indefinitely without
    ever being rewritten, so a file that arrived world-readable must be
    narrowed on load, not only when something happens to save it."""

    def test_a_world_readable_config_is_narrowed_on_read(self, tmp_path):
        from lilbee.core import settings

        settings.set_value(tmp_path, "api_key", "sk-secret")
        path = tmp_path / "config.toml"
        path.chmod(0o644)

        assert settings.load(tmp_path) == {"api_key": "sk-secret"}
        assert _mode(path) == 0o600

    def test_an_unchmoddable_config_warns_instead_of_failing_the_read(
        self, tmp_path, monkeypatch, caplog
    ):
        from lilbee.core import security, settings

        settings.set_value(tmp_path, "api_key", "sk-secret")
        (tmp_path / "config.toml").chmod(0o644)

        def refuse(*_args, **_kwargs):
            raise PermissionError("not owner")

        monkeypatch.setattr(security.Path, "chmod", refuse)
        with caplog.at_level("WARNING"):
            assert settings.load(tmp_path) == {"api_key": "sk-secret"}
        assert "Could not restrict permissions" in caplog.text
