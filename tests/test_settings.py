"""Tests for persistent settings (config.toml)."""

from unittest import mock

from lilbee import settings


class TestLoad:
    def test_load_missing_file_returns_empty(self, tmp_path):
        with mock.patch.object(settings, "_config_path", return_value=tmp_path / "config.toml"):
            assert settings.load() == {}

    def test_load_existing_file(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('chat_model = "llama3"\n')
        with mock.patch.object(settings, "_config_path", return_value=path):
            assert settings.load() == {"chat_model": "llama3"}


class TestSave:
    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "config.toml"
        with mock.patch.object(settings, "_config_path", return_value=path):
            settings.save({"chat_model": "llama3"})
        assert path.exists()
        assert 'chat_model = "llama3"' in path.read_text()

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "config.toml"
        with mock.patch.object(settings, "_config_path", return_value=path):
            settings.save({"key": "value"})
        assert path.exists()

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "config.toml"
        with mock.patch.object(settings, "_config_path", return_value=path):
            settings.save({"chat_model": "phi3", "top_k": "20"})
            result = settings.load()
        assert result == {"chat_model": "phi3", "top_k": "20"}


class TestGet:
    def test_get_existing_key(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('chat_model = "llama3"\n')
        with mock.patch.object(settings, "_config_path", return_value=path):
            assert settings.get("chat_model") == "llama3"

    def test_get_missing_key(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('chat_model = "llama3"\n')
        with mock.patch.object(settings, "_config_path", return_value=path):
            assert settings.get("nonexistent") is None

    def test_get_missing_file(self, tmp_path):
        with mock.patch.object(settings, "_config_path", return_value=tmp_path / "nope.toml"):
            assert settings.get("anything") is None


class TestSetValue:
    def test_set_value_creates_file(self, tmp_path):
        path = tmp_path / "config.toml"
        with mock.patch.object(settings, "_config_path", return_value=path):
            settings.set_value("chat_model", "mistral")
            assert settings.get("chat_model") == "mistral"

    def test_set_value_preserves_existing(self, tmp_path):
        path = tmp_path / "config.toml"
        path.write_text('existing = "keep"\n')
        with mock.patch.object(settings, "_config_path", return_value=path):
            settings.set_value("chat_model", "phi3")
            result = settings.load()
        assert result == {"existing": "keep", "chat_model": "phi3"}

    def test_set_value_overwrites_key(self, tmp_path):
        path = tmp_path / "config.toml"
        with mock.patch.object(settings, "_config_path", return_value=path):
            settings.set_value("chat_model", "llama3")
            settings.set_value("chat_model", "mistral")
            assert settings.get("chat_model") == "mistral"
