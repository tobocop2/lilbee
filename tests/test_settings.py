"""Tests for persistent settings (config.toml)."""

from lilbee import settings


class TestLoad:
    def test_load_missing_file_returns_empty(self, tmp_path):
        assert settings.load(tmp_path) == {}

    def test_load_existing_file(self, tmp_path):
        (tmp_path / "config.toml").write_text('chat_model = "llama3"\n')
        assert settings.load(tmp_path) == {"chat_model": "llama3"}


class TestSave:
    def test_save_creates_file(self, tmp_path):
        settings.save(tmp_path, {"chat_model": "llama3"})
        assert (tmp_path / "config.toml").exists()
        assert 'chat_model = "llama3"' in (tmp_path / "config.toml").read_text()

    def test_save_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "nested" / "dir"
        settings.save(nested, {"key": "value"})
        assert (nested / "config.toml").exists()

    def test_save_load_roundtrip(self, tmp_path):
        settings.save(tmp_path, {"chat_model": "phi3", "top_k": "20"})
        result = settings.load(tmp_path)
        assert result == {"chat_model": "phi3", "top_k": "20"}


class TestGet:
    def test_get_existing_key(self, tmp_path):
        (tmp_path / "config.toml").write_text('chat_model = "llama3"\n')
        assert settings.get(tmp_path, "chat_model") == "llama3"

    def test_get_missing_key(self, tmp_path):
        (tmp_path / "config.toml").write_text('chat_model = "llama3"\n')
        assert settings.get(tmp_path, "nonexistent") is None

    def test_get_missing_file(self, tmp_path):
        assert settings.get(tmp_path, "anything") is None


class TestSetValue:
    def test_set_value_creates_file(self, tmp_path):
        settings.set_value(tmp_path, "chat_model", "mistral")
        assert settings.get(tmp_path, "chat_model") == "mistral"

    def test_set_value_preserves_existing(self, tmp_path):
        (tmp_path / "config.toml").write_text('existing = "keep"\n')
        settings.set_value(tmp_path, "chat_model", "phi3")
        result = settings.load(tmp_path)
        assert result == {"existing": "keep", "chat_model": "phi3"}

    def test_set_value_overwrites_key(self, tmp_path):
        settings.set_value(tmp_path, "chat_model", "llama3")
        settings.set_value(tmp_path, "chat_model", "mistral")
        assert settings.get(tmp_path, "chat_model") == "mistral"


class TestDeleteValue:
    def test_delete_existing_key(self, tmp_path):
        settings.set_value(tmp_path, "temperature", "0.5")
        settings.delete_value(tmp_path, "temperature")
        assert settings.get(tmp_path, "temperature") is None

    def test_delete_preserves_other_keys(self, tmp_path):
        settings.set_value(tmp_path, "chat_model", "llama3")
        settings.set_value(tmp_path, "temperature", "0.5")
        settings.delete_value(tmp_path, "temperature")
        assert settings.get(tmp_path, "chat_model") == "llama3"

    def test_delete_missing_key_is_noop(self, tmp_path):
        settings.set_value(tmp_path, "chat_model", "llama3")
        settings.delete_value(tmp_path, "nonexistent")
        assert settings.get(tmp_path, "chat_model") == "llama3"

    def test_delete_from_empty_file(self, tmp_path):
        settings.delete_value(tmp_path, "anything")
        assert settings.load(tmp_path) == {}


class TestTomlEscaping:
    def test_escape_double_quotes(self, tmp_path):
        settings.set_value(tmp_path, "prompt", 'say "hello"')
        assert settings.get(tmp_path, "prompt") == 'say "hello"'

    def test_escape_backslashes(self, tmp_path):
        settings.set_value(tmp_path, "path", r"C:\Users\test")
        assert settings.get(tmp_path, "path") == r"C:\Users\test"

    def test_escape_newlines(self, tmp_path):
        settings.set_value(tmp_path, "msg", "line1\nline2")
        assert settings.get(tmp_path, "msg") == "line1\nline2"

    def test_escape_tab(self, tmp_path):
        settings.set_value(tmp_path, "msg", "col1\tcol2")
        assert settings.get(tmp_path, "msg") == "col1\tcol2"

    def test_escape_mixed(self, tmp_path):
        val = 'He said "hello" at C:\\home\n'
        settings.set_value(tmp_path, "mixed", val)
        assert settings.get(tmp_path, "mixed") == val

    def test_escape_preserves_normal_values(self, tmp_path):
        settings.set_value(tmp_path, "model", "qwen3:8b")
        assert settings.get(tmp_path, "model") == "qwen3:8b"

    def test_escape_empty_string(self, tmp_path):
        settings.set_value(tmp_path, "key", "")
        assert settings.get(tmp_path, "key") == ""

    def test_escape_toml_string_function(self):
        from lilbee.settings import _escape_toml_string

        assert _escape_toml_string('say "hi"') == r"say \"hi\""
        assert _escape_toml_string(r"C:\path") == r"C:\\path"
        assert _escape_toml_string("a\nb") == r"a\nb"
        assert _escape_toml_string("a\tb") == r"a\tb"
        assert _escape_toml_string("normal") == "normal"
        assert _escape_toml_string("") == ""
