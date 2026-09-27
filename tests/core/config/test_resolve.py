"""The settings resolver: one value and one source per field."""

from pathlib import Path

from lilbee.core.config import Config
from lilbee.core.config.enums import SettingSource
from lilbee.core.config.resolve import (
    SettingLayers,
    builtin_value,
    read_layers,
    resolve,
    resolve_all,
)


def _write(root: Path, text: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.toml").write_text(text, encoding="utf-8")
    return root


def test_env_beats_user_and_user_value_loses(tmp_path, monkeypatch):
    root = _write(tmp_path / "root", "top_k = 7\nchunk_size = 900\n")
    monkeypatch.setenv("LILBEE_TOP_K", "9")
    layers = read_layers(root)
    assert resolve("top_k", layers).value == "9"
    assert resolve("top_k", layers).source is SettingSource.ENV
    # The losing layer still holds its own value; the other user key is untouched by env.
    assert layers.user["top_k"] == 7
    assert resolve("chunk_size", layers).value == 900
    assert resolve("chunk_size", layers).source is SettingSource.USER


def test_user_beats_profile_table_value_which_loses(tmp_path):
    root = _write(
        tmp_path / "root",
        'top_k = 7\n[profile]\nname = "x"\n[profile.values]\ntop_k = 3\nchunk_overlap = 50\n',
    )
    layers = read_layers(root)
    assert resolve("top_k", layers).value == 7
    assert resolve("top_k", layers).source is SettingSource.USER
    assert resolve("chunk_overlap", layers).value == 50
    assert resolve("chunk_overlap", layers).source is SettingSource.PROFILE


def test_profile_table_beats_built_in(tmp_path):
    root = _write(tmp_path / "root", "[profile.values]\ntemperature = 0.7\n")
    layers = read_layers(root)
    assert resolve("temperature", layers).value == 0.7
    assert resolve("temperature", layers).source is SettingSource.PROFILE
    assert resolve("top_p", layers).source is SettingSource.BUILT_IN
    assert "profile" not in layers.user


def test_missing_profile_table_falls_to_built_in(tmp_path):
    root = _write(tmp_path / "root", "top_k = 7\n")
    layers = read_layers(root)
    assert layers.profile == {}
    assert resolve("temperature", layers).value == builtin_value("temperature") == 0.1
    assert resolve("temperature", layers).source is SettingSource.BUILT_IN
    assert resolve("top_k", layers).source is SettingSource.USER


def test_profile_table_that_is_not_a_table_contributes_nothing(tmp_path):
    root = _write(tmp_path / "root", 'profile = "fast"\ntop_k = 7\n')
    layers = read_layers(root)
    assert layers.profile == {}
    assert "profile" not in layers.user
    assert resolve("top_k", layers).source is SettingSource.USER


def test_profile_values_that_are_not_a_table_contribute_nothing(tmp_path):
    root = _write(tmp_path / "root", '[profile]\nname = "x"\nvalues = 3\n')
    assert read_layers(root).profile == {}


def test_empty_env_var_is_not_a_source(tmp_path, monkeypatch):
    root = _write(tmp_path / "root", "top_k = 7\n")
    monkeypatch.setenv("LILBEE_TOP_K", "")
    monkeypatch.setenv("LILBEE_CHUNK_SIZE", "900")
    layers = read_layers(root)
    assert "top_k" not in layers.env
    assert resolve("top_k", layers).source is SettingSource.USER
    assert resolve("chunk_size", layers).source is SettingSource.ENV


def test_empty_string_in_config_toml_is_not_a_source(tmp_path):
    root = _write(tmp_path / "root", 'chat_model = ""\n[profile.values]\nseed = ""\ntop_k = 4\n')
    layers = read_layers(root)
    assert resolve("chat_model", layers).source is SettingSource.BUILT_IN
    assert resolve("seed", layers).source is SettingSource.BUILT_IN
    assert resolve("top_k", layers).source is SettingSource.PROFILE


def test_derived_field_at_default_is_auto_and_set_is_user(tmp_path):
    layers = read_layers(_write(tmp_path / "unset", "top_k = 7\n"))
    assert resolve("num_ctx", layers).source is SettingSource.AUTO
    assert resolve("ingest_workers", layers).source is SettingSource.AUTO
    assert resolve("temperature", layers).source is SettingSource.BUILT_IN
    set_layers = read_layers(_write(tmp_path / "set", "num_ctx = 8192\n"))
    assert resolve("num_ctx", set_layers).value == 8192
    assert resolve("num_ctx", set_layers).source is SettingSource.USER


def test_skip_toml_drops_user_and_profile_layers(tmp_path, monkeypatch):
    root = _write(tmp_path / "root", "top_k = 7\n[profile.values]\nchunk_overlap = 50\n")
    monkeypatch.setenv("LILBEE_CHUNK_SIZE", "900")
    monkeypatch.setenv("LILBEE_SKIP_TOML_CONFIG", "1")
    layers = read_layers(root)
    assert layers.user == {}
    assert layers.profile == {}
    assert resolve("chunk_size", layers).source is SettingSource.ENV


def test_unreadable_config_toml_drops_user_and_profile_layers(tmp_path, caplog):
    root = _write(tmp_path / "root", "top_k = [\n")
    layers = read_layers(root)
    assert layers.user == {}
    assert layers.profile == {}
    assert any("config.toml" in record.getMessage() for record in caplog.records)


def test_missing_config_toml_gives_empty_file_layers(tmp_path):
    layers = read_layers(tmp_path / "absent")
    assert layers.user == {}
    assert layers.profile == {}


def test_every_field_resolves_to_one_source(tmp_path, monkeypatch):
    root = _write(tmp_path / "root", "top_k = 7\n[profile.values]\nchunk_overlap = 50\n")
    monkeypatch.setenv("LILBEE_CHUNK_SIZE", "900")
    resolved = resolve_all(read_layers(root))
    assert resolved
    assert set(resolved) == set(Config.model_fields)
    assert all(isinstance(entry.source, SettingSource) for entry in resolved.values())
    seen = {entry.source for entry in resolved.values()}
    assert seen == set(SettingSource)


def test_builtin_value_calls_the_default_factory():
    assert builtin_value("linked_roots") == {}
    assert builtin_value("linked_roots") is not builtin_value("linked_roots")


def test_resolve_reads_only_the_layers_it_is_given():
    layers = SettingLayers(env={}, user={"top_k": 7}, profile={"top_k": 3})
    assert resolve("top_k", layers).value == 7
    assert resolve("top_k", SettingLayers(env={}, user={}, profile={"top_k": 3})).value == 3


def test_blank_env_var_is_not_a_source(tmp_path, monkeypatch):
    root = _write(tmp_path / "root", "top_k = 7\n")
    monkeypatch.setenv("LILBEE_TOP_K", " \t ")
    layers = read_layers(root)
    assert "top_k" not in layers.env
    assert resolve("top_k", layers).source is SettingSource.USER


def test_blank_env_var_clears_a_model_role_that_can_be_off(tmp_path, monkeypatch):
    root = _write(tmp_path / "root", 'vision_model = "org/Toml-GGUF/toml-Q4_K_M.gguf"\n')
    monkeypatch.setenv("LILBEE_VISION_MODEL", " \t ")
    resolved = resolve("vision_model", read_layers(root))
    assert resolved.source is SettingSource.ENV
    assert resolved.value.strip() == ""


def test_host_computed_default_is_auto(tmp_path):
    layers = read_layers(tmp_path / "absent")
    assert resolve("chat_n_ctx_target", layers).source is SettingSource.AUTO


def test_fixed_default_is_built_in(tmp_path):
    layers = read_layers(tmp_path / "absent")
    assert resolve("main_gpu", layers).source is SettingSource.BUILT_IN
