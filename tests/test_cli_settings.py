"""The ``lilbee settings`` command group: list, get, set, and unset, as text and --json."""

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from lilbee.app import services as svc_mod
from lilbee.cli.app import app
from lilbee.core.config import cfg
from tests.conftest import make_mock_services

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.delenv("LILBEE_DATA", raising=False)
    snapshot = cfg.model_copy()
    svc_mod.set_services(make_mock_services())
    yield
    svc_mod.set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


@pytest.fixture()
def project(tmp_path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    return root


def _invoke(project: Path, *args: str, json_mode: bool = False):
    prefix = ["--json"] if json_mode else []
    return runner.invoke(app, [*prefix, "settings", *args, "--data-dir", str(project)])


def _json(project: Path, *args: str) -> dict:
    result = _invoke(project, *args, json_mode=True)
    return json.loads(result.output)


def _stored(project: Path) -> dict:
    import tomllib

    path = project / "config.toml"
    return tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# -- list -----------------------------------------------------------------


def test_list_shows_every_setting_in_the_group_with_value_and_source(project):
    result = _invoke(project, "list", "--group", "Retrieval")
    assert result.exit_code == 0, result.output
    assert "top_k" in result.output
    assert "built in" in result.output
    payload = _json(project, "list", "--group", "Retrieval")
    row = {entry["key"]: entry for entry in payload["settings"]}["top_k"]
    assert row == {
        "key": "top_k",
        "value": 12,
        "default": 12,
        "type": "int",
        "nullable": False,
        "group": "Retrieval",
        "help": "Number of chunks returned by search",
        "choices": None,
        "reindex_required": False,
        "source": "built_in",
        "advanced": False,
    }


def test_list_excludes_write_only_api_keys(project):
    keys = {entry["key"] for entry in _json(project, "list")["settings"]}
    assert "hf_token" not in keys
    assert "openai_api_key" not in keys


def test_list_reflects_a_user_value_and_its_source(project):
    (project / "config.toml").write_text("top_k = 3\n", encoding="utf-8")
    row = {e["key"]: e for e in _json(project, "list")["settings"]}["top_k"]
    assert row["value"] == 3
    assert row["source"] == "user"


def test_list_reflects_a_profile_value_and_its_source(project):
    (project / "config.toml").write_text(
        '[profile]\nname = "custom"\n\n[profile.values]\ntop_k = 77\n', encoding="utf-8"
    )
    row = {e["key"]: e for e in _json(project, "list")["settings"]}["top_k"]
    assert row["value"] == 77
    assert row["source"] == "profile"


def test_list_unknown_group_is_an_error(project):
    result = _invoke(project, "list", "--group", "bogus")
    assert result.exit_code == 1
    assert "Unknown setting group" in result.output
    payload = _json(project, "list", "--group", "bogus")
    assert "Unknown setting group" in payload["error"]


# -- get --------------------------------------------------------------------


def test_get_shows_value_source_and_help(project):
    result = _invoke(project, "get", "top_k")
    assert result.exit_code == 0, result.output
    assert "top_k = 12" in result.output
    assert "source: built in" in result.output
    assert "Number of chunks returned by search" in result.output
    assert _json(project, "get", "top_k")["source"] == "built_in"


def test_get_reflects_a_profile_value_and_its_source(project):
    (project / "config.toml").write_text(
        '[profile]\nname = "custom"\n\n[profile.values]\ntop_k = 77\n', encoding="utf-8"
    )
    result = _invoke(project, "get", "top_k")
    assert result.exit_code == 0, result.output
    assert "top_k = 77" in result.output
    assert "source: profile" in result.output
    assert _json(project, "get", "top_k")["source"] == "profile"


def test_get_unknown_key_is_an_error(project):
    result = _invoke(project, "get", "not-a-real-setting")
    assert result.exit_code == 1
    assert "Unknown or read-only setting" in result.output


def test_get_refuses_write_only_fields(project):
    result = _invoke(project, "get", "hf_token")
    assert result.exit_code == 1
    assert "write-only" in result.output


# -- set ----------------------------------------------------------------------


def test_set_persists_the_value_and_confirms_it(project):
    result = _invoke(project, "set", "top_k", "9")
    assert result.exit_code == 0, result.output
    assert "Set top_k to 9." in result.output
    assert _stored(project) == {"top_k": 9}
    assert cfg.top_k == 9


def test_set_json_reuses_the_config_update_shape(project):
    payload = _json(project, "set", "top_k", "9")
    assert payload == {"updated": ["top_k"], "reindex_required": False, "warnings": []}


def test_set_a_reindex_field_prints_the_rebuild_hint(project):
    result = _invoke(project, "set", "chunk_size", "900")
    assert result.exit_code == 0, result.output
    assert "Run lilbee rebuild so the index uses the new values." in result.output
    assert _json(project, "set", "chunk_size", "900")["reindex_required"] is True


def test_set_warns_when_it_leaves_ocr_off_with_a_vision_model(project):
    vision_model = "org/Test-Vision-GGUF/test-vision-Q4_K_M.gguf"
    (project / "config.toml").write_text(f'vision_model = "{vision_model}"\n', encoding="utf-8")
    result = _invoke(project, "set", "enable_ocr", "false")
    assert result.exit_code == 0, result.output
    assert vision_model in result.output
    (project / "config.toml").write_text(f'vision_model = "{vision_model}"\n', encoding="utf-8")
    warnings = _json(project, "set", "enable_ocr", "false")["warnings"]
    assert len(warnings) == 1 and "enable_ocr" in warnings[0]


def test_set_masks_a_secret_value_in_its_confirmation(project):
    result = _invoke(project, "set", "hf_token", "hf_super_secret_value")
    assert result.exit_code == 0, result.output
    assert "hf_super_secret_value" not in result.output
    assert "************" in result.output
    assert cfg.hf_token == "hf_super_secret_value"


def test_set_does_not_mask_a_non_secret_value(project):
    result = _invoke(project, "set", "top_k", "9")
    assert "9" in result.output
    assert "*" not in result.output


def test_set_a_setting_without_a_settings_map_entry_is_not_masked(project):
    """embed_batch_sequences is writable but absent from SETTINGS_MAP: the
    secret check must not crash on a key with no settings-map definition."""
    result = _invoke(project, "set", "embed_batch_sequences", "32")
    assert result.exit_code == 0, result.output
    assert "32" in result.output
    assert cfg.embed_batch_sequences == 32


def test_set_a_list_field_splits_the_value_on_newlines(project):
    result = _invoke(project, "set", "crawl_exclude_patterns", "a.*\nb.*")
    assert result.exit_code == 0, result.output
    assert cfg.crawl_exclude_patterns == ["a.*", "b.*"]
    payload = _json(project, "get", "crawl_exclude_patterns")
    assert payload["value"] == ["a.*", "b.*"]


def test_set_rejects_an_unknown_key(project):
    result = _invoke(project, "set", "not-a-real-setting", "1")
    assert result.exit_code == 1
    assert "Unknown or read-only setting" in result.output


def test_set_refuses_a_model_role_field(project):
    result = _invoke(project, "set", "chat_model", "some/Model-GGUF/model.gguf")
    assert result.exit_code == 1
    assert "dedicated model route" in result.output


def test_set_rejects_overlap_at_or_above_chunk_size(project):
    (project / "config.toml").write_text("chunk_size = 512\n", encoding="utf-8")
    result = _invoke(project, "set", "chunk_overlap", "1024")
    assert result.exit_code == 1
    assert "chunk_overlap" in result.output


def test_set_reports_a_failed_write_with_the_cause(project, monkeypatch):
    def refuse(_src, _dst):
        raise PermissionError(13, "The process cannot access the file")

    monkeypatch.setattr(os, "replace", refuse)
    result = _invoke(project, "set", "top_k", "9")
    assert result.exit_code == 1
    assert "Could not write config.toml" in result.output
    assert cfg.top_k != 9


# -- unset ----------------------------------------------------------------


def test_unset_reports_the_value_and_source_it_falls_back_to(project):
    (project / "config.toml").write_text("top_k = 3\n", encoding="utf-8")
    result = _invoke(project, "unset", "top_k")
    assert result.exit_code == 0, result.output
    assert "top_k = 12 (built in)" in result.output
    assert cfg.top_k == 12
    assert "top_k" not in _stored(project)


def test_unset_json_reuses_the_config_update_shape(project):
    (project / "config.toml").write_text("top_k = 3\n", encoding="utf-8")
    payload = _json(project, "unset", "top_k")
    assert payload == {"updated": ["top_k"], "reindex_required": False, "warnings": []}


def test_unset_several_keys_at_once(project):
    (project / "config.toml").write_text("top_k = 3\nmax_distance = 0.4\n", encoding="utf-8")
    result = _invoke(project, "unset", "top_k", "max_distance")
    assert result.exit_code == 0, result.output
    assert "top_k = 12 (built in)" in result.output
    assert "max_distance = 0.75 (built in)" in result.output


def test_unset_of_a_reindex_field_prints_the_rebuild_hint(project):
    (project / "config.toml").write_text("chunk_size = 900\n", encoding="utf-8")
    result = _invoke(project, "unset", "chunk_size")
    assert result.exit_code == 0, result.output
    assert "Run lilbee rebuild so the index uses the new values." in result.output


def test_unset_a_write_only_field_hides_its_new_value(project):
    (project / "config.toml").write_text('hf_token = "hf_secret"\n', encoding="utf-8")
    result = _invoke(project, "unset", "hf_token")
    assert result.exit_code == 0, result.output
    assert "hf_secret" not in result.output
    assert "write-only" in result.output
    assert cfg.hf_token == ""


def test_unset_refuses_a_field_with_no_reset_target(project):
    result = _invoke(project, "unset", "documents_dir")
    assert result.exit_code == 1
    assert "no default to reset to" in result.output


def test_unset_refuses_a_model_role_field(project):
    result = _invoke(project, "unset", "chat_model")
    assert result.exit_code == 1
    assert "dedicated model route" in result.output


# -- unknown key, across commands ------------------------------------------


def test_get_set_and_unset_print_the_same_unknown_key_message(project):
    expected = "Error: Unknown or read-only setting: not-a-real-setting\n"
    get_result = _invoke(project, "get", "not-a-real-setting")
    set_result = _invoke(project, "set", "not-a-real-setting", "1")
    unset_result = _invoke(project, "unset", "not-a-real-setting")
    assert get_result.output == expected
    assert set_result.output == expected
    assert unset_result.output == expected
