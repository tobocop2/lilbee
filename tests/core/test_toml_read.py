"""A TOML document nested too deep reads as a malformed one on every path that reads it."""

import logging
import tomllib

import pytest
from tests.conftest import clean_env

from lilbee.core import profile_files, settings
from lilbee.core.config import Config
from lilbee.core.config.resolve import read_config_table, toml_is_readable
from lilbee.core.project_state import read_state
from lilbee.core.toml_read import load_toml, loads_toml

DEEP = "a = " + "[" * 3000 + "]" * 3000 + "\n"
MALFORMED = "top_k = = 9\n"


def test_a_deep_document_is_a_decode_error() -> None:
    with pytest.raises(tomllib.TOMLDecodeError, match="nested too deeply"):
        loads_toml(DEEP)


def test_a_malformed_document_is_the_same_error_class() -> None:
    with pytest.raises(tomllib.TOMLDecodeError):
        loads_toml(MALFORMED)


def test_a_valid_document_parses_from_text_and_from_a_file(tmp_path) -> None:
    path = tmp_path / "a.toml"
    path.write_text("top_k = 4\n", encoding="utf-8")
    with path.open("rb") as handle:
        assert load_toml(handle) == {"top_k": 4}
    assert loads_toml("top_k = 4\n") == {"top_k": 4}


def test_a_deep_config_toml_reads_as_a_malformed_one_on_every_path(tmp_path) -> None:
    deep = tmp_path / "deep"
    broken = tmp_path / "broken"
    for folder, text in ((deep, DEEP), (broken, MALFORMED)):
        folder.mkdir()
        (folder / "config.toml").write_text(text, encoding="utf-8")
    with pytest.raises(tomllib.TOMLDecodeError):
        settings.load(deep)
    with pytest.raises(tomllib.TOMLDecodeError):
        settings.load(broken)
    assert read_config_table(deep / "config.toml") is None
    assert read_config_table(broken / "config.toml") is None
    assert toml_is_readable(deep / "config.toml") is False
    assert toml_is_readable(broken / "config.toml") is False


def test_a_deep_config_toml_starts_with_the_defaults_and_a_warning(
    tmp_path, monkeypatch, caplog
) -> None:
    (tmp_path / "config.toml").write_text(DEEP, encoding="utf-8")
    for name, value in clean_env(tmp_path).items():
        monkeypatch.setenv(name, value)
    with caplog.at_level(logging.WARNING, logger="lilbee.core.config"):
        built = Config()
    assert built.top_k == Config.model_fields["top_k"].default
    assert any("Failed to read" in record.message for record in caplog.records)


def test_a_deep_state_toml_reads_as_empty(tmp_path) -> None:
    (tmp_path / "state.toml").write_text(DEEP, encoding="utf-8")
    state = read_state(tmp_path)
    assert (state.analyzed_at, state.tip_dismissed) == (None, False)


def test_a_deep_profile_text_is_a_profile_file_error() -> None:
    with pytest.raises(profile_files.ProfileFileError, match="Not valid TOML"):
        profile_files.parse_text(DEEP, "deep", profile_files.ProfileFolder.GLOBAL)
