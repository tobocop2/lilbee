"""Every config.toml example in docs/usage.md must set what it says it sets."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from lilbee.core.config.enums import SettingSource
from lilbee.core.config.resolve import PROFILE_TABLE, read_layers, resolve

USAGE_GUIDE = Path(__file__).resolve().parent.parent / "docs" / "usage.md"
GUIDE_TEXT = USAGE_GUIDE.read_text(encoding="utf-8")
FENCES = re.findall(r"```toml\n(.*?)```", GUIDE_TEXT, re.DOTALL)


def test_the_guide_has_config_toml_examples():
    assert len(FENCES) >= 4  # profile file, config file, semantic chunking, ocr_language


@pytest.mark.parametrize("example", FENCES, ids=[f"fence-{i}" for i in range(len(FENCES))])
def test_every_value_in_an_example_reaches_the_resolver(example, tmp_path):
    (tmp_path / "config.toml").write_text(example, encoding="utf-8")
    expected = tomllib.loads(example)
    layers = read_layers(tmp_path)
    for key, value in expected.items():
        if key == PROFILE_TABLE:
            continue  # the one table config.toml holds; the profile commands write it
        assert not isinstance(value, dict), f"[{key}] is a table; the resolver reads top-level keys"
        resolved = resolve(key, layers)
        assert resolved.source is SettingSource.USER, f"{key} does not reach the resolver"
        assert resolved.value == value


def test_prose_names_no_setting_as_a_dotted_config_key():
    assert not re.findall(r"`cfg\.\w+ = ", GUIDE_TEXT)
