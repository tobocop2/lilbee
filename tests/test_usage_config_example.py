"""The config.toml example in docs/usage.md must set what it says it sets."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from lilbee.core.config.enums import SettingSource
from lilbee.core.config.resolve import read_layers, resolve

USAGE_GUIDE = Path(__file__).resolve().parent.parent / "docs" / "usage.md"


def _config_example() -> str:
    guide = USAGE_GUIDE.read_text(encoding="utf-8")
    section = guide.split("\n## Config file\n", 1)[1].split("\n## ", 1)[0]
    match = re.search(r"```toml\n(.*?)```", section, re.DOTALL)
    assert match, "the Config file section has no toml example"
    return match.group(1)


def test_every_value_in_the_example_reaches_the_resolver(tmp_path):
    example = _config_example()
    (tmp_path / "config.toml").write_text(example, encoding="utf-8")
    expected = tomllib.loads(example)
    assert expected, "the example sets nothing"
    layers = read_layers(tmp_path)
    for key, value in expected.items():
        assert not isinstance(value, dict), f"[{key}] is a table; the resolver reads top-level keys"
        resolved = resolve(key, layers)
        assert resolved.source is SettingSource.USER, f"{key} does not reach the resolver"
        assert resolved.value == value
