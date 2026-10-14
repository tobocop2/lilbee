"""The library the suite falls back to is a temporary folder, never the developer's own."""

import os
import tempfile
from pathlib import Path

from lilbee.core import settings
from lilbee.core.config import Config
from lilbee.core.system import canonical_data_root, default_data_dir

_PROBE_KEY = "suite_data_root_probe"


def _real_root() -> Path:
    return canonical_data_root(default_data_dir())


def _assert_temporary(root: Path) -> None:
    root = canonical_data_root(root)
    assert root.is_relative_to(canonical_data_root(tempfile.gettempdir())), root
    assert not root.is_relative_to(_real_root()), root


def test_the_library_the_environment_names_is_temporary():
    _assert_temporary(Path(os.environ["LILBEE_DATA"]))


def test_a_config_a_test_builds_uses_a_temporary_library():
    _assert_temporary(Config().data_root)


def test_a_setting_saved_to_the_default_library_leaves_the_real_config_alone():
    real_config = _real_root() / "config.toml"
    before = real_config.stat().st_mtime_ns if real_config.exists() else None
    restored = Config().data_root
    # Checked before the save, so a run that fails here never writes the real file.
    _assert_temporary(restored)

    settings.set_value(restored, _PROBE_KEY, "saved")
    try:
        assert settings.get(restored, _PROBE_KEY) == "saved"
    finally:
        settings.delete_value(restored, _PROBE_KEY)

    assert (real_config.stat().st_mtime_ns if real_config.exists() else None) == before
