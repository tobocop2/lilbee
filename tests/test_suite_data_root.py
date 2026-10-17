"""The suite's library, engine slot and cache are temporary, never the developer's."""

import os
import tempfile
from pathlib import Path

from huggingface_hub import constants as hf_constants

from lilbee.catalog.hf_client import hf_token
from lilbee.core import settings
from lilbee.core.config import Config
from lilbee.core.system import canonical_data_root, default_data_dir
from lilbee.providers import gguf_meta
from lilbee.runtime.engine_lock import machine_engine_dir
from tests._suite_data_root import SUITE_DATA_ROOT

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


def test_the_machine_engine_slot_is_temporary():
    _assert_temporary(machine_engine_dir())


def test_the_gguf_metadata_cache_is_temporary():
    # The key feeds the digest; the model file never exists.
    cache_file = gguf_meta._disk_cache_file(("/models/qwen.gguf", 0, 0))
    assert cache_file is not None
    assert cache_file.is_relative_to(Path(SUITE_DATA_ROOT.name)), cache_file


def test_the_agent_harnesses_fetch_is_switched_off():
    assert hf_constants.HF_HUB_OFFLINE is True


def test_a_saved_hugging_face_token_stays_readable(monkeypatch):
    home = Path(hf_constants.HF_HOME)
    assert not home.is_relative_to(Path(SUITE_DATA_ROOT.name)), home
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("LILBEE_HF_TOKEN", raising=False)
    token_file = Path(hf_constants.HF_TOKEN_PATH)
    saved = token_file.read_text(encoding="utf-8").strip() if token_file.exists() else None
    saved = saved or None
    assert hf_token() == saved


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
