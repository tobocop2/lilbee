"""A real pull of a tiny safetensors repo with no engine installed.

Moves about 0.5 MB from HuggingFace:

    uv run pytest tests/integration/test_pull_without_engine_e2e.py -v -m slow
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from lilbee.app import engine_choice
from lilbee.catalog.hf_client import HfClient
from lilbee.catalog.types import ModelSource
from lilbee.core.config import cfg
from lilbee.modelhub.model_manager import ModelManager
from lilbee.modelhub.registry import ModelRegistry
from lilbee.providers.fleet.engines.choice import NoEngineError

pytestmark = pytest.mark.slow

_REF = "hf-internal-testing/tiny-random-gpt2#safetensors"


@pytest.fixture
def models_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setattr(engine_choice, "installed_engines", lambda: {})
    original = cfg.models_dir
    cfg.models_dir = tmp_path / "models"
    cfg.models_dir.mkdir()
    yield cfg.models_dir
    cfg.models_dir = original


def test_the_download_needs_no_engine_and_the_load_names_the_fix(models_dir: Path) -> None:
    path = ModelManager(models_dir, cfg, hf_client=HfClient(cfg)).pull(_REF, ModelSource.NATIVE)

    assert path is not None
    assert path.name == "model.safetensors"
    assert path.stat().st_size > 0
    assert [m.ref for m in ModelRegistry(models_dir).list_installed()] == [_REF]
    with pytest.raises(NoEngineError, match="No installed engine loads the safetensors format"):
        engine_choice.engine_for_ref(_REF, cfg)
