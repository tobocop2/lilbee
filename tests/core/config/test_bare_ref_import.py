"""A model ref like <org>/<repo> must not stop the config from loading in a fresh process."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REF = "myorg/model"
FIELDS = ["chat_model", "embedding_model", "reranker_model", "vision_model"]
LOAD = "from lilbee.core.config import cfg; print(getattr(cfg, {field!r}))"


def _run(
    field: str, data_root: Path, env_extra: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("LILBEE_")}
    env["LILBEE_DATA"] = str(data_root)
    env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-c", LOAD.format(field=field)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        stdin=subprocess.DEVNULL,
        check=False,
    )


@pytest.mark.parametrize("field", FIELDS)
def test_a_bare_org_repo_ref_in_config_toml_loads(field: str, tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(f'{field} = "{REF}"\n', encoding="utf-8")
    result = _run(field, tmp_path, {})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == REF


@pytest.mark.parametrize("field", FIELDS)
def test_a_bare_org_repo_ref_in_the_environment_loads(field: str, tmp_path: Path) -> None:
    result = _run(field, tmp_path, {f"LILBEE_{field.upper()}": REF})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == REF
