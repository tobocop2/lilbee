"""A config.toml write after the CLI loads the settings and before a follower starts is taken."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from litestar.testing import TestClient

import lilbee
from lilbee.app import settings as app_settings
from lilbee.app.services import set_services
from lilbee.cli.app import apply_overrides
from lilbee.core import settings as persistent_settings
from lilbee.core.config import CONFIG_FILE_NAME, cfg
from tests.conftest import make_mock_services

_MTIME_NS = 1_700_000_000_000_000_000


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    snapshot = cfg.model_copy()
    root = tmp_path / "project"
    root.mkdir()
    cfg.data_root = root
    cfg.data_dir = root / "data"
    cfg.documents_dir = root / "documents"
    cfg.lancedb_dir = cfg.data_dir / "lancedb"
    monkeypatch.setenv("LILBEE_DATA", str(root))
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    set_services(make_mock_services())
    yield root
    set_services(None)
    for name in type(cfg).model_fields:
        setattr(cfg, name, getattr(snapshot, name))


def _write_from_another_process(key: str, value: Any, tick: int) -> None:
    """Write config.toml the way the CLI does, with an mtime no earlier write had."""
    persistent_settings.set_value(cfg.data_root, key, value)
    stamp = _MTIME_NS + tick * 1_000_000_000
    os.utime(cfg.data_root / CONFIG_FILE_NAME, ns=(stamp, stamp))


def _start_the_cli(**flags: Any) -> None:
    """Load the settings the way a CLI command does at startup, then write top_k 21 in the gap."""
    _write_from_another_process("top_k", 12, tick=1)
    apply_overrides(**flags)
    assert cfg.top_k == 12
    _write_from_another_process("top_k", 21, tick=2)


def _serve_client() -> TestClient:
    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    auth_mod.session_manager.disable()
    return TestClient(create_app())


def test_serve_takes_a_write_made_before_its_app_was_built():
    _start_the_cli()
    client = _serve_client()
    assert client.get("/api/config").json()["top_k"] == 21
    assert client.get("/api/config").json()["top_k"] == 21


def test_serve_takes_a_write_made_before_its_app_was_built_with_data_dir(isolated_env):
    _start_the_cli(data_dir=isolated_env)
    client = _serve_client()
    assert client.get("/api/config").json()["top_k"] == 21


async def test_the_stdio_mcp_takes_a_write_made_before_it_started():
    from mcp.types import CallToolResult

    from lilbee.mcp_server import build_mcp_server

    _start_the_cli()
    server = build_mcp_server()
    server.follow_config_file()
    result = await server.call_tool("settings_get", {"key": "top_k"})
    assert isinstance(result, CallToolResult), result
    assert result.structured_content is not None, result.content
    assert result.structured_content["setting"]["value"] == 21


def test_the_tui_takes_a_write_made_before_it_started():
    from lilbee.cli.tui.app import LilbeeApp

    _start_the_cli()
    app = LilbeeApp()
    assert app.adopt_external_settings() is True
    assert cfg.top_k == 21


def test_the_tui_engine_retry_takes_the_file_and_shows_it_on_the_app_thread(monkeypatch):
    from lilbee.app import services as svc_mod
    from lilbee.cli.tui import app as tui_app_mod
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    assert svc_mod._state.config_refresh == app.take_external_settings_now
    shown: list[tuple[Any, ...]] = []
    monkeypatch.setattr(tui_app_mod, "call_from_thread", lambda *args: shown.append(args))
    svc_mod.refresh_config()
    assert shown == []  # nothing changed, nothing to show
    _write_from_another_process("top_k", 23, tick=3)
    svc_mod.refresh_config()
    assert cfg.top_k == 23
    assert len(shown) == 1
    node, show, result = shown[0]
    assert node is app
    assert show == app._show_taken_settings
    assert result.updated == ["top_k"]
    assert app.adopt_external_settings() is False  # the retry's take moved the reading


# A fresh interpreter starts the CLI the way `lilbee` does on the default path: the import
# loads config.toml (top_k 12) from the platform data dir, the callback applies the
# overrides, and an outside write of top_k 21 lands before the product reads the file.
_DEFAULT_PATH_PROBE = """
import asyncio, os, sys
from pathlib import Path

mode, repo, home = sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve()
sys.path.insert(0, repo)
from lilbee.cli.app import apply_overrides
from lilbee.core import settings as persistent_settings
from lilbee.core.config import CONFIG_FILE_NAME, cfg

if not cfg.data_root.resolve().is_relative_to(home):
    sys.exit(f"not a tmp data root: {cfg.data_root}")
apply_overrides()
assert cfg.top_k == 12, cfg.top_k
persistent_settings.set_value(cfg.data_root, "top_k", 21)
stamp = int(sys.argv[4])
os.utime(cfg.data_root / CONFIG_FILE_NAME, ns=(stamp, stamp))

if mode == "serve":
    from litestar.testing import TestClient

    import lilbee.server.auth as auth_mod
    from lilbee.server.app import create_app

    auth_mod.session_manager.disable()
    client = TestClient(create_app())
    print(client.get("/api/config").json()["top_k"])
elif mode == "mcp":
    from lilbee.mcp_server import build_mcp_server

    server = build_mcp_server()
    server.follow_config_file()
    result = asyncio.run(server.call_tool("settings_get", {"key": "top_k"}))
    print(result.structured_content["setting"]["value"])
else:
    from lilbee.cli.tui.app import LilbeeApp

    app = LilbeeApp()
    app.adopt_external_settings()
    print(cfg.top_k)
"""

_PROBE_BUDGET_S = 120.0


@pytest.mark.parametrize("mode", ["serve", "mcp", "tui"])
@pytest.mark.parametrize("data_env", [None, ""], ids=["unset", "blank"])
def test_the_default_path_takes_a_write_made_before_the_product_started(
    tmp_path, monkeypatch, mode, data_env
):
    from tests.conftest import clean_env

    home = tmp_path / "home"
    home.mkdir()
    platform_dirs = {
        "HOME": str(home),
        "USERPROFILE": str(home),
        "XDG_DATA_HOME": str(home / "share"),
        "LOCALAPPDATA": str(home / "local"),
    }
    for name, value in platform_dirs.items():
        monkeypatch.setenv(name, value)
    from lilbee.core.system import default_data_dir

    root = default_data_dir()
    assert root.is_relative_to(home)
    root.mkdir(parents=True)
    persistent_settings.set_value(root, "top_k", 12)
    first = _MTIME_NS + 1_000_000_000
    os.utime(root / CONFIG_FILE_NAME, ns=(first, first))

    env = {
        **clean_env(),
        **platform_dirs,
        "HF_HUB_OFFLINE": "1",
        "LILBEE_ENGINE_DIR": str(home / "engine-slot"),
        # The child runs in another directory; it must import the lilbee under test.
        "PYTHONPATH": str(Path(lilbee.__file__).resolve().parents[1]),
    }
    if data_env is not None:
        env["LILBEE_DATA"] = data_env
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", _DEFAULT_PATH_PROBE, mode, str(repo), str(home), str(first * 2)],
        env=env,
        cwd=home,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        timeout=_PROBE_BUDGET_S,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "21"


@pytest.mark.parametrize(
    ("given", "expected"),
    [(None, "root"), ("", "root"), ("  ", "root"), ("/kept", "/kept")],
    ids=["unset", "blank", "spaces", "user-set"],
)
def test_the_import_names_the_root_unless_the_user_named_one(
    isolated_env, monkeypatch, given, expected
):
    from lilbee.core.config.model import export_data_root

    if given is None:
        monkeypatch.delenv("LILBEE_DATA")
    else:
        monkeypatch.setenv("LILBEE_DATA", given)
    export_data_root()
    assert os.environ["LILBEE_DATA"] == (str(isolated_env) if expected == "root" else expected)


def test_a_one_off_flag_stays_after_a_later_write_to_another_key():
    _write_from_another_process("temperature", 0.3, tick=1)
    apply_overrides(temperature=0.9)
    client = _serve_client()
    assert client.get("/api/config").json()["temperature"] == 0.9
    _write_from_another_process("top_k", 21, tick=2)
    body = client.get("/api/config").json()
    assert body["top_k"] == 21
    assert body["temperature"] == 0.9


def test_a_follower_on_another_root_reads_that_root_afresh(tmp_path):
    _write_from_another_process("top_k", 12, tick=1)
    apply_overrides()
    other = tmp_path / "other"
    other.mkdir()
    cfg.data_root = other
    _write_from_another_process("top_k", 33, tick=1)
    reading = app_settings.start_config_reading()
    assert reading.values["top_k"] == 33


def test_a_settle_that_raises_after_setting_keys_still_drops_their_caches(monkeypatch):
    from lilbee.catalog import picks

    reading = app_settings.read_config_state()
    resets: list[bool] = []
    monkeypatch.setattr(picks, "reset_picks", lambda: resets.append(True))

    def rederive_fails(keys: object) -> None:
        raise RuntimeError("rederive failed")

    monkeypatch.setattr(app_settings, "_rederive_from_resolved", rederive_fails)
    flipped = not cfg.include_uncensored
    _write_from_another_process("include_uncensored", flipped, tick=1)
    with pytest.raises(RuntimeError, match="rederive failed"):
        app_settings.adopt_external_changes(reading)
    assert cfg.include_uncensored is flipped
    assert resets == [True]
