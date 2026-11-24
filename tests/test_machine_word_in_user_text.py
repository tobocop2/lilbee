"""User text names the machine lilbee runs on "this computer", never "this host"."""

from __future__ import annotations

import pytest

from lilbee.app import engine_service
from lilbee.app.presets import RESERVE_REMOTE_NOTE
from lilbee.catalog.types import ModelFormat
from lilbee.cli import preset as preset_cli
from lilbee.cli.tui import messages
from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import choose_engine
from lilbee.providers.fleet.engines.host_gate import PLATFORM_UNREAD
from lilbee.providers.fleet.host_caps import collect

RENDERED = {
    "reserve note": RESERVE_REMOTE_NOTE,
    "reserve help": preset_cli._reserve_help,
    "reserve placeholder": messages.PRESET_DIALOG_RESERVE_PLACEHOLDER,
    "unreadable platform": PLATFORM_UNREAD.reason,
}

IK = EngineName.IK_LLAMA
GGUF = ModelFormat.GGUF
NO_AVX2 = collect(
    system="Linux", machine="x86_64", cpu_info={"flags": ["sse4_2"], "brand_raw": "Xeon"}
)


@pytest.mark.parametrize("name", sorted(RENDERED))
def test_rendered_text_says_this_computer(name: str) -> None:
    text = RENDERED[name]
    assert "this host" not in text
    assert "this computer" in text


def test_the_install_refusal_says_this_computer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(engine_service, "refresh_session_host", lambda: NO_AVX2)
    with pytest.raises(engine_service.EngineRefusedError) as exc:
        engine_service.install(IK.value)
    assert "this host" not in str(exc.value)
    assert "cannot run on this computer." in str(exc.value)


def test_the_override_refusal_says_this_computer() -> None:
    got = choose_engine(
        fmt=GGUF, host=lambda: NO_AVX2, installed={IK: {GGUF}}, preferred="auto", override=IK.value
    )
    assert got.notice is not None
    assert "this host" not in got.notice
    assert "cannot run on this computer." in got.notice
    assert "can run on this computer." in got.notice


def test_the_default_refusal_says_this_computer() -> None:
    got = choose_engine(
        fmt=GGUF, host=lambda: NO_AVX2, installed={IK: {GGUF}}, preferred="auto", override=None
    )
    assert got.notice is not None
    assert "this host" not in got.notice
    assert "but cannot run on this computer." in got.notice
