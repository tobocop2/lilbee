"""User text names the machine lilbee runs on "this computer", never "this host"."""

from __future__ import annotations

import ast
import re
from pathlib import Path

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
    system="Linux",
    machine="x86_64",
    cpu_info={"flags": ["sse4_2"], "brand_raw": "Xeon"},
    nvidia=None,
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


ROOT = Path(__file__).resolve().parents[1]
THIS_HOST = re.compile(r"this host", re.IGNORECASE)
LOGGER_NAMES = frozenset({"log", "logger", "logging"})
# Network sense: the login of a crawl goes to one web host, not to the user's machine.
NETWORK_SENSE = {("src/lilbee/cli/tui/messages.py", "sent to this host only")}


def _is_logger_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in LOGGER_NAMES
    )


def _user_strings_with_this_host(path: Path) -> list[tuple[int, str]]:
    """String literals with "this host" that no logger call takes and no docstring holds."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if not THIS_HOST.search(node.value) or isinstance(parents.get(node), ast.Expr):
            continue
        chain = []
        up = parents.get(node)
        while up is not None:
            chain.append(up)
            up = parents.get(up)
        if not any(_is_logger_call(ancestor) for ancestor in chain):
            found.append((node.lineno, node.value))
    return found


def test_no_user_string_in_src_says_this_host() -> None:
    offenders = []
    for path in sorted((ROOT / "src" / "lilbee").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        for line, text in _user_strings_with_this_host(path):
            if not any(rel == site and part in text for site, part in NETWORK_SENSE):
                offenders.append(f"{rel}:{line}: {text[:60]!r}")
    assert offenders == []


def test_no_doc_says_this_host() -> None:
    offenders = [
        f"{path.relative_to(ROOT).as_posix()}:{number}"
        for path in sorted((ROOT / "docs").rglob("*.md"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if THIS_HOST.search(line)
    ]
    assert offenders == []
