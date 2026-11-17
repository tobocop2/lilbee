"""The embed and vision architecture readers refuse a non-GGUF path as the chat reader does."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from lilbee.cli.tui.screens.status import _build_arch_content
from lilbee.core.config import cfg
from lilbee.modelhub import model_info
from lilbee.providers import engine_params, gguf_meta

_GGUF_HEADER = b"GGUF\x03\x00\x00\x00"


def _directory(root: Path) -> Path:
    path = root / "dir"
    path.mkdir()
    return path


def _missing(root: Path) -> Path:
    return root / "missing.gguf"


def _text_file(root: Path) -> Path:
    path = root / "model.gguf"
    path.write_text("this is not a model\n", encoding="utf-8")
    return path


def _other_magic(root: Path) -> Path:
    path = root / "model.gguf"
    path.write_bytes(b"GGML" + bytes(32))
    return path


NOT_GGUF: dict[str, Callable[[Path], Path]] = {
    "directory": _directory,
    "missing path": _missing,
    "text file": _text_file,
    "other magic": _other_magic,
}


@pytest.fixture
def parse_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record each header read; the parser answers nothing, as it does for a cut file."""
    calls: list[Path] = []

    def read_header(path: Path) -> None:
        calls.append(path)

    monkeypatch.setattr(gguf_meta, "read_gguf_metadata", read_header)
    monkeypatch.setattr(gguf_meta, "read_mmproj_projector_type", lambda path: read_header(path))
    return calls


def _serve(monkeypatch: pytest.MonkeyPatch, role: str, path: Path) -> None:
    configured = f"org/{role}-GGUF/{role}.gguf"

    def resolve(ref: str, registry: object = None) -> Path:
        assert ref == configured
        return path

    monkeypatch.setattr(engine_params, "resolve_model_path", resolve)
    monkeypatch.setattr(cfg, "chat_model", "")
    monkeypatch.setattr(cfg, "embedding_model", configured if role == "embed" else "")
    monkeypatch.setattr(cfg, "vision_model", configured if role == "vision" else "")


def _projector_beside(path: Path) -> None:
    """Give a vision model a projector file to find, so only the magic check can stop the read."""
    if path.parent.is_dir():
        (path.parent / "mmproj-x.gguf").write_bytes(_GGUF_HEADER)


def _shown(role: str) -> str:
    """The status screen line that the reader of *role* fills."""
    read = model_info._read_embed_arch if role == "embed" else model_info._read_vision_arch
    label = "Embed arch" if role == "embed" else "Vision proj"
    info = read(model_info.ModelArchInfo())
    lines = str(_build_arch_content(info)).splitlines()
    (line,) = [ln for ln in lines if ln.startswith(label)]
    return line


@pytest.mark.parametrize("role", ["embed", "vision"])
@pytest.mark.parametrize("kind", NOT_GGUF)
def test_a_wrong_path_shows_unknown_and_no_header_is_read(
    role: str,
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parse_calls: list[Path],
) -> None:
    path = NOT_GGUF[kind](tmp_path)
    _projector_beside(path)
    _serve(monkeypatch, role, path)
    assert _shown(role).split()[-1] == "unknown"
    assert parse_calls == []


@pytest.mark.parametrize("role", ["embed", "vision"])
def test_a_gguf_with_a_cut_header_reaches_the_parser_and_shows_unknown(
    role: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parse_calls: list[Path],
) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(_GGUF_HEADER)
    _projector_beside(path)
    _serve(monkeypatch, role, path)
    assert _shown(role).split()[-1] == "unknown"
    assert parse_calls == [path if role == "embed" else path.parent / "mmproj-x.gguf"]


def test_a_gguf_embed_model_reports_its_header_architecture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "embed.gguf"
    path.write_bytes(_GGUF_HEADER)
    monkeypatch.setattr(gguf_meta, "read_gguf_metadata", lambda p: {"architecture": "bert"})
    _serve(monkeypatch, "embed", path)
    assert _shown("embed").split()[-1] == "bert"


def test_a_gguf_embed_header_without_an_architecture_shows_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "embed.gguf"
    path.write_bytes(_GGUF_HEADER)
    monkeypatch.setattr(gguf_meta, "read_gguf_metadata", lambda p: {"context_length": "4096"})
    _serve(monkeypatch, "embed", path)
    assert _shown("embed").split() == ["Embed", "arch", "unknown"]


def test_a_gguf_vision_model_reports_its_projector_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "vision.gguf"
    path.write_bytes(_GGUF_HEADER)
    _projector_beside(path)
    monkeypatch.setattr(gguf_meta, "read_mmproj_projector_type", lambda p: "resampler")
    _serve(monkeypatch, "vision", path)
    assert _shown("vision").split()[-1] == "resampler"
