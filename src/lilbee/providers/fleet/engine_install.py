"""Install advice for a missing engine, in the form that fits how lilbee was installed."""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path

from lilbee._frozen import is_frozen


class InstallMethod(StrEnum):
    """How this lilbee was put on the machine."""

    UV_TOOL = "uv_tool"
    UV_ENV = "uv_env"
    PIPX = "pipx"
    PIP = "pip"
    HOMEBREW = "homebrew"
    STANDALONE = "standalone"
    UNKNOWN = "unknown"


# Hardware label and index name, one line each in the advice. One index per
# hardware because the builds are not interchangeable: cpu has every GPU backend
# off, metal ships only a macOS arm64 wheel, vulkan only Linux and Windows ones.
_INDEXES: tuple[tuple[str, str], ...] = (
    ("NVIDIA (CUDA)", "cu125"),
    ("AMD (ROCm)", "rocm"),
    ("Apple silicon", "metal"),
    ("Other GPUs", "vulkan"),
    ("No GPU", "cpu"),
)
_INDEX_URL = "https://lilbee.sh/{index}/"
_COMMANDS: dict[InstallMethod, str] = {
    InstallMethod.PIP: "pip install --pre 'lilbee[engine]' --extra-index-url {url}",
    InstallMethod.UV_ENV: (
        "uv pip install --prerelease=allow 'lilbee[engine]' --extra-index-url {url}"
    ),
    InstallMethod.UV_TOOL: (
        "uv tool install --force --prerelease=allow 'lilbee[engine]' --extra-index-url {url}"
    ),
    InstallMethod.PIPX: (
        "pipx install --force --pip-args='--pre --extra-index-url {url}' 'lilbee[engine]'"
    ),
}
_BUNDLED_FORMS = (
    "Install a self-contained build again; it carries its own engine: "
    "'brew reinstall lilbee' (Homebrew), 'npm install -g lilbee' (npm launcher), "
    "or the standalone binary from https://github.com/tobocop2/lilbee/releases/latest."
)
_BRING_YOUR_OWN = (
    "To bring your own engine, set LILBEE_LLAMA_SERVER_PATH to a llama-server binary "
    "and put llama-swap / gguf-parser on PATH."
)


def detect_install_method() -> InstallMethod:
    """Tell how lilbee was installed from the interpreter's own location."""
    if is_frozen():
        exe = Path(sys.executable).as_posix().lower()
        return (
            InstallMethod.HOMEBREW
            if ("/cellar/" in exe or "/homebrew/" in exe)
            else (InstallMethod.STANDALONE)
        )
    prefix = Path(sys.prefix).as_posix().lower()
    if "/uv/tools/" in prefix:
        return InstallMethod.UV_TOOL
    if "/pipx/venvs/" in prefix:
        return InstallMethod.PIPX
    if sys.prefix == sys.base_prefix:
        return InstallMethod.UNKNOWN
    try:
        lines = (Path(sys.prefix) / "pyvenv.cfg").read_text(encoding="utf-8").splitlines()
    except OSError:
        return InstallMethod.PIP
    made_by_uv = any(line.partition("=")[0].strip() == "uv" for line in lines)
    return InstallMethod.UV_ENV if made_by_uv else InstallMethod.PIP


def _index_lines(method: InstallMethod) -> list[str]:
    template = _COMMANDS[method]
    return [
        f"  {label + ':':<15}{template.format(url=_INDEX_URL.format(index=index))}"
        for label, index in _INDEXES
    ]


def engine_install_hint(method: InstallMethod | None = None) -> str:
    """The advice for installing the engine, for *method* or the detected install."""
    method = detect_install_method() if method is None else method
    if method in (InstallMethod.HOMEBREW, InstallMethod.STANDALONE):
        return f"{_BUNDLED_FORMS} {_BRING_YOUR_OWN}"
    intro = (
        "The engine is lilbee's 'engine' extra, published on lilbee.sh rather than "
        "PyPI, so the index is part of the command:"
    )
    if method is InstallMethod.UNKNOWN:
        blocks = [
            "With pip:",
            *_index_lines(InstallMethod.PIP),
            "With uv:",
            *_index_lines(InstallMethod.UV_TOOL),
        ]
    else:
        blocks = _index_lines(method)
    return "\n".join([intro, *blocks, _BUNDLED_FORMS, _BRING_YOUR_OWN])
