"""Ensure hermes has its optional ``mcp`` extra (HTTP MCP) before lilbee wires in over it."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# The module hermes imports to enable Streamable-HTTP MCP; its absence is what
# produces "MCP Servers (0) connected".
_HTTP_MCP_PROBE = "import mcp.client.streamable_http"
# hermes's own documented command (tools/mcp_tool.py); pins the mcp + starlette
# versions hermes expects. Shown when we don't (or can't) auto-install.
MCP_EXTRA_HINT = (
    "Enable lilbee's search in hermes by installing hermes's MCP extra:\n"
    "  pip install 'hermes-agent[mcp]'\n"
    "(or `uv tool install 'hermes-agent[mcp]'` / `pipx inject hermes-agent mcp`)."
)
# Reads hermes's pinned `[mcp]` extra requirements from its own metadata, so we
# install exactly what hermes expects rather than an unpinned `mcp`.
_EXTRA_REQS_SNIPPET = (
    "import importlib.metadata as m, json\n"
    "try: reqs = m.requires('hermes-agent') or []\n"
    "except Exception: reqs = []\n"
    "out = [r.split(';')[0].strip() for r in reqs\n"
    '       if len(r.split(";")) > 1 and "extra" in r.split(";")[1] and "mcp" in r.split(";")[1]]\n'
    "print(json.dumps(out))"
)


# Suffixes pip uses for Windows console-script wrappers; neither carries a shebang.
_WINDOWS_WRAPPER_SUFFIXES = (".cmd", ".exe")


def hermes_interpreter(binary: str) -> str | None:
    """The Python that runs hermes, read from its console-script shebang (or None).

    On Windows, ``.cmd``/``.exe`` wrappers have no ``#!python`` line.  We parse
    the embedded Python path from the ``.cmd`` (pip writes it as a comment), and
    fall back to ``sys.executable`` (shared-venv assumption) when that fails.
    """
    try:
        first_line = Path(binary).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return None
    if first_line.startswith("#!") and "python" in first_line:
        return first_line[2:].strip().split()[0] or None
    # Windows .cmd/.exe wrappers carry no shebang; detect by suffix + platform.
    if sys.platform == "win32" and Path(binary).suffix.lower() in _WINDOWS_WRAPPER_SUFFIXES:
        # pip's generated .cmd embeds the interpreter on the first line as
        # `@"<path>\python.exe" ...`; try to parse that before falling back.
        if first_line.startswith('@"') or first_line.startswith("@'"):
            candidate = first_line[2:].split('"')[0].split("'")[0]
            if "python" in candidate.lower():
                return candidate or None
        # Shared-venv fallback: the lilbee process and hermes share an environment
        # (e.g. both installed via `pip install` into the same venv), so the
        # running interpreter is the right one to use for hermes's packages too.
        return sys.executable
    return None


def has_http_mcp(interpreter: str) -> bool:
    """Whether ``interpreter`` can import hermes's Streamable-HTTP MCP client."""
    try:
        return (
            subprocess.run(  # noqa: S603 - hermes's own resolved interpreter, fixed argv
                [interpreter, "-c", _HTTP_MCP_PROBE],
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )
    except OSError:
        return False


def _mcp_extra_requirements(interpreter: str) -> list[str]:
    """hermes's pinned ``[mcp]`` extra requirements, or ``["mcp"]`` if unreadable."""
    try:
        result = subprocess.run(  # noqa: S603 - hermes's own resolved interpreter, fixed argv
            [interpreter, "-c", _EXTRA_REQS_SNIPPET],
            capture_output=True,
            text=True,
            check=False,
        )
        reqs = json.loads(result.stdout or "[]")
    except (OSError, json.JSONDecodeError):
        reqs = []
    return reqs or ["mcp"]


def ensure_hermes_http_mcp(
    binary: str, *, allow_lazy_installs: bool, echo: Callable[[str], None]
) -> bool:
    """Make sure hermes can speak HTTP MCP, returning whether it ends up supported.

    When support is missing: auto-installs hermes's pinned ``[mcp]`` extra into
    hermes's own environment (only if ``allow_lazy_installs``), otherwise echoes
    hermes's documented install command. Idempotent and cheap when already present."""
    interpreter = hermes_interpreter(binary)
    if interpreter is None:
        echo(MCP_EXTRA_HINT)
        return False
    if has_http_mcp(interpreter):
        return True
    if not allow_lazy_installs:
        # Respect the user's hermes security setting; don't pip-install behind it.
        echo(MCP_EXTRA_HINT)
        return False
    echo("Setting up hermes MCP support (installing hermes's mcp extra)...")
    try:
        subprocess.run(  # noqa: S603 - hermes's own resolved interpreter, fixed argv
            [interpreter, "-m", "pip", "install", *_mcp_extra_requirements(interpreter)],
            capture_output=True,
            check=False,
        )
    except OSError:
        echo(MCP_EXTRA_HINT)
        return False
    if has_http_mcp(interpreter):
        echo("hermes MCP support ready.")
        return True
    echo(MCP_EXTRA_HINT)
    return False
