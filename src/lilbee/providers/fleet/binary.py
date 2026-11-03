"""Resolve the bundled engine binaries (llama-server, llama-swap, gguf-parser)."""

from __future__ import annotations

import functools
import hashlib
import shutil
import subprocess
from enum import StrEnum
from importlib.metadata import version as _pkg_version
from pathlib import Path

from lilbee.providers.base import ProviderError, ProviderErrorKind
from lilbee.providers.fleet.engine_install import engine_install_hint

# --help exits from arg parsing, before any backend or model load, so this is
# generous; a binary that cannot even print usage in this time is not one the
# fleet should silently trust either way.
_HELP_PROBE_TIMEOUT_S = 15.0


class EngineTool(StrEnum):
    """A bundled engine executable resolved from the ``lilbee-engine`` wheel."""

    LLAMA_SERVER = "llama-server"
    LLAMA_SWAP = "llama-swap"
    GGUF_PARSER = "gguf-parser"


_BUNDLED_ACCESSORS = {
    EngineTool.LLAMA_SERVER: "get_llama_server_path",
    EngineTool.LLAMA_SWAP: "get_llama_swap_path",
    EngineTool.GGUF_PARSER: "get_gguf_parser_path",
}


def _bundled_tool(tool: EngineTool) -> Path | None:
    """Path to *tool* from the ``lilbee-engine`` wheel, or ``None`` if absent."""
    try:
        import lilbee_engine
    except ImportError:
        return None
    accessor = getattr(lilbee_engine, _BUNDLED_ACCESSORS[tool], None)
    if accessor is None:  # a wheel that predates this tool lacks the accessor
        return None
    path = Path(accessor())
    return path if path.is_file() else None


def engine_pin() -> str:
    """Identity of the engine this lilbee would spawn; sharing keys on it.

    Three dimensions must match for two processes to share one engine: the engine
    BUILD (a configured ``LILBEE_LLAMA_SERVER_PATH`` is its own identity so a
    bring-your-own engine never silently shares with a bundled one), the
    load-affecting CONFIG baked into the launch argv (kv-cache type, expert
    offload, n-gpu-layers, ctx target, ...), and the revision of the llama-swap
    config the engine runs under. A process whose pin differs has
    ``contract_matches`` refuse the bind and overflows to its own engine rather
    than silently running on the incumbent's flags. Total: never raises, because
    it runs on every state write.
    """
    # circular: swap_config -> binary via readback.engine_help
    from lilbee.providers.fleet.swap_config import SWAP_CONFIG_REVISION

    return f"{engine_build_id()}|{_load_config_signature()}|swap-config={SWAP_CONFIG_REVISION}"


def engine_build_id() -> str:
    """The engine build's identity: configured path, wheel pin, PATH, or unpinned.

    A BYO (``custom:``) or PATH-resolved (``path:``) binary is identified by its
    location AND a cheap build fingerprint (size + mtime), so replacing the binary
    in place (a brew upgrade, a re-download) changes the pin and never binds a new
    process to an engine spawned from the old build. The bundled wheel needs no
    fingerprint: its pin already encodes the build.
    """
    from lilbee.core.config import cfg

    if cfg.llama_server_path:
        return f"custom:{cfg.llama_server_path}@{_binary_signature(Path(cfg.llama_server_path))}"
    try:
        import lilbee_engine
    except ImportError:
        lilbee_engine = None
    if lilbee_engine is not None:
        try:
            return str(lilbee_engine.get_engine_pin())
        except AttributeError:  # pre-pin wheels lack the accessor
            return f"wheel:{_engine_wheel_version()}"
    found = shutil.which(EngineTool.LLAMA_SERVER.value)
    if found is not None:
        return f"path:{found}@{_binary_signature(Path(found))}"
    return "unpinned"


def _engine_wheel_version() -> str:
    """The engine wheel's version, or a marker when it has no distribution metadata.

    ``lilbee_engine`` can be importable with nothing to look up: an extracted
    wheel on sys.path, a vendored copy, or a distribution registered under a name
    that does not normalize to ``lilbee-engine``. Since this feeds the pin, and
    the pin is computed on every state write, a missing version degrades to a
    marker rather than raising out of ``engine_pin``.
    """
    from importlib.metadata import PackageNotFoundError

    try:
        return _pkg_version("lilbee-engine")
    except PackageNotFoundError:
        return "unknown"


def _binary_signature(path: Path) -> str:
    """A cheap build fingerprint of the binary at *path*: size and mtime.

    An in-place replacement changes both, so the pin stops matching the old build.
    Best-effort and total (engine_pin runs on every state write): an unstatable
    path degrades to a fixed marker rather than raising.
    """
    try:
        st = path.stat()
    except OSError:
        return "unstatable"
    return f"{st.st_size}-{st.st_mtime_ns}"


def engine_binary_identity(binary: Path) -> str:
    """Identity of the engine file at *binary*: its location and a digest of its bytes.

    The file rather than the wheel, because a stub wheel can carry the version of
    the real one; only the bytes that answered a probe identify what answered it.
    The bytes rather than the stat fields ``engine_pin`` matches on: an upgrade
    rewrites the file in place, the rewrite keeps the inode, and a build of the
    same size written inside one mtime tick moves no stat field at all, so
    metadata can repeat across a real engine change. ``llama-server`` is a
    launcher of a few tens of kilobytes that loads the backend libraries at run
    time, and this is read once per planning pass, so the digest costs far less
    than the device probe it decides to re-run.
    """
    try:
        with binary.open("rb") as handle:
            return f"{binary}@{hashlib.file_digest(handle, 'sha256').hexdigest()}"
    except OSError:
        return f"{binary}@unreadable"


# Ctx sizing keys share by window coverage (contract.chat_ctx_covers), not
# value equality: a running window that covers the demand serves both peers.
# chat_n_ctx_target in particular defaults per process from its cgroup-capped
# RAM, so exact equality here restarted a warm engine per co-tenant.
_CTX_SIZING_KEYS = frozenset({"num_ctx", "num_ctx_max", "chat_n_ctx_target"})


def _load_config_signature() -> str:
    """A deterministic digest of the settings an engine bakes in at launch.

    These decide cross-process sharing, since an engine launched with one set
    cannot serve a peer that configured another: the ``LOAD_AFFECTING_KEYS`` a
    single process reloads on (minus the ctx sizing keys, matched by coverage
    instead), plus the placement keys that fix which devices a launch uses, so
    a peer with different placement binds its own engine.
    """
    from lilbee.core.config import cfg
    from lilbee.core.config.keys import LOAD_AFFECTING_KEYS, PLACEMENT_PIN_KEYS

    keys = (LOAD_AFFECTING_KEYS - _CTX_SIZING_KEYS) | PLACEMENT_PIN_KEYS
    return ";".join(f"{key}={getattr(cfg, key, None)}" for key in sorted(keys))


def resolve_engine_tool(tool: EngineTool) -> Path:
    """Resolve *tool*: configured llama-server path, then bundled wheel, then PATH.

    Never downloads anything; the binaries arrive via the bundled ``lilbee-engine``
    wheel or bring-your-own. Only llama-server honors ``LILBEE_LLAMA_SERVER_PATH``
    (an explicit setting beats the bundled wheel); the other tools resolve from the
    wheel, then ``PATH``.
    """
    if tool is EngineTool.LLAMA_SERVER:
        from lilbee.core.config import cfg

        if cfg.llama_server_path:
            configured = Path(cfg.llama_server_path)
            if not configured.is_file():
                raise ProviderError(f"LILBEE_LLAMA_SERVER_PATH is not a file: {configured}")
            return configured

    bundled = _bundled_tool(tool)
    if bundled is not None:
        return bundled

    found = shutil.which(tool.value)
    if found is not None:
        return Path(found)

    # Only llama-server carries NOT_FOUND: it marks the engine-less host that
    # legitimately serves nothing. A missing sibling tool (gguf-parser) must not
    # take that kind, or the sizing fallback would misreport it as a model that
    # isn't installed.
    kind = (
        ProviderErrorKind.NOT_FOUND
        if tool is EngineTool.LLAMA_SERVER
        else ProviderErrorKind.UNKNOWN
    )
    raise ProviderError(f"{tool.value} binary not found. {engine_install_hint()}", kind=kind)


def resolve_llama_server() -> Path:
    """Resolve the ``llama-server`` executable."""
    return resolve_engine_tool(EngineTool.LLAMA_SERVER)


def resolve_llama_swap() -> Path:
    """Resolve the ``llama-swap`` executable."""
    return resolve_engine_tool(EngineTool.LLAMA_SWAP)


def resolve_gguf_parser() -> Path:
    """Resolve the ``gguf-parser`` executable."""
    return resolve_engine_tool(EngineTool.GGUF_PARSER)


def llama_server_runtime_env() -> dict[str, str]:
    """Extra environment for a spawned ``llama-server``.

    The bundled wheel ships its own ggml/llama/mtmd next to the binary with a baked
    rpath (``@loader_path`` on macOS, ``$ORIGIN`` on Linux), but a CUDA build also
    links the CUDA 12 runtime, which driver-only GPU images omit. On Linux this
    adds any installed CUDA-runtime wheel libs to ``LD_LIBRARY_PATH``; elsewhere it
    is empty.
    """
    from lilbee.providers.fleet.cuda_runtime import cuda_runtime_env

    return cuda_runtime_env()


@functools.lru_cache(maxsize=8)
def engine_help(binary: Path) -> str:
    """The usage text *binary* prints for ``--help``, stdout then stderr, read once per path.

    Empty when the binary cannot run or answer in time. A gated flag then reads as
    unsupported; the optional flags are all written, since no help text proves
    them missing.
    """
    from lilbee.providers.fleet.gpu_env import engine_environ

    try:
        done = subprocess.run(  # noqa: S603 - argv is the resolved engine binary and a literal
            [str(binary), "--help"],
            env=engine_environ(),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_HELP_PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return f"{done.stdout}\n{done.stderr}"


# Option lines start at the margin or a few columns in; wrapped descriptions start further right.
_MAX_OPTION_INDENT = 12


def help_lists_option(help_text: str, flag: str) -> bool:
    """Whether an option line of *help_text* names *flag*.

    An option line starts with ``-`` within the first few columns and names its flags
    up to the first word that is not one, so a flag quoted in a description does not
    count. A flag written ``--name(s)`` names both ``--name`` and ``--names``.
    """
    for line in help_text.splitlines():
        text = line.lstrip()
        if not text.startswith("-") or len(line) - len(text) > _MAX_OPTION_INDENT:
            continue
        for token in text.split():
            if not token.startswith("-"):
                break
            name = token.rstrip(",")
            if flag in (name.replace("(s)", ""), name.replace("(s)", "s")):
                return True
    return False
