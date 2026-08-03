"""Resolve the bundled engine binaries (llama-server, llama-swap, gguf-parser)."""

from __future__ import annotations

import shutil
from enum import StrEnum
from importlib.metadata import version as _pkg_version
from pathlib import Path

from lilbee.providers.base import ProviderError, ProviderErrorKind

# Names the extra rather than a command. The engine is published per backend and
# the right index depends on the machine, so a single one baked in here goes
# stale and points somebody at the wrong build; the README's install table is
# where that choice belongs.
_INSTALL_HINT = (
    "The bundled engine ships as the 'engine' extra: reinstall lilbee with it "
    "using the index for your hardware (see the install table in the README), or "
    "set LILBEE_LLAMA_SERVER_PATH to a llama-server binary (a llama.cpp release "
    "download or `brew install llama.cpp`) and put llama-swap / gguf-parser on PATH."
)


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
    path = Path(getattr(lilbee_engine, _BUNDLED_ACCESSORS[tool])())
    return path if path.is_file() else None


def engine_pin() -> str:
    """Identity of the engine this lilbee would spawn; sharing keys on it.

    Two dimensions must match for two processes to share one engine: the engine
    BUILD (a configured ``LILBEE_LLAMA_SERVER_PATH`` is its own identity so a
    bring-your-own engine never silently shares with a bundled one) and the
    load-affecting CONFIG baked into the launch argv (kv-cache type, expert
    offload, n-gpu-layers, ctx target, ...). A process whose load config differs
    computes a different pin, so ``contract_matches`` refuses the bind and it
    overflows to its own engine rather than silently running on the incumbent's
    flags. Total: never raises, because it runs on every state write.
    """
    return f"{_engine_build_id()}|{_load_config_signature()}"


def _engine_build_id() -> str:
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
    raise ProviderError(f"{tool.value} binary not found. {_INSTALL_HINT}", kind=kind)


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
