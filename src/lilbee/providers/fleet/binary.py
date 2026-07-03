"""Resolve the bundled engine binaries (llama-server, llama-swap, gguf-parser)."""

from __future__ import annotations

import shutil
from enum import StrEnum
from pathlib import Path

from lilbee.core.config import cfg
from lilbee.providers.base import ProviderError

_INSTALL_HINT = (
    "Reinstall lilbee to get the bundled engine, or set LILBEE_LLAMA_SERVER_PATH "
    "to a llama-server binary (a llama.cpp release download or `brew install llama.cpp`) "
    "and put llama-swap / gguf-parser on PATH."
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


def resolve_engine_tool(tool: EngineTool) -> Path:
    """Resolve *tool*: configured llama-server path, then bundled wheel, then PATH.

    Never downloads anything; the binaries arrive via the bundled ``lilbee-engine``
    wheel or bring-your-own. Only llama-server honors ``LILBEE_LLAMA_SERVER_PATH``
    (an explicit setting beats the bundled wheel); the other tools resolve from the
    wheel, then ``PATH``.
    """
    if tool is EngineTool.LLAMA_SERVER:
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

    raise ProviderError(f"{tool.value} binary not found. {_INSTALL_HINT}")


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
