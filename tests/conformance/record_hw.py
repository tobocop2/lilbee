"""Record a managed engine on a GPU, from the pack install to the recording.

    python -m tests.conformance.record_hw --engine tei --cell linux-x86_64-cuda-sm75 \\
        --backend cuda --name tei-cuda-sm75 --models /var/tmp/models --logs /var/tmp/logs

The run installs the pinned pack, stops if the pack is not the build of --cell, and starts
one server for each role of the engine with the adapter's own command line on the GPU. It
checks that every server reports a device of --backend, then runs
tests.conformance.record_presets with --strict against them and stops the servers. Without
a device of --backend, no recording is written. A recording with every step passed gets its
row in the results page and the name of the card in its log. Arguments that this tool does
not know go to the recorder.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO
from unittest import mock

from lilbee.app import engine_service
from lilbee.catalog import runnable
from lilbee.providers.engine_params import resolve_n_gpu_layers
from lilbee.providers.fleet.engine_manifest import cell_key
from lilbee.providers.fleet.engines.base import PORT_SLOT, EngineAdapter, EngineName, LaunchRequest
from lilbee.providers.fleet.engines.choice import ADAPTERS
from lilbee.providers.fleet.gpu_select import PCIVendorID
from lilbee.providers.fleet.host_caps import HostCaps
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole
from tests.conformance import record_presets
from tests.conformance.record_cpu_compare import (
    _ENGINE_MODELS,
    _MMPROJ,
    _ORACLE_MODELS,
    Model,
)
from tests.conformance.record_cpu_compare import _fetch as fetch_model

_BACKENDS = ("cuda", "vulkan", "rocm", "metal")
_HEALTH_SECONDS = 1200
_POLL_SECONDS = 2
_STOP_SECONDS = 60
_MISTRAL_SERVED_ID = "default"
_VULKAN_BACKEND = "vulkan"
_TABLE_HEAD = "| Recording | Roles |"

# The cross-encoder of the TEI record in the CPU comparison.
_RERANK_MODEL = Model(
    "bge-reranker-v2-m3",
    "BAAI/bge-reranker-v2-m3",
    "model.safetensors",
    "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286",
    "F32",
    16384,
)

# The roles each engine records, in the order the servers start. The first role holds --url.
_TEI_MODELS = {
    WorkerRole.EMBED: _ENGINE_MODELS[WorkerRole.EMBED],
    WorkerRole.RERANK: _RERANK_MODEL,
}
_MODELS: Mapping[EngineName, Mapping[WorkerRole, Model]] = {
    EngineName.MISTRAL: _ENGINE_MODELS,
    EngineName.TEI: _TEI_MODELS,
    EngineName.LLAMA: _ORACLE_MODELS,
    EngineName.IK_LLAMA: {
        role: _ORACLE_MODELS[role] for role in (WorkerRole.CHAT, WorkerRole.EMBED)
    },
}
# The vector size of each embed model, by its Hugging Face repo; the embed step checks it.
_EMBED_DIMS = {
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-0.6B-GGUF": 1024,
}
# The rerank wire of an engine that does not answer /v1/rerank. TEI answers /rerank.
_WIRES = {EngineName.TEI: "tei"}
_URL_FLAGS = {
    WorkerRole.EMBED: "--embed-url",
    WorkerRole.VISION: "--vision-url",
    WorkerRole.RERANK: "--rerank-url",
}


@dataclass(frozen=True)
class Part:
    """One server of the run: a role, its model on disk and the id it answers to."""

    role: WorkerRole
    model: Model
    path: Path
    served: str


@dataclass
class Running:
    """A started server: its address, its log, and the command that started it."""

    part: Part
    url: str
    log: Path
    argv: list[str]
    proc: subprocess.Popen[bytes]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0] if __doc__ else None, allow_abbrev=False
    )
    parser.add_argument("--engine", required=True, choices=[n.value for n in _MODELS])
    parser.add_argument(
        "--cell", required=True, help="The cell the pack must be, as in coverage.json."
    )
    parser.add_argument("--backend", required=True, choices=_BACKENDS)
    parser.add_argument("--name", required=True, help="Recording name, such as tei-cuda-sm75.")
    parser.add_argument("--models", type=Path, required=True, help="Folder of the model files.")
    parser.add_argument("--logs", type=Path, required=True, help="Folder of the engine logs.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tests/conformance/fixtures/engines"),
        help="Folder that receives <name>.json and <name>.log.",
    )
    parser.add_argument(
        "--results-doc",
        type=Path,
        default=Path("docs/engine-results.md"),
        help="The results page that receives the row of a good recording.",
    )
    parser.add_argument(
        "--gpu",
        default="",
        help="Name of the card for the row; the default is the first NVIDIA card of the host.",
    )
    parser.add_argument("--base-port", type=int, default=8501)
    parser.add_argument(
        "--as-vulkan-host",
        action="store_true",
        help="Install the Vulkan pack of llama.cpp on a host whose card has a CUDA pack.",
    )
    return parser


def vulkan_host() -> HostCaps:
    """The live host as a Vulkan-only machine: no CUDA card, a Vulkan loader, an AMD display."""
    host = runnable.session_host()
    return dataclasses.replace(
        host,
        gpus=(),
        amd_gfx=(),
        vulkan_loader=True,
        display_vendors=frozenset({PCIVendorID.AMD}),
    )


@contextmanager
def presented_as(host: HostCaps | None) -> Iterator[None]:
    """Make *host* the session host while the block runs; ``None`` keeps the live host."""
    if host is None:
        yield
        return
    try:
        with mock.patch.object(runnable, "detect", lambda: host):
            runnable.refresh_session_host()
            yield
    finally:
        runnable.refresh_session_host()


def install_pinned(adapter: EngineAdapter, cell: str) -> tuple[Path, str]:
    """Install the pack of *adapter*; stop unless it is the build of *cell*.

    Returns the server binary and the build version.
    """
    build = engine_service.install(adapter.name)
    found = cell_key(runnable.session_host(), build.backend)
    if found != cell:
        raise SystemExit(
            f"the install of {adapter.name} chose {found}, not {cell}: "
            "this host does not run that cell, so nothing is recorded"
        )
    return adapter.binary(), build.version


def launch_argv(
    adapter: EngineAdapter, part: Part, binary: Path, mmproj: Path | None, port: int
) -> list[str]:
    """The command line that lilbee starts *part* with on the GPU, with *port* in place."""
    spec = adapter.role_spec(part.role, RerankMode.CROSS_ENCODER, None)
    target = part.path if part.path.suffix == ".gguf" else part.path.parent
    request = LaunchRequest(
        binary=binary,
        model_path=target,
        devices=(0,),
        n_gpu_layers=resolve_n_gpu_layers(embedding=True),
        slots=1,
        ctx_per_slot=part.model.context,
        mmproj=mmproj,
        memory_endpoint=adapter.check_readback(binary),
        target=RoleTarget.GPU,
    )
    argv = adapter.launch_argv(spec, request)
    if PORT_SLOT in argv:
        argv = [str(port) if arg == PORT_SLOT else arg for arg in argv]
    else:
        argv += ["--port", str(port)]
    if adapter.name in (EngineName.LLAMA, EngineName.IK_LLAMA):
        argv += ["--alias", part.served]
    return argv


def _parts(engine: EngineName, models: Path) -> list[Part]:
    parts: list[Part] = []
    for role, model in _MODELS[engine].items():
        served = {
            EngineName.MISTRAL: _MISTRAL_SERVED_ID,
            EngineName.TEI: model.folder,
        }.get(engine, Path(model.file).stem)
        parts.append(Part(role, model, fetch_model(model, models), served))
    return parts


def _start(
    adapter: EngineAdapter, part: Part, binary: Path, mmproj: Path | None, port: int, logs: Path
) -> Running:
    argv = launch_argv(adapter, part, binary, mmproj, port)
    log = logs / f"{adapter.name}-{part.role.value}-{port}.log"
    env = {**os.environ, **adapter.runtime_env(), **adapter.target_env(RoleTarget.GPU, 1)}
    print("start", " ".join(argv), flush=True)
    handle: IO[bytes] = log.open("wb")
    try:
        proc = subprocess.Popen(argv, stdout=handle, stderr=subprocess.STDOUT, env=env)
    finally:
        handle.close()
    return Running(part, f"http://127.0.0.1:{port}", log, argv, proc)


def _wait(server: Running) -> None:
    for _ in range(_HEALTH_SECONDS // _POLL_SECONDS):
        if server.proc.poll() is not None:
            raise SystemExit(
                f"the {server.part.role} server exited with {server.proc.returncode}; "
                f"see {server.log}"
            )
        try:
            with urllib.request.urlopen(server.url + "/health", timeout=3):  # noqa: S310  a local http URL
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(_POLL_SECONDS)
    raise SystemExit(f"the {server.part.role} server at {server.url} did not answer /health")


def _stop(servers: list[Running]) -> None:
    for server in servers:
        server.proc.terminate()
    for server in servers:
        try:
            server.proc.wait(timeout=_STOP_SECONDS)
        except subprocess.TimeoutExpired:
            server.proc.kill()
            server.proc.wait()


def require_device(server: Running, backend: str) -> list[str]:
    """The device labels of *server*; SystemExit unless they name *backend*."""
    devices, _source = record_presets.device_evidence(server.url, None, server.log)
    reported = record_presets.reported_backend(devices)
    record_presets.check_backend(backend, reported)
    return devices


def recorder_command(
    args: argparse.Namespace, version: str, servers: list[Running], extra: list[str]
) -> list[str]:
    """The record_presets command for the started servers."""
    first = servers[0]
    command = [
        sys.executable,
        "-m",
        "tests.conformance.record_presets",
        "--strict",
        "--engine",
        args.engine,
        "--backend",
        args.backend,
        "--server",
        args.name,
        "--version",
        version,
        "--url",
        first.url,
        "--roles",
        ",".join(s.part.role.value for s in servers),
        "--models",
        ",".join(f"{s.part.role.value}={s.part.served}" for s in servers),
        "--engine-log",
        str(first.log),
        "--server-flags",
        "; ".join(f"{s.part.role.value}: {' '.join(s.argv)}" for s in servers),
        "--out",
        str(args.out_dir / f"{args.name}.json"),
    ]
    for server in servers:
        if server.part.role is WorkerRole.EMBED:
            command += ["--embed-dim", str(_EMBED_DIMS[server.part.model.repo])]
    if wire := _WIRES.get(EngineName(args.engine)):
        command += ["--wire", wire]
    for server in servers[1:]:
        command += [_URL_FLAGS[server.part.role], server.url]
    return [*command, *extra]


def card_name(args: argparse.Namespace) -> str:
    """The card of the row: --gpu, else the first NVIDIA card of the live host."""
    if args.gpu:
        return str(args.gpu)
    gpus = runnable.session_host().gpus
    if not gpus:
        raise SystemExit("this host lists no NVIDIA card: name the card with --gpu")
    return gpus[0].name


def record_row(args: argparse.Namespace, card: str) -> None:
    """Name the card in the log of the recording and put its row in the results page.

    A row of the same name, such as a "not recorded" row, is replaced; any other row is new
    and goes at the end of the table. The row says what the recording holds, as the tests of
    the page check.
    """
    out = args.out_dir / f"{args.name}.json"
    doc = json.loads(out.read_text(encoding="utf-8"))
    steps = len(doc["results"])
    row = (
        f"| {args.name} | {', '.join(doc['roles'])} | {card} | pass, {steps} of {steps} steps "
        f"| {doc['version']} | `{out.name}` |"
    )
    log = out.with_suffix(".log")
    log_lines = log.read_text(encoding="utf-8").split("\n")
    first_step = next(n for n, line in enumerate(log_lines) if line.startswith("STEP "))
    log_lines.insert(first_step, f"gpu: {card}")
    log.write_text("\n".join(log_lines), encoding="utf-8")
    lines = args.results_doc.read_text(encoding="utf-8").split("\n")
    start = next(n for n, line in enumerate(lines) if line.startswith(_TABLE_HEAD)) + 2
    end = start
    while lines[end].startswith("|"):
        end += 1
    mine = f"| {args.name} |"
    held = [n for n in range(start, end) if lines[n].startswith(mine)]
    if held:
        lines[held[0]] = row
    else:
        lines.insert(end, row)
    args.results_doc.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace, extra: list[str]) -> int:
    """Install, start, check the devices, record and stop; the return code is the recorder's.

    A recording with every step passed gets its row in the results page; any other keeps no row.
    """
    adapter = ADAPTERS[EngineName(args.engine)]
    if not args.results_doc.is_file():
        raise SystemExit(
            f"the results page {args.results_doc} is missing: "
            "run from the repo root or pass --results-doc"
        )
    card = card_name(args)
    args.logs.mkdir(parents=True, exist_ok=True)
    host = vulkan_host() if args.as_vulkan_host else None
    with presented_as(host):
        binary, version = install_pinned(adapter, args.cell)
        parts = _parts(adapter.name, args.models)
        mmproj = fetch_model(_MMPROJ, args.models) if adapter.name is EngineName.LLAMA else None
        servers: list[Running] = []
        try:
            for port, part in enumerate(parts, start=args.base_port):
                vision = mmproj if part.role is WorkerRole.VISION else None
                servers.append(_start(adapter, part, binary, vision, port, args.logs))
            for server in servers:
                _wait(server)
            for server in servers:
                print(f"{server.part.role}: {','.join(require_device(server, args.backend))}")
            label = f"{adapter.name} {version} {args.cell}"
            command = recorder_command(args, label, servers, extra)
            code = subprocess.call(command)
            if code == 0:
                record_row(args, card)
            return code
        finally:
            _stop(servers)


def main(argv: list[str] | None = None) -> int:
    """Parse the command line and run; unknown arguments go to the recorder."""
    args, extra = _parser().parse_known_args(argv)
    if args.as_vulkan_host and args.engine != EngineName.LLAMA:
        raise SystemExit("--as-vulkan-host applies to llama.cpp only")
    if args.as_vulkan_host and args.backend != _VULKAN_BACKEND:
        raise SystemExit("--as-vulkan-host records the vulkan backend")
    return run(args, extra)


if __name__ == "__main__":
    raise SystemExit(main())
