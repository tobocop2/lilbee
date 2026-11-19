"""Record a preset check against a running server: python -m tests.conformance.record_presets."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import tempfile
import urllib.request
from enum import StrEnum
from pathlib import Path
from typing import Any

from tools.qa.engine_demo.scrub import scrub_recording

from lilbee.providers.fleet.readback import _is_host_device, parse_memory_rows
from tests.conformance.preset_check import (
    Outcome,
    Setup,
    configured,
    fixture_json,
    recording,
    run_steps,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument(
        "--token-env",
        default="",
        help="Name of the environment variable that holds the bearer token for --url.",
    )
    parser.add_argument(
        "--embed-url",
        default="",
        help="Server for /v1/embeddings when --url has no such route, such as the engine proxy.",
    )
    parser.add_argument(
        "--vision-url",
        default="",
        help="Server for chat requests that carry an image, such as the proxy of the vision role.",
    )
    parser.add_argument("--roles", required=True, help="Comma-separated roles of the server.")
    parser.add_argument("--models", required=True, help="Model ids by role: chat=ID,embed=ID.")
    parser.add_argument("--wire", default="v1")
    parser.add_argument("--embed-dim", type=int, default=0, help="Vector size of the embed model.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Run the strict steps and the named step; every engine recording sets it.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="Window the engine serves, for a server that has no overflow check of its own.",
    )
    parser.add_argument("--backend", default="cpu", help="Backend of the run: cpu, cuda, metal.")
    parser.add_argument(
        "--device-url",
        default="",
        help="Server whose /memory reply lists the devices, when --url is a proxy without it.",
    )
    parser.add_argument(
        "--engine-log",
        type=Path,
        help="Startup output of the engine, for an engine whose server has no /memory route.",
    )
    parser.add_argument("--front", default="", help="Proxy in front of the server, if any.")
    parser.add_argument("--out", required=True, type=Path)
    return parser


def setup_from(args: argparse.Namespace) -> Setup:
    models = dict(pair.split("=", 1) for pair in args.models.split(","))
    return Setup(
        tuple(args.roles.split(",")),
        models,
        args.wire,
        args.embed_dim,
        args.strict,
        args.server,
        args.window,
    )


_NO_REPORT = "none"
_CPU = "cpu"
_PROBE_TIMEOUT_S = 10
# The word a device label starts with ("CUDA0", "MTL0", "Vulkan1") and the backend it names.
_LABEL_BACKENDS = {
    "cuda": "cuda",
    "mtl": "metal",
    "metal": "metal",
    "vulkan": "vulkan",
    "rocm": "rocm",
    "hip": "rocm",
    "sycl": "sycl",
    "openvino": "openvino",
}


# "Starting FlashBert model on Cuda(CudaDevice(DeviceId(1)))" names the device by kind and index.
_LOG_DEVICE_ID = re.compile(r"\bmodel on (?P<kind>[A-Za-z]+)\(\w+\(DeviceId\((?P<index>\d+)\)\)\)")
# "llama_prepare_model_devices: using device CUDA0 (NVIDIA GeForce RTX 3090)" names the label.
_LOG_DEVICE_LABEL = re.compile(r"\busing device (?P<label>\S+)")


class DeviceSource(StrEnum):
    """Where the recorder read the devices of a run."""

    MEMORY = "memory"
    ENGINE_LOG = "engine-log"
    NONE = "none"


def reported_devices(url: str, token: str | None) -> list[str]:
    """The device labels in the ``/memory`` reply of the engine at *url*; empty if none."""
    # S310: the URL is the engine address the operator passes on the command line, never a reply.
    request = urllib.request.Request(url.rstrip("/") + "/memory")  # noqa: S310
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=_PROBE_TIMEOUT_S) as resp:  # noqa: S310
            payload = json.loads(resp.read())
    except (OSError, ValueError):
        return []
    return sorted(label for label, size in parse_memory_rows(payload).items() if size > 0)


def logged_devices(path: Path) -> list[str]:
    """The device labels that the startup lines in *path* name; SystemExit if unreadable."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SystemExit(f"--engine-log {path}: {exc.strerror}") from exc
    labels = {f"{m['kind']}{m['index']}" for m in _LOG_DEVICE_ID.finditer(text)}
    labels.update(m["label"] for m in _LOG_DEVICE_LABEL.finditer(text))
    return sorted(labels)


def device_evidence(
    url: str, token: str | None, engine_log: Path | None
) -> tuple[list[str], DeviceSource]:
    """The device labels of the run and where they came from; the engine reply wins over the log."""
    from_log = logged_devices(engine_log) if engine_log else []
    from_reply = reported_devices(url, token)
    if from_reply and from_log and reported_backend(from_reply) != reported_backend(from_log):
        raise SystemExit(
            f"--engine-log {engine_log}: the log names {reported_backend(from_log)}, "
            f"the engine reply names {reported_backend(from_reply)}"
        )
    if from_reply:
        return from_reply, DeviceSource.MEMORY
    if from_log:
        return from_log, DeviceSource.ENGINE_LOG
    return [], DeviceSource.NONE


def reported_backend(devices: list[str]) -> str:
    """The backend of the device labels that hold GPU memory; cpu when only host rows remain."""
    gpus: set[str] = set()
    for label in devices:
        if _is_host_device(label):
            continue
        word = re.match(r"[A-Za-z]+", label)
        key = word.group().lower() if word else label.lower()
        gpus.add(_LABEL_BACKENDS.get(key, key))
    if gpus:
        return "+".join(sorted(gpus))
    return _CPU if devices else _NO_REPORT


def check_backend(asked: str, reported: str) -> None:
    """SystemExit unless the backend the engine reported is the backend the run asked for."""
    if reported == asked or (reported == _NO_REPORT and asked == _CPU):
        return
    if reported == _NO_REPORT:
        raise SystemExit(
            f"--backend {asked}: the engine lists no device in its reply, "
            f"so the run is not shown to use {asked}"
        )
    raise SystemExit(f"--backend {asked}: the engine reports {reported}")


def run_log(
    args: argparse.Namespace,
    results: dict[str, Outcome],
    devices: list[str],
    source: DeviceSource,
) -> str:
    """The text saved beside a recording: engine, version, host, backends, devices, steps."""
    lines = [
        f"engine: {args.server}",
        f"version: {args.version}",
        f"host: {platform.platform()}",
        f"backend: {args.backend}",
        f"reported-backend: {reported_backend(devices)}",
        f"devices: {','.join(devices) or _NO_REPORT}",
        f"devices-from: {source}",
        f"models: {args.models}",
    ]
    if args.front:
        lines.append(f"front: {args.front}")
    lines += [f"STEP {args.server} {step} ok={o.ok} {o.detail}" for step, o in results.items()]
    return "\n".join(lines) + "\n"


def write_recording(out: Path, doc: dict[str, Any], log: str) -> str:
    """Write the recording and its log with each home directory and the hostname scrubbed."""
    hostname = socket.gethostname()
    out.parent.mkdir(parents=True, exist_ok=True)
    scrubbed = scrub_recording(doc, hostname)
    out.write_text(json.dumps(scrubbed, indent=1) + "\n", encoding="utf-8")
    log = scrub_recording(log, hostname)
    out.with_suffix(".log").write_text(log, encoding="utf-8")
    return log


def token_from(args: argparse.Namespace) -> str | None:
    """The bearer token held in the variable that --token-env names; SystemExit when it is empty."""
    if not args.token_env:
        return None
    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit(f"--token-env {args.token_env}: the variable is empty or not set")
    return token


def main() -> int:
    args = build_parser().parse_args()
    setup = setup_from(args)
    token = token_from(args)
    with (
        tempfile.TemporaryDirectory() as root,
        recording(
            args.url,
            token=token,
            embed_url=args.embed_url or None,
            vision_url=args.vision_url or None,
        ) as (
            proxy_url,
            tape,
        ),
        configured(proxy_url, setup, Path(root)),
    ):
        results = run_steps(setup)
    device_url = args.device_url or args.url
    devices, source = device_evidence(
        device_url, token if device_url == args.url else None, args.engine_log
    )
    check_backend(args.backend, reported_backend(devices))
    doc = fixture_json(args.server, args.version, setup, results, tape)
    log = write_recording(args.out, doc, run_log(args, results, devices, source))
    print(log, end="")
    return 0 if all(o.ok for o in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
