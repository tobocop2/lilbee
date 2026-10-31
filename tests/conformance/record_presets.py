"""Record a preset check against a running server: python -m tests.conformance.record_presets."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from pathlib import Path

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
    parser.add_argument("--roles", required=True, help="Comma-separated roles of the server.")
    parser.add_argument("--models", required=True, help="Model ids by role: chat=ID,embed=ID.")
    parser.add_argument("--wire", default="v1")
    parser.add_argument("--embed-dim", type=int, default=0, help="Vector size of the embed model.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Run the strict steps and the named step; every engine recording sets it.",
    )
    parser.add_argument("--backend", default="cpu", help="Backend of the run: cpu, cuda, metal.")
    parser.add_argument("--front", default="", help="Proxy in front of the server, if any.")
    parser.add_argument("--out", required=True, type=Path)
    return parser


def setup_from(args: argparse.Namespace) -> Setup:
    models = dict(pair.split("=", 1) for pair in args.models.split(","))
    return Setup(
        tuple(args.roles.split(",")), models, args.wire, args.embed_dim, args.strict, args.server
    )


def run_log(args: argparse.Namespace, results: dict[str, Outcome]) -> str:
    """The text saved beside a recording: engine, version, host, backend, models and each step."""
    lines = [
        f"engine: {args.server}",
        f"version: {args.version}",
        f"host: {platform.platform()}",
        f"backend: {args.backend}",
        f"models: {args.models}",
    ]
    if args.front:
        lines.append(f"front: {args.front}")
    lines += [f"STEP {args.server} {step} ok={o.ok} {o.detail}" for step, o in results.items()]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = build_parser().parse_args()
    setup = setup_from(args)
    with (
        tempfile.TemporaryDirectory() as root,
        recording(args.url) as (proxy_url, tape),
        configured(proxy_url, setup, Path(root)),
    ):
        results = run_steps(setup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc = fixture_json(args.server, args.version, setup, results, tape)
    args.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    log = run_log(args, results)
    args.out.with_suffix(".log").write_text(log, encoding="utf-8")
    print(log, end="")
    return 0 if all(o.ok for o in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
