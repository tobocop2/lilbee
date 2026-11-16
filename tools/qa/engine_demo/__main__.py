"""Run one engine through the lilbee CLI, or judge a transcript a run wrote.

    python -m tools.qa.engine_demo run --engine mistral.rs --backend cpu --out mistral-rs-cpu.json
    python -m tools.qa.engine_demo report mistral-rs-cpu.json

``run`` records and exits 0 when it wrote a transcript, whatever the engine did;
``report`` judges it and exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from tools.qa.engine_demo.observe import DEFAULT_QUESTION, RunPlan, observe
from tools.qa.engine_demo.oracles import Transcript, judge
from tools.qa.engine_demo.scrub import scrub_urls

from lilbee.providers.fleet.engines.base import EngineName
from lilbee.providers.fleet.engines.choice import ADAPTERS
from lilbee.providers.roles import WorkerRole

# The model each role runs when --model does not name one.
DEFAULT_MODELS = {
    WorkerRole.CHAT: "Qwen/Qwen3-4B-GGUF",
    WorkerRole.EMBED: "Qwen/Qwen3-Embedding-0.6B-GGUF",
    WorkerRole.RERANK: "gpustack/bge-reranker-v2-m3-GGUF",
    WorkerRole.VISION: "ggml-org/SmolVLM-256M-Instruct-GGUF",
}
# An engine that loads a format other than GGUF runs that form of its models; the
# "#<format>" suffix is how a pull picks it. Every repo is public.
ENGINE_MODELS: dict[str, dict[WorkerRole, str]] = {
    EngineName.TEI: {
        WorkerRole.EMBED: "Qwen/Qwen3-Embedding-0.6B#safetensors",
        WorkerRole.RERANK: "BAAI/bge-reranker-v2-m3#safetensors",
    },
    EngineName.MISTRAL: {
        WorkerRole.EMBED: "Qwen/Qwen3-Embedding-0.6B#safetensors",
        WorkerRole.VISION: "HuggingFaceTB/SmolVLM-256M-Instruct#safetensors",
    },
    EngineName.MLX: {WorkerRole.CHAT: "mlx-community/Qwen3-0.6B-4bit#mlx"},
    EngineName.LEMONADE: {
        WorkerRole.CHAT: "qwen3:0.6b#xdna",
        WorkerRole.EMBED: "embed-gemma:300m#xdna",
    },
}
# Vision is left out: it needs --image-pdf, so a run names it.
_DEFAULT_ROLES = (WorkerRole.CHAT, WorkerRole.EMBED, WorkerRole.RERANK)


def default_models(engine: str, roles: Sequence[WorkerRole]) -> dict[WorkerRole, str]:
    """The model of each role for *engine*: the format it loads where that is not GGUF."""
    overrides = ENGINE_MODELS.get(engine, {})
    return {role: overrides.get(role, DEFAULT_MODELS[role]) for role in roles}


def served_roles(engine: str) -> frozenset[WorkerRole] | None:
    """The roles the adapter of *engine* serves, or None when no adapter has that name."""
    adapter = next((a for name, a in ADAPTERS.items() if name == engine), None)
    return None if adapter is None else frozenset(adapter.served_roles)


def _role_model(value: str) -> tuple[WorkerRole, str]:
    role, _, ref = value.partition("=")
    if role not in list(WorkerRole) or not ref:
        raise argparse.ArgumentTypeError(
            f"{value!r} must be ROLE=REF, with ROLE one of {', '.join(WorkerRole)}"
        )
    return WorkerRole(role), ref


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="engine_demo", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one engine and write a transcript")
    run.add_argument("--engine", required=True, help="engine name, as lilbee engine list shows it")
    run.add_argument("--backend", required=True, help="backend of the install, such as cpu")
    run.add_argument("--out", required=True, type=Path, help="transcript file to write")
    run.add_argument("--cli", default="lilbee", help="command that starts lilbee")
    run.add_argument(
        "--roles",
        nargs="+",
        type=WorkerRole,
        choices=list(WorkerRole),
        help="roles the engine claims; the default is every role of chat, embed and rerank "
        "that the engine serves",
    )
    run.add_argument(
        "--model",
        action="append",
        default=[],
        type=_role_model,
        metavar="ROLE=REF",
        help="model for a claimed role, in place of the default",
    )
    run.add_argument("--image-pdf", type=Path, help="image-only PDF; the vision role needs it")
    run.add_argument("--question", default=DEFAULT_QUESTION, help="question for ask and chat")
    run.add_argument("--work-dir", type=Path, help="folder for the fresh data folder of this run")
    report = commands.add_parser("report", help="judge a transcript")
    report.add_argument("transcript", type=Path)
    return parser


def _planned_roles(parser: argparse.ArgumentParser, args: argparse.Namespace) -> list[WorkerRole]:
    """The roles to run: those named, or the default roles the engine serves."""
    served = served_roles(args.engine)
    requested = args.roles or list(_DEFAULT_ROLES)
    unserved = [role for role in requested if served is not None and role not in served]
    if args.roles:
        if unserved:
            parser.error(f"{args.engine} does not serve {', '.join(unserved)}")
        return list(requested)
    if unserved:
        print(f"roles skipped: {', '.join(unserved)} ({args.engine} does not serve them)")
    planned = [role for role in requested if role not in unserved]
    if not planned:
        parser.error(f"{args.engine} serves none of the default roles: pass --roles")
    return planned


def _run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    roles = _planned_roles(parser, args)
    models = default_models(args.engine, roles)
    for role, ref in args.model:
        if role not in models:
            parser.error(f"--model names {role}, which --roles does not claim")
        models[role] = ref
    try:
        plan = RunPlan(
            cli=tuple(shlex.split(args.cli)),
            engine=args.engine,
            backend=args.backend,
            models=models,
            image_pdf=args.image_pdf,
            question=args.question,
        )
    except ValueError as exc:
        parser.error(f"{exc}: pass --image-pdf")
    run_dir = Path(tempfile.mkdtemp(prefix="engine-demo-", dir=args.work_dir))
    transcript = observe(plan, run_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(transcript.to_json(), indent=2) + "\n"
    args.out.write_text(scrub_urls(text), encoding="utf-8")
    print(f"run folder: {run_dir}")
    print(f"transcript: {args.out}")
    return 0


def _report(args: argparse.Namespace) -> int:
    try:
        transcript = Transcript.from_json(json.loads(args.transcript.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {args.transcript}: {exc}") from exc
    failures = judge(transcript)
    label = f"{transcript.engine} {transcript.backend} {transcript.pin}"
    print(f"{'FAIL' if failures else 'PASS'} {label}")
    for failure in failures:
        print(f"  {failure.rule}: {failure.detail}")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    return _run(parser, args) if args.command == "run" else _report(args)


if __name__ == "__main__":
    sys.exit(main())
