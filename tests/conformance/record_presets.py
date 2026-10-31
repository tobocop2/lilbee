"""Record a preset check against a running server: python -m tests.conformance.record_presets."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from tests.conformance.preset_check import Setup, configured, fixture_json, recording, run_steps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--roles", required=True, help="Comma-separated roles the server documents.")
    parser.add_argument("--models", required=True, help="Model ids by role: chat=ID,embed=ID.")
    parser.add_argument("--wire", default="v1")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    models = dict(pair.split("=", 1) for pair in args.models.split(","))
    setup = Setup(tuple(args.roles.split(",")), models, args.wire)
    with tempfile.TemporaryDirectory() as root, recording(args.url) as (proxy_url, tape):
        with configured(proxy_url, setup, Path(root)):
            results = run_steps(setup)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc = fixture_json(args.server, args.version, setup, results, tape)
    args.out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    for step, outcome in results.items():
        print(f"STEP {args.server} {step} ok={outcome.ok} {outcome.detail}")
    return 0 if all(o.ok for o in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
