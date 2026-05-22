"""Re-order ``tools/qa/opencode/models.toml`` largest-first + unskip GPU-only cells.

Run once on a GPU cloud box right before the matrix kicks off:

    uv run python tools/qa/reorder-largest-first.py

Effects:
- Sorts the cells by descending ``size_gb`` so peak disk pressure clears
  early and the slowest cells (glm-air, qwen3-coder) don't sit at the tail
  where a budget cutoff could orphan them.
- Drops ``skip = true`` from cells that the M1 Pro can't run but a CUDA
  box can (``glm-air``, ``phi4mini``). Leaves cells skipped for model-quality
  reasons (``mistral`` v0.3, ``gemma2``) untouched.

Idempotent: re-running on an already-reordered file is a no-op.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

GPU_UNBLOCK = frozenset({"glm-air", "phi4mini"})


def main() -> None:
    p = Path(__file__).parent / "opencode" / "models.toml"
    data = tomllib.loads(p.read_text())
    for entry in data["model"]:
        if entry["family"] in GPU_UNBLOCK and entry.get("skip"):
            entry["skip"] = False

    data["model"].sort(key=lambda m: (-float(m["size_gb"]), m["family"]))

    lines = [
        "# QA matrix model list. Local-only; not pulled in by CI.",
        "# Reordered largest-first by tools/qa/reorder-largest-first.py for the",
        "# cloud GPU sweep. Cleanup-per-cell keeps disk usage flat at the peak",
        "# single-model size.",
        "",
    ]
    q = '"'
    for entry in data["model"]:
        lines.append("[[model]]")
        lines.append(f"family = {q}{entry['family']}{q}")
        lines.append(f"ref = {q}{entry['ref']}{q}")
        lines.append(f"size_gb = {entry['size_gb']}")
        if entry.get("skip"):
            lines.append("skip = true")
        lines.append("")
    p.write_text("\n".join(lines))

    print(f"reordered; {len([m for m in data['model'] if not m.get('skip')])} active cells in run order:")
    for entry in data["model"]:
        flag = "  [SKIP]" if entry.get("skip") else ""
        print(f"  {entry['family']:<15} {entry['size_gb']:>5}GB{flag}")


if __name__ == "__main__":
    main()
