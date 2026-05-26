"""Validate a model's demo prompt before recording.

Downloads the model, stands up the demo (llama-server + lilbee serve + the
lilbee_search-only opencode.json via giant_demo.sh), then runs the prompt
headless through `opencode run` and prints the full transcript so the answer
can be read and judged. No recording happens here.

    python validate.py <family> ["optional prompt override to iterate"]

If no override is given, the prompt comes from reel_config.PROMPTS[family].
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from models import ROSTER  # noqa: E402
from probe import _borrow_template, _download  # noqa: E402
from reel_config import PROMPTS, display_name  # noqa: E402

DEMO_SH = str(HERE / "giant_demo.sh")
PROJ = "/root/demo-proj"


def main() -> None:
    fam = sys.argv[1]
    override = sys.argv[2] if len(sys.argv) > 2 else None
    spec = {s.family: s for s in ROSTER}.get(fam)
    if spec is None:
        sys.exit(f"unknown family {fam!r}")
    prompt = override or PROMPTS.get(fam)
    if not prompt:
        sys.exit(f"no prompt for {fam!r}")
    print(f"===== {fam}: download =====", flush=True)
    gguf = _download(spec)
    disp = display_name(spec.gguf)
    template = _borrow_template(spec)
    env = {**os.environ, "MULTIGPU": "1"} if spec.multi_gpu_only else None
    print(f"===== {fam} ({disp}): setup =====", flush=True)
    cmd = ["bash", DEMO_SH, fam, str(gguf), disp] + ([str(template)] if template else [])
    if subprocess.run(cmd, env=env).returncode != 0:
        sys.exit("SETUP_FAIL")
    print(f"\n===== PROMPT ({disp}) =====\n{prompt}\n\n===== TRANSCRIPT =====", flush=True)
    # Force our model so opencode doesn't fall back to its built-in build agent.
    r = subprocess.run(
        ["opencode", "run", "-m", f"lilbee/{disp}", prompt],
        cwd=PROJ,
        capture_output=True,
        text=True,
        timeout=900,
    )
    print(r.stdout)
    if r.stderr.strip():
        print("\n----- stderr (tail) -----\n" + r.stderr[-2000:])
    print(f"\n===== rc={r.returncode} =====")


if __name__ == "__main__":
    main()
