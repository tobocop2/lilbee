"""QA + demo every supported model in one pass.

For each model: download the GGUF, stand up llama-server (native tools) + lilbee
serve (lilbee_search MCP over src/lilbee) via giant_demo.sh, drive the real opencode
TUI through a VHS tape (the demo), `vhs publish` the gif, and verify tool calling
actually fired (lilbee serve logged a CallToolRequest during the run) = the QA pass.

Run on the pod:
    HF_TOKEN=... PATH=$HOME/.opencode/bin:$PATH /root/pv/bin/python record_reel.py
Writes gifs/webms + reel_manifest.txt (model -> QA verdict + published URL) to
/root/demos-out/.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from models import ROSTER  # noqa: E402
from probe import _borrow_template, _download  # noqa: E402

DEMO_SH = str(HERE / "giant_demo.sh")
TAPE_TMPL = HERE / "giant_demo.tape.tmpl"
OUT = Path("/root/demos-out")
SERVE_LOG = Path("/tmp/lilbee-serve.log")
MANIFEST = OUT / "reel_manifest.txt"
PROJ = "/root/demo-proj"

PROMPT = (
    "Add support for a new model family to lilbee whose tool calls look like "
    "<tool_call name=search>{json args}</tool_call>. Implement the family profile and "
    "the response-parser schema following the existing families, and wire it into the parser."
)
# Tool-call-passing models (FINDINGS). Cached giants + smalls (pulled). Smalls and
# cached giants first; the big single-GPU pulls (gpt-oss, glm-air) last. qwen3-235b
# was already recorded.
REEL = [
    "qwen3", "llama3", "mistral-nemo", "gemma4", "qwen3-coder", "hermes",
    "functionary", "minimax-m2", "glm-4.6", "gpt-oss", "glm-air",
]


def _calltool_count() -> int:
    try:
        return SERVE_LOG.read_text(errors="replace").count("CallToolRequest")
    except FileNotFoundError:
        return 0


def _load_manifest() -> dict[str, tuple[str, str, str]]:
    """Prior verdicts, so a resume run preserves already-recorded models."""
    out: dict[str, tuple[str, str, str]] = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            parts = line.split("\t")
            if len(parts) == 3:
                out[parts[0]] = (parts[0], parts[1], parts[2])
    return out


def _write_manifest(results: dict[str, tuple[str, str, str]]) -> None:
    """Write in canonical REEL order, keeping any extra families at the end."""
    order = REEL + [f for f in results if f not in REEL]
    rows = [results[f] for f in order if f in results]
    MANIFEST.write_text("\n".join(f"{f}\t{q}\t{u}" for f, q, u in rows) + "\n")


def _cleanup_repo(repo: str) -> None:
    try:
        from huggingface_hub import scan_cache_dir

        for r in scan_cache_dir().repos:
            if r.repo_id == repo:
                shutil.rmtree(r.repo_path, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    specs = {s.family: s for s in ROSTER}
    # Optional resume subset: families on argv (else the full reel). Prior manifest
    # verdicts are preserved so a resume doesn't clobber already-recorded models.
    run = [f for a in sys.argv[1:] for f in a.replace(",", " ").split()] or REEL
    results = _load_manifest()
    for fam in run:
        spec = specs.get(fam)
        if spec is None:
            results[fam] = (fam, "NO_SPEC", "")
            continue
        print(f"\n===== {fam}: download =====", flush=True)
        try:
            gguf = _download(spec)
        except Exception as exc:  # noqa: BLE001
            print(f"[{fam}] download failed: {exc}")
            results[fam] = (fam, f"SKIP(download: {str(exc)[:40]})", "")
            continue
        template = _borrow_template(spec)
        print(f"===== {fam}: setup =====", flush=True)
        cmd = ["bash", DEMO_SH, fam, str(gguf)] + ([str(template)] if template else [])
        # Only the 200GB giants span both GPUs; small models stay on one (see giant_demo.sh).
        env = {**os.environ, "MULTIGPU": "1"} if spec.multi_gpu_only else None
        if subprocess.run(cmd, env=env).returncode != 0:
            results[fam] = (fam, "SETUP_FAIL", "")
            _write_manifest(results)
            continue
        stem = str(OUT / f"reel-{fam}")
        tape = OUT / f"{fam}.tape"
        tape.write_text(
            TAPE_TMPL.read_text()
            .replace("__OUT__", stem)
            .replace("__PROMPT__", PROMPT)
            .replace("__GENSLEEP__", "90s")
        )
        before = _calltool_count()
        print(f"===== {fam}: record =====", flush=True)
        subprocess.run(["vhs", str(tape)], cwd=PROJ)
        qa = "PASS" if _calltool_count() > before else "FAIL(no tool call)"
        pub = subprocess.run(["vhs", "publish", f"{stem}.gif"], capture_output=True, text=True)
        url = next(
            (ln.strip() for ln in (pub.stdout or "").splitlines() if "vhs.charm.sh" in ln), ""
        )
        results[fam] = (fam, qa, url)
        _write_manifest(results)
        print(f"[{fam}] -> {qa}  {url}", flush=True)
        if not spec.multi_gpu_only:  # keep cached giants for the heavy demo
            _cleanup_repo(spec.repo)
    print("\n===== REEL DONE =====")
    for f, q, u in results.values():
        print(f"  {f}: {q}  {u}")


if __name__ == "__main__":
    main()
