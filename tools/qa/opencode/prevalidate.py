"""Pre-record QA gate: for EVERY models.toml cell, assert the model actually answers
the demo prompt well and collect empirical timings -- BEFORE any demo is recorded.

This is prerequisite #1 for the reel: no model gets a demo recorded until it has
produced a good, complete answer here. It also measures the numbers the demo tape
needs (cold-start load time + generation time) so the tape is built on real data,
not guesses.

Per cell it:
  1. pulls the model (resolves the actually-installed quant) + writes the workspace
  2. boots ``lilbee serve`` as a subprocess with LILBEE_DATA pinned to that workspace
     (fresh process => the per-cell chat_model is honored; cfg is import-time)
  3. WARM-UP completion (max_tokens=1) -> times the cold model load (cold_s)
  4. runs the tier prompt through the real chat -> tool_calls -> /api/search ->
     feed-back loop, capturing the FINAL ANSWER TEXT, generation time (gen_s), the
     number of tool calls, and any error
  5. tears the serve down; deletes big models to keep disk flat
  6. writes report.md (full prompt + answer per model, for a human/agent to read and
     assert quality) and report.json (machine timings the tape build consumes)

A cell auto-FAILS if it errored, never tool-called, or returned an empty answer.
Auto-PASS only means "produced a non-empty answer via a real tool call" -- the
quality call (correct, grounded, complete) is made by reading report.md.

Run on the pod (cached models + indexed Godot corpus)::

    LILBEE_MODELS_DIR=/root/models LILBEE_QA_CORPUS=/root/godot_corpus \
      uv run python tools/qa/opencode/prevalidate.py --out /root/prevalidate
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matrix  # noqa: E402

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "lilbee_search",
        "description": "Search the indexed Godot 4 class reference; returns cited chunks.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
    },
}
_SYSTEM = (
    "You are a Godot 4.4 GDScript developer. Your training data is outdated, so you "
    "MUST call lilbee_search to confirm every class, method, and signature before you "
    "use it, then answer using only confirmed APIs."
)
_HOP_CAP = 6
_HEALTH_TIMEOUT_S = 240.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _boot_serve(workspace: Path, port: int) -> tuple[subprocess.Popen[bytes], str]:
    """Spawn ``lilbee serve`` on *port* with LILBEE_DATA=*workspace*; return (proc, token)."""
    env = os.environ.copy()
    env["LILBEE_DATA"] = str(workspace / ".lilbee")
    proc = subprocess.Popen(
        ["uv", "run", "lilbee", "serve", "--port", str(port)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + _HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/api/health", timeout=5).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(1)
    server_json = workspace / ".lilbee" / "data" / "server.json"
    token = json.loads(server_json.read_text())["token"] if server_json.exists() else ""
    return proc, token


@dataclass
class CellReport:
    family: str
    ref: str
    tier: str
    cold_s: float
    gen_s: float
    tool_calls: int
    answer: str
    error: str
    ok: bool


def _warmup_cold_s(base: str, headers: dict[str, str], model: str) -> float:
    """Time a 1-token completion: forces the cold model load. Returns seconds (-1 on error)."""
    t0 = time.monotonic()
    try:
        r = httpx.post(
            f"{base}/v1/chat/completions", headers=headers,
            json={"model": model, "messages": [{"role": "user", "content": "hi"}],
                  "max_tokens": 1, "stream": False},
            timeout=_HEALTH_TIMEOUT_S,
        )
        return time.monotonic() - t0 if r.status_code == 200 else -1.0
    except httpx.HTTPError:
        return -1.0


def _run_prompt(base: str, headers: dict[str, str], model: str, prompt: str) -> CellReport:
    msgs = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}]
    tool_calls = 0
    t0 = time.monotonic()
    with httpx.Client() as client:
        for _hop in range(_HOP_CAP):
            try:
                r = client.post(
                    f"{base}/v1/chat/completions", headers=headers,
                    json={"model": model, "messages": msgs, "max_tokens": 700,
                          "stream": False, "tools": [_SEARCH_TOOL]},
                    timeout=300,
                )
            except httpx.HTTPError as exc:
                return CellReport(model, model, "", 0, time.monotonic() - t0, tool_calls,
                                  "", f"chat exception: {exc}", False)
            if r.status_code != 200:
                return CellReport(model, model, "", 0, time.monotonic() - t0, tool_calls,
                                  "", f"chat {r.status_code}: {r.text[:200]}", False)
            msg = r.json()["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                answer = (msg.get("content") or "").strip()
                gen_s = time.monotonic() - t0
                ok = bool(answer) and tool_calls > 0
                return CellReport(model, model, "", 0, gen_s, tool_calls, answer, "", ok)
            msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for tc in calls:
                tool_calls += 1
                try:
                    qa = json.loads(tc.get("function", {}).get("arguments") or "{}")
                except (ValueError, TypeError):
                    qa = {}
                try:
                    sr = client.get(f"{base}/api/search", headers=headers,
                                    params={"q": qa.get("query", "Godot Node"), "top_k": 5}, timeout=90)
                    items = sr.json() if sr.status_code == 200 else []
                    result = "\n".join(f"- {it.get('source','?')}: {str(it.get('text', it))[:300]}"
                                       for it in items[:5]) or "(no results)"
                except httpx.HTTPError as exc:
                    result = f"(search error: {exc})"
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", "0"), "content": result})
    return CellReport(model, model, "", 0, time.monotonic() - t0, tool_calls,
                      "(tool loop hit hop cap without a final answer)", "hop_cap", False)


def validate_cell(cell, args: argparse.Namespace) -> CellReport:
    if not args.no_pull:
        pull_ref = matrix._pull_ref_for(cell.ref)
        matrix._run_pull_with_group_kill(pull_ref)
        installed = matrix._installed_ref_for_repo(pull_ref)
        if installed is not None and installed != cell.ref:
            cell.ref = installed
    workspace = matrix.write_per_cell_workspace(cell.family, cell.ref)
    port = _free_port()
    subprocess.run(["pkill", "-f", "lilbee serve"], check=False)
    subprocess.run(["pkill", "-f", "llama-server"], check=False)
    time.sleep(2)
    proc, token = _boot_serve(workspace, port)
    base = f"http://127.0.0.1:{port}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        cold_s = _warmup_cold_s(base, headers, cell.ref)
        rep = _run_prompt(base, headers, cell.ref, matrix._TIER_PROMPTS[cell.tier])
        rep.family, rep.ref, rep.tier, rep.cold_s = cell.family, cell.ref, cell.tier, cold_s
        return rep
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        subprocess.run(["pkill", "-f", "lilbee serve"], check=False)
        subprocess.run(["pkill", "-f", "llama-server"], check=False)
        if cell.size_gb > 20:
            matrix.cleanup_cell_model(cell)


def _write_reports(reports: list[CellReport], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps([asdict(r) for r in reports], indent=2))
    lines = ["# Pre-record QA matrix: answer quality + timings", "",
             "Auto-PASS = non-empty answer via a real tool call. Quality (correct, "
             "grounded, complete) is asserted by READING each answer below.", ""]
    for r in reports:
        verdict = "AUTO-PASS" if r.ok else "FAIL"
        lines += [f"## {r.family} ({r.tier}) -- {verdict}",
                  f"- ref: `{r.ref}`",
                  f"- cold_s: {r.cold_s:.1f}  gen_s: {r.gen_s:.1f}  tool_calls: {r.tool_calls}",
                  (f"- error: {r.error}" if r.error else "- error: none"),
                  "", "### prompt", matrix._TIER_PROMPTS.get(r.tier, ""), "",
                  "### answer", "```", r.answer or "(empty)", "```", ""]
    (out / "report.md").write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", help="comma-separated allowlist")
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("prevalidate_out"))
    args = ap.parse_args()
    cells = [c for c in matrix.load_models(matrix.QA_DIR / "models.toml") if not c.skip]
    if args.families:
        wanted = {f.strip() for f in args.families.split(",")}
        cells = [c for c in cells if c.family in wanted]
    if not args.no_pull:
        matrix.ensure_embedding_model_pulled()
    reports: list[CellReport] = []
    for cell in cells:
        print(f"[prevalidate] {cell.family} ({cell.tier}) {cell.ref}", flush=True)
        try:
            rep = validate_cell(cell, args)
        except Exception as exc:  # noqa: BLE001
            rep = CellReport(cell.family, cell.ref, cell.tier, 0, 0, 0, "", f"{type(exc).__name__}: {exc}", False)
        reports.append(rep)
        print(f"  -> {'AUTO-PASS' if rep.ok else 'FAIL'} cold_s={rep.cold_s:.1f} "
              f"gen_s={rep.gen_s:.1f} tools={rep.tool_calls} {rep.error}", flush=True)
    _write_reports(reports, args.out)
    n_ok = sum(1 for r in reports if r.ok)
    print(f"\n{n_ok}/{len(reports)} auto-pass. READ {args.out}/report.md to assert answer quality "
          f"before recording any demo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
