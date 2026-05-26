"""Probe ``llama-server --jinja`` native tool-call coverage across lilbee's roster.

For each model: download the GGUF, launch ``llama-server --jinja``, POST a
``/v1/chat/completions`` request carrying a ``tools`` array, and check whether the
response comes back with a structured ``tool_calls`` (llama.cpp's own parser
working) versus the call leaking into ``content`` as text (parser missed it).

No lilbee, no opencode, no MCP. This is a direct capability probe of the bundled
llama.cpp server, to decide whether the multi-GPU fleet (and possibly everything)
can rely on native parsing instead of lilbee's hand-rolled ``response_parser``.

Usage (on a GPU host, after building llama-server):
    python tools/qa/llama_server_toolcall/probe.py --server-bin /path/to/llama-server
    python tools/qa/llama_server_toolcall/probe.py --families qwen3,gpt-oss --keep
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from models import ROSTER, ModelSpec  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"

# A lilbee_search-shaped tool. The model only has to decide to CALL it; we never
# run it (there is no backend). Emitting a structured call is the whole test.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lilbee_search",
            "description": "Search the indexed documentation corpus and return matching chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "the search query"},
                    "top_k": {"type": "integer", "description": "max number of results"},
                },
                "required": ["query"],
            },
        },
    }
]
PROMPT = (
    "Use the lilbee_search tool to find documentation about the chat worker, "
    "then tell me what you searched for."
)


@dataclass
class ProbeResult:
    family: str
    verdict: str  # PASS | FAIL | LOAD_FAIL | SKIP
    detail: str
    tool_name: str = ""
    tool_args: str = ""
    content_excerpt: str = ""
    load_s: float = 0.0
    probe_s: float = 0.0
    inprocess: str = ""


def _find_server_bin(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("LILBEE_LLAMA_SERVER")
    if env:
        return env
    repo = Path(__file__).resolve().parents[3]
    bundled = repo / "packaging/llama-server-wheel/lilbee_llama_server/bin/llama-server"
    if bundled.exists():
        return str(bundled)
    found = shutil.which("llama-server")
    if found:
        return found
    raise SystemExit("llama-server not found; pass --server-bin or set LILBEE_LLAMA_SERVER")


def _download(spec: ModelSpec) -> Path:
    """Fetch every shard matching the spec; return the first-shard path on disk."""
    from huggingface_hub import snapshot_download

    snap = snapshot_download(
        repo_id=spec.repo,
        allow_patterns=list(spec.allow_patterns),
        token=os.environ.get("HF_TOKEN"),
    )
    path = Path(snap) / spec.gguf
    if not path.exists():
        raise FileNotFoundError(f"{spec.family}: expected {path} after download")
    return path


def _borrow_template(spec: ModelSpec) -> Path | None:
    """Return a path to a chat template borrowed from *template_repo*, or None."""
    if spec.template_repo is None:
        return None
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    try:
        return Path(hf_hub_download(spec.template_repo, "chat_template.jinja", token=token))
    except Exception as exc:  # noqa: BLE001
        # No standalone jinja file: pull the template out of tokenizer_config.json.
        try:
            cfg_path = hf_hub_download(spec.template_repo, "tokenizer_config.json", token=token)
        except Exception:  # noqa: BLE001
            print(f"  [{spec.family}] could not borrow template: {exc}")
            return None
        cfg = json.loads(Path(cfg_path).read_text())
        tmpl = cfg.get("chat_template")
        if not isinstance(tmpl, str):
            return None
        out = RESULTS_DIR / f"{spec.family}.chat_template.jinja"
        out.write_text(tmpl)
        return out


def _launch(server_bin: str, gguf: Path, template: Path | None, port: int) -> subprocess.Popen[bytes]:
    argv = [
        server_bin,
        "--jinja",
        "-m", str(gguf),
        "-ngl", "999",
        "--host", "127.0.0.1",
        "--port", str(port),
        "-c", "8192",
        "--no-webui",
    ]
    if template is not None:
        argv += ["--chat-template-file", str(template)]
    log = (RESULTS_DIR / "_server.log").open("wb")
    return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def _wait_healthy(port: int, proc: subprocess.Popen[bytes], timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            if httpx.get(url, timeout=3).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(2)
    return False


def _probe(port: int, model_name: str) -> dict[str, Any]:
    resp = httpx.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [{"role": "user", "content": PROMPT}],
            "tools": TOOLS,
            "tool_choice": "auto",
            "max_tokens": 512,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def _kill(proc: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def _evaluate(family: str, inprocess: str, body: dict[str, Any]) -> ProbeResult:
    msg = body.get("choices", [{}])[0].get("message", {})
    tool_calls = msg.get("tool_calls") or []
    content = (msg.get("content") or "")[:400]
    if tool_calls:
        fn = tool_calls[0].get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "")
        args_ok = True
        if isinstance(raw_args, str):
            try:
                json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                args_ok = False
        if name and args_ok:
            return ProbeResult(family, "PASS", "native tool_calls emitted", name,
                               str(raw_args)[:200], content, inprocess=inprocess)
        return ProbeResult(family, "FAIL", f"tool_calls present but name/args bad (name={name!r})",
                           name, str(raw_args)[:200], content, inprocess=inprocess)
    # No structured tool_calls. Did the model emit the call as text in content?
    looks_like_call = any(m in content for m in ("lilbee_search", "tool_call", "<|", "functions.", "```json"))
    detail = "emitted call as TEXT, native parser missed it" if looks_like_call else "no tool call at all (model declined)"
    return ProbeResult(family, "FAIL", detail, content_excerpt=content, inprocess=inprocess)


def run_one(spec: ModelSpec, server_bin: str, port: int, keep: bool) -> ProbeResult:
    print(f"[{spec.family}] downloading {spec.repo}")
    try:
        gguf = _download(spec)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult(spec.family, "SKIP", f"download failed: {exc}", inprocess=spec.inprocess)
    template = _borrow_template(spec)
    if spec.template_repo and template is None:
        print(f"  [{spec.family}] WARNING no template borrowed; --jinja may fail")
    print(f"[{spec.family}] launching llama-server (template={'yes' if template else 'embedded'})")
    t0 = time.time()
    proc = _launch(server_bin, gguf, template, port)
    healthy = _wait_healthy(port, proc, timeout_s=420)
    load_s = time.time() - t0
    try:
        if not healthy:
            tail = (RESULTS_DIR / "_server.log").read_text(errors="replace")[-600:]
            return ProbeResult(spec.family, "LOAD_FAIL", f"server never healthy ({load_s:.0f}s); log tail: {tail}",
                               load_s=load_s, inprocess=spec.inprocess)
        print(f"[{spec.family}] probing tool call")
        t1 = time.time()
        try:
            body = _probe(port, spec.gguf)
        except Exception as exc:  # noqa: BLE001
            return ProbeResult(spec.family, "FAIL", f"probe request error: {exc}",
                               load_s=load_s, probe_s=time.time() - t1, inprocess=spec.inprocess)
        result = _evaluate(spec.family, spec.inprocess, body)
        result.load_s = load_s
        result.probe_s = time.time() - t1
        (RESULTS_DIR / f"{spec.family}.json").write_text(
            json.dumps({"result": asdict(result), "raw": body}, indent=2)
        )
        return result
    finally:
        _kill(proc)
        if not keep:
            from huggingface_hub import scan_cache_dir

            try:
                info = scan_cache_dir()
                for repo in info.repos:
                    if repo.repo_id == spec.repo:
                        shutil.rmtree(repo.repo_path, ignore_errors=True)
            except Exception:  # noqa: BLE001
                pass


def write_report(results: list[ProbeResult]) -> None:
    lines = [
        "# llama-server --jinja native tool-call coverage",
        "",
        "| Family | Native verdict | Detail | In-process (lilbee parser) |",
        "|--------|----------------|--------|----------------------------|",
    ]
    for r in results:
        lines.append(f"| {r.family} | {r.verdict} | {r.detail[:80]} | {r.inprocess} |")
    (RESULTS_DIR / "coverage.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-bin", default=None)
    ap.add_argument("--families", default=None, help="comma-separated subset")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--keep", action="store_true", help="don't delete GGUFs after each cell")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    server_bin = _find_server_bin(args.server_bin)
    print(f"llama-server: {server_bin}")
    wanted = set(args.families.split(",")) if args.families else None
    specs = [s for s in ROSTER if not s.multi_gpu_only and (wanted is None or s.family in wanted)]

    results: list[ProbeResult] = []
    for spec in specs:
        r = run_one(spec, server_bin, args.port, args.keep)
        print(f"[{spec.family}] -> {r.verdict}: {r.detail[:90]}")
        results.append(r)
        write_report(results)  # rewrite after each so a crash still leaves partials

    passed = sum(1 for r in results if r.verdict == "PASS")
    print(f"\n{passed}/{len(results)} families emit native tool_calls")


if __name__ == "__main__":
    main()
