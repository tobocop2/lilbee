#!/usr/bin/env python3
"""Embed the canaries and a retrieval corpus on one llama-server build, for each GGUF and scenario.

Standard library only, so that it runs on a host with no lilbee install. The request body and the
server flags copy what lilbee sends for an embed role at the pinned base. Writes one JSON file per
engine, model and scenario. Each file holds the raw base64 float32 vectors and the proof of the
device that served them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CTX = 512
BIG_CTX = 2048
DOC_PREFIX = {"nomic": "search_document: "}
QUERY_PREFIX = {"nomic": "search_query: "}
PORT_WAIT_S = 180
REQUEST_TIMEOUT_S = 900


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class Server:
    """One llama-server process, stopped on exit, with its log kept for the device proof."""

    def __init__(self, binary: str, model: Path, flags: list[str], log: Path, env: dict[str, str]):
        self.port = free_port()
        self.log = log
        argv = [binary, "--model", str(model), "--host", "127.0.0.1", "--port", str(self.port)]
        argv += ["--embeddings", "--cont-batching", *flags]
        self.argv = argv
        with log.open("w", encoding="utf-8") as sink:
            self.proc = subprocess.Popen(
                argv,
                stdout=sink,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
                cwd=str(log.parent),
            )

    def wait_ready(self) -> None:
        deadline = time.time() + PORT_WAIT_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited rc={self.proc.returncode}")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/health", timeout=3
                ) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                time.sleep(0.5)
        raise RuntimeError("server not ready")

    def embed(self, texts: list[str]) -> list[str]:
        body = {
            "model": "x",
            "input": texts,
            "embd_normalize": -1,
            "encoding_format": "base64",
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/embeddings",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            data = json.load(resp)["data"]
        return [item["embedding"] for item in data]

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def device_proof(self) -> list[str]:
        wanted = re.compile(
            r"(offload|using device|device_|GPU name|system_info|n_threads =|pooling type|"
            r"Metal|Vulkan|CUDA|buffer size)"
        )
        lines: list[str] = []
        with self.log.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if wanted.search(line):
                    lines.append(line.strip()[:220])
        return lines[:40]


def scenarios(kind: str) -> list[tuple[str, list[str], str]]:
    """Name, extra flags, and the way the canaries are sent."""
    gpu = "0" if kind == "cpu" else "999"
    base = ["--n-gpu-layers", gpu, "--parallel", "1"]
    sized = ["--ctx-size", str(CTX), "--batch-size", str(CTX), "--ubatch-size", str(CTX)]
    big = ["--ctx-size", str(BIG_CTX * 4), "--batch-size", str(BIG_CTX), "--ubatch-size", str(BIG_CTX)]
    rows = [
        ("base", base + sized, "batch"),
        ("restart", base + sized, "batch"),
        ("single", base + sized, "single"),
        ("reversed", base + sized, "reversed"),
        ("repeat3", base + sized, "repeat3"),
        ("parallel4", ["--n-gpu-layers", gpu, "--parallel", "4", *big], "batch"),
        ("threads1", ["--threads", "1", *base, *sized], "batch"),
        ("ctxbig", ["--n-gpu-layers", gpu, "--parallel", "1", *big], "batch"),
    ]
    rows.append(("proof", ["-lv", "4", *base, *sized], "batch"))
    if kind != "cpu":
        rows.append(("ngl0", ["--n-gpu-layers", "0", "--parallel", "1", *sized], "batch"))
        rows.append(("ngl3", ["--n-gpu-layers", "3", "--parallel", "1", *sized], "batch"))
    return rows


def default_pooling(stem: str) -> list[str]:
    return ["--pooling", "last"] if stem.startswith("Qwen3") else []


def wrong_pooling(stem: str) -> list[str]:
    return ["--pooling", "mean"] if stem.startswith("Qwen3") else ["--pooling", "cls"]


def family(stem: str) -> str:
    for name in ("nomic", "MiniLM", "bge", "Qwen3"):
        if name in stem:
            return name
    raise ValueError(stem)


def send(server: Server, texts: list[str], how: str) -> dict[str, object]:
    if how == "batch":
        return {"vectors": server.embed(texts)}
    if how == "single":
        return {"vectors": [server.embed([t])[0] for t in texts]}
    if how == "reversed":
        return {"vectors": list(reversed(server.embed(list(reversed(texts)))))}
    if how == "repeat3":
        server.embed(texts)
        server.embed(texts)
        return {"vectors": server.embed(texts)}
    raise ValueError(how)


def run_one(args, stem, model, name, flags, how, texts, corpus):
    target = args.out / f"{args.engine}__{stem}__{name}.json"
    if target.exists():
        print(f"HAVE {target.name}", flush=True)
        return
    log = args.out / f"{args.engine}__{stem}__{name}.log"
    env = dict(os.environ)
    if args.libdir:
        env["LD_LIBRARY_PATH"] = args.libdir
        env["DYLD_LIBRARY_PATH"] = args.libdir
    server = Server(args.server, model, flags, log, env)
    started = time.time()
    record: dict[str, object] = {
        "engine": args.engine,
        "kind": args.kind,
        "model": stem,
        "scenario": name,
        "how": how,
        "argv": server.argv,
    }
    try:
        server.wait_ready()
        record.update(send(server, texts, how))
        if corpus and name == "base":
            fam = family(stem)
            doc = DOC_PREFIX.get(fam, "")
            qry = QUERY_PREFIX.get(fam, "")
            docs = corpus["docs"][: args.corpus_limit]
            vectors: list[str] = []
            for start in range(0, len(docs), 16):
                vectors += server.embed([doc + t for t in docs[start : start + 16]])
            record["corpus"] = vectors
            record["queries"] = server.embed([qry + q for q in corpus["queries"]])
    except (RuntimeError, urllib.error.URLError, OSError) as exc:
        record["error"] = repr(exc)
    finally:
        server.stop()
    record["seconds"] = round(time.time() - started, 1)
    record["device_proof"] = server.device_proof()
    target.write_text(json.dumps(record), encoding="utf-8")
    state = "ERROR " + str(record["error"])[:120] if "error" in record else "ok"
    print(f"RUN {target.name} {record['seconds']}s {state}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--kind", required=True, choices=["cpu", "gpu"])
    parser.add_argument("--server", required=True)
    parser.add_argument("--libdir", default="")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--canaries", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--corpus-limit", type=int, default=400)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default="", help="comma list of scenarios")
    parser.add_argument("--match", default="", help="regex on the model stem")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    texts = json.loads(args.canaries.read_text(encoding="utf-8"))
    corpus = json.loads(args.corpus.read_text(encoding="utf-8")) if args.corpus else None
    only = {s for s in args.only.split(",") if s}
    for model in sorted(args.models.glob("*.gguf")):
        stem = model.stem
        if args.match and not re.search(args.match, stem):
            continue
        for name, flags, how in scenarios(args.kind):
            if (only and name not in only) or (name == "proof" and not only):
                continue
            run_one(args, stem, model, name, flags + default_pooling(stem), how, texts, corpus)
        if not only or "wrongpool" in only:
            flags = ["--n-gpu-layers", "0" if args.kind == "cpu" else "999", "--parallel", "1"]
            flags += ["--ctx-size", str(CTX), "--batch-size", str(CTX), "--ubatch-size", str(CTX)]
            run_one(args, stem, model, "wrongpool", flags + wrong_pooling(stem), "batch", texts, None)
    print("MEASURE-LOOP-END", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
