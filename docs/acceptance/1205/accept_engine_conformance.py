"""Acceptance rows for #1205: the engine integration document and its conformance suite.

Run: PYTHONPATH=<checkout>/src python -B accept_engine_conformance.py [--base SHA] [--work DIR]
Prints one line per cell: ROW <id> <GREEN|RED> <detail>. Exit 1 when any row is red.

The rule has four parts. Each part has rows.
  DOC    docs/engine-integration.md states every requirement by tier, with its evidence rows.
  MAP    every one of the 55 rows of reports/spike-1188-failures.md maps to a requirement
         or to "none" with a reason; the document carries the same map.
  SUITE  tests/conformance holds the harness. Contract the rows pin (names below are fixed):
           tests.conformance.harness.REQUIREMENTS            dict id -> object with .tier .ci
           tests.conformance.harness.run(url, *, claims, models, health_path=None,
                                         quality=False, timeout=60.0) -> {id: verdict}
           tests.conformance.harness.run_managed(binary, *, pin, timeout=30.0) -> {id: verdict}
         verdict is "pass", "fallback", "fail" or "skip". claims is a set from
         {"chat","embed","rerank","vision","reasoning"}. models maps a claim to a model id.
         quality=False runs the protocol rows only; quality=True adds the rows that compare
         content (determinism, rerank order, the image answer). pytest options:
         --engine-url, --claims, --engine-pin. Lilbee-side rows (L*) are pytest tests in
         tests/conformance/test_lilbee_behaviour.py, strict xfail, one per owning item.
  FAKE   the oracle. This script serves its own fake engines from the measured rows (never the
         suite's fake) and states the expected verdict for each (profile, requirement) cell.
         Profiles replay the spike rows; mutants break one thing in the reference engine.
         The "fid" rows check each fake against the spike row it copies, with raw HTTP, so a red
         verdict row cannot be a bad fake.
Context that must not change: nothing under src/ and no existing test file differs from the base.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE_DEFAULT = "c9a8ff550c487444b3dbbeb626ab964823c10154"
ROWS: list[tuple[str, bool, str]] = []


def row(rid: str, ok: bool, detail: str = "") -> None:
    ROWS.append((rid, bool(ok), detail))
    print(f"ROW {rid} {'GREEN' if ok else 'RED'} {detail}")


# ---------------------------------------------------------------- requirements (the oracle's table)
# id: (tier, kind, spike rows, ci_no_model).  E = what the engine serves, M = managed engine,
# L = what lilbee does about it (pytest, strict xfail until the owner item lands).
REQS: dict[str, tuple[int, str, tuple[int, ...], bool]] = {
    "E1": (1, "E", (1, 12, 22, 38), True),
    "E2": (1, "E", (36, 37), True),
    "E3": (1, "E", (13, 34, 38), True),
    "E4": (1, "E", (6, 26, 36), True),
    "E5": (1, "E", (36,), True),
    "E6": (1, "E", (6, 26, 48, 50, 51), True),
    "E7": (1, "E", (9, 29, 48, 55), True),
    "E8": (1, "E", (10, 17, 30, 32, 44, 46, 54), True),
    "E9": (1, "E", (10, 19, 20, 21, 32, 33, 46, 47), True),
    "E10": (1, "E", (), True),
    "E11": (1, "E", (), False),
    "E12": (1, "E", (8, 28, 43, 52), True),
    "E13": (1, "E", (33,), True),
    "M1": (2, "M", (), True),
    "M2": (2, "M", (5, 16, 25, 42), True),
    "M3": (2, "M", (4, 16, 25, 37, 41), True),
    "M4": (2, "M", (11,), True),
    "M5": (2, "M", (), True),
    "M6": (2, "M", (8,), True),
    "L1": (1, "L", (2, 14, 15, 23, 31, 39, 45, 53), True),
    "L2": (1, "L", (3, 16, 24, 40), True),
    "L3": (1, "L", (6,), True),
    "L4": (1, "L", (43, 52), True),
    "L5": (1, "L", (36,), True),
    "L6": (1, "L", (19, 20, 32, 33, 46, 47), True),
    "L7": (1, "L", (13,), True),
    "L8": (1, "L", (37,), True),
}
# Rows that need no requirement, with the reason the document must print.
NONE_ROWS = {
    7: "prompt-progress frames are an llama-server extra; the client works without them",
    18: "an engine that truncates an oversize embedding input itself breaks nothing",
    27: "works",
    35: "works",
    49: "prompt-progress frames are an llama-server extra; the client works without them",
}
# Rows whose only requirement is a reference to a row above (work, or not probed).
ROW_REFS = {r for _, (_, _, rows, _) in REQS.items() for r in rows}


def row_map_problems() -> list[str]:
    return [f"row {n} has neither a requirement nor a reason" for n in range(1, 56)
            if n not in ROW_REFS and n not in NONE_ROWS]


# ---------------------------------------------------------------- fake engine (tier 1)
CTX_CHARS = 100_000
DEFAULT = dict(
    health=200, live=404, models=200, models_ids=True, chat="ok", stream="ok", tools="native",
    reasoning="reasoning_content", reasoning_on=True, embed="ok", embed_norm=True, rerank="v1",
    vision="ok", overflow="400-ctx", deterministic=True, order_ok=True, junk_embed=False,
    dim=8,
)


def png(rgb: tuple[int, int, int], size: int = 8) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def colour_of(data_url: str) -> str:
    blob = base64.b64decode(data_url.split(",", 1)[1])
    pos = blob.index(b"IDAT")
    length = struct.unpack(">I", blob[pos - 4:pos])[0]
    r, g, b = zlib.decompress(blob[pos + 4:pos + 4 + length])[1:4]
    return "red" if r > 200 and g < 60 and b < 60 else "other"


def vec(text: str, dim: int, norm: bool) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    v = [(h[i % 32] - 127.5) / 127.5 for i in range(dim)]
    if norm:
        n = sum(x * x for x in v) ** 0.5
        v = [x / n for x in v]
    return v


def make_handler(cfg: dict):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_a):
            pass

        def send(self, code: int, obj=None, raw: bytes | None = None, ctype="application/json"):
            body = raw if raw is not None else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            envelope = {"error": {"message": "The requested endpoint does not exist", "path": self.path}}
            if self.path == "/health":
                return self.send(cfg["health"], {"status": "ok"} if cfg["health"] == 200 else envelope)
            if self.path == "/live":
                return self.send(cfg["live"], {"status": "ok"} if cfg["live"] == 200 else envelope)
            if self.path == "/v1/models":
                if cfg["models"] != 200:
                    return self.send(cfg["models"], raw=b"")
                data = [{"id": "m-chat"}, {"id": "m-embed"}, {"id": "m-rerank"}] if cfg["models_ids"] else []
                return self.send(200, {"object": "list", "data": data})
            self.send(404, raw=b"Not Found", ctype="text/plain")

        def do_POST(self):
            req = self.body()
            path = self.path
            if path == "/v1/chat/completions":
                return self.chat(req)
            if path == "/v1/embeddings" and cfg["embed"] != "404":
                return self.embed(req)
            if path == "/v1/rerank" and cfg["rerank"] in ("v1", "zero", "badkey", "501"):
                return self.rerank(req)
            if path == "/rerank" and cfg["rerank"] == "tei":
                texts = req["texts"]
                return self.send(200, [{"index": i, "score": 0.5} for i in range(len(texts))])
            self.send(404, raw=b"Not Found", ctype="text/plain")

        def chat(self, req):
            prompt = json.dumps(req["messages"])
            if cfg["chat"] == "503":
                err = {"error": {"message": "The service is temporarily unavailable.",
                                 "type": "server_error", "code": "service_unavailable"}}
                if req.get("stream"):
                    raw = f"data: {json.dumps(err)}\n\ndata: [DONE]\n\n".encode()
                    return self.send(200, raw=raw, ctype="text/event-stream")
                return self.send(503, err)
            if len(prompt) > CTX_CHARS:
                mode = cfg["overflow"]
                if mode == "hang":
                    time.sleep(6)
                if mode == "500":
                    return self.send(500, {"error": {"message": "boom"}})
                if mode == "hang":
                    return self.send(200, {"choices": [{"message": {"content": "late"}}]})
                msg = ("request (300000 tokens) exceeds the available context size (40960 tokens)"
                       if mode == "400-ctx" else
                       "Prompt sequence length is greater than 40960, perhaps consider using truncate_sequence?")
                return self.send(400, {"error": {"message": msg, "type": "invalid_request_error"}})
            message: dict = {"role": "assistant", "content": self.answer(req, prompt)}
            parts = [p for m in req["messages"] if isinstance(m.get("content"), list) for p in m["content"]]
            images = [p for p in parts if p.get("type") == "image_url"]
            if images:
                if cfg["vision"] == "400":
                    return self.send(400, {"error": {"message": "images are not supported"}})
                message["content"] = "I see " + (colour_of(images[0]["image_url"]["url"])
                                                 if cfg["vision"] == "ok" else "nothing")
            finish = "stop"
            if req.get("tools"):
                name = req["tools"][0]["function"]["name"]
                if cfg["tools"] == "native":
                    message = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "c1", "type": "function",
                         "function": {"name": name, "arguments": json.dumps({"city": "Paris"})}}]}
                    finish = "tool_calls"
                elif cfg["tools"] == "bare":
                    message["content"] = json.dumps({"name": name, "arguments": {"city": "Paris"}})
            if cfg["reasoning_on"] and req.get("thinking", True) and "content" in message and message["content"]:
                if cfg["reasoning"] == "inline":
                    message["content"] = "<think>hm</think>" + message["content"]
                elif cfg["reasoning"]:
                    message[cfg["reasoning"]] = "hm"
            if req.get("stream"):
                return self.stream(message, finish)
            if cfg["chat"] == "nochoices":
                return self.send(200, {"id": "x"})
            self.send(200, {"choices": [{"index": 0, "message": message, "finish_reason": finish}],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})

        def answer(self, req, prompt):
            if req.get("temperature") == 0 and req.get("seed") is not None and cfg["deterministic"]:
                return "ans-" + hashlib.sha1((prompt + str(req["seed"])).encode()).hexdigest()[:6]
            return "ans-" + os.urandom(3).hex() if not cfg["deterministic"] else "ok"

        def stream(self, message, finish):
            frames = []
            if message.get("tool_calls"):
                frames.append({"choices": [{"index": 0, "delta": {"tool_calls": [dict(message["tool_calls"][0], index=0)]}}]})
            else:
                for key in ("reasoning_content", "reasoning", cfg["reasoning"] or ""):
                    if key and key in message:
                        frames.append({"choices": [{"index": 0, "delta": {key: message[key]}}]})
                        break
                if message.get("content"):
                    frames.append({"choices": [{"index": 0, "delta": {"content": message["content"]}}]})
            frames.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            frames.append({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
            raw = "".join(f"data: {json.dumps(f)}\n\n" for f in frames)
            if cfg["stream"] == "no-done":
                pass
            elif cfg["stream"] == "bad-sse":
                raw = "".join(json.dumps(f) + "\n" for f in frames)
            else:
                raw += "data: [DONE]\n\n"
            self.send(200, raw=raw.encode(), ctype="text/event-stream")

        def embed(self, req):
            inputs = req["input"] if isinstance(req["input"], list) else [req["input"]]
            if cfg["embed"] == "count":
                inputs = inputs[:-1] or inputs
            out = []
            for i, text in enumerate(inputs):
                dim = cfg["dim"] + (1 if cfg["embed"] == "drift" and i else 0)
                v = [0.0] * dim if cfg["embed"] == "zero" or (cfg["embed"] == "zero_first" and i == 0) else vec(text, dim, cfg["embed_norm"])
                if req.get("encoding_format") == "base64":
                    v = base64.b64encode(struct.pack(f"<{len(v)}f", *v)).decode()
                out.append({"object": "embedding", "index": i, "embedding": v})
            self.send(200, {"object": "list", "data": out})

        def rerank(self, req):
            if cfg["rerank"] == "501":
                return self.send(501, {"error": {"message": "This server does not support reranking"}})
            q = set(req["query"].lower().split())
            res = []
            for i, d in enumerate(req["documents"]):
                w = set(d.lower().split())
                s = len(q & w) / max(len(q | w), 1)
                if not cfg["order_ok"]:
                    s = 1 - s
                if cfg["rerank"] == "zero":
                    s = 0.0
                key = "score" if cfg["rerank"] == "badkey" else "relevance_score"
                res.append({"index": i, key: s})
            self.send(200, {"results": sorted(res, key=lambda r: -r.get(next(k for k in r if k != "index"), 0))})

    return H


class Fake:
    def __init__(self, **over):
        self.cfg = {**DEFAULT, **over}
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.cfg))
        self.srv.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def http(method: str, url: str, body: dict | None = None, timeout: float = 5.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


# ---------------------------------------------------------------- expected verdicts (hand-stated)
ALL = {"chat", "embed", "rerank", "vision", "reasoning"}
MODELS = {"chat": "m-chat", "embed": "m-embed", "rerank": "m-rerank", "vision": "m-chat"}


def default_expect(claims: set[str], quality: bool) -> dict[str, str | None]:
    exp = {k: "pass" for k in ("E1", "E2", "E3", "E13")}
    for k, need in (("E4", "chat"), ("E5", "chat"), ("E7", "chat"), ("E12", "chat"), ("E8", "embed"),
                    ("E9", "rerank"), ("E10", "vision"), ("E6", "reasoning")):
        exp[k] = "pass" if need in claims else "skip"
    exp["E11"] = "pass" if quality and "chat" in claims else "skip"
    return exp


# (name, config override, claims, {req: verdict or None=not asserted}, {req: verdict} extra when quality)
PROFILES = [
    ("ref", {}, ALL, {}, {}),
    # constant scores: with quality on the order check also fails, so the row must run with it off
    ("rerank_zero", dict(rerank="zero"), ALL, {"E9": "fail"}, {}),
    # rows 1, 6, 8, 9: mlx_lm.server; /v1/models not probed in the spike, so E3 is not asserted
    ("mlx_lm", dict(reasoning="reasoning", overflow="hang", vision="400"), {"chat", "reasoning"},
     {"E12": "fail", "E3": None}, {}),
    # rows 12, 13, 17: TEI embed; always normalized, no /v1/models
    ("tei_embed", dict(models=404), {"embed"}, {"E3": "fallback"}, {}),
    # rows 19 to 21: TEI rerank answers only the native route
    ("tei_rerank", dict(models=404, rerank="tei"), {"rerank"},
     {"E3": "fallback", "E9": "fail", "E2": "fail"}, {}),
    # rows 22 to 29, 33: Lemonade chat; /health is 404, /live is 200
    ("lemonade_chat", dict(health=404, live=200), {"chat", "reasoning"}, {"E1": "fallback"}, {}),
    ("lemonade_chat_declared", dict(health=404, live=200), {"chat", "reasoning"}, {}, {}),
    ("lemonade_embed", dict(health=404, live=200), {"embed"}, {"E1": "fallback"}, {}),
    ("lemonade_rerank", dict(health=404, live=200), {"rerank"}, {"E1": "fallback"}, {}),
    # row 32: the reranker model on the embeddings route answers 200 with zeros
    ("lemonade_reranker_as_embed", dict(health=404, live=200, embed="zero"), {"embed"},
     {"E1": "fallback", "E8": "fail"}, {}),
    # rows 49 to 55 (CPU): row 55, tools, is not asserted; the suite measures it
    ("mistralrs_cpu", dict(overflow="400-seq", reasoning="reasoning_content"), {"chat", "embed", "reasoning"},
     {"E7": None}, {}),
    # rows 36 to 38, 44 (CUDA): every chat call is 503, embed works
    ("mistralrs_cuda", dict(chat="503"), {"chat", "embed"},
     {"E2": "fail", "E4": "fail", "E5": "fail", "E7": "fail", "E12": "fail"}, {"E11": "fail"}),
]
MUTANTS = [
    ("health_500_models_ok", dict(health=500), ALL, {"E1": "fallback"}, {}),
    ("health_500_models_404", dict(health=500, models=404), ALL, {"E1": "fail", "E3": "fallback"}, {}),
    ("models_empty", dict(models_ids=False), ALL, {"E3": "fail"}, {}),
    ("chat_nochoices", dict(chat="nochoices"), ALL, {"E4": "fail", "E2": "fail", "E7": "fail", "E6": "fail",
                                                      "E10": "fail"}, {"E11": "fail"}),
    ("stream_no_done", dict(stream="no-done"), ALL, {"E5": "fail"}, {}),
    ("stream_bad_sse", dict(stream="bad-sse"), ALL, {"E5": "fail"}, {}),
    ("tools_none", dict(tools="none"), ALL, {"E7": "fail"}, {}),
    ("tools_bare_json", dict(tools="bare"), ALL, {"E7": "fallback"}, {}),
    ("reasoning_alt_field", dict(reasoning="reasoning"), ALL, {}, {}),
    ("reasoning_inline", dict(reasoning="inline"), ALL, {}, {}),
    ("reasoning_thoughts", dict(reasoning="thoughts"), ALL, {"E6": "fail"}, {}),
    ("embed_zero", dict(embed="zero"), ALL, {"E8": "fail"}, {}),
    ("embed_zero_first", dict(embed="zero_first"), ALL, {"E8": "fail"}, {}),
    ("embed_count", dict(embed="count"), ALL, {"E8": "fail"}, {}),
    ("embed_dim_drift", dict(embed="drift"), ALL, {"E8": "fail"}, {}),
    ("embed_unnormalized", dict(embed_norm=False), ALL, {}, {}),
    ("embed_404", dict(embed="404"), ALL, {"E8": "fail", "E2": "fail"}, {}),
    ("rerank_501", dict(rerank="501"), ALL, {"E9": "fail", "E2": "fail"}, {}),
    ("rerank_bad_key", dict(rerank="badkey"), ALL, {"E9": "fail"}, {}),
    ("rerank_reversed", dict(order_ok=False), ALL, {}, {"E9": "fail"}),
    ("vision_blind", dict(vision="blind"), ALL, {}, {"E10": "fail"}),
    ("vision_400", dict(vision="400"), ALL, {"E10": "fail"}, {}),
    ("nondeterministic", dict(deterministic=False), ALL, {}, {"E11": "fail"}),
    ("overflow_hang", dict(overflow="hang"), ALL, {"E12": "fail"}, {}),
    ("overflow_500", dict(overflow="500"), ALL, {"E12": "fail"}, {}),
    ("overflow_seq_message", dict(overflow="400-seq"), ALL, {}, {}),
    ("unclaimed_junk_embed", dict(embed="zero"), {"chat"}, {"E13": "fail"}, {}),
    ("unclaimed_embed_ok", {}, {"chat"}, {}, {}),
]


def import_harness(proto: str | None = None):
    if proto:  # a throwaway harness file, to prove the expected table can be met
        spec = importlib.util.spec_from_file_location("proto_harness", proto)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    sys.path.insert(0, str(ROOT))
    try:
        return importlib.import_module("tests.conformance.harness")
    except Exception as exc:  # absent at the base
        return exc


def run_tier1(h) -> None:
    n = 0
    for quality_modes, group in ((("F", "T"), PROFILES), (("T",), MUTANTS)):
        for name, over, claims, exp_over, exp_q in group:
            for mode in quality_modes:
                quality = mode == "T"
                exp = default_expect(claims, quality)
                exp.update(exp_over)
                if quality:
                    exp.update(exp_q)
                health_path = "/live" if name.endswith("_declared") else None
                fake = Fake(**over)
                try:
                    got = None
                    if not isinstance(h, Exception):
                        got = h.run(fake.url, claims=set(claims), models=MODELS, health_path=health_path,
                                    quality=quality, timeout=3.0)
                finally:
                    fake.close()
                for rid, want in exp.items():
                    if want is None:
                        continue
                    n += 1
                    have = "no harness" if got is None else got.get(rid, "missing")
                    row(f"v.{name}.{mode}.{rid}", have == want, f"want {want} got {have}")


# ---------------------------------------------------------------- tier 2 (fake binary)
FAKE_BIN = r'''#!{py}
import json, signal, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
CFG = json.loads({cfg!r})
a = sys.argv[1:]
if "--help" in a:
    print("usage: engine " + " ".join(CFG["flags"])); sys.exit(CFG.get("help_rc", 0))
if "--list-devices" in a:
    if CFG["devices"] == "llama":
        print("Available devices:\n  CUDA0: NVIDIA Test GPU (8192 MiB, 8000 MiB free)"); sys.exit(0)
    sys.stderr.write("error: unexpected argument '--list-devices' found\n"); sys.exit(2)
if "--estimate" in a:
    if CFG["estimate"]:
        print("123456789"); sys.exit(0)
    sys.exit(1)
if "--port" in a:
    port = int(a[a.index("--port") + 1])
    if CFG["term"] == "ignore":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    class H(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_GET(self):
            self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0)); time.sleep(30)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
sys.exit(2)
'''
SHA = "a" * 64
GOOD_PIN = {
    "engine": "fake", "version": "1.2.3", "backend": None,
    "launch": ["{binary}", "--port", "{port}"],
    "platforms": {"linux-x86_64": {"url": "https://example.org/fake-linux.tar.gz", "sha256": SHA, "min_os": None},
                  "macos-arm64": {"url": "https://example.org/fake-mac.tar.gz", "sha256": SHA, "min_os": "14.0"}},
    "memory": {"sizing": ["{binary}", "--estimate", "{model}"], "rule": None},
    "licenses": ["MIT", "Apache-2.0", "BSD-3-Clause"],
}


def pin_with(**ch) -> dict:
    p = json.loads(json.dumps(GOOD_PIN))
    for k, v in ch.items():
        if k == "plat":
            p["platforms"].update(v)
        else:
            p[k] = v
    return p


def tier2_cases():
    nodev = dict(devices="usage")
    return [
        ("ref", {}, GOOD_PIN, {}),
        ("no_devices_pin_backend", nodev, pin_with(backend="metal"), {"M2": "fallback"}),
        ("no_devices_no_backend", nodev, GOOD_PIN, {"M2": "fail"}),
        ("launch_without_port", {}, pin_with(launch=["{binary}", "--serve"]), {"M1": "fail", "M6": "fail"}),
        ("no_help", dict(help_rc=2), GOOD_PIN, {"M1": "fail"}),
        ("estimate_fails", dict(estimate=False), GOOD_PIN, {"M3": "fail"}),
        ("memory_rule_only", dict(estimate=False), pin_with(memory={"sizing": None, "rule": "file_size"}),
         {"M3": "fallback"}),
        ("no_memory_rule", {}, pin_with(memory={"sizing": None, "rule": None}), {"M3": "fail"}),
        ("pin_no_sha", {}, pin_with(plat={"linux-x86_64": {"url": "https://e.org/x", "sha256": "", "min_os": None}}),
         {"M4": "fail"}),
        ("pin_http_url", {}, pin_with(plat={"linux-x86_64": {"url": "http://e.org/x", "sha256": SHA, "min_os": None}}),
         {"M4": "fail"}),
        ("pin_mac_no_floor", {}, pin_with(plat={"macos-arm64": {"url": "https://e.org/x", "sha256": SHA, "min_os": None}}),
         {"M4": "fail"}),
        ("pin_no_platforms", {}, pin_with(platforms={}), {"M4": "fail"}),
        ("licence_gpl", {}, pin_with(licenses=["MIT", "GPL-3.0-only"]), {"M5": "fail"}),
        ("licence_unknown", {}, pin_with(licenses=["MIT", ""]), {"M5": "fail"}),
        ("licence_none_listed", {}, pin_with(licenses=[]), {"M5": "fail"}),
        ("ignores_sigterm", dict(term="ignore"), GOOD_PIN, {"M6": "fail"}),
    ]


def run_tier2(h, work: Path) -> None:
    for name, over, pin, exp_over in tier2_cases():
        cfg = dict(flags=["--port", "--ctx-size", "--list-devices"], devices="llama", estimate=True, term="exit")
        cfg.update(over)
        binary = work / f"fake-{name}"
        binary.write_text(FAKE_BIN.format(py=sys.executable, cfg=json.dumps(cfg)))
        binary.chmod(0o755)
        exp = {k: "pass" for k in ("M1", "M2", "M3", "M4", "M5", "M6")}
        exp.update(exp_over)
        got = None
        if not isinstance(h, Exception):
            got = h.run_managed(binary, pin=pin, timeout=15.0)
        for rid, want in exp.items():
            have = "no harness" if got is None else got.get(rid, "missing")
            row(f"v2.{name}.{rid}", have == want, f"want {want} got {have}")


# ---------------------------------------------------------------- fidelity: the fakes copy the spike rows
def fidelity() -> None:
    f = Fake(health=404, live=200, models=404)
    try:
        s, b = http("GET", f.url + "/health")
        row("fid.row22", s == 404 and b"does not exist" in b, "Lemonade /health 404 JSON envelope")
        row("fid.row22b", http("GET", f.url + "/live")[0] == 200, "Lemonade /live 200")
        row("fid.row13", http("GET", f.url + "/v1/models")[0] == 404, "TEI /v1/models 404")
        row("fid.row2", http("POST", f.url + "/tokenize", {"content": "x"})[0] == 404, "no /tokenize")
    finally:
        f.close()
    f = Fake(chat="503")
    try:
        s, b = http("POST", f.url + "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]})
        row("fid.row36", s == 503 and b"service_unavailable" in b, "mistral.rs CUDA chat 503")
        s, b = http("GET", f.url + "/health")
        row("fid.row37", s == 200, "health answers while chat is down")
    finally:
        f.close()
    f = Fake(rerank="tei")
    try:
        s, b = http("POST", f.url + "/rerank", {"query": "q", "texts": ["a", "b"]})
        row("fid.row20", s == 200 and isinstance(json.loads(b), list), "TEI /rerank list of index/score")
        row("fid.row21", http("POST", f.url + "/v1/rerank", {"query": "q", "documents": ["a"]})[0] == 404, "no /v1/rerank")
    finally:
        f.close()
    f = Fake(overflow="400-seq")
    try:
        s, b = http("POST", f.url + "/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "w " * 60000}]})
        row("fid.row43", s == 400 and b"sequence length is greater than" in b, "mistral.rs overflow text")
    finally:
        f.close()
    f = Fake(reasoning="reasoning")
    try:
        s, b = http("POST", f.url + "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]})
        msg = json.loads(b)["choices"][0]["message"]
        row("fid.row6", "reasoning" in msg and "reasoning_content" not in msg, "mlx_lm names the field reasoning")
    finally:
        f.close()
    f = Fake(embed="zero")
    try:
        s, b = http("POST", f.url + "/v1/embeddings", {"input": ["a"], "encoding_format": "float"})
        row("fid.row32", s == 200 and set(json.loads(b)["data"][0]["embedding"]) == {0.0}, "zero vector, 200")
    finally:
        f.close()
    blob = "data:image/png;base64," + base64.b64encode(png((255, 0, 0))).decode()
    row("fid.png", colour_of(blob) == "red", "the generated image reads back red")


# ---------------------------------------------------------------- document, map and suite files
def doc_rows() -> None:
    doc = ROOT / "docs" / "engine-integration.md"
    if not doc.exists():
        row("doc.exists", False, "docs/engine-integration.md absent")
        for rid in REQS:
            row(f"doc.{rid}", False, "no document")
        for n in range(1, 56):
            row(f"map.row{n}", False, "no document")
        return
    text = doc.read_text(encoding="utf-8")
    row("doc.exists", True, str(len(text)) + " chars")
    t1, t2 = text.find("## Tier 1"), text.find("## Tier 2")
    row("doc.tier_order", 0 <= t1 < t2, "Tier 1 heading before Tier 2 heading")
    for rid, (tier, _kind, rows, _ci) in REQS.items():
        m = re.search(rf"^### {rid}\b.*?(?=^### |^## |\Z)", text, re.M | re.S)
        ok = m is not None
        detail = "section found"
        if m:
            sec = m.group(0)
            where = m.start()
            ok = (tier == 1 and t1 < where < t2) or (tier == 2 and where > t2)
            missing = [r for r in rows if not re.search(rf"\brow {r}\b", sec)]
            if "Evidence:" not in sec:
                ok, detail = False, "no Evidence line"
            elif missing:
                ok, detail = False, f"rows not cited: {missing}"
            elif not ok:
                detail = "section under the wrong tier"
        else:
            detail = "no section"
        row(f"doc.{rid}", ok, detail)
    e9 = re.search(r"^### E9\b.*?(?=^### |^## |\Z)", text, re.M | re.S)
    row("doc.rerank_shape", bool(e9 and "/v1/rerank" in e9.group(0) and "relevance_score" in e9.group(0)),
        "E9 names the route and the score field")
    for n in range(1, 56):
        hit = re.search(rf"^\|\s*{n}\s*\|[^\n]*\|", text, re.M)
        row(f"map.row{n}", bool(hit), "row in the failure-row table" if hit else "row not mapped in the document")
    cited = set(re.findall(r"`((?:src|tests|docs|tools)/[\w./-]+\.\w+)`", text))
    gone = sorted(p for p in cited if not (ROOT / p).exists())
    row("doc.paths_exist", not gone and bool(cited), f"cited {len(cited)} paths; missing {gone}")


def suite_rows(h) -> None:
    base = ROOT / "tests" / "conformance"
    for rel in ("__init__.py", "harness.py", "conftest.py", "test_lilbee_behaviour.py"):
        row(f"suite.file.{rel}", (base / rel).exists(), "exists" if (base / rel).exists() else "absent")
    goldens = sorted(p.stem for p in (base / "golden").glob("*.json")) if (base / "golden").is_dir() else []
    need = sorted(k for k, v in REQS.items() if v[1] in "EM")
    row("suite.golden_files", goldens == need, f"want {need} got {goldens}")
    if isinstance(h, Exception):
        row("suite.requirements_match", False, f"harness import failed: {type(h).__name__}")
    else:
        got = sorted(h.REQUIREMENTS)
        row("suite.requirements_match", got == need, f"want {need} got {got}")
        bad = [k for k in need if k not in h.REQUIREMENTS
               or getattr(h.REQUIREMENTS[k], "ci", None) is not REQS[k][3]
               or getattr(h.REQUIREMENTS[k], "tier", None) != REQS[k][0]]
        row("suite.tier_and_ci_flags", not bad, f"mismatch {bad}")
    lb = base / "test_lilbee_behaviour.py"
    ltxt = lb.read_text(encoding="utf-8") if lb.exists() else ""
    lids = sorted(k for k, v in REQS.items() if v[1] == "L")
    miss = [k for k in lids if not re.search(rf"\b{k}\b", ltxt)]
    row("suite.lilbee_rows", lb.exists() and not miss and "strict=True" in ltxt, f"L ids missing {miss}")
    env = {**os.environ, "UV_NO_SYNC": "1", "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/conformance", "--help", "-p", "no:cacheprovider"],
                       cwd=ROOT, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    opts = [o for o in ("--engine-url", "--claims", "--engine-pin") if o in r.stdout]
    row("suite.pytest_options", len(opts) == 3, f"found {opts}")


def unchanged_rows(base: str) -> None:
    def diff(*paths: str) -> list[str]:
        out = subprocess.run(["git", "diff", "--name-only", base, "HEAD", "--", *paths], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
        return out
    src = diff("src", "pyproject.toml", "uv.lock", "Makefile")
    row("same.src", not src or src == ["pyproject.toml"], f"changed {src}")
    tests = [p for p in diff("tests") if not p.startswith("tests/conformance/")]
    row("same.tests", not tests, f"changed outside tests/conformance {tests}")
    row("same.integration_stub", not diff("tests/integration"), "tests/integration untouched")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_DEFAULT)
    ap.add_argument("--work", default=None)
    ap.add_argument("--proto", default=None, help="a throwaway harness file, for the proof run")
    args = ap.parse_args()
    work = Path(args.work or tempfile.mkdtemp(prefix="accept1205."))
    work.mkdir(parents=True, exist_ok=True)
    for problem in row_map_problems():
        row("oracle.map", False, problem)
    row("oracle.map", not row_map_problems(), "all 55 spike rows are mapped in the oracle table")
    fidelity()
    h = import_harness(args.proto)
    doc_rows()
    suite_rows(h)
    run_tier1(h)
    run_tier2(h, work)
    unchanged_rows(args.base)
    red = [r for r in ROWS if not r[1]]
    print(f"SUMMARY inputs={len(ROWS)} green={len(ROWS) - len(red)} red={len(red)} open=0")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
