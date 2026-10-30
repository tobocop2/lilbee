"""Acceptance script: chat, embed and rerank reach a server preset by its name prefix.

Prints one line per cell: ROW <status> <section> <input> expected=<...> actual=<...>.
PASS and RED are verdicts. The expected column is typed here from rulings/routing.md
(Expected behaviour); the oracle is a fake OpenAI-compatible server that records each request,
and hand-typed tables. It does not read the registry under test.
Names the head must offer are listed in rulings/routing.md under "Names the script reads".
A row whose name is absent at the base reads actual=absent:<error> and is RED.
The script starts no model server: its fake is a stdlib HTTP handler on 127.0.0.1.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import re
import socket
import subprocess
import sys
import tempfile
import threading
from collections import Counter
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

REPO = Path(os.environ.get("ACCEPT_REPO") or Path(__file__).resolve().parents[2])
BASE_SHA = "76ce7cf886fa6ea3f0e50e47d64c61d476235a5b"
KEY = "SENTINEL-preset-key-41d2"
LLM_KEY = "SENTINEL-llm-api-key-9c07"
ROLES = ("chat", "embed", "rerank", "vision")
LOCAL = "local"
OK = "ok"
REFUSE = "refuse"

# ---- Oracle tables, typed from the rulings. ----
BUILTIN_REFS = {  # ref -> (provider, is_api)
    "openai/gpt-4o": ("openai", True),
    "anthropic/claude-3": ("anthropic", True),
    "gemini/gemini-pro": ("gemini", True),
    "ollama/llama3.2:1b": ("ollama", False),
    "lm_studio/qwen2.5-7b": ("lm_studio", False),
    "bartowski/Qwen3-GGUF": (LOCAL, False),  # a bare <org>/<repo>: a registry ref
    "TheBloke/x/y.gguf": (LOCAL, False),
}
SHAPES = {  # ref suffix after "<name>/" -> (expected provider is the preset?, expected name)
    "model": (True, "model"),
    "model:tag": (True, "model:tag"),
    "org/model": (True, "org/model"),
    "x.gguf": (True, "x.gguf"),  # two segments: not the native shape
    "org/repo/file.gguf": (False, None),  # three segments and .gguf: native, local wins
}
EDGE_NAMES = ("a", "0", "9lives", "my-server", "my_server", "x1-y2_z3", "bge", "tei")
NAME_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-_"


def generated_names(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    names = list(EDGE_NAMES)
    while len(names) < len(EDGE_NAMES) + count:
        text = rng.choice(NAME_ALPHABET[:36]) + "".join(
            rng.choice(NAME_ALPHABET) for _ in range(rng.randint(1, 10))
        )
        if text not in names and text not in {"local", "openai", "ollama", "gemini", "mistral"}:
            names.append(text)
    return names


def overlap_score(query: str, doc: str) -> float:
    """The rerank oracle: shared lower-case words, then a hash tie-break below 0.001."""
    q, d = set(query.lower().split()), set(doc.lower().split())
    tie = int(hashlib.sha256(doc.encode()).hexdigest()[:4], 16) / 65536 / 1000
    return len(q & d) + tie


def embed_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [b / 255 for b in digest[:4]]


# ---- The fake server: records each request, answers by path. ----
class Fake:
    def __init__(self, label: str) -> None:
        self.label = label
        self.log: list[dict[str, Any]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                size = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(size) or b"{}")
                fake.log.append(
                    {"path": self.path, "auth": self.headers.get("authorization"), "body": body}
                )
                out = fake.answer(self.path, body)
                raw = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def answer(self, path: str, body: dict[str, Any]) -> Any:
        if path.endswith("/embeddings"):
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            data = [
                {"object": "embedding", "index": i, "embedding": embed_vector(t)}
                for i, t in enumerate(texts)
            ]
            return {"object": "list", "model": body["model"], "data": data,
                    "usage": {"prompt_tokens": 1, "total_tokens": 1}}
        if path.endswith("/rerank"):
            docs = body.get("documents") or body.get("texts")
            scored = sorted(
                ((overlap_score(body["query"], d), i) for i, d in enumerate(docs)), reverse=True
            )
            if "texts" in body:  # the TEI native shape
                return [{"index": i, "score": s} for s, i in scored]
            return {"results": [{"index": i, "relevance_score": s} for s, i in scored]}
        return {"id": "c", "object": "chat.completion", "created": 1, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": f"from {self.label}"}}]}

    def clear(self) -> None:
        self.log.clear()


class Dns:
    """Replace socket.getaddrinfo, the layer the preset check uses, with a table."""

    def __init__(self) -> None:
        self.table: dict[str, list[str]] = {}
        self.real = socket.getaddrinfo
        socket.getaddrinfo = self.lookup

    def lookup(self, host: Any, port: Any, *args: Any, **kwargs: Any) -> list[Any]:
        name = str(host).lower()
        if name in self.table:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, 0)) for a in self.table[name]]
        return self.real(host, port, *args, **kwargs)


class Report:
    def __init__(self, out: Any) -> None:
        self.out = out
        self.counts: Counter[str] = Counter()
        self.sections: Counter[tuple[str, str]] = Counter()
        self.inputs = 0

    def row(self, section: str, ident: str, expected: Any, actual: Any, note: str = "") -> None:
        self.inputs += 1
        status = "PASS" if expected == actual else "RED"
        self.counts[status] += 1
        self.sections[(section, status)] += 1
        extra = f" note={note}" if note else ""
        print(f"ROW {status} {section} {ident!r} expected={expected} actual={actual}{extra}",
              file=self.out)


def attempt(fn: Callable[[], Any]) -> Any:
    """The value, or absent:<error> for a missing product name, or crash:<type>."""
    try:
        return fn()
    except (ImportError, AttributeError, TypeError, KeyError) as exc:
        return f"absent:{type(exc).__name__}"
    except Exception as exc:
        return f"crash:{type(exc).__name__}"


def verdict(fn: Callable[[], Any]) -> str:
    """ok, refuse (a ProviderError or ValueError), or absent/crash."""
    try:
        fn()
    except (ImportError, AttributeError, TypeError, KeyError) as exc:
        return f"absent:{type(exc).__name__}"
    except Exception as exc:
        return REFUSE if type(exc).__name__ in {"ProviderError", "ValueError"} else (
            f"crash:{type(exc).__name__}")
    return OK


def mod(name: str) -> Any:
    return importlib.import_module(name)


def fresh_root(work: Path) -> Path:
    from lilbee.core.config import cfg

    root = Path(tempfile.mkdtemp(dir=work))
    cfg.data_root = root
    cfg.server_presets = None
    cfg.llm_api_key = ""
    return root


def parsed(ref: str) -> tuple[str, bool, bool, str] | str:
    def go() -> tuple[str, bool, bool, str]:
        p = mod("lilbee.providers.model_ref").parse_model_ref(ref)
        return (p.provider, p.is_remote, p.is_api, p.name)

    return attempt(go)


def add(name: str, url: str, roles: tuple[str, ...] = ROLES, **kw: Any) -> Any:
    return mod("lilbee.app.presets").add_preset(name, url, roles=roles, **kw)


# ---- Sections. ----
def classify(rep: Report, work: Path, names: list[str], fake: Fake) -> None:
    """Each shape under each name: bare before the add, the preset after, bare after the remove."""
    for name in names:
        fresh_root(work)
        for suffix, (preset, bare) in SHAPES.items():
            ref = f"{name}/{suffix}"
            before = parsed(ref)
            rep.row("classify-before-add", ref, LOCAL,
                    before[0] if isinstance(before, tuple) else before,
                    "control: equals the base")
        add(name, fake.url)
        for suffix, (preset, bare) in SHAPES.items():
            ref = f"{name}/{suffix}"
            got = parsed(ref)
            want = (name, True, False, bare) if preset else (LOCAL, False, False, ref)
            rep.row("classify-after-add", ref, want, got)
        mod("lilbee.app.presets").remove_preset(name)
        got = parsed(f"{name}/model")
        rep.row("classify-after-remove", f"{name}/model", LOCAL,
                got[0] if isinstance(got, tuple) else got, "no stale copy of the name")


def config_validator(rep: Report, work: Path, fake: Fake) -> None:
    """A preset ref in a model field survives the field validator, which runs before any store read."""
    from lilbee.core.config import cfg

    fresh_root(work)
    attempt(lambda: add("kept", fake.url))
    for field in ("chat_model", "embedding_model", "reranker_model"):
        def go(f: str = field) -> str:
            setattr(cfg, f, "kept/some-model")
            return getattr(cfg, f)

        rep.row("config-validator", field, "kept/some-model", attempt(go))
    cfg.chat_model = cfg.embedding_model = cfg.reranker_model = ""


def other_process(rep: Report, work: Path, fake: Fake) -> None:
    """A preset another process writes or removes is seen at once, with no restart."""
    root = fresh_root(work)
    env = {**os.environ, "LILBEE_DATA": str(root), "PYTHONPATH": str(REPO / "src")}
    script = Path(tempfile.mkdtemp(dir=work)) / "other.py"
    action = "add_preset('other', sys.argv[2], roles=['chat'])"
    script.write_text(
        "import sys\nfrom lilbee.core.config import cfg\nfrom lilbee.app.presets import *\n"
        "cfg.data_root = __import__('pathlib').Path(sys.argv[1])\n"
        f"{action} if sys.argv[3] == 'add' else remove_preset('other')\n"
    )
    py = sys.executable
    for step, want in (("add", "other"), ("remove", LOCAL)):
        subprocess.run([py, "-I", str(script), str(root), fake.url, step], env=env,
                       check=False, capture_output=True, stdin=subprocess.DEVNULL)
        got = parsed("other/model")
        rep.row("other-process", step, want, got[0] if isinstance(got, tuple) else got)


def builtin_controls(rep: Report, work: Path, fake: Fake) -> None:
    """Built-in prefixes and registry refs equal the base with no preset and with presets."""
    for context in ("no-presets", "with-presets"):
        fresh_root(work)
        if context == "with-presets":
            for name in ("alpha", "bartowski", "ollama2"):
                attempt(lambda n=name: add(n, fake.url))
        for ref, (provider, is_api) in BUILTIN_REFS.items():
            if context == "with-presets" and ref == "bartowski/Qwen3-GGUF":
                continue  # the shadow cell below
            got = parsed(ref)
            rep.row(f"builtin-{context}", ref, (provider, is_api),
                    (got[0], got[2]) if isinstance(got, tuple) else got)
        for bad in ("noslash", "model.gguf"):
            ok = verdict(lambda b=bad: mod("lilbee.providers.model_ref").parse_model_ref(b))
            rep.row(f"builtin-{context}", bad, REFUSE, ok)
        rep.row(f"builtin-{context}", "translate_options keeps raw options for openai/", True,
                attempt(lambda: _options_kept("openai/gpt-4o", api=True)))


def _options_kept(ref: str, *, api: bool) -> bool:
    mr = mod("lilbee.providers.model_ref")
    out = mr.translate_options({"top_k": 5, "num_predict": 9}, mr.parse_model_ref(ref))
    return ("top_k" in out) is (not api)


def shadow(rep: Report, work: Path, fake: Fake) -> None:
    """OPEN, default taken: a preset named like a Hugging Face org wins a two-segment ref."""
    fresh_root(work)
    attempt(lambda: add("bartowski", fake.url))
    got = parsed("bartowski/Qwen3-GGUF")
    rep.row("shadow-default", "bartowski/Qwen3-GGUF", "bartowski",
            got[0] if isinstance(got, tuple) else got, "owner question 3: prefix wins")
    got = parsed("bartowski/Qwen3/Q4.gguf")
    rep.row("shadow-default", "bartowski/Qwen3/Q4.gguf", LOCAL,
            got[0] if isinstance(got, tuple) else got, "native shape wins, as for openai/")
    got = attempt(lambda: mod("lilbee.modelhub.role_validator")._is_registry_ref("bartowski/Qwen3-GGUF"))
    rep.row("shadow-default", "role validator reads the same registry", False, got)


def roles(rep: Report, work: Path, fake: Fake) -> None:
    """check_role over every non-empty role subset and every role; vision needs no wire."""
    subsets = [
        tuple(r for i, r in enumerate(ROLES) if mask >> i & 1) for mask in range(1, 16)
    ]
    for n, served in enumerate(subsets):
        fresh_root(work)
        name = f"srv{n}"
        attempt(lambda: add(name, fake.url, served))
        for role in ROLES:
            def go() -> None:
                sr = mod("lilbee.providers.server_registry")
                wr = mod("lilbee.providers.roles").WorkerRole
                ref = mod("lilbee.providers.model_ref").parse_model_ref(f"{name}/m")
                sr.check_role(ref, wr(role))

            rep.row("check-role", f"{'+'.join(served)} asked {role}",
                    OK if role in served else REFUSE, verdict(go))


class Wire:
    """Drive the real provider against the fake; at the base a preset ref is not routed."""

    def __init__(self) -> None:
        from lilbee.core.config import cfg
        self.cfg = cfg

    def call(self, role: str, ref: str) -> Any:
        from lilbee.providers.routing_provider import RoutingProvider

        provider = RoutingProvider()
        if role == "chat":
            return provider.chat([{"role": "user", "content": "hi"}], model=ref).text
        if role == "embed":
            self.cfg.embedding_model = ref
            return [list(map(float, v)) for v in provider.embed(["alpha", "beta"])]
        self.cfg.reranker_model = ref
        return provider.rerank("red apple", ["green pear", "red apple pie", "apple"])


def wire(rep: Report, work: Path, fake: Fake) -> None:
    w = Wire()
    docs = ["green pear", "red apple pie", "apple"]
    want_scores = [overlap_score("red apple", d) for d in docs]
    cells = [  # role, rerank_wire, expected path
        ("chat", None, "/v1/chat/completions"),
        ("embed", None, "/v1/embeddings"),
        ("rerank", None, "/v1/rerank"),
        ("rerank", "tei", "/rerank"),
    ]
    for role, rw, path in cells:
        for keyed in (True, False):
            fresh_root(work)
            w.cfg.llm_api_key = LLM_KEY
            fake.clear()
            kw: dict[str, Any] = {"api_key": KEY} if keyed else {}
            if rw:
                kw["rerank_wire"] = rw
            ident = f"{role}/{rw or 'default'}/{'key' if keyed else 'nokey'}"
            made = attempt(lambda: add("wired", fake.url, ROLES, **kw))
            if isinstance(made, str):
                rep.row("wire", ident, "added", made)
                continue
            ref = "wired/some-model"
            routed = parsed(ref)
            if not (isinstance(routed, tuple) and routed[0] == "wired"):
                rep.row("wire", ident, "routed to wired", f"parsed as {routed}",
                        "not called: a local ref would start the engine")
                continue
            got = attempt(lambda: w.call(role, ref))
            seen = fake.log[-1] if fake.log else {}
            rep.row("wire-path", ident, path, seen.get("path"))
            model_field = seen.get("body", {}).get("model")
            want_model = "some-model" if rw != "tei" else "huggingface/some-model"
            rep.row("wire-model-field", ident, want_model, model_field,
                    "the preset name never reaches the server")
            auth = seen.get("auth") or ""
            rep.row("wire-key", ident, f"Bearer {KEY}" if keyed else False,
                    auth if keyed else (LLM_KEY in auth),
                    "a keyless preset never sends cfg.llm_api_key")
            if role == "rerank":
                rep.row("wire-rerank-order", ident, want_scores, got,
                        "scores in input order, from a reordered reply")
            if role == "embed":
                rep.row("wire-embed", ident, [embed_vector(t) for t in ("alpha", "beta")],
                        [[round(x, 6) for x in v] for v in got] if isinstance(got, list) else got)
            if role == "chat":
                rep.row("wire-chat", ident, "from " + fake.label, got)


def harness_control(rep: Report, work: Path, fake: Fake) -> None:
    """The fake and the driver reach a built-in local server through the layer presets will use."""
    from lilbee.core.config import cfg

    w = Wire()
    fresh_root(work)
    cfg.lm_studio_base_url = fake.url
    for role, path in (("chat", "/v1/chat/completions"), ("embed", "/v1/embeddings"),
                       ("rerank", "/v1/rerank")):
        fake.clear()
        got = attempt(lambda: w.call(role, "lm_studio/m"))
        seen = fake.log[-1] if fake.log else {}
        rep.row("harness-control", f"lm_studio {role}", path, seen.get("path"),
                f"driver returned {type(got).__name__}")
    cfg.lm_studio_base_url = ""


def two_servers(rep: Report, work: Path) -> None:
    """The same model name under two presets reaches only the server of its own preset."""
    fresh_root(work)
    a, b = Fake("A"), Fake("B")
    attempt(lambda: add("one", a.url, ("chat",)))
    attempt(lambda: add("two", b.url, ("chat",)))
    w = Wire()
    for name, hit, miss, label in (("one", a, b, "A"), ("two", b, a, "B")):
        hit.clear(); miss.clear()
        routed = parsed(f"{name}/m")
        got = (attempt(lambda: w.call("chat", f"{name}/m"))
               if isinstance(routed, tuple) and routed[0] == name else f"parsed as {routed}")
        rep.row("two-servers", name, (f"from {label}", 1, 0), (got, len(hit.log), len(miss.log)))


def unserved_role_sends_nothing(rep: Report, work: Path, fake: Fake) -> None:
    """A role the preset does not serve is refused before any request leaves."""
    w = Wire()
    for served, role in ((("chat",), "embed"), (("chat", "embed"), "rerank"), (("embed",), "chat")):
        fresh_root(work)
        fake.clear()
        attempt(lambda: add("narrow", fake.url, served))
        routed = parsed("narrow/m")
        if not (isinstance(routed, tuple) and routed[0] == "narrow"):
            rep.row("unserved-no-request", f"{'+'.join(served)} asked {role}",
                    (REFUSE, 0), f"parsed as {routed}", "not called")
            continue
        res = verdict(lambda: w.call(role, "narrow/m"))
        rep.row("unserved-no-request", f"{'+'.join(served)} asked {role}",
                (REFUSE, 0), (res, len(fake.log)))


def request_time_check(rep: Report, work: Path, dns: Dns, fake: Fake) -> None:
    """Ruling 1 of 1193a: the address is checked before each call; a name that moved is refused."""
    w = Wire()
    host = "moves.example.test"
    for blocked in ("169.254.169.254", "0.0.0.0", "224.0.0.1", "240.0.0.1"):
        fresh_root(work)
        dns.table[host] = ["8.8.8.8"]
        attempt(lambda: add("moves", f"http://{host}:{fake.server.server_port}", ("chat",),
                            api_key=KEY))
        dns.table[host] = [blocked]
        fake.clear()
        routed = parsed("moves/m")
        if not (isinstance(routed, tuple) and routed[0] == "moves"):
            rep.row("request-time", blocked, (REFUSE, 0, False), f"parsed as {routed}", "not called")
            continue
        text = ""
        try:
            w.call("chat", "moves/m")
        except Exception as exc:
            text = f"{type(exc).__name__}:{exc}"
        rep.row("request-time", blocked,
                (REFUSE, 0, False),
                (REFUSE if text.startswith("ProviderError") else text[:40], len(fake.log), KEY in text))


def canary(rep: Report, work: Path, fake: Fake) -> None:
    """The embedder trigger names the preset, its URL and its model; no secret enters it."""
    from lilbee.core.config import cfg

    def trigger(model: str) -> Any:
        def go() -> str:
            cp = mod("lilbee.retrieval.canary_probe")
            reg = mod("lilbee.modelhub.registry").ModelRegistry(cfg.models_dir)
            cfg.embedding_model = model
            return cp.embedder_trigger(cfg, reg)
        return attempt(go)

    fresh_root(work)
    other = Fake("other")
    attempt(lambda: add("e1", fake.url, ("embed",), api_key=KEY))
    attempt(lambda: add("e2", other.url, ("embed",)))
    t_a, t_a2, t_b, t_c = trigger("e1/m"), trigger("e1/m"), trigger("e1/n"), trigger("e2/m")
    good = lambda t: isinstance(t, str) and not t.startswith(("absent", "crash"))  # noqa: E731
    rep.row("canary-trigger", "a named string", True, good(t_a))
    rep.row("canary-trigger", "stable in one process", True, good(t_a) and t_a == t_a2)
    rep.row("canary-trigger", "model change differs", True, good(t_a) and good(t_b) and t_a != t_b)
    rep.row("canary-trigger", "same model, other preset URL differs", True,
            good(t_a) and good(t_c) and t_a != t_c)
    attempt(lambda: mod("lilbee.app.presets").edit_preset("e1", url=other.url))
    t_edit = trigger("e1/m")
    rep.row("canary-trigger", "URL edit under the same name differs", True,
            good(t_a) and good(t_edit) and t_a != t_edit)
    attempt(lambda: mod("lilbee.app.presets").edit_preset("e1", api_key="another-key"))
    rep.row("canary-trigger", "key change keeps the trigger", True,
            good(t_edit) and trigger("e1/m") == t_edit)
    rep.row("canary-trigger", "no key text", True,
            good(t_a) and KEY not in t_a and "another-key" not in str(trigger("e1/m")))
    rep.row("canary-trigger", "builtin remote still process-scoped", True,
            good(trigger("openai/text-embedding-3-small")) and "openai" in str(
                trigger("openai/text-embedding-3-small")))


def census(rep: Report) -> None:
    """Callers of the fixed prefix set and of the local server registry, by grep."""
    def files(pattern: str) -> list[str]:
        out = subprocess.run(["grep", "-rlE", pattern, "src"], cwd=REPO, capture_output=True,
                             text=True, check=False).stdout.split()
        return sorted(out)

    control = files("class LocalServerSpec")
    rep.row("census", "control: grep finds a known class", ["src/lilbee/providers/local_servers/spec.py"],
            control)
    fixed = files(r"\bPROVIDER_PREFIXES\b")
    rep.row("census", "files naming PROVIDER_PREFIXES", [], fixed,
            "the registry replaces the set; no caller keeps it")
    sdk = files(r"\bLOCAL_SERVER_KEYS\b")
    rep.row("census", "files naming LOCAL_SERVER_KEYS outside the registry", [],
            [f for f in sdk if "providers/server_registry.py" not in f
             and "providers/local_servers/" not in f])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extras", type=Path, required=True, help="a directory holding liter_llm")
    parser.add_argument("--seed", type=int, default=34)
    parser.add_argument("--names", type=int, default=12)
    args = parser.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)
    os.environ["LILBEE_DATA"] = str(Path(tempfile.mkdtemp(dir=args.work)))
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(1, str(args.extras))
    import lilbee

    loaded_from = Path(lilbee.__file__).resolve()
    if REPO not in loaded_from.parents:
        print(f"REFUSE: lilbee loaded from {loaded_from}, not from {REPO}", file=sys.stderr)
        return 2
    mod("liter_llm")  # the SDK path under test; a missing module stops the run

    dns = Dns()
    names = generated_names(args.seed, args.names)
    fake = Fake("fake")
    with args.out.open("w", encoding="utf-8") as out:
        rep = Report(out)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
        print(f"HEAD {head} base={BASE_SHA} seed={args.seed} names={len(names)}", file=out)
        classify(rep, args.work, names, fake)
        config_validator(rep, args.work, fake)
        other_process(rep, args.work, fake)
        builtin_controls(rep, args.work, fake)
        shadow(rep, args.work, fake)
        roles(rep, args.work, fake)
        wire(rep, args.work, fake)
        harness_control(rep, args.work, fake)
        two_servers(rep, args.work)
        unserved_role_sends_nothing(rep, args.work, fake)
        request_time_check(rep, args.work, dns, fake)
        canary(rep, args.work, fake)
        census(rep)
        print("DEFER canary-stop-end-to-end: needs the 1187 store on this branch", file=out)
        print("GAP request-time: the check runs before the call; liter-llm resolves the host "
              "itself, so a change between check and connect is not refused", file=out)
        for (section, status), count in sorted(rep.sections.items()):
            print(f"SECTION {section} {status} {count}", file=out)
        total = f"TOTAL inputs={rep.inputs} PASS={rep.counts['PASS']} RED={rep.counts['RED']} OPEN=0"
        print(total, file=out)
    print(total)
    return 0 if rep.counts["RED"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
