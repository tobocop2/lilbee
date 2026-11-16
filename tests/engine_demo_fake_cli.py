"""A stand-in for the lilbee CLI that prints fixed output and log lines.

tests/test_engine_demo_observe.py runs the engine demo observer against this
script. FAKE_MODE breaks one behavior at a time. Like the real CLI, `engine use`
refuses a role whose model is not set or a ref other than the stored one, and the
server wants the token it wrote.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PIN = "b6123"
ROOT = Path(os.environ["LILBEE_DATA"])
MODE = os.environ.get("FAKE_MODE", "")
SERVED_ENGINE = os.environ.get("FAKE_SERVED_ENGINE", "llama.cpp")
MARKER = ROOT / "installed"
PREFERRED = ROOT / "preferred"
ROLES_FILE = ROOT / "roles.json"
MODELS_FILE = ROOT / "models.json"
CALLS_FILE = ROOT / "calls.txt"
UTF8 = "utf-8"
# The ref suffix that picks the safetensors format of a repo.
SAFETENSORS_SUFFIX = "#safetensors"
# The route that sets each role's model on the server.
ROUTE_ROLES = {"chat": "chat", "reranker": "rerank", "vision": "vision"}
# Every role, in the order engine roles lists them.
ALL_ROLES = ("chat", "embed", "rerank", "vision")


def log(message: str) -> None:
    print(f"INFO lilbee.fake: {message}", file=sys.stderr, flush=True)


def roles() -> dict[str, dict[str, object]]:
    return json.loads(ROLES_FILE.read_text(encoding=UTF8)) if ROLES_FILE.exists() else {}


def models() -> dict[str, str]:
    return json.loads(MODELS_FILE.read_text(encoding=UTF8)) if MODELS_FILE.exists() else {}


def set_model(role: str, ref: str) -> str:
    """Store the model of *role*; a bare repo ref becomes a file ref, as the real settings do."""
    keeps_ref = ref.endswith(".gguf") or ref.endswith(SAFETENSORS_SUFFIX)
    stored = ref if keeps_ref else f"{ref}/{ref.rsplit('/', 1)[-1]}.gguf"
    MODELS_FILE.write_text(json.dumps({**models(), role: stored}), encoding=UTF8)
    return stored


def served(role: str) -> None:
    model = roles().get(role, {}).get("model", "unassigned")
    log(f"served by engine={SERVED_ENGINE} version={PIN} build=none role={role} model={model}")


def listing() -> dict[str, object]:
    builds = (
        [{"engine": "llama.cpp", "version": PIN, "backend": "cpu", "not_run_on": []}]
        if MARKER.exists() and MODE != "no-build"
        else []
    )
    return {"command": "engine list", "builds": builds, "not_installed": []}


INGEST_OUT = "Added: 3\nUpdated: 0\nRemoved: 0\nUnchanged: 0\nSkipped: 0\nFailed: 0\nTruncated: 0"

ASK = {
    "command": "ask",
    "question": "q",
    "answer": "A forager bee does a waggle dance [1].",
    "sources": [{"source": "bees.md"}],
    "cited_sources": [{"source": "bees.md", "label": "bees.md"}],
    "retrieval_query": "q",
    "dropped_sources": [],
}


class Api(BaseHTTPRequestHandler):
    token = ""

    def reply(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_request(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.headers["Authorization"] != f"Bearer {self.token}":
            self.reply(401, {"detail": "bad token"})
        elif self.command == "PUT" and self.path.startswith("/api/models/"):
            stored = set_model(ROUTE_ROLES[self.path.rsplit("/", 1)[1]], body["model"])
            self.reply(200, {"model": stored})
        elif self.command == "POST" and self.path == "/api/chat":
            self.chat(body["question"])
        else:
            self.reply(404, {"detail": "no route"})

    def chat(self, question: str) -> None:
        if MODE == "chat-hang":
            time.sleep(30)
        if MODE == "chat-drop":
            self.close_connection = True
            return
        for role in ("chat",) if MODE == "ask-silent" else ("embed", "rerank", "chat"):
            served(role)
        reply = {
            "answer": f"echo {question}: bees do a waggle dance [1].",
            "sources": ASK["sources"],
            "cited_sources": ASK["cited_sources"],
        }
        self.reply(500 if MODE == "chat-500" else 200, reply)

    def do_POST(self) -> None:
        self.handle_request()

    def do_PUT(self) -> None:
        self.handle_request()

    def log_message(self, *args: object) -> None:
        return


def serve(argv: list[str]) -> None:
    (ROOT / "serve.pid").write_text(str(os.getpid()), encoding=UTF8)
    if MODE == "serve-dies":
        sys.exit(0)
    if MODE == "serve-silent":
        time.sleep(60)
    if MODE == "serve-stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    if MODE == "leaves-child":
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        (ROOT / "child.pid").write_text(str(child.pid), encoding=UTF8)
    port = int(argv[argv.index("--port") + 1])
    Api.token = secrets.token_hex(8)
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "server.json").write_text(json.dumps({"token": Api.token}), encoding=UTF8)
    (ROOT / "data" / "server.port").write_text(str(port), encoding=UTF8)
    time.sleep(0.4)
    ThreadingHTTPServer(("127.0.0.1", port), Api).serve_forever()


def engine_list(argv: list[str]) -> None:
    print(json.dumps(listing()))


def engine_install(argv: list[str]) -> None:
    print("begun", flush=True)
    if MODE == "hang-install":
        time.sleep(60)
    MARKER.write_text(argv[2], encoding=UTF8)


def engine_prefer(argv: list[str]) -> None:
    assert argv[-1] == "--yes", argv
    if not MARKER.exists():
        sys.exit(f"Engine {argv[2]} is not installed.")
    PREFERRED.write_text(argv[2], encoding=UTF8)
    print(f"The preferred engine is {argv[2]}.")


def model_pull(argv: list[str]) -> None:
    """Refuse a safetensors ref until an engine is preferred, as the real pull does."""
    if argv[2].endswith(SAFETENSORS_SUFFIX) and not PREFERRED.exists():
        sys.exit(f"No engine is chosen by default for the safetensors format of {argv[2]}.")
    print(f"pulled {argv[2]} {os.environ.get('FAKE_PULL_URL', '')}".rstrip())


def use_embedder(argv: list[str]) -> None:
    set_model("embed", argv[1])
    print(f"Now embedding with {argv[1]}.")


def engine_use(argv: list[str]) -> None:
    assert argv[-1] == "--yes", argv
    role, ref = argv[2], argv[3]
    if role not in models():
        sys.exit(f"No {role} model is set. Choose a {role} model before you choose its engine.")
    if models()[role] != ref:
        sys.exit(f"The {role} model is {models()[role]}, not {ref}.")
    views = roles()
    views[role] = {"role": role, "model": ref, "format": "gguf", "engine": argv[4], "notice": None}
    ROLES_FILE.write_text(json.dumps(views), encoding=UTF8)


def add(argv: list[str]) -> None:
    if "--ocr" in argv:
        served("vision")
    print(INGEST_OUT)


def sync(argv: list[str]) -> None:
    served("embed")
    print(INGEST_OUT)


def ask(argv: list[str]) -> None:
    if MODE != "ask-silent":
        for role in ("embed", "rerank", "chat"):
            served(role)
    print(json.dumps(ASK))


def engine_roles(argv: list[str]) -> None:
    chosen, stored = roles(), models()
    views = [
        {
            "role": role,
            "model": stored.get(role, ""),
            "format": "gguf",
            "engine": chosen.get(role, {}).get("engine"),
            "notice": None,
        }
        for role in ALL_ROLES
    ]
    print(json.dumps({"command": "engine roles", "roles": views}))


COMMANDS = {
    ("engine", "list"): engine_list,
    ("engine", "install"): engine_install,
    ("engine", "prefer"): engine_prefer,
    ("model", "pull"): model_pull,
    ("use-embedder",): use_embedder,
    ("engine", "use"): engine_use,
    ("add",): add,
    ("sync",): sync,
    ("ask",): ask,
    ("engine", "roles"): engine_roles,
    ("serve",): serve,
}


def main(argv: list[str]) -> None:
    with CALLS_FILE.open("a", encoding=UTF8) as calls:
        calls.write(f"{os.environ['LILBEE_DATA']}|{os.environ['LILBEE_LOG_LEVEL']}\n")
    log("start " + " ".join(argv))
    if argv[0] == "--json":
        argv = argv[1:]
    for words, handler in COMMANDS.items():
        if tuple(argv[: len(words)]) == words:
            handler(argv)
            return
    sys.exit(f"unknown command {argv}")


main(sys.argv[1:])
