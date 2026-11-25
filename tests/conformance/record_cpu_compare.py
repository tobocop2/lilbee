"""Record the Linux CPU comparison of mistral.rs against llama.cpp, from fetch to record.

    python -m tests.conformance.record_cpu_compare --mistralrs BIN --llama-server BIN \\
        --models DIR --logs DIR --host-machine TYPE --host-os OS --host-cpu CPU

The run takes the two binaries from the packs that the tree pins. It downloads each model
it does not find in --models and stops if a file has another sha256 than the record states.
It starts every server with the adapter's own command line on the CPU, the engine on cores 0-7
and the oracle on cores 8-15, each with 8 threads. Then it runs tests.conformance.compare_run
with the cross_format embed bound, and stops the servers. compare_run writes the record and refuses a stale pin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from lilbee.providers.fleet.engines.base import PORT_SLOT, EngineAdapter, LaunchRequest
from lilbee.providers.fleet.engines.llama import LLAMA
from lilbee.providers.fleet.engines.mistral import MISTRAL
from lilbee.providers.roles import RerankMode, RoleTarget, WorkerRole

_HF = "https://huggingface.co"
_ENGINE_CORES = "0-7"
_ORACLE_CORES = "8-15"
_THREADS = 8
_HEALTH_TRIES = 600
_FETCH_TRIES = 4


@dataclass(frozen=True)
class Model:
    """One model file: the folder it lives in, its Hugging Face repo and its sha256."""

    folder: str
    repo: str
    file: str
    sha256: str
    quant: str
    context: int


# The weights of the record of commit 3df1d204f. A safetensors model is a folder: the run
# also fetches the small files at the top of its repo, and the pooling config of an embedder.
# mistral.rs answers with its model folder as the model name, so the engine folders keep the
# names of that run. The record refuses a path under /home: keep --models under /var/tmp.
_ENGINE_MODELS = {
    WorkerRole.CHAT: Model(
        "chat17",
        "Qwen/Qwen3-1.7B-GGUF",
        "Qwen3-1.7B-Q8_0.gguf",
        "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a",
        "Q8_0",
        16384,
    ),
    WorkerRole.EMBED: Model(
        "st_embed",
        "Qwen/Qwen3-Embedding-0.6B",
        "model.safetensors",
        "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
        "BF16",
        16384,
    ),
    WorkerRole.VISION: Model(
        "st_vision",
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        "model.safetensors",
        "74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e",
        "BF16",
        8192,
    ),
}
_ORACLE_MODELS = {
    WorkerRole.CHAT: _ENGINE_MODELS[WorkerRole.CHAT],
    WorkerRole.EMBED: Model(
        "qwen3-embedding-0.6b-gguf",
        "Qwen/Qwen3-Embedding-0.6B-GGUF",
        "Qwen3-Embedding-0.6B-Q8_0.gguf",
        "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439",
        "Q8_0",
        16384,
    ),
    WorkerRole.RERANK: Model(
        "bge-reranker-v2-m3-gguf",
        "gpustack/bge-reranker-v2-m3-GGUF",
        "bge-reranker-v2-m3-Q8_0.gguf",
        "a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3",
        "Q8_0",
        16384,
    ),
    WorkerRole.VISION: Model(
        "smolvlm-256m-instruct-gguf",
        "ggml-org/SmolVLM-256M-Instruct-GGUF",
        "SmolVLM-256M-Instruct-Q8_0.gguf",
        "2a31195d3769c0b0fd0a4906201666108834848db768af11de1d2cef7cd35e65",
        "Q8_0",
        8192,
    ),
}
# The oracle's embed model at another quant: the cross_format bound compares the two.
_QUANT_EMBED = Model(
    "qwen3-embedding-0.6b-gguf",
    "Qwen/Qwen3-Embedding-0.6B-GGUF",
    "Qwen3-Embedding-0.6B-f16.gguf",
    "421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340",
    "F16",
    16384,
)
_MMPROJ = Model(
    "smolvlm-256m-instruct-gguf",
    "ggml-org/SmolVLM-256M-Instruct-GGUF",
    "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
    "7e943f7c53f0382a6fc41b6ee0c2def63ba4fded9ab8ed039cc9e2ab905e0edd",
    "Q8_0",
    0,
)
_SAFETENSORS_EXTRAS = ("1_Pooling/config.json",)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(_FETCH_TRIES):
        try:
            # The url is an https URL of the Hugging Face hub, built from the model table.
            with urllib.request.urlopen(url, timeout=120) as answer, part.open("wb") as out:  # noqa: S310
                while block := answer.read(1 << 20):
                    out.write(block)
            part.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"fetch {url} failed ({exc}); try {attempt + 1} of {_FETCH_TRIES}", flush=True)
            time.sleep(10)
    raise SystemExit(f"could not fetch {url}")


def _repo_files(repo: str) -> list[str]:
    url = f"{_HF}/api/models/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(url, timeout=60) as answer:  # noqa: S310  an https hub URL
        entries: list[dict[str, Any]] = json.load(answer)
    return [e["path"] for e in entries if e["type"] == "file"]


def _fetch(model: Model, root: Path) -> Path:
    """The local path of *model*, downloaded when absent, with its sha256 checked."""
    folder = root / model.folder
    path = folder / model.file
    if model.file.endswith(".safetensors"):
        wanted = [f for f in _repo_files(model.repo) if "/" not in f or f in _SAFETENSORS_EXTRAS]
        for name in wanted:
            if not (folder / name).exists():
                _download(f"{_HF}/{model.repo}/resolve/main/{name}", folder / name)
    elif not path.exists():
        _download(f"{_HF}/{model.repo}/resolve/main/{model.file}", path)
    found = _sha256(path)
    if found != model.sha256:
        raise SystemExit(f"{path} has sha256 {found}, the record states {model.sha256}")
    return path


@dataclass
class _Servers:
    logs: Path
    procs: list[subprocess.Popen[bytes]]
    handles: list[IO[bytes]]

    def start(
        self,
        adapter: EngineAdapter,
        binary: Path,
        role: WorkerRole,
        model: Path,
        context: int,
        port: int,
        cores: str,
        alias: str | None,
        mmproj: Path | None = None,
    ) -> str:
        spec = adapter.role_spec(role, RerankMode.CROSS_ENCODER, None)
        request = LaunchRequest(
            binary=binary,
            model_path=model,
            devices=(),
            n_gpu_layers=0,
            slots=1,
            ctx_per_slot=context,
            mmproj=mmproj,
            target=RoleTarget.CPU,
        )
        argv = adapter.launch_argv(spec, request)
        if PORT_SLOT in argv:
            argv = [str(port) if arg == PORT_SLOT else arg for arg in argv]
        else:
            argv += ["--port", str(port)]
        if alias:
            # llama.cpp starts one thread per physical core of the host by default. More threads
            # than pinned cores run at 0.4 tokens per second, and an answer passes the timeout.
            argv += ["--alias", alias, "--threads", str(_THREADS)]
        log = (self.logs / f"{adapter.name.value}-{role.value}-{port}.log").open("wb")
        self.handles.append(log)
        print("start", " ".join(argv), flush=True)
        env = {**os.environ, "RAYON_NUM_THREADS": str(_THREADS)}
        command = ["taskset", "-c", cores, *argv]
        self.procs.append(
            subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        )
        return f"http://127.0.0.1:{port}"

    def stop(self) -> None:
        for proc in self.procs:
            proc.terminate()
        for proc in self.procs:
            proc.wait(timeout=60)
        for handle in self.handles:
            handle.close()


def _wait(url: str) -> None:
    for _ in range(_HEALTH_TRIES):
        try:
            with urllib.request.urlopen(url + "/health", timeout=3):  # noqa: S310  a local http URL
                return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(2)
    raise SystemExit(f"the server at {url} did not answer /health")


def _entry(url: str, served_id: str, model: Model, path: Path) -> dict[str, str]:
    return {"url": url, "id": served_id, "file": str(path), "quant": model.quant}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--mistralrs", type=Path, required=True, help="the pinned engine binary")
    parser.add_argument("--llama-server", type=Path, required=True, help="the pinned oracle")
    parser.add_argument("--models", type=Path, required=True, help="the folder of the models")
    parser.add_argument("--logs", type=Path, required=True, help="the folder of the server logs")
    for part in ("machine", "os", "cpu"):
        parser.add_argument(f"--host-{part}", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Fetch, start, compare and stop; the return code is compare_run's."""
    args = _parser().parse_args(argv)
    args.logs.mkdir(parents=True, exist_ok=True)
    engine_paths = {role: _fetch(m, args.models) for role, m in _ENGINE_MODELS.items()}
    oracle_paths = {role: _fetch(m, args.models) for role, m in _ORACLE_MODELS.items()}
    quant_path = _fetch(_QUANT_EMBED, args.models)
    mmproj = _fetch(_MMPROJ, args.models)
    servers = _Servers(args.logs, [], [])
    try:
        engine: dict[str, dict[str, str]] = {}
        for port, (role, model) in enumerate(_ENGINE_MODELS.items(), start=8301):
            path = engine_paths[role]
            target = path if path.suffix == ".gguf" else path.parent
            url = servers.start(
                MISTRAL, args.mistralrs, role, target, model.context, port, _ENGINE_CORES, None
            )
            engine[role.value] = _entry(url, "default", model, path)
        oracle: dict[str, dict[str, str]] = {}
        for port, (role, model) in enumerate(_ORACLE_MODELS.items(), start=8401):
            alias = Path(model.file).stem
            url = servers.start(
                LLAMA,
                args.llama_server,
                role,
                oracle_paths[role],
                model.context,
                port,
                _ORACLE_CORES,
                alias,
                mmproj if role is WorkerRole.VISION else None,
            )
            oracle[role.value] = _entry(url, alias, model, oracle_paths[role])
        quant_alias = Path(_QUANT_EMBED.file).stem
        quant_url = servers.start(
            LLAMA,
            args.llama_server,
            WorkerRole.EMBED,
            quant_path,
            _QUANT_EMBED.context,
            8409,
            _ORACLE_CORES,
            quant_alias,
        )
        for side in (engine, oracle):
            for entry in side.values():
                _wait(entry["url"])
        _wait(quant_url)
        spec = args.logs / "models.json"
        quant = {"embed": _entry(quant_url, quant_alias, _QUANT_EMBED, quant_path)}
        spec.write_text(json.dumps({"engine": engine, "oracle": oracle, "quant": quant}))
        command = [
            sys.executable,
            "-m",
            "tests.conformance.compare_run",
            "--engine",
            "mistral.rs",
            "--backend",
            "linux-x86_64-cpu",
            "--embed-kind",
            "cross_format",
            "--models",
            str(spec),
            "--host-machine",
            args.host_machine,
            "--host-os",
            args.host_os,
            "--host-cpu",
            args.host_cpu,
            "--host-gpu",
            "none",
        ]
        return subprocess.call(command)
    finally:
        servers.stop()


if __name__ == "__main__":
    raise SystemExit(main())
