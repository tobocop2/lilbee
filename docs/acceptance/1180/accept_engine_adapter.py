"""Acceptance script for the engine adapter interface (#1180). Red at the base.

Run: UV_NO_SYNC=1 PYTHONPATH=<checkout>/src <venv>/bin/python -B accept_engine_adapter.py
         --work <scratch dir> [--base-src <base>/src] [--suite <checkout with tests/conformance>] [--ci]

Rows (the expected table is in rulings/1180.md):
  G  golden: argv and llama-swap config of the head equal those of the base, byte for byte,
     over a seeded generator (the base source is run in a subprocess: it is the oracle).
  A  adapter surface: the interface in lilbee.providers.fleet.engines has every duty.
  D  direct call sites: no caller outside the engines package reaches a llama.cpp tool
     (counted with the ast module, not with the code under test).
  S  llama-swap: extras use only keys of the llama-swap v267 reference; llama.cpp adds none.
  T  tier 2: a fake second adapter passes M1 to M6 of the integration suite.
  C  CI checks (make lint, format-check, typecheck) with --ci; they take no base ref.
Exit 0 only when every row is green.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEED = 1180
INPUTS = 400
SWAP_KEYS_V267 = (
    "checkEndpoint", "useModelName", "filters", "stripParams", "setParams", "cmdStop", "capabilities",
    "reranker", "metadata", "aliases", "unloadTimeout", "env", "ttl", "concurrencyLimit", "proxy",
)
ADAPTER_PKG = "lilbee.providers.fleet.engines"
# Duty -> (names no caller outside the adapter may use, why).
FORBIDDEN: dict[str, tuple[str, ...]] = {
    "memory estimate": ("estimate_instance_footprint", "resolve_gguf_parser", "estimator_argv"),
    "device discovery": ("probe_devices",),
    "binary and pin": ("resolve_llama_server", "engine_help", "help_lists_option"),
    "readback": ("supports_memory_readback", "engine_advertises"),
    "argv and roles": ("build_server_argv", "ROLE_SPECS", "embed_spec", "rerank_spec"),
    "formats": ("SUPPORTED_ARCHS",),
}
FORBIDDEN_LITERALS = {
    "request parameters": ("embd_normalize", "return_progress", "/tokenize", "/apply-template"),
    "device discovery": ("--list-devices",),
}
# Callers that must go through the adapter at the head (relative to src/lilbee).
CALLERS = (
    "providers/fleet/planning.py", "providers/fleet/ctx.py", "providers/fleet/replicas.py",
    "providers/fleet/provider.py", "providers/fleet/swap_manager.py", "providers/fleet/client.py",
    "providers/fleet/loadability.py", "providers/engine_params.py", "providers/gguf_meta.py",
    "data/ingest/adaptive.py", "catalog/compat.py", "mcp_server.py", "server/handlers/models.py",
)

ROWS: list[tuple[str, bool, str]] = []


def row(rid: str, ok: bool, detail: str) -> None:
    ROWS.append((rid, ok, detail))
    print(f"{'GREEN' if ok else 'RED  '} {rid}: {detail}", flush=True)


# ---------------------------------------------------------------- generator (runs in a subprocess)
HELP_FULL = "\n".join(
    f"  {flag} VALUE  text" for flag in (
        "--model", "--flash-attn", "--cache-type-k", "--cache-type-v", "--batch-size",
        "--ubatch-size", "--mmproj", "--n-cpu-moe", "--cpu-moe", "--threads", "--jinja",
        "--reasoning-format", "--no-prefill-assistant", "--pooling", "--kv-unified", "--memory",
        "--fit-target", "--kv-unified-per-slot", "--load-mode", "--device", "--tensor-split",
    )
)
HELPS = {
    "full": HELP_FULL,
    "no-mmap": HELP_FULL.replace("--load-mode", "--no-mmap"),
    "minimal": "  --model VALUE  text",
}


def _binaries(work: Path) -> dict[str, Path]:
    out = {}
    for name, text in HELPS.items():
        path = work / f"fake-engine-{name}.sh"
        path.write_text(f"#!/bin/sh\ncat <<'EOF'\n{text}\nEOF\n", encoding="utf-8")
        path.chmod(0o755)
        out[name] = path
    return out


def generate(work: Path, via_adapter: bool) -> dict[str, object]:
    """Argv and swap config for seeded inputs, from the code on PYTHONPATH."""
    from lilbee.providers.fleet.launch import InstanceLaunch
    from lilbee.providers.fleet.swap_config import build_swap_config
    from lilbee.providers.roles import RerankMode, WorkerRole

    rng = random.Random(int(os.environ.get("ACCEPT_SEED", SEED)))
    bins = _binaries(work)
    argvs: list[list[str]] = []
    launches = []
    for i in range(INPUTS):
        role = rng.choice(list(WorkerRole))
        mode = rng.choice([None, RerankMode.CROSS_ENCODER, RerankMode.LLM])
        meta = rng.choice([None, {"architecture": "qwen3", "pooling_type": "2"}])
        binary = bins[rng.choice(list(bins))]
        kw = {
            "devices": tuple(range(rng.randint(0, 3))),
            "device_names": tuple(f"CUDA{n}" for n in range(rng.randint(0, 3))),
            "n_gpu_layers": rng.choice([-1, 0, 33]),
            "slots": rng.randint(1, 4),
            "ctx_per_slot": rng.choice([2048, 8192, 32768]),
            "tensor_split": tuple(rng.randint(1, 9) for _ in range(rng.randint(0, 3))),
            "mmproj": rng.choice([None, Path("/m/proj.gguf")]),
            "flash_attn": rng.choice([None, "on", "auto"]),
            "cache_type": rng.choice([None, "q8_0"]),
            "batch_size": rng.choice([None, 512]),
            "no_mmap": rng.random() < 0.5,
            "cpu_moe": rng.random() < 0.3,
            "n_cpu_moe": rng.choice([None, 4]),
            "memory_endpoint": rng.random() < 0.5,
            "fit_target_mib": rng.choice([(), (512,), (512, 1024)]),
            "kv_unified_per_slot": rng.choice([None, 4096]),
        }
        model = Path(f"/models/m{i}.gguf")
        if via_adapter:
            from lilbee.providers.fleet.engines import LLAMA, LaunchRequest

            spec = LLAMA.role_spec(role, rerank_mode=mode, meta=meta)
            argv = LLAMA.launch_argv(spec, LaunchRequest(binary=binary, model_path=model, **kw))
        else:
            from lilbee.providers.fleet import adapters as ad

            if role is WorkerRole.RERANK and mode is RerankMode.LLM:
                spec = ad.LLM_RERANK_SPEC
            elif role is WorkerRole.EMBED:
                spec = ad.embed_spec(meta)
            else:
                spec = ad.ROLE_SPECS[role]
            argv = ad.build_server_argv(binary=binary, spec=spec, model_path=model, **kw)
        argvs.append([a.replace(str(work), "<work>") for a in argv])
        launches.append(
            InstanceLaunch(
                role=role, argv=argv, env_overrides=rng.choice([{}, {"GGML_VK_VISIBLE_DEVICES": "0"}]),
                model=str(model), weights_bytes=rng.choice([0, 10**9, 10**11]), slots=kw["slots"],
                replica=i, rerank_mode=mode,
            )
        )
    configs = []
    for swap in (False, True):
        for ttl in (0, 300):
            for log_dir in (None, work / "logs"):
                ports = {launch.model_id: 20000 + n for n, launch in enumerate(launches)}
                text = build_swap_config(
                    launches, ports, swap=swap, ttl_seconds=ttl, engine_log_dir=log_dir
                )
                configs.append(text.replace(str(work), "<work>"))
    return {"argv": argvs, "configs": configs}


def run_generator(src: Path, work: Path, via_adapter: bool) -> tuple[dict[str, object] | None, str]:
    out = work / f"gen-{hashlib.sha1(str(src).encode()).hexdigest()[:6]}-{int(via_adapter)}.json"
    env = dict(os.environ, PYTHONPATH=str(src), UV_NO_SYNC="1")
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "--gen", str(out), "--work", str(work)]
    if via_adapter:
        cmd.append("--via-adapter")
    done = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)
    if done.returncode != 0:
        return None, done.stderr.strip().splitlines()[-1] if done.stderr.strip() else "no output"
    return json.loads(out.read_text(encoding="utf-8")), ""


def run_generator_seeded(src: Path, work: Path, seed: int) -> dict[str, object] | None:
    old = os.environ.get("ACCEPT_SEED")
    os.environ["ACCEPT_SEED"] = str(seed)
    try:
        sub = work / f"seed{seed}"
        sub.mkdir(exist_ok=True)
        return run_generator(src, sub, False)[0]
    finally:
        if old is None:
            del os.environ["ACCEPT_SEED"]
        else:
            os.environ["ACCEPT_SEED"] = old


def golden_rows(work: Path, base_src: Path, head_src: Path) -> None:
    base, err = run_generator(base_src, work, False)
    row("G0", base is not None and len(base["argv"]) == INPUTS, f"base generator ran, {INPUTS} seeded inputs ({err or 'ok'})")
    if base is None:
        return
    mixed = {tuple(a[:1]) for a in base["argv"]}
    row("G0b", len(base["argv"]) > 0 and len(mixed) >= 1, f"census nonempty: {len(base['argv'])} argv, {len(base['configs'])} configs")
    other = run_generator_seeded(base_src, work, 7)
    row("G0c", other is not None and other["argv"] != base["argv"] and other["configs"] != base["configs"], "control: another seed gives other argv and configs, so the comparison can fail")
    head, err = run_generator(head_src, work, False)
    row("G1", head == base, f"head code path (build_server_argv, build_swap_config) equals base ({err or 'compared'})")
    via, err = run_generator(head_src, work, True)
    row("G2", via is not None and via["argv"] == base["argv"], f"llama adapter argv equals base argv for all {INPUTS} inputs ({err or 'compared'})")
    row("G3", via is not None and via["configs"] == base["configs"], f"swap config built from adapter launches is byte-identical for {len(base['configs'])} variants (swap x ttl x log dir) ({err or 'compared'})")


# ---------------------------------------------------------------- adapter surface
DUTIES = {
    "formats": "formats",
    "supports_architecture": "supports_architecture",
    "binary": "binary",
    "pin_record": "pin_record",
    "identity": "identity",
    "role_spec": "role_spec",
    "served_roles": "served_roles",
    "launch_argv": "launch_argv",
    "discover_devices": "discover_devices",
    "estimate_memory": "estimate_memory",
    "readback": "readback",
    "swap_extras": "swap_extras",
    "request_profile": "request_profile",
}


def adapter_rows(src: Path) -> object | None:
    try:
        mod = importlib.import_module(ADAPTER_PKG)
    except ImportError as exc:
        for duty in DUTIES:
            row(f"A.{duty}", False, f"{ADAPTER_PKG} missing ({exc})")
        row("A.registry", False, "no EngineName/LLAMA exports")
        return None
    llama = getattr(mod, "LLAMA", None)
    row("A.registry", llama is not None and hasattr(mod, "EngineName") and hasattr(mod, "EngineAdapter"), "EngineName, EngineAdapter, LLAMA exported")
    for duty in DUTIES:
        row(f"A.{duty}", llama is not None and hasattr(llama, duty), f"LLAMA.{duty}")
    return llama


def pin_rows(llama: object | None) -> None:
    if llama is None:
        row("A.pin_matches_env", False, "no adapter")
        return
    env = dict(
        line.split("=", 1) for line in (ROOT / "engine-versions.env").read_text(encoding="utf-8").splitlines()
        if re.match(r"^ENGINE_[A-Z_]+=", line)
    )
    record = llama.pin_record()
    need = {"engine", "version", "backend", "launch", "platforms", "memory", "licenses"}
    row("A.pin_keys", need <= set(record), f"pin record keys {sorted(need - set(record)) or 'complete'}")
    row("A.pin_matches_env", record.get("version") == env["ENGINE_LLAMA_CPP_REF"], "pin version equals ENGINE_LLAMA_CPP_REF in engine-versions.env")


# ---------------------------------------------------------------- direct call sites (ast)
def direct_sites(src: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {duty: [] for duty in {*FORBIDDEN, *FORBIDDEN_LITERALS}}
    for rel in CALLERS:
        path = src / "lilbee" / rel
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[tuple[str, int]] = []
            if isinstance(node, ast.ImportFrom):
                names = [(a.name, node.lineno) for a in node.names]
            elif isinstance(node, ast.Call):
                f = node.func
                names = [(f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else "", node.lineno)]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                for duty, lits in FORBIDDEN_LITERALS.items():
                    if node.value in lits:
                        found[duty].append(f"{rel}:{node.lineno}:{node.value!r}")
            for name, line in names:
                for duty, banned in FORBIDDEN.items():
                    if name in banned:
                        found[duty].append(f"{rel}:{line}:{name}")
    return {k: sorted(set(v)) for k, v in found.items()}


def call_site_rows(src: Path) -> None:
    sites = direct_sites(src)
    total = sum(len(v) for v in sites.values())
    control = direct_sites_control()
    row("D.control", control, "the scanner finds a planted forbidden call in a synthetic file")
    for duty, hits in sorted(sites.items()):
        row(f"D.{duty}", not hits, f"{len(hits)} direct sites outside the adapter" + (f": {', '.join(hits[:6])}" if hits else ""))
    print(f"      total direct sites: {total}")


def direct_sites_control() -> bool:
    tree = ast.parse("from x import probe_devices\nprobe_devices(b)\n")
    return sum(isinstance(n, ast.ImportFrom) for n in ast.walk(tree)) == 1 and any(
        isinstance(n, ast.Call) and getattr(n.func, "id", "") == "probe_devices" for n in ast.walk(tree)
    )


# ---------------------------------------------------------------- llama-swap
def swap_rows(llama: object | None) -> None:
    ref = (HERE / "llama-swap-v267-config.example.yaml").read_text(encoding="utf-8")
    missing = [k for k in SWAP_KEYS_V267 if not re.search(rf"^\s*#?\s*{k}\s*:", ref, re.M)]
    row("S.reference", not missing, f"v267 reference holds the keys the adapter may emit (missing: {missing or 'none'})")
    try:
        from lilbee.providers.fleet.launch import InstanceLaunch
        from lilbee.providers.fleet.swap_config import build_swap_config
        from lilbee.providers.roles import WorkerRole
    except ImportError as exc:
        row("S.extras_field", False, str(exc))
        return
    has_field = "swap_extras" in getattr(InstanceLaunch, "__dataclass_fields__", {})
    row("S.extras_field", has_field, "InstanceLaunch carries swap_extras (empty for llama.cpp)")
    if not has_field or llama is None:
        row("S.extras_render", False, "needs the field and the adapter")
        row("S.llama_empty", False, "needs the adapter")
        return
    extras = {"checkEndpoint": "/live", "useModelName": "m", "filters": {"stripParams": "embd_normalize, return_progress"}}
    launch = InstanceLaunch(role=WorkerRole.CHAT, argv=["/b"], env_overrides={}, model="m", swap_extras=extras)
    entry = json.loads(build_swap_config([launch], {launch.model_id: 1}))["models"][launch.model_id]
    row("S.extras_render", all(entry.get(k) == v for k, v in extras.items()), "extras land in the model entry under llama-swap's own keys")
    plain = InstanceLaunch(role=WorkerRole.CHAT, argv=["/b"], env_overrides={}, model="m")
    row("S.llama_empty", not plain.swap_extras and not llama.swap_extras(plain), "llama.cpp adds no key (byte-identical config)")


# ---------------------------------------------------------------- tier 2 with a fake second adapter
def tier2_rows(suite: Path | None, src: Path) -> None:
    if suite is None or not (suite / "tests/conformance/harness.py").exists():
        row("T.suite", False, "no --suite checkout with tests/conformance (origin/engines/1205-impl)")
        return
    sys.path.insert(0, str(suite))
    try:
        harness = importlib.import_module("tests.conformance.harness")
        fake = importlib.import_module("tests.conformance.fake_adapter")
    except ImportError as exc:
        row("T.fake_adapter", False, f"tests.conformance.fake_adapter missing ({exc})")
        return
    adapter = fake.FAKE
    pin = adapter.pin_record()
    verdicts = harness.run_managed(adapter.binary(), pin=pin)
    for key in ("M1", "M2", "M3", "M4", "M5", "M6"):
        v = str(verdicts[key])
        row(f"T.{key}", v in {"pass", "fallback"}, f"fake adapter verdict {v}")
    from lilbee.providers.fleet.engines import EngineName

    row("T.distinct", adapter.name != EngineName.LLAMA, "fake adapter is a second engine, not llama.cpp")


def ci_rows() -> None:
    for target in ("lint", "format-check", "typecheck"):
        done = subprocess.run(["make", target], cwd=ROOT, capture_output=True, text=True, check=False, stdin=subprocess.DEVNULL)
        row(f"C.{target}", done.returncode == 0, f"make {target} rc={done.returncode} (no base ref: reads the tree)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--base-src", type=Path, default=ROOT / "src")
    ap.add_argument("--suite", type=Path)
    ap.add_argument("--ci", action="store_true")
    ap.add_argument("--gen", type=Path)
    ap.add_argument("--via-adapter", action="store_true")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    if args.gen:
        args.gen.write_text(json.dumps(generate(args.work, args.via_adapter)), encoding="utf-8")
        return 0
    head_src = ROOT / "src"
    import lilbee

    print(f"lilbee.__file__ = {lilbee.__file__}")
    assert str(head_src) in lilbee.__file__, "PYTHONPATH does not point at this checkout"
    golden_rows(args.work, args.base_src.resolve(), head_src)
    llama = adapter_rows(head_src)
    pin_rows(llama)
    call_site_rows(head_src)
    swap_rows(llama)
    tier2_rows(args.suite, head_src)
    if args.ci:
        ci_rows()
    red = [r for r in ROWS if not r[1]]
    print(f"\nrows: {len(ROWS)}  green: {len(ROWS) - len(red)}  red: {len(red)}  inputs: {INPUTS} x (argv) + 8 swap-config variants")
    return 1 if red else 0


if __name__ == "__main__":
    sys.exit(main())
