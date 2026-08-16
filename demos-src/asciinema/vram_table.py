#!/usr/bin/env python3
"""Compute real VRAM requirements per (model, ctx, gpu-count) with gguf-parser
and record per-device numbers under a real tensor split — the estimator's
single-device number collapses on multi-GPU (the documented tensor-split OOM
bug). scatter.py launch() cross-checks these rows against each candidate's
per-GPU memory. Runs on the prep pod (gguf-parser needs no GPU).

Usage: vram_table.py reels.yaml vram_table.json
"""

import json
import pathlib
import re
import subprocess
import sys

import yaml

MODELS_DIR = pathlib.Path("/workspace/models")
GPU_COUNTS = (1, 2, 3, 4)


def gguf_for(ref: str, quant: str) -> pathlib.Path:
    d = MODELS_DIR / ("models--" + ref.replace("/", "--"))
    cands = [p for p in d.rglob("*.gguf")
             if quant.lower() in p.name.lower()
             and not re.search(r"mmproj|draft|-mtp|\bmtp\b", p.name, re.I)]
    if not cands:
        sys.exit(f"no gguf for {ref} {quant} under {d}")
    return sorted(cands)[0]  # first shard for split models


def estimate(source: list[str], ctx: int, gpu_count: int) -> dict:
    cmd = ["gguf-parser", *source, "--ctx-size", str(ctx), "--json"]
    if gpu_count > 1:
        cmd += ["--tensor-split", ",".join(["1"] * gpu_count)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    est = data.get("estimate", {})
    items = est.get("items") or [est]
    per_device = []
    for it in items:
        for v in it.get("vrams", []):
            per_device.append(v.get("nonuma", 0))
    if not per_device:
        sys.exit(f"gguf-parser returned no vram rows for {gguf} (JSON shape drift?)")
    return {
        "max_device_gb": round(max(per_device) / 1e9, 1),
        "total_gb": round(sum(per_device) / 1e9, 1),
        "devices": len(per_device),
    }


def main() -> None:
    manifest = yaml.safe_load(open(sys.argv[1]))
    table = {}
    refs_path = pathlib.Path(__file__).parent / "pull_refs.json"
    pull_refs = json.loads(refs_path.read_text()) if refs_path.exists() else {}
    for key, m in manifest["models"].items():
        if MODELS_DIR.exists() and list(MODELS_DIR.glob(f"models--{m['ref'].replace('/', '--')}*")):
            source = ["--path", str(gguf_for(m["ref"], m["quant"]))]
        else:
            # fresh-DC: estimate remotely from the resolved HF ref
            ref = pull_refs.get(key)
            if not ref:
                sys.exit(f"no local gguf and no pull ref for {key}")
            hf_file = ref[len(m["ref"]) + 1:]
            source = ["--hf-repo", m["ref"], "--hf-file", hf_file]
        rows = {}
        for n in GPU_COUNTS:
            rows[str(n)] = estimate(source, m["ctx"], n)
            print(f"{key} x{n}: max/dev {rows[str(n)]['max_device_gb']} GB, "
                  f"total {rows[str(n)]['total_gb']} GB @ ctx {m['ctx']}")
        table[key] = {"source": " ".join(source), "ctx": m["ctx"], "by_gpu_count": rows}
    json.dump(table, open(sys.argv[2], "w"), indent=1)
    print(f"wrote {sys.argv[2]}")


if __name__ == "__main__":
    main()
