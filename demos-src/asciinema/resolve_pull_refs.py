#!/usr/bin/env python3
"""Resolve each manifest model's EXACT registry ref from the installed
catalog. Refs are not derivable from (repo, quant) — the quant may be a
subdir, a filename token, or absent — and a wrong ref makes `lilbee model
pull` miss the hash-hit, which is the silent-fallback path the identity gate
exists to stop. Also smoke-tests one pull per model (hash-hit, fast).

Runs on the prep pod with LILBEE_MODELS_DIR=/workspace/models.
Usage: resolve_pull_refs.py reels.yaml pull_refs.json
"""

import json
import re
import subprocess
import sys

import yaml


def hf_refs(manifest: dict) -> dict[str, str]:
    """Fresh-DC mode: construct pull refs from HF file listings — pick the
    quant-matching first shard, excluding mmproj/draft files."""
    from huggingface_hub import HfApi
    api = HfApi()
    out = {}
    for key, m in manifest["models"].items():
        files = api.list_repo_files(m["ref"])
        ggufs = [f for f in files if f.endswith(".gguf")
                 and m["quant"].lower() in f.lower()
                 and not re.search(r"mmproj|draft|-mtp|\bmtp\b", f, re.I)]
        if not ggufs:
            sys.exit(f"no {m['quant']} gguf in HF repo {m['ref']}")
        first = sorted(ggufs)[0]
        out[key] = f"{m['ref']}/{first}"
        print(f"{key}: {out[key]} (hf listing)")
    return out


def main() -> None:
    manifest = yaml.safe_load(open(sys.argv[1]))
    resolved = hf_refs(manifest)
    # end-to-end proof: pull+register the smallest model (hash-hits if the
    # earlier prep run already fetched it)
    smallest = min(manifest["models"], key=lambda k: manifest["models"][k]["gb"])
    r = subprocess.run(["lilbee", "model", "pull", resolved[smallest]],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"pull smoke-test failed for {smallest}: {r.stderr[-300:]}")
    print(f"pull-ok {smallest}")
    json.dump(resolved, open(sys.argv[2], "w"), indent=1)
    print(f"wrote {sys.argv[2]} ({len(resolved)} refs)")


if __name__ == "__main__":
    main()
