#!/usr/bin/env python3
"""Copy this pod group's inputs from the volume (or HF fallback) to local NVMe.

Sources come from /workspace/golden/transfer.json, written by the prep pod
after measuring volume throughput: {"mode": "volume"|"hf", "measured_MBps": N}.
Many-small-file trees ship as tar.zst; GGUFs copy raw (sequential reads are
fine on MFS). Only THIS group's models are transferred.

Usage: materialize.py reels.yaml --group <G> --models-dst /root/models
"""

import json
import pathlib
import shutil
import subprocess
import sys
import time

import yaml

VOLUME_MODELS = pathlib.Path("/workspace/models")
GOLDEN = pathlib.Path("/workspace/golden")


def sh(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    manifest = yaml.safe_load(open(sys.argv[1]))
    group = sys.argv[sys.argv.index("--group") + 1]
    models_dst = pathlib.Path(sys.argv[sys.argv.index("--models-dst") + 1])
    transfer = json.loads((GOLDEN / "transfer.json").read_text())

    reels = {k: r for k, r in manifest["reels"].items() if r.get("pod_group") == group}
    if not reels:
        sys.exit(f"no reels for group {group}")

    model_keys = set()
    kbs = set()
    for r in reels.values():
        for k in ([r.get("model")] if r.get("model") else r.get("models", [])):
            if k:
                model_keys.add(k)
        if r.get("kb"):
            kbs.add(r["kb"])

    # KB snapshots (small, tar.zst from golden)
    kb_root = pathlib.Path("/root/kb")
    kb_root.mkdir(exist_ok=True)
    for kb in kbs:
        sh(f"tar --zstd -xf {GOLDEN}/kb-{kb}.tar.zst -C {kb_root}")
    # stable local doc paths for the live-/add reels (tapes reference these)
    doc_links = {
        "manual": ("kb-manual-docs/cv-manual.pdf", "/root/cv-manual.pdf"),
        "corpus": ("kb-corpus-docs", "/root/corpus"),
        "godot": ("kb-godot-docs", "/root/godot/doc/classes"),
    }
    for kb in kbs:
        src, dst = doc_links[kb]
        sh(f"rm -rf {dst} && mkdir -p {pathlib.Path(dst).parent} && cp -r {kb_root}/{src} {dst}")

    # agent reels run opencode in a pod-local checkout copy
    if any("agent" in k or "godot-with" in k or "godot-without" in k for k in reels):
        sh(f"tar --zstd -xf {GOLDEN}/lilbee-src.tar.zst -C /root")
    # mcp-manual runs opencode from a project dir holding the pdf
    if "mcp-manual" in reels:
        sh("rm -rf /root/opencode-manual && mkdir -p /root/opencode-manual && "
           f"cp {kb_root}/kb-manual-docs/cv-manual.pdf /root/opencode-manual/")
    # godot launcher demos run from small project dirs (AGENTS.md steers the
    # agent to lilbee search / bare codesearch); result file must not pre-exist
    if any(k.startswith("godot-") or k.endswith("-godot") for k in reels):
        sh(f"rm -rf /root/godot-project /root/godot-project-bare && "
           f"cp -r {GOLDEN}/godot-project /root/godot-project && "
           f"cp -r {GOLDEN}/godot-project-bare /root/godot-project-bare && "
           f"rm -f /root/godot-project*/level_generator.gd")

    # embedder always
    models_dst.mkdir(parents=True, exist_ok=True)
    emb = [p for p in VOLUME_MODELS.glob("models--*") if "embed" in p.name.lower()]
    if not emb:
        sys.exit("embedder model dir not found on volume")
    for e in emb:
        shutil.copytree(e, models_dst / e.name, dirs_exist_ok=True, symlinks=True)

    for key in model_keys:
        m = manifest["models"][key]
        dirname = "models--" + m["ref"].replace("/", "--")
        t0 = time.time()
        if transfer["mode"] == "hf":
            # stage.py's `lilbee model pull` downloads AND registers this
            # model natively on first stage (later attempts hash-hit)
            print(f"DEFERRED {key}: hf mode — stage's lilbee model pull fetches it")
            continue
        else:
            src = VOLUME_MODELS / dirname
            if not src.exists():
                sys.exit(f"model dir missing on volume: {src}")
            sh(f"cp -r {src} {models_dst}/{dirname}")
        rate = m["gb"] * 1024 / max(time.time() - t0, 1)
        print(f"MATERIALIZED {key} {m['gb']}GB at {rate:.0f} MB/s via {transfer['mode']}")

    print("MATERIALIZE_OK")


if __name__ == "__main__":
    main()
