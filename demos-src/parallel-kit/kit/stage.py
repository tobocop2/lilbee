#!/usr/bin/env python3
"""Full pod-private state reset before a take. Ports the proven stage/reset
logic to local-NVMe paths: rebuild LILBEE_DATA from the reel's KB snapshot,
write config.toml for the reel's model, always recreate an empty lancedb dir
(deleting it entirely re-triggers the setup wizard mid-take).

Usage: stage.py reels.yaml <reel-name>
"""

import json
import pathlib
import shutil
import subprocess
import sys

import yaml

DATA = pathlib.Path("/root/demo-data")
KB_SRC = pathlib.Path("/root/kb")  # materialize.py untars snapshots here

# flat keys, same shape the proven k5b boots wrote; chat lines only when the
# reel has a chat model (code-search is embedder-only)
CONFIG_BASE = 'embedding_model = "Qwen/Qwen3-Embedding-8B-GGUF/Qwen3-Embedding-8B-Q8_0.gguf"\ntop_k = {top_k}\n'
CONFIG_CHAT = 'chat_model = "{model_ref}"\nchat_n_ctx_target = {ctx}\n'


def main() -> None:
    manifest = yaml.safe_load(open(sys.argv[1]))
    reel = manifest["reels"][sys.argv[2]]
    shutil.rmtree(DATA, ignore_errors=True)
    (DATA / "data" / "lancedb").mkdir(parents=True)
    (DATA / "documents").mkdir(parents=True)

    kb = reel.get("kb")
    empty_boot = reel.get("empty_kb_boot", False)
    if kb and not empty_boot:
        docs = KB_SRC / f"kb-{kb}-docs"
        lance = KB_SRC / f"kb-{kb}-lancedb"
        if docs.exists():
            shutil.copytree(docs, DATA / "documents", dirs_exist_ok=True)
        if lance.exists():
            shutil.rmtree(DATA / "data" / "lancedb")
            shutil.copytree(lance, DATA / "data" / "lancedb")

    model_key = reel.get("model") or (reel.get("models") or [None])[0]
    config = CONFIG_BASE.format(top_k=20)
    if model_key:
        m = manifest["models"][model_key]
        refs_path = pathlib.Path(__file__).parent / "pull_refs.json"
        refs = json.loads(refs_path.read_text())
        ref = refs.get(model_key)
        if not ref:
            sys.exit(f"STAGE_FAIL no resolved pull ref for {model_key}")
        config += CONFIG_CHAT.format(model_ref=ref, ctx=m["ctx"])
    (DATA / "config.toml").write_text(config)
    if model_key:
        # re-register against the pod-private data dir (hash-hits local cache;
        # unregistered models silently fall back AND rewrite config.toml)
        r = subprocess.run(["lilbee", "model", "pull", ref],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"STAGE_FAIL model pull {ref}: {r.stderr[-300:]}")
    print(f"STAGE_OK {sys.argv[2]}")


if __name__ == "__main__":
    main()
