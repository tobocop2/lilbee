#!/usr/bin/env python3
"""Copy this pod group's inputs from the volume (or HF fallback) to local NVMe.

Sources come from /workspace/golden/transfer.json, written by the prep pod
after measuring volume throughput: {"mode": "volume"|"hf", "measured_MBps": N}.
Many-small-file trees ship as tar.zst; GGUFs copy raw (sequential reads are
fine on MFS). Only THIS group's models are transferred.

Usage: materialize.py reels.yaml --group <G> --models-dst /root/models
                       [--reels r1,r2,...]   # stage exactly these reels (else group's)
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

    # Stage based on the EXACT reels being recorded (passed after --reels), not
    # the pod group. A reel may be recorded on a different group's pod (e.g. a
    # rehearsal running group-B agent reels on a group-A pod); staging decisions
    # (lilbee-src checkout, godot project, KB snapshots, models) must follow the
    # reels actually rolling, not the group's manifest membership. Falls back to
    # the group's reels when --reels is not supplied.
    if "--reels" in sys.argv:
        want = sys.argv[sys.argv.index("--reels") + 1].split(",")
        reels = {k: manifest["reels"][k] for k in want if k in manifest["reels"]}
    else:
        reels = {k: r for k, r in manifest["reels"].items() if r.get("pod_group") == group}
    if not reels:
        sys.exit(f"no reels to materialize (group={group})")

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
    # NOTE: hermes truncates a >20k-char project AGENTS.md and prints a ⚠ on
    # camera. Pre-seeding context_file_max_chars in config.yaml does NOT work —
    # `lilbee launch hermes` rewrites config.yaml and drops keys not in its own
    # fragment. The real fix lives in lilbee (hermes_config emits the key); the
    # volume's lilbee must carry it for a fresh hermes re-run to render clean.
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
        dst = models_dst / e.name
        # idempotent: copytree(symlinks=True) raises FileExistsError re-creating
        # an already-present snapshot->blob symlink, so a job retry / rehearsal
        # re-run on a warm pod would crash. Clear the dest first.
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(e, dst, symlinks=True)

    for key in model_keys:
        m = manifest["models"][key]
        dirname = "models--" + m["ref"].replace("/", "--")
        t0 = time.time()
        if transfer["mode"] == "hf":
            # download+register the model ONCE here (with retries) via the
            # exact pull ref, so every reel's stage hash-hits. Deferring to the
            # first reel's stage lost reels when that big download flaked.
            refs = json.loads((pathlib.Path(__file__).parent / "pull_refs.json").read_text())
            ref = refs[key]
            for attempt in range(4):
                r = subprocess.run(["lilbee", "model", "pull", ref], text=True)
                if r.returncode == 0:
                    break
                print(f"materialize pull {key} attempt {attempt} rc={r.returncode}, retrying")
                time.sleep(15)
            else:
                sys.exit(f"materialize: model pull failed for {key} after retries")
            print(f"MATERIALIZED {key} via hf pull")
            continue
        else:
            src = VOLUME_MODELS / dirname
            if not src.exists():
                sys.exit(f"model dir missing on volume: {src}")
            # idempotent: cp -r into an existing dir nests it as {dirname}/{dirname};
            # clear the dest so a job retry / rehearsal re-run copies clean.
            sh(f"rm -rf {models_dst}/{dirname} && cp -r {src} {models_dst}/{dirname}")
        rate = m["gb"] * 1024 / max(time.time() - t0, 1)
        print(f"MATERIALIZED {key} {m['gb']}GB at {rate:.0f} MB/s via {transfer['mode']}")

    print("MATERIALIZE_OK")


if __name__ == "__main__":
    main()
