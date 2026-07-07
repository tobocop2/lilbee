#!/usr/bin/env python3
"""Parallel reel orchestrator. Runs on the Mac.

- preflight: advisory GLOBAL stock report (per-DC stock is only testable
  by a create attempt; launch() falls through candidates)
- launches one pod per group via GraphQL (browser UA; runpodctl's runtime
  field lies and is never consulted), records every created pod in a ledger
- pushes the API key + launches /root/kit/job.sh in tmux over ssh
- monitors /workspace/reels-out (via a monitor pod or post-hoc gather) and
  pod liveness; job pods self-terminate, this sweep is the billing backstop
- NEVER touches pods outside its own ledger (the user's ingest pod stays up)

Usage:
  scatter.py reels.yaml preflight            # advisory global stock report
  scatter.py reels.yaml launch --groups A,B,C
  scatter.py reels.yaml prep-pod | gather [dest.tar]
  scatter.py reels.yaml watch                # deadline-enforced babysitter
  scatter.py reels.yaml sweep                # terminate every ledger pod still alive
"""

import json
import pathlib
import subprocess
import sys
import time

import requests
import yaml

HERE = pathlib.Path(__file__).parent
LEDGER = HERE / "pod_ledger.json"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
SSH_KEY = pathlib.Path.home() / ".runpod/ssh/runpodctl-ssh-key"


def api_key() -> str:
    cfg = (pathlib.Path.home() / ".runpod/config.toml").read_text()
    for line in cfg.splitlines():
        if line.strip().startswith("apikey"):
            return line.split("=", 1)[1].strip().strip("\"'")
    sys.exit("no apikey in ~/.runpod/config.toml")


def gql(query: str, variables: dict | None = None) -> dict:
    r = requests.post(
        f"https://api.runpod.io/graphql?api_key={api_key()}",
        json={"query": query, "variables": variables or {}},
        headers={"User-Agent": UA, "Content-Type": "application/json"},
        timeout=60,
    )
    r.raise_for_status()
    out = r.json()
    if out.get("errors"):
        raise RuntimeError(out["errors"])
    return out["data"]


def load_ledger() -> dict:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else {}


def save_ledger(ledger: dict) -> None:
    LEDGER.write_text(json.dumps(ledger, indent=1))


def stock(gpu_type_id: str, count: int, dc: str) -> str:
    """GLOBAL stock signal only — the dataCenterId-filtered stockStatus query
    returns null in every DC (verified against 4 DCs with in-stock types), so
    per-DC availability is only truly testable by a create attempt. Preflight
    uses this to skip globally-dry types; launch() falls through candidates
    on create failure."""
    del dc
    q = """query($id: String!, $input: GpuLowestPriceInput) {
      gpuTypes(input: {id: $id}) { lowestPrice(input: $input) { stockStatus uninterruptablePrice } }
    }"""
    data = gql(q, {"id": gpu_type_id, "input": {"gpuCount": count}})
    types = data.get("gpuTypes") or []
    lp = (types[0] or {}).get("lowestPrice") or {} if types else {}
    return lp.get("stockStatus") or "None"


def parse_gpu(spec: str) -> tuple[str, int]:
    name, _, count = spec.rpartition(" x")
    return name, int(count)


def preflight(manifest: dict) -> dict[str, tuple[str, int]]:
    g = manifest["globals"]
    picks = {}
    for group, spec in manifest["pod_groups"].items():
        chosen = None
        for cand in spec["gpus"]:
            name, count = parse_gpu(cand)
            status = stock(name, count, g["datacenter"])
            print(f"{group}: {name} x{count} -> {status}")
            if status in ("High", "Medium", "Low"):
                chosen = (name, count)
                break
        if not chosen:
            print(f"WARNING {group}: no candidate in stock in {g['datacenter']} — regroup needed")
        picks[group] = chosen
    return picks


def create_pod(manifest: dict, group: str, gpu: tuple[str, int]) -> str | None:
    """Create a pod in the volume's DC. Returns None when this GPU type has
    no capacity there (the only reliable per-DC availability signal)."""
    g = manifest["globals"]
    spec = manifest["pod_groups"][group]
    name = f"reelv2-{group.lower()}"
    q = """mutation($input: PodFindAndDeployOnDemandInput) {
      podFindAndDeployOnDemand(input: $input) { id machineId }
    }"""
    payload = {"input": {
        "cloudType": "SECURE",
        "name": name,
        "gpuTypeId": gpu[0],
        "gpuCount": gpu[1],
        "minVcpuCount": spec["min_vcpu"],
        "minMemoryInGb": spec["min_vcpu"] * 2,
        "imageName": g["container_image"],
        "containerDiskInGb": spec["disk_gb"],
        "volumeInGb": 0,
        "networkVolumeId": g["volume_id"],
        "volumeMountPath": "/workspace",
        "dataCenterId": g["datacenter"],
        "ports": "22/tcp",
        # image default start.sh handles sshd; PUBLIC_KEY is RunPod's standard
        # key-injection env (custom dockerArgs would break built-in ssh)
        "env": [{"key": "PUBLIC_KEY",
                 "value": (SSH_KEY.with_suffix(".pub")).read_text().strip()}],
    }}
    try:
        data = gql(q, payload)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "no instances" in msg or "not available" in msg or "no longer any instances" in msg:
            print(f"{group}: {gpu[0]} x{gpu[1]} has no capacity in {g['datacenter']}")
            return None
        raise
    pod = data.get("podFindAndDeployOnDemand")
    if not pod or not pod.get("id"):
        print(f"{group}: create returned no pod for {gpu[0]} x{gpu[1]} (treating as no capacity)")
        return None
    pod_id = pod["id"]
    ledger = load_ledger()
    reels = [r for r in manifest["reels"].values() if r.get("pod_group") == group]
    budget_s = 0
    for r in reels:
        w = r.get("windows") or {}
        d = r.get("duration_s") or {}
        budget_s += (int(w.get("boot", 120)) + 3 * int(d.get("max", 300)) + 600) * 2
    model_keys = {k for r in reels
                  for k in ([r.get("model")] if r.get("model") else r.get("models", [])) if k}
    model_gb = sum(manifest["models"][k]["gb"] for k in model_keys)
    deadline_min = 45 + budget_s / 60 + model_gb / 8  # boot+takes(2 attempts)+copy at >=133MB/s
    ledger[pod_id] = {"group": group, "gpu": f"{gpu[0]} x{gpu[1]}", "created": time.time(),
                      "status": "created", "deadline_min": round(deadline_min)}
    save_ledger(ledger)
    print(f"created {pod_id} for group {group} ({gpu[0]} x{gpu[1]})")
    return pod_id


def pod_ssh(pod_id: str) -> tuple[str, int] | None:
    q = """query($id: String!) { pod(input: {podId: $id}) {
      desiredStatus runtime { ports { ip isIpPublic privatePort publicPort } } } }"""
    data = gql(q, {"id": pod_id})
    pod = data.get("pod") or {}
    runtime = pod.get("runtime") or {}
    for p in runtime.get("ports") or []:
        if p["privatePort"] == 22 and p["isIpPublic"]:
            return p["ip"], p["publicPort"]
    return None


def terminate_pod(pod_id: str, reason: str) -> None:
    try:
        gql("""mutation($id: String!) { podTerminate(input: {podId: $id}) }""", {"id": pod_id})
    except Exception as exc:
        print(f"{pod_id}: podTerminate failed ({exc}) — retry via sweep")
    ledger = load_ledger()
    if pod_id in ledger:
        ledger[pod_id]["status"] = f"terminated:{reason}"
        save_ledger(ledger)


def ssh(pod_id: str, cmd: str, timeout: int = 120, stdin: str | None = None) -> subprocess.CompletedProcess:
    addr = pod_ssh(pod_id)
    if not addr:
        raise RuntimeError(f"{pod_id}: ssh not ready")
    return subprocess.run(
        ["ssh", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
         "-o", "ConnectTimeout=15", "-p", str(addr[1]), f"root@{addr[0]}", cmd],
        capture_output=True, text=True, timeout=timeout, input=stdin,
    )


def gpu_memory_gb() -> dict[str, int]:
    data = gql("query { gpuTypes { id memoryInGb } }")
    return {g["id"]: g["memoryInGb"] or 0 for g in data["gpuTypes"]}


def candidate_fits(manifest: dict, group: str, gpu: tuple[str, int],
                   vram: dict, memory: dict[str, int]) -> bool:
    """Per-device VRAM check from the gguf-parser table (real tensor split).
    Utility groups and modelless reels always fit."""
    reels = [r for r in manifest["reels"].values() if r.get("pod_group") == group]
    keys = {k for r in reels for k in ([r.get("model")] if r.get("model") else r.get("models", [])) if k}
    if not keys:
        return True
    dev_gb = memory.get(gpu[0], 0)
    for key in keys:
        rows = vram.get(key, {}).get("by_gpu_count", {})
        row = rows.get(str(gpu[1]))
        if not row:
            print(f"{group}: no vram row for {key} x{gpu[1]} — allowing, pretake will catch")
            continue
        need = row["max_device_gb"] + 3.0   # embedder + KV headroom margin
        if need > dev_gb * 0.95:
            print(f"{group}: {key} needs {need:.0f}GB/device on x{gpu[1]}, "
                  f"{gpu[0]} has {dev_gb}GB — skipping candidate")
            return False
    return True


def create_first_available(manifest: dict, group: str) -> str | None:
    """Try each GPU candidate in order (VRAM-fit filtered when the table
    exists); the create attempt is the only reliable per-DC availability test."""
    vram_path = HERE / "vram_table.json"
    vram = json.loads(vram_path.read_text()) if vram_path.exists() else {}
    memory = gpu_memory_gb() if vram else {}
    for cand in manifest["pod_groups"][group]["gpus"]:
        gpu = parse_gpu(cand)
        if vram and not candidate_fits(manifest, group, gpu, vram, memory):
            continue
        pod_id = create_pod(manifest, group, gpu)
        if pod_id:
            return pod_id
    return None


def launch(manifest: dict, groups: list[str]) -> None:
    vram = HERE / "vram_table.json"
    if not vram.exists() or not json.loads(vram.read_text()):
        sys.exit("vram_table.json missing/empty: run the gguf-parser pass on the prep pod first")
    for group in groups:
        pod_id = create_first_available(manifest, group)
        if not pod_id:
            print(f"SKIP {group}: no candidate has capacity in the volume DC; regroup or retry")
            continue
        reels = [k for k, r in manifest["reels"].items() if r.get("pod_group") == group]
        for _ in range(90):   # cold image pulls can push past 10 min
            time.sleep(10)
            if pod_ssh(pod_id):
                break
        else:
            print(f"{pod_id}: ssh never came up — terminating")
            terminate_pod(pod_id, "ssh_timeout")
            continue
        steps = [
            ("push key", "umask 077 && cat > /root/.runpod_key"),
            ("extract kit", "tar -xzf /workspace/golden/kit.tar.gz -C /root && test -f /root/kit/job.sh"),
            ("start job", f"RUNPOD_POD_ID={pod_id} nohup bash /root/kit/job.sh {group} {' '.join(reels)} "
                          f"> /root/job-nohup.log 2>&1 & sleep 2 && pgrep -f '[j]ob.sh' >/dev/null && echo started"),
        ]
        failed = False
        for label, cmd in steps:
            r = ssh(pod_id, cmd, stdin=api_key() if label == "push key" else None)
            if r.returncode != 0 or (label == "start job" and "started" not in r.stdout):
                print(f"{pod_id}: {label} failed ({r.stderr.strip()[:200]}) — terminating")
                terminate_pod(pod_id, f"{label}_failed")
                failed = True
                break
        if failed:
            continue
        ledger = load_ledger()
        ledger[pod_id]["status"] = "running"
        ledger[pod_id]["reels"] = reels
        save_ledger(ledger)
        print(f"{pod_id}: job started for {reels}")


def watch(manifest: dict, deadline_min: int = 150) -> None:
    """Deadline-enforced babysitter: a ledger pod alive past its deadline is
    terminated from here (the billing backstop of record). A single null poll
    is treated as transient; two consecutive nulls = pod gone."""
    ledger = load_ledger()
    active = {p: e for p, e in ledger.items()
              if e["status"] in ("created", "running") and e.get("group") not in ("PREP", "GATHER")}
    null_polls: dict[str, int] = {}
    while active:
        time.sleep(60)
        for pod_id in list(active):
            q = """query($id: String!) { pod(input: {podId: $id}) { desiredStatus } }"""
            try:
                data = gql(q, {"id": pod_id})
            except Exception as exc:
                print(f"{pod_id}: poll error {exc}")
                continue
            pod = data.get("pod")
            if not pod or pod.get("desiredStatus") in ("TERMINATED", "EXITED"):
                null_polls[pod_id] = null_polls.get(pod_id, 0) + 1
                if null_polls[pod_id] < 2:
                    continue
                print(f"{pod_id}: gone (self-terminated)")
                ledger[pod_id]["status"] = "done"
                save_ledger(ledger)
                del active[pod_id]
                continue
            null_polls[pod_id] = 0
            age_min = (time.time() - active[pod_id]["created"]) / 60
            pod_deadline = active[pod_id].get("deadline_min", deadline_min)
            if age_min > pod_deadline:
                print(f"{pod_id}: {age_min:.0f} min old > {pod_deadline} min deadline — terminating")
                terminate_pod(pod_id, "watch_deadline")
                del active[pod_id]
        alive = [f"{e['group']}:{p}" for p, e in active.items()]
        print(f"alive: {alive or 'none'}")
    print("all ledger pods down")


def sweep(manifest: dict) -> None:
    ledger = load_ledger()
    for pod_id, entry in ledger.items():
        q = """query($id: String!) { pod(input: {podId: $id}) { desiredStatus } }"""
        state = None
        try:
            data = gql(q, {"id": pod_id})
            pod = data.get("pod")
            state = pod.get("desiredStatus") if pod else "GONE"
        except Exception as exc:
            print(f"{pod_id}: state poll failed ({exc}); terminating defensively")
        if state in ("TERMINATED", "EXITED", "GONE"):
            continue
        print(f"terminating straggler {pod_id} ({entry['group']}, state={state})")
        terminate_pod(pod_id, "swept")
        entry["status"] = "swept"
    save_ledger(ledger)
    print("sweep complete; ledger pods only — nothing else touched")


def utility_pod(manifest: dict, group: str) -> str:
    """Create a utility pod (PREP/GATHER) and wait for ssh. No job launched."""
    pod_id = create_first_available(manifest, group)
    if not pod_id:
        sys.exit(f"no {group} candidate has capacity in {manifest['globals']['datacenter']}")
    for _ in range(90):   # cold image pulls can push past 10 min
        time.sleep(10)
        addr = pod_ssh(pod_id)
        if addr:
            print(f"{pod_id} ssh ready: ssh -i {SSH_KEY} -p {addr[1]} root@{addr[0]}")
            return pod_id
    terminate_pod(pod_id, "ssh_timeout")
    sys.exit(f"{pod_id}: ssh never came up (pod terminated)")


def gather(manifest: dict, dest: str) -> None:
    pod_id = utility_pod(manifest, "GATHER")
    try:
        addr = pod_ssh(pod_id)
        with open(dest, "wb") as f:
            subprocess.run(
                ["ssh", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=no",
                 "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                 "-p", str(addr[1]), f"root@{addr[0]}", "tar -C /workspace -cf - reels-out"],
                stdout=f, check=True, timeout=1800,
            )
    finally:
        terminate_pod(pod_id, "gather_done")
    print(f"gathered to {dest}; gather pod terminated")


def main() -> None:
    manifest = yaml.safe_load(open(sys.argv[1]))
    cmd = sys.argv[2]
    if cmd == "preflight":
        preflight(manifest)
    elif cmd == "launch":
        groups = sys.argv[sys.argv.index("--groups") + 1].split(",")
        launch(manifest, groups)
    elif cmd == "prep-pod":
        utility_pod(manifest, "PREP")
    elif cmd == "gather":
        dest = sys.argv[3] if len(sys.argv) > 3 else "reels-out.tar"
        gather(manifest, dest)
    elif cmd == "watch":
        watch(manifest)
    elif cmd == "sweep":
        sweep(manifest)
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
