# Renting a pod on the MS MARCO index volume

How to get a machine with the full 8,841,823-passage lilbee index mounted, in
about five minutes, to validate a change against real data instead of a fixture.

Everything here is from doing it, including the parts that wasted time.

## What you are attaching to

RunPod network volume **`stbaz85u3i`**, 600 GB, datacenter **`EUR-IS-3`**. It
mounts at `/workspace` automatically and outlives every pod.

```
/workspace/kb/.lilbee/          the index. data/ is ~146GiB, config.toml beside it
/workspace/kb/.lilbee/shards/   ~147GiB of per-worker resume state, NOT needed to
                                read the index. Deletable, but only after checking
                                the merged index is complete at 8,841,823 rows.
/workspace/models/              Qwen3-Embedding-8B-Q8_0 gguf, already pulled (8GB)
/workspace/prof/                the build run's samplers and traces
/workspace/status/              counts, run.env, the config that built it
/workspace/export_fixtest/      exports from commit b08d9189c, useful as an A/B
                                baseline when a change touches the export
```

The venv does **not** survive: it lives on the container disk. Installing lilbee
is the only setup step.

## The datacenter constraint, which is the thing that wastes time

A network volume pins the pod to its datacenter, and **EUR-IS-3 serves only H100
SXM**. There are no CPU pods there and no consumer cards. One publish job spent
ten retries looking for a CPU pod and another twenty looking for an RTX 4090,
neither of which that datacenter has ever had.

The cheapest machine that can reach this volume is **one H100, about $2.99/hr**.
Do not try to find something cheaper; there isn't one.

## Provision

```bash
AT=$(python3 -c "from datetime import datetime,timedelta,UTC
print((datetime.now(UTC)+timedelta(hours=3)).strftime('%Y-%m-%dT%H:%M:%SZ'))")

for i in $(seq 1 30); do
  POD=$(runpodctl pod create --name my-check \
    --image runpod/pytorch:1.0.7-rc.138-cu1281-torch271-ubuntu2404 \
    --gpu-id "NVIDIA H100 80GB HBM3" --gpu-count 1 \
    --network-volume-id stbaz85u3i --data-center-ids EUR-IS-3 \
    --container-disk-in-gb 60 --terminate-after "$AT" --ports 22/tcp 2>/dev/null \
    | sed -n 's/.*"id": *"\([^"]*\)".*/\1/p' | head -1)
  [ -n "$POD" ] && break
  echo "attempt $i: no capacity"; sleep 20
done
```

Always pass `--terminate-after`. It is the only thing standing between a
forgotten pod and a surprise bill.

## Getting the SSH endpoint

Not available immediately. `runtime` is **present-but-null** while the pod boots,
so every level needs `or {}` and you have to poll rather than read it once.

```python
q = 'query{pod(input:{podId:"%s"}){runtime{ports{ip isIpPublic privatePort publicPort}}}}' % pod
for _ in range(90):
    d = graphql(q)
    ports = (((d.get("data") or {}).get("pod") or {}).get("runtime") or {}).get("ports") or []
    for p in ports:
        if p.get("privatePort") == 22 and p.get("isIpPublic"):
            return p["ip"], p["publicPort"]
    time.sleep(10)
```

Key is `~/.ssh/runpod_qa`. Use `ssh -n` for anything that backgrounds work on the
pod, or the ssh call holds the channel open and your caller blocks.

## Install lilbee

```bash
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --seed --python 3.12 /root/venv
```

**Install from a git clone, not a pip URL**, if you need to record which tree you
tested. A branch can be force-pushed during review and a pip URL leaves nothing
to `rev-parse`.

```bash
git clone -q https://github.com/tobocop2/lilbee /root/src
cd /root/src && git fetch -q origin <branch> && git checkout -q <commit>
git rev-parse HEAD                      # put THIS in your findings, not the branch name
uv pip install -q --python /root/venv/bin/python --prerelease=allow /root/src
```

**The engine wheel is needed only if your change embeds anything.** Reading the
store, exporting, or inspecting the index does not. Searching does, because a
query has to be embedded:

```bash
W=lilbee_engine-0.6.90b420.dev728-1.cu124-py3-none-manylinux_2_17_x86_64.whl
curl -fsSL -o /tmp/$W \
  https://github.com/tobocop2/lilbee/releases/download/v0.6.90b420.dev728/$W
uv pip install -q --python /root/venv/bin/python /tmp/$W
```

## Environment

```bash
export LILBEE_DATA=/workspace/kb/.lilbee
export LILBEE_MODELS_DIR=/workspace/models   # or an 8GB re-download
```

## Do not run `lilbee sync` against this data root

`config.toml` still points `documents_dir` at `/root/corpus/documents`, which
existed only on the machine that built the index. On a fresh pod that path is
absent or empty, and a sync against an empty documents dir is not a no-op: it is
how an index gets pruned.

Nothing about validating a read path needs the corpus. If you genuinely need it,
it is `corpus/msmarco-passage-full.tar.gz` in
[beeberg/msmarco-ingest](https://huggingface.co/datasets/beeberg/msmarco-ingest),
about eight minutes to download and unpack 8.8M files.

## Measuring peak memory honestly

Sample `MemAvailable` for the whole run and slice it per phase afterwards, so
each phase gets its own peak instead of sharing one number:

```bash
( while :; do
    printf '%s,%s\n' "$(date -u +%s)" \
      "$(awk '/^MemAvailable:/{print int($2/1024)}' /proc/meminfo)"
    sleep 5
  done ) > mem.csv &
```

Write a `ts phase_name` line at each boundary, then take
`max(MemAvailable) - min(MemAvailable within the window)` as peak used above
idle. `evals/infra/exportcheck.sh` is a worked example.

Report peak against the size of what was produced. A cost that scales with the
output is an encoding cost; a cost that is the same for a 1.6GB output and a
4.2GB one is the in-memory table, which is a different problem.

## Reference numbers on this index

From the export verification of 2026-08-01 at commit `c98c0fadc`, so a later
change has something to move against:

| | parquet | jsonl |
|---|---|---|
| wall | 66 s | 119 s |
| size | 1,651,435,665 B | 4,150,639,318 B |
| peak above idle | 22.7 GB | 22.6 GB |

Row count is 8,841,823 and any export that does not produce exactly that is a
bug. The index carries **zero duplicate source rows**
(`count(*) == count(DISTINCT filename)`), and `title`, `authors` and
`created_at` are null throughout because MS MARCO passages have none. Both facts
limit what this corpus can test; say so rather than reporting a pass they do not
support.

## Gotchas

- `lancedb`'s `Table.to_lance()` needs `pylance`, which is not a lilbee
  dependency. Use `table.head(table.count_rows())` for a full columnar read.
- `df` on `/workspace` reports the whole MooseFS cluster (630T of 851T), not the
  volume. Use `du`.
- Never `scp` over a script that is currently running. bash reads a script by
  byte offset and it will resume mid-token.
- `sysctl -w` prints the value it did not set when `/proc/sys` is read-only.
  Read it back rather than trusting the echo.
- The pod has ~1.5 TB of RAM. That is not typical, so a memory measurement taken
  here says what a change costs, not that the cost is acceptable elsewhere.

## Tear down

```bash
runpodctl pod delete "$POD"     # keep the volume; it is the point
```

The volume costs roughly $42/month at 600 GB and holds the only copy of the index
outside HuggingFace. Delete pods freely; leave the volume alone.
