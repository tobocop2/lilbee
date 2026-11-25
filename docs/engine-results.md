# Engine results

This page records what each managed engine did in its recorded run. Most rows are results of the strict output check. The check is the same one that [Preset results](preset-results.md) describes, with more steps. Two rows are manual runs of one chat request, and two rows are logs of a device list or an abort.

## How the check runs

A recording proxy sits between lilbee and the engine. It saves each request and each answer in `tests/conformance/fixtures/engines/<recording>.json`. The recorder also writes `<recording>.log` beside it. The log holds the engine version, the host, the backend, the model ids and one line for each step.

A test replays each recording with no engine and checks that the results are the same. A step is "pass" only when a run on the real engine gave it.

| Role | Steps |
|---|---|
| chat | Chat template, tool call with arguments, `json_object`, `json_schema`, reasoning split, context overflow |
| embed | Vectors in input order, one size, finite, no near copies |
| rerank | One score for each document, and the best score is for the right document |
| vision | A red image answers "red" and a blue image does not |
| all | `named`: the log line and the status name the engine that served the request |

A role that the engine does not claim has no row. A missing claim is `skip`, not `pass`.

## Recordings

A recording is taken when an engine is added and on each pin bump. The Fixture column names the recording of each row. A row with no fixture has no recording yet. A test compares each row with its fixture and fails when they differ. It also fails when a fixture has no row.

The Machine column names the machine that the recording names. The Version column is the engine version that the recorder saved.

| Recording | Roles | Machine | Result | Version | Fixture |
|---|---|---|---|---|---|
| llama.cpp-cpu | chat, embed, rerank, vision | CPU host | pass, 10 of 10 steps | memory-20261019-152820 62be051e9 llama-server 0.7.0-dev (build 1, commit 62be051) | `llama.cpp-cpu.json` |
| llama.cpp-cuda | chat, embed, rerank, vision | NVIDIA GeForce RTX 3090 | pass, 10 of 10 steps | memory-20261019-152820 62be051e9 llama-server 0.7.0-dev (build 1, commit 62be051) | `llama.cpp-cuda.json` |
| llama.cpp-cpu-inline | chat | CPU host | pass, 7 of 7 steps | version: 0.8.0-dev (build 1, commit dde879b) | `llama.cpp-cpu-inline.json` |
| mistralrs-cpu | chat, embed, vision | CPU host | pass, 9 of 9 steps | mistralrs 0.9.5 v0.9.5 cpu | `mistralrs-cpu.json` |
| tei-cpu | embed, rerank | CPU host | pass, 3 of 3 steps | 1.9.4 BAAI/bge-small-en-v1.5,BAAI/bge-reranker-base cpu-1.9 | `tei-cpu.json` |
| tei-cuda | embed, rerank | NVIDIA GeForce RTX 3090 | pass, 3 of 3 steps | 1.9.4 BAAI/bge-small-en-v1.5,BAAI/bge-reranker-base cuda-source-1.9.4 | `tei-cuda.json` |
| openvino-cpu | chat | Intel i7-10750H | manual run: 1 chat request, HTTP 200, no strict steps | llama-server 0.8.0-dev (build 12020, commit 0ec4027a2) openvino 2026.4.1 | `openvino-cpu.json` |
| openvino-gpu | chat | Intel i7-10750H | manual run: 1 chat request, HTTP 200, no strict steps | llama-server 0.8.0-dev (build 12020, commit 0ec4027a2) openvino 2026.4.1 | `openvino-gpu.json` |
| openvino-gpu-no-runtime | none | Intel i7-10750H | fail: the server aborts and does not fall back to the CPU | llama.cpp b12020 ubuntu-openvino-2026.4.1-x64 | `openvino-gpu-no-runtime.log` |
| openvino-two-gpu | none | AMD Ryzen 5 8645HS | pass: `--list-devices` shows one CPU and two GPU devices |  | `openvino-two-gpu.list-devices.log` |
| lemonade-ryzen-ai | chat, embed | Ryzen AI machine with an XDNA2 NPU | not recorded, no machine |  |  |
| ik-llama-cuda | chat, embed | NVIDIA GeForce RTX 4090 | pass, 7 of 7 compared steps against llama.cpp | 0+git13cffc006b82 cuda | `ik_llama.cpp-linux-x86_64-cuda/0+git13cffc006b82.json` |
| mlx-embed-rerank-metal | embed, rerank | Apple M3 Pro | pass, 3 of 3 steps | 0.32.2 | `mlx-embed-rerank-metal.json` |
| mlx-metal | chat, embed, rerank | Apple M3 Pro | pass, 9 of 9 steps | 0.32.2 | `mlx-metal.json` |
| llama-metal | chat, embed, rerank | Apple M3 Pro | pass, 9 of 9 steps | 1.12032.1 | `llama-metal.json` |
| ovms-intel-igpu | chat | Intel machine | not recorded, no machine |  |  |
| ovms-intel-npu | embed, rerank | Intel NPU machine | not recorded, no machine |  |  |
| geniex-snapdragon-npu | chat | Snapdragon X machine | not recorded, no machine |  |  |

**Experimental.** An engine is experimental on a device while its row in the coverage table (`src/lilbee/providers/fleet/engines/coverage.json`) has no record. A record is a comparison record (`compared`) or a hardware recording (`recorded`). `lilbee engine list` ends the line of such an engine with ", experimental: not run on" and the device names, for example ", experimental: not run on Intel Arc graphics and Intel NPU", and then prints this note: "Experimental: built and tested without the real device. If you run it, report the result at https://github.com/tobocop2/lilbee/issues." A cell with no row in the table has no record either, and the line names it by its raw key, for example ", experimental: not run on linux-arm64-cpu". The JSON list and `/api/engines` carry the same devices in the `experimental` field, by engine. `auto` never chooses an experimental engine, except the bundled llama.cpp. GenieX and FastFlowLM are servers and do not appear in `lilbee engine list`. If you run lilbee on such a device, report the result in the [GitHub issues](https://github.com/tobocop2/lilbee/issues).
