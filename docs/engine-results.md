# Engine results

This page records what each managed engine did when lilbee ran its strict output check. The check is the same one that [Preset results](preset-results.md) describes, with more steps.

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

No recording is taken yet. Each row changes when its recording lands. A recording is taken when an engine is added and on each pin bump.

| Recording | Roles | Machine | Result |
|---|---|---|---|
| llama.cpp-cpu | chat, embed, rerank, vision | CPU host | not recorded |
| llama.cpp-cuda | chat, embed, rerank, vision | NVIDIA GPU | not recorded |
| mistralrs-cpu | chat, embed, vision | CPU host | not recorded |
| mistralrs-cuda | chat, embed, vision | NVIDIA GPU | not recorded |
| tei-cpu | embed, rerank | CPU host | not recorded |
| tei-cuda | embed, rerank | NVIDIA GPU | not recorded |
| lemonade-cpu | chat, embed, rerank, vision | CPU host | not recorded |
| lemonade-ryzen-ai | chat, embed, rerank | Ryzen AI machine | not recorded, no machine |
| ik-llama-cuda | chat, embed | NVIDIA GPU | not recorded |
| mlx-chat-metal | chat | Apple silicon Mac | not recorded |
| mlx-embed-rerank-metal | embed, rerank | Apple silicon Mac | not recorded |
| ovms-intel-igpu | chat | Intel machine | not recorded, no machine |
| ovms-intel-npu | embed, rerank | Intel NPU machine | not recorded, no machine |
| geniex-snapdragon-npu | chat | Snapdragon X machine | not recorded, no machine |
