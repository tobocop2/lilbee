# Preset results

This page records what each server did when lilbee ran its output check through a stored server preset. See [Server presets](presets.md) for how to add a preset.

## How the check runs

The check uses the real lilbee provider. It stores a preset, points the model settings at it, and runs one step for each role the server documents.

| Role | Steps |
|---|---|
| chat | Chat template, tool call, context overflow |
| embed | Two inputs give two vectors of the same size |
| rerank | Three documents give three scores, and the best score is for the right document |
| vision | One image chat names the colour of a red image |

A recording proxy sits between lilbee and the server. It saves each request and each answer in `tests/conformance/fixtures/presets/`. A test replays each fixture with no server and checks that the results are the same. A result is "pass" only when a run on a real server gave it.

## Results

All runs used CPU only, on a Linux host. Each version is the version of the server in the run.

| Server | Role | Result | Version |
|---|---|---|---|
| TEI | embed | pass (384 values per vector) | 1.9.4, `cpu-1.9` image, model `BAAI/bge-small-en-v1.5` |
| TEI | rerank (wire `tei`) | pass | 1.9.4, `cpu-1.9` image, model `BAAI/bge-reranker-base` |
| TEI | rerank (wire `v1`) | not supported for this wire: TEI documents no `POST /v1/rerank` route, and the server answers HTTP 404 | 1.9.4 |
| TEI | chat, vision | not supported for this role: TEI documents no route | 1.9.4 |
| Lemonade | chat template | pass | 2026.44.1, llama.cpp b11393 CPU |
| Lemonade | tool call | pass | 2026.44.1, llama.cpp b11393 CPU |
| Lemonade | context overflow | pass | 2026.44.1, llama.cpp b11393 CPU |
| Lemonade | embed | pass (1024 values per vector) | 2026.44.1, llama.cpp b11393 CPU |
| Lemonade | rerank (wire `v1`) | pass | 2026.44.1, llama.cpp b11393 CPU |
| Lemonade | vision | pass | 2026.44.1, llama.cpp b11393 CPU |
| mistral.rs | chat template | pass | 0.9.5 CPU |
| mistral.rs | tool call | pass | 0.9.5 CPU |
| mistral.rs | context overflow | fail | 0.9.5 CPU |
| mistral.rs | embed | pass (1024 values per vector) | 0.9.5 CPU |
| mistral.rs | vision | pass | 0.9.5 CPU |
| mistral.rs | rerank | not supported for this role: its `openapi.json` lists no rerank route | 0.9.5 CPU |
| mlx-lm | all roles | not run yet | none |
| oMLX | all roles | not run yet | none |
| vMLX | all roles | not run yet | none |
| vllm-mlx | all roles | not run yet | none |
| GenieX | all roles | not verified | none |

The models in the runs:

| Server | Chat | Embed | Rerank | Vision |
|---|---|---|---|---|
| TEI | none | `BAAI/bge-small-en-v1.5` | `BAAI/bge-reranker-base` | none |
| Lemonade | `Qwen3-0.6B-GGUF` | `Qwen3-Embedding-0.6B-GGUF` | `bge-reranker-v2-m3-GGUF` | `Qwen2.5-VL-3B-Instruct-GGUF` |
| mistral.rs | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-Embedding-0.6B` | none | `HuggingFaceTB/SmolVLM-256M-Instruct` |

## Findings

**mistral.rs context overflow.** The server refuses an oversize prompt with the text "Prompt sequence length is greater than 40960". lilbee reports this error as a bad request, not as a context overflow. The step fails for this reason. A separate issue tracks the error classifier. This page does not change the result.

**TEI rerank wire.** TEI answers `POST /rerank` with `texts`. It answers `POST /v1/rerank` with HTTP 404. Use the `tei` wire for a TEI preset.

**Lemonade backend install.** In the run, `lemonade backends install llamacpp:cpu` printed a "Download cancelled" error once. The server log then shows the server installing the llama.cpp backend during the check, and all steps ran.

## Servers not run

oMLX, vMLX and vllm-mlx need macOS 15 or newer. mlx-lm runs on a Mac. The runs for these four servers are not done. Their rows read "not run yet" until a Mac run adds a fixture.

GenieX is not verified. FastFlowLM and Infinity are targets for presets, but this check does not cover them.
