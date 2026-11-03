# Preset results

This page records what each server did when lilbee ran its output check through a stored server preset. See [Server presets](presets.md) for how to add a preset.

## How the check runs

The check uses the real lilbee provider. It stores a preset, points the model settings at it, and runs one step for each role the server documents.

| Role | Steps |
|---|---|
| chat | Chat template (the answer names Rome), tool call, context overflow |
| embed | Two inputs give two vectors of the recorded size |
| rerank | Three documents give three scores, and the best score is for the right document |
| vision | One image chat answers with the colour "red" for a red image |

A recording proxy sits between lilbee and the server. It saves each request and each answer in `tests/conformance/fixtures/presets/`. A test replays each fixture with no server and checks that the results are the same. A result is "pass" only when a run on a real server gave it.

The Fixture column names the fixture that backs each pass row and each fail row. A fixture that records the other result than the row has the word "(fail)" or "(pass)" after its name. For example, the vMLX vision row passes with `vmlx-vision-mllm`, and `vmlx-vision` (fail) records the run without `--is-mllm`. A test replays each fixture and fails when a row and the replayed result disagree, when a row names no fixture, or when a step of a fixture has no row. The fixture also holds the status and the body of each error answer. The replay fails when one of them changes.

## Results

The TEI, Lemonade and mistral.rs runs used CPU only, on a Linux host. The MLX server runs used the host class "hosted macos-15, Metal, paravirtual GPU". Each version is the version of the server in the run.

| Server | Role | Result | Version | Fixture |
|---|---|---|---|---|
| TEI | embed | pass (384 values per vector) | 1.9.4, `cpu-1.9` image, model `BAAI/bge-small-en-v1.5` | `tei-embed` |
| TEI | rerank (wire `tei`) | pass | 1.9.4, `cpu-1.9` image, model `BAAI/bge-reranker-base` | `tei-rerank` |
| TEI | rerank (wire `v1`) | fail: TEI documents no `POST /v1/rerank` route and answers HTTP 404, and lilbee reports "The provider doesn't offer ... on your account" | 1.9.4 | `tei-rerank-v1` |
| TEI | chat, vision | not supported for this role: TEI documents no route | 1.9.4 |  |
| Lemonade | chat template | pass | 2026.44.1, llama.cpp b11393 CPU | `lemonade` |
| Lemonade | tool call | pass | 2026.44.1, llama.cpp b11393 CPU | `lemonade` |
| Lemonade | context overflow | pass | 2026.44.1, llama.cpp b11393 CPU | `lemonade` |
| Lemonade | embed | pass (1024 values per vector) | 2026.44.1, llama.cpp b11393 CPU | `lemonade` |
| Lemonade | rerank (wire `v1`) | pass | 2026.44.1, llama.cpp b11393 CPU | `lemonade` |
| Lemonade | vision | pass | 2026.44.1, llama.cpp b11393 CPU | `lemonade` |
| mistral.rs | chat template | pass | 0.9.5 CPU | `mistralrs-chat` |
| mistral.rs | tool call | pass | 0.9.5 CPU | `mistralrs-chat` |
| mistral.rs | context overflow | pass | 0.9.5 CPU | `mistralrs-chat` |
| mistral.rs | embed | pass (1024 values per vector) | 0.9.5 CPU | `mistralrs-embed` |
| mistral.rs | vision | pass | 0.9.5 CPU | `mistralrs-vision` |
| mistral.rs | rerank | not supported for this role: its `openapi.json` lists no rerank route | 0.9.5 CPU |  |
| mlx-lm | chat template | pass; the fixture of this recording ends with an overflow step that timed out at 120 s, and its HTTP 504 comes from the recording shim, not the server | 0.32.2, hosted macos-15, Metal, paravirtual GPU | `mlxlm-chat` |
| mlx-lm | tool call | pass; the fixture of this recording ends with an overflow step that timed out at 120 s, and its HTTP 504 comes from the recording shim, not the server | 0.32.2, hosted macos-15, Metal, paravirtual GPU | `mlxlm-chat` |
| mlx-lm | context overflow | fail: the server accepts the 60008-token prompt and gives no answer within 120 s. The recording shows HTTP 504 "the server gave no answer within 120 s" from the bounding shim, and lilbee reports `kind=server`. See the findings | 0.32.2, hosted macos-15, Metal, paravirtual GPU | `mlxlm-chat` |
| mlx-lm | embed, rerank | not supported for this role: `mlx_lm/server.py` at tag v0.32.2 routes only `/v1/completions`, `/v1/chat/completions` and `/chat/completions` (lines 1244 to 1246, other paths answer 404 "Not Found" at lines 1249 to 1253), and `mlx_lm/SERVER.md` documents only chat and `/v1/models` | 0.32.2, hosted macos-15, Metal, paravirtual GPU |  |
| mlx-lm | vision | fail with `mlx-community/Qwen3-VL-2B-Instruct-4bit` (a model type mlx-lm loads): `POST /v1/chat/completions` with an image part answers HTTP 404 "Only 'text' content type is supported." The server takes images only for a model that has a processor in `mlx_lm/processors/__init__.py`, and at tag v0.32.2 that list holds only Muse Spark. Its public MLX conversions have 816B parameters, too large for the runner. A tiny random-weight test checkpoint exists, but it cannot name an image colour, so the pass path was not run | 0.32.2, hosted macos-15, Metal, paravirtual GPU | `mlxlm-vision` |
| oMLX | chat template | pass | 0.7.3, hosted macos-15, Metal, paravirtual GPU | `omlx-chat` |
| oMLX | tool call | pass | 0.7.3, hosted macos-15, Metal, paravirtual GPU | `omlx-chat` |
| oMLX | context overflow | pass | 0.7.3, hosted macos-15, Metal, paravirtual GPU | `omlx-chat` |
| oMLX | embed | pass (1024 values per vector) | 0.7.3, hosted macos-15, Metal, paravirtual GPU | `omlx-embed` |
| oMLX | rerank (wire `v1`) | pass, with the memory guard on (the default) and with `--memory-guard off`; both answer HTTP 200 | 0.7.3, hosted macos-15, Metal, paravirtual GPU | `omlx-rerank` |
| oMLX | vision | fail with `Qwen2-VL-2B-Instruct-4bit`: the model answers "Blue" for a red image | 0.7.3, hosted macos-15, Metal, paravirtual GPU | `omlx-vision` |
| oMLX | vision | pass with `Qwen2.5-VL-3B-Instruct-4bit`: the model answers "Red." | 0.7.3, hosted macos-15, Metal, paravirtual GPU | `omlx-vision-3b` |
| vMLX | chat template | pass; the fixture of this recording ends with an overflow step that timed out at 120 s, and its HTTP 504 comes from the recording shim, not the server | 1.6.83 with mlx-lm 0.31.3, hosted macos-15, Metal, paravirtual GPU | `vmlx-chat` |
| vMLX | tool call | pass; the fixture of this recording ends with an overflow step that timed out at 120 s, and its HTTP 504 comes from the recording shim, not the server | 1.6.83 with mlx-lm 0.31.3, hosted macos-15, Metal, paravirtual GPU | `vmlx-chat` |
| vMLX | context overflow | fail: the server logs that it declares 40960 tokens and attempts longer prompts, and gives no answer to the 60008-token prompt within 120 s. The recording shows HTTP 504 "the server gave no answer within 120 s" from the bounding shim, and lilbee reports `kind=server`. See the findings | 1.6.83 with mlx-lm 0.31.3, hosted macos-15, Metal, paravirtual GPU | `vmlx-chat` |
| vMLX | embed | pass (384 values per vector), started with `--embedding-model`; the request must name the embedding model | 1.6.83 with mlx-lm 0.31.3, hosted macos-15, Metal, paravirtual GPU | `vmlx-embed` |
| vMLX | rerank | fail with `BAAI/bge-reranker-v2-m3`: `POST /v1/rerank` answers HTTP 500 "Received 393 parameters not in model" and lists `classifier.*` and `roberta.*` weights. vMLX documents encoder rerankers (BGE, XLM-RoBERTa) and loads the model named in the request. The MLX conversion `flaglow/BAAI-bge-reranker-v2-m3-mlx-fp16` fails the same way (HTTP 500, 393 parameters not in model). No other reranker was tried | 1.6.83 with mlx-lm 0.31.3, hosted macos-15, Metal, paravirtual GPU | `vmlx-rerank`, `vmlx-rerank-flaglow` |
| vMLX | vision | pass with `--is-mllm`; fail without it: `vmlx serve` loads a vision model as text-only and answers HTTP 400 "unsupported media modality image because the loaded runtime is text-only" | 1.6.83 with mlx-lm 0.31.3, hosted macos-15, Metal, paravirtual GPU | `vmlx-vision-mllm`, `vmlx-vision` (fail) |
| vllm-mlx | chat template | pass; the fixture of this recording ends with an overflow step that timed out at 120 s, and its HTTP 504 comes from the recording shim, not the server | 0.4.1, hosted macos-15, Metal, paravirtual GPU | `vllmmlx-chat` |
| vllm-mlx | tool call | pass; the fixture of this recording ends with an overflow step that timed out at 120 s, and its HTTP 504 comes from the recording shim, not the server | 0.4.1, hosted macos-15, Metal, paravirtual GPU | `vllmmlx-chat` |
| vllm-mlx | context overflow | fail: the server accepts the 60008-token prompt and gives no answer within 120 s. The recording shows HTTP 504 "the server gave no answer within 120 s" from the bounding shim, and lilbee reports `kind=server`. See the findings | 0.4.1, hosted macos-15, Metal, paravirtual GPU | `vllmmlx-chat` |
| vllm-mlx | embed | pass (384 values per vector), started with `--embedding-model` | 0.4.1, hosted macos-15, Metal, paravirtual GPU | `vllmmlx-embed` |
| vllm-mlx | rerank | pass with `--rerank-model BAAI/bge-reranker-v2-m3`; the request must name that model, and the name "default" answers HTTP 400. Without `--rerank-model` the route answers HTTP 404 "No reranker model loaded" | 0.4.1, hosted macos-15, Metal, paravirtual GPU | `vllmmlx-rerank-bge`, `vllmmlx-rerank` (fail) |
| vllm-mlx | vision | pass with `mlx-community/Qwen2.5-VL-3B-Instruct-4bit`, no extra flag | 0.4.1, hosted macos-15, Metal, paravirtual GPU | `vllmmlx-vision` |
| GenieX | all roles | not verified | none |  |

The models in the runs:

| Server | Chat | Embed | Rerank | Vision |
|---|---|---|---|---|
| TEI | none | `BAAI/bge-small-en-v1.5` | `BAAI/bge-reranker-base` | none |
| Lemonade | `Qwen3-0.6B-GGUF` | `Qwen3-Embedding-0.6B-GGUF` | `bge-reranker-v2-m3-GGUF` | `Qwen2.5-VL-3B-Instruct-GGUF` |
| mistral.rs | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-Embedding-0.6B` | none | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| mlx-lm | `mlx-community/Qwen3-0.6B-4bit` | none | none | `mlx-community/Qwen3-VL-2B-Instruct-4bit` (fails) |
| oMLX | `Qwen3-0.6B-4bit` (`mlx-community/Qwen3-0.6B-4bit`) | `Qwen3-Embedding-0.6B-4bit-DWQ` (`mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`) | `bge-reranker-v2-m3` (`flaglow/BAAI-bge-reranker-v2-m3-mlx-fp16`) | `Qwen2-VL-2B-Instruct-4bit` (`mlx-community/Qwen2-VL-2B-Instruct-4bit`), `Qwen2.5-VL-3B-Instruct-4bit` (`mlx-community/Qwen2.5-VL-3B-Instruct-4bit`) |
| vMLX | `mlx-community/Qwen3-0.6B-4bit` | `mlx-community/all-MiniLM-L6-v2-4bit` | `BAAI/bge-reranker-v2-m3` and `flaglow/BAAI-bge-reranker-v2-m3-mlx-fp16` (both fail) | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` |
| vllm-mlx | `mlx-community/Qwen3-0.6B-4bit` | `mlx-community/all-MiniLM-L6-v2-4bit` | `BAAI/bge-reranker-v2-m3` | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` |

## Findings

**mistral.rs context overflow.** The server refuses an oversize prompt with the text "Prompt sequence length is greater than 40960". lilbee reports this error as a context overflow. The step passes.

**TEI embed limit.** The output check passes for TEI embed. The server answers `encoding_format` values `float` and `base64` with HTTP 200, and the base64 text holds 384 floats. A stock TEI server refuses more than 32 inputs in one request: it answers HTTP 422 with "batch size 33 > maximum allowed batch size 32". lilbee sends at most 32 inputs per embed request to a preset, and joins the vectors in input order, so an ingest never meets this limit. The wire check E8 of the conformance harness still fails on a stock TEI server, because its largest batch holds 33 inputs and the harness sends it straight to the server. Start TEI with `--max-client-batch-size 64` only to make E8 pass.

**TEI rerank wire.** TEI answers `POST /rerank` with `texts`. It answers `POST /v1/rerank` with HTTP 404. lilbee reports this 404 as "The provider doesn't offer ... on your account", which blames the account, not the wire. The fixture `tei-rerank-v1.json` records this failure. Use the `tei` wire for a TEI preset.

**Lemonade backend install.** In the run, `lemonade backends install llamacpp:cpu` printed a "Download cancelled" error once. The pulls of `Qwen3-0.6B-GGUF` and `Qwen3-Embedding-0.6B-GGUF` also failed with "Could not connect". The server then downloaded both models on first use, and all steps ran.

## MLX server findings

**Context overflow.** mlx-lm, vMLX and vllm-mlx do not refuse the oversize prompt. They start to process all 60008 tokens and give no answer. oMLX refuses the prompt, and lilbee reports a context overflow. The recorder talks to a shim that waits 120 s for the server and then answers HTTP 504, so the recording ends with a failure that replays. A direct request to each server with no shim also got no answer in 150 s.

- mlx-lm: `mlx_lm/server.py` at tag v0.32.2 has no prompt-length check on the chat route. The server log shows "Prompt processing progress: 30720/60009".
- vMLX: the log reads "model declares 40960 tokens ... Advice only: longer prompts are attempted, not rejected". `vmlx_engine/server.py` at tag v1.6.83 has an HTTP 413 "prompt_too_long" response (line 1191 onward), but it did not fire for this prompt. The cause is not known.
- vllm-mlx: `vllm_mlx/server.py` in release 0.4.1 checks only `max_tokens` (line 286) and has no prompt limit. The log shows "CLIENT DISCONNECTED (non-stream) elapsed=150.3s".

The fixtures `mlxlm-chat.json`, `vmlx-chat.json` and `vllmmlx-chat.json` hold template, tool call and the failed overflow. The overflow exchanges in these three fixtures are not server answers: the 504 comes from the recording shim. The fixture format has no field for a note, so the fixtures carry none, and the shim 504 is stated here.

**oMLX rerank.** The reranker answers HTTP 200 with the memory guard on and with `--memory-guard off`, after a chat and an embed model load. The runner has 7.5 GB of memory. With the memory guard on, the pool evicts the chat and embed models first to make room for the reranker (the log reads "Evicting 'Qwen3-0.6B-4bit' to fit 'bge-reranker-v2-m3'"). The fixture `omlx-rerank.json` was recorded with the guard off.

**vMLX install.** vMLX 1.6.83 needs `SequenceStateMachine` from mlx-lm. mlx-lm 0.32.2 lacks it, so `vmlx serve` fails with an ImportError. The run used mlx-lm 0.31.3.

**oMLX vision.** The failure with `Qwen2-VL-2B-Instruct-4bit` comes from the model. The chat request that lilbee sends is the same as the request in the Lemonade vision fixture, which passes: one user message, an `image_url` part with a PNG data URL, then the text part, with `stream` false. `Qwen2.5-VL-3B-Instruct-4bit` answers "Red." to that request.

**vMLX vision.** `vmlx serve` needs `--is-mllm` (`--is-mllm` is the option in `vmlx_engine/cli.py`; the server log text calls it "--mllm") to load a vision model with images. Without the flag, the server answers HTTP 400 for an image part. The fixture `vmlx-vision.json` records this failure, and `vmlx-vision-mllm.json` records the pass.

**vllm-mlx rerank.** The model name `mlx-community/jina-reranker-v2-base-multilingual` in the `--help` text does not exist on the Hub. `BAAI/bge-reranker-v2-m3` loads and ranks correctly. The server accepts only the name given to `--rerank-model`. The fixture `vllmmlx-rerank-bge.json` records the pass. The fixture `vllmmlx-rerank.json` records a server started without `--rerank-model`, which answers HTTP 404 "No reranker model loaded". The HTTP 400 for the name "default" is in the job log only.

**vMLX rerank.** The earlier HTTP 500 row used a chat model as the reranker. With `BAAI/bge-reranker-v2-m3` the route fails with HTTP 500 "Received 393 parameters not in model". The fixture `vmlx-rerank.json` records this for the original Hub checkpoint. The MLX conversion `flaglow/BAAI-bge-reranker-v2-m3-mlx-fp16` gives the same HTTP 500 and the same list of 393 parameters, and `vmlx-rerank-flaglow.json` records it. The same conversion works on oMLX.

**mlx-lm vision.** `mlx_lm.server` at tag v0.32.2 reads image parts only when the loaded model has a processor in `mlx_lm/processors/__init__.py`. Only Muse Spark has one. For `mlx-community/Qwen3-VL-2B-Instruct-4bit` the server answers HTTP 404 "Only 'text' content type is supported." The fixture `mlxlm-vision.json` records this. lilbee shows the 404 as "The provider doesn't offer ... on your account".

## Servers not run

GenieX is not verified. FastFlowLM and Infinity are targets for presets, but this check does not cover them.
