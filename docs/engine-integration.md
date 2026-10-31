# Engine integration

This document says what an inference engine must do before lilbee can use it. Each requirement has an id, a failing case that proves it, and a golden file in `tests/conformance/golden/`. The suite in `tests/conformance/` checks every requirement.

## How to read it

An E requirement is for the engine. An M requirement is for a managed engine, which lilbee starts and stops. An L requirement is for lilbee itself: lilbee must work around an engine that lacks an optional route.

A verdict is `pass`, `fallback`, `fail` or `skip`. `fallback` means the engine lacks an optional route and the stated rule still works. `skip` means the engine does not claim the role. An engine is supported when no claimed role is `fail`.

Llama-server extras are not requirements. These are `/tokenize`, `/detokenize`, `/apply-template`, `/memory`, `/props`, prompt-progress frames and `embd_normalize`. If lilbee needs one, the need is an L requirement.

A "row" is a numbered observation from the engine spike, which measured four engines: mlx_lm.server, TEI, Lemonade and mistral.rs. Each Evidence line states what the rows showed, so no other file is needed.

## Run the suite

Run tier 1 against a server that is already up. `--models` gives the model id of each claimed role. `--health-path` is needed only when the health path is not `/health`. It applies to a running engine. A pin run starts the engine and waits for 200 on `/health`, so a pinned engine answers `/health`.

```bash
pytest tests/conformance/test_real_engine.py \
  --engine-url URL \
  --claims chat,embed,rerank,vision,reasoning \
  --models chat=CHAT_ID,embed=EMBED_ID,rerank=RERANK_ID,vision=VISION_ID
```

Run tier 2 against a pinned binary with `--engine-pin PATH`. `PATH` is a JSON file, the pin record, with the keys `binary` (the engine file), `launch` (the argument list, with `{binary}` and `{port}`), and the M requirement fields (`version`, `platforms`, `backend`, `memory`, `licenses`). The suite starts the binary from `launch`, runs tier 1 against it, then runs the M requirements. Give `--claims` and `--models` as for tier 1.

A real run takes minutes. The module `tests/conformance/test_real_engine.py` sets its own pytest timeout of one hour, so the repo timeout of 60 seconds does not stop it. Add `--record FILE` to keep the verdicts as JSON lines.

Without these options the suite runs against the fake engine in `tests/conformance/fake_engine.py` and the fake binary in `tests/conformance/fake_binary.py`. No model is needed. The content requirements (E9 order, E10 colour) run against the fake and against a real engine. E11 needs a real engine; against the fake it only shows that the check can fail.

## Tier 1

Tier 1 is a server that lilbee does not start. The user runs it.

### E1 Health

The engine answers 200 on its health path, which is `/health` unless the run gives `--health-path`. Lilbee has no health path setting today; the option names the path for the E1 probe only. If the path does not answer, `GET /v1/models` with 200 gives `fallback`.

Evidence: row 1, row 12, row 22, row 38. Lemonade answers 404 on `/health` and 200 on `/live` (row 22). The caller is `src/lilbee/providers/fleet/swap_manager.py`.

### E2 Ready means served

For each claimed role, one request that asks for one token gets 200 with the shape of that role. A health answer alone is not ready.

Evidence: row 36, row 37. The CUDA mistral.rs answered health 200 and chat 503.

### E3 Model list

`GET /v1/models` lists every claimed model id. A 404 gives `fallback`: the id comes from the preset, and every surface says the list is unavailable.

Evidence: row 13, row 34, row 38. The callers are `src/lilbee/providers/backends/liter_llm/backend.py` and `src/lilbee/providers/local_servers/registry.py`.

### E4 Chat

`POST /v1/chat/completions` without streaming returns `choices[0].message.content` as a non-empty string and `finish_reason` `stop` for a system, user, assistant, user conversation. The request allows 1024 tokens, so a reasoning model has room to think and answer. An engine that sends reasoning and no `content` fails, because lilbee shows `content` to the user.

Evidence: row 6, row 26, row 36. The caller is `src/lilbee/providers/fleet/client.py`.

### E5 Stream

A stream sends `data:` frames with `choices[0].delta` and ends with `data: [DONE]`. It sends no `error` frame. At least one frame carries text in `content`. Text in `reasoning_content` or `reasoning` alone counts only when the stream ends with `finish_reason` `length`: the request allows 32 tokens, and a reasoning model can spend them all on thinking.

Evidence: row 36. The CUDA mistral.rs sent an error frame and then `[DONE]`.

### E6 Reasoning

An engine that claims reasoning puts the reasoning text in `message.reasoning_content` or `message.reasoning`, or inline in `content` between `<think>` tags.

Evidence: row 6, row 26, row 48, row 50, row 51. mlx_lm.server names the field `reasoning` (row 6). mistral.rs on CPU uses `reasoning_content` (row 50). Rows 50 and 51 close row 48.

### E7 Tool calls

The engine returns native `message.tool_calls` with a function name and JSON arguments, also in the stream form. Bare JSON in `content` gives `fallback`, because the client recovers it.

Evidence: row 9, row 29, row 48, row 55. Row 55 is the real mistral.rs CPU run of the golden request. The fake checks the shape. Only a real engine proves that a model emits a tool call.

### E8 Embeddings

`POST /v1/embeddings` returns N vectors in input order, as float and as base64. The suite checks the order at batches of 3, 9 and 33 texts: vector i of a batch must match the vector of text i embedded alone. All vectors have one dimension, hold finite numbers, are not all zero, and differ for different inputs. A normalized or a raw vector both pass.

Evidence: row 10, row 17, row 30, row 32, row 44, row 46, row 54. Row 32 and row 46 answered 200 with zeros or a vector that is not a score. Row 17 is not a defect for embeddings. Row 10 was not probed.

### E9 Rerank

`POST /v1/rerank` with `{model, query, documents}` returns `{"results": [{"index": int, "relevance_score": float}]}`. Each `index` is an integer, and the indexes are exactly 0 to N-1 for N documents, because lilbee reads the score of document i at `index` i. The scores are finite and not all equal. The relevant document scores higher.

Evidence: row 10, row 19, row 20, row 21, row 32, row 33, row 46, row 47. Lemonade serves the route (row 33). TEI serves `/rerank` with other field names (row 20) and not `/v1/rerank` (row 21). mistral.rs serves no route (row 47).

### E10 Vision

An engine that claims vision answers 200 to one chat with an `image_url` part that holds a PNG data URL. The answer names the colour of a generated solid red image.

Evidence: no spike row. The callers are `src/lilbee/vision.py` and `src/lilbee/providers/backends/liter_llm/config.py`.

### E11 Deterministic output

Two chats at temperature 0 with a fixed seed return the same text. The text is the answer and the reasoning. The answer is not empty. Reasoning alone counts only when `finish_reason` is `length`, because the request allows 32 tokens. The check also runs against the fake engine, to show that it can fail. A pass counts as a result for a real engine only.

Evidence: no spike row. The golden comparison of the next engine items needs it.

### E12 Prompt past the context

A prompt longer than the context gets a 4xx answer within the timeout (default 60 s). The engine does not hang and does not answer 5xx.

Evidence: row 8, row 28, row 43, row 52. mlx_lm.server hung until the client timeout (row 8). The wording of the 400 differs by engine (row 43, row 52). The caller is `src/lilbee/providers/fleet/client.py`.

### E13 Route not served

A role the engine does not claim answers 400, 404, 405 or 501, or answers with a valid body of that role. A 200 with a degenerate body fails. Every other status fails, including 401, 403, 429 and 5xx. An embedding server that answers a chat call with 400 "model not found" passes.

Evidence: row 32, row 33, row 46, row 47. Row 33 is a chat model that answers 501 on rerank.

### E14 Response format and JSON schema

A chat request with `response_format` of type `json_object` gets content that parses whole to an object, with no code fence and no trailing text. A request of type `json_schema` gets content that also holds every required key with the declared type and no other key. An engine that ignores the field fails. The row runs only when the engine claims `json`. No caller in `src/lilbee` sends the field today, so a failure does not refuse the engine.

Evidence: no spike row. Issue 1190 lists the field.

### L1 No tokenize route

Without `/tokenize` or `/detokenize` the client estimates the token count. Embed and rerank do not fail.

Evidence: row 2, row 14, row 15, row 23, row 31, row 39, row 45, row 53. The code is in `src/lilbee/providers/fleet/client.py`. The test is a strict expected failure until the client item lands.

### L2 No apply-template route

A 404 with a JSON error envelope on `/apply-template` falls back to an estimate.

Evidence: row 3, row 16, row 24, row 40. Lemonade answers with an envelope that the client does not treat as an absent route (row 24).

### L3 Reasoning field name

The client reads `reasoning` as well as `reasoning_content`.

Evidence: row 6.

### L4 Overflow wording

The overflow classifier knows the code `context_length_exceeded` and the text "sequence length is greater than".

Evidence: row 43, row 52.

### L5 Error-only stream

A stream whose only frame is an `error` object raises.

Evidence: row 36.

### L6 Rerank route

An engine that claims rerank gets `/v1/rerank`. Rank pooling over embeddings stays for the managed llama engine only.

Evidence: row 19, row 20, row 32, row 33, row 46, row 47. TEI refuses rank pooling (row 19).

### L7 Unavailable role

A claimed role whose E requirements fail is unavailable. Every surface (REST, MCP, CLI, TUI and status) says so by name.

Evidence: row 13.

### L8 Ready after serving

A managed start is ready after E2, not after E1.

Evidence: row 37.

## Tier 2

Tier 2 is a managed engine. It meets all of tier 1 and the requirements below.

### M1 Launch and help

The engine has a launch template with `{port}`. Its `--help` exits 0 and prints text. Lilbee passes only the flags that the help text lists.

Evidence: row 5, row 16, row 25. The help text of mlx_lm.server, TEI and Lemonade lacks `--memory`. The code is in `src/lilbee/providers/fleet/adapters.py` and `src/lilbee/providers/fleet/swap_config.py`. The test runs against the fake binary.

### M2 Device list

`--list-devices` prints the llama-server line format, or the pin declares `backend` and the verdict is `fallback`. Devices then come from host detection. An engine with neither fails.

Evidence: row 5, row 16, row 25, row 42. The CUDA device was not found and the backend was unknown (row 42). The code is in `src/lilbee/providers/fleet/devices.py`.

### M3 Memory estimate

The engine has a memory estimate before launch: a sizing command that prints bytes, or a declared rule (`file_size`, verdict `fallback`). A readback after launch is optional.

Evidence: row 4, row 16, row 25, row 37, row 41. No engine in the spike served a readback (row 41). The CUDA engine sized its KV cache and then had no room (row 37). The code is in `src/lilbee/providers/fleet/vram.py` and `src/lilbee/providers/fleet/readback.py`.

### M4 Pin

A pin has a version. Each platform has an https URL and a 64-character sha256. A macOS platform has `min_os`.

Evidence: row 11. The Metal archive of mistral.rs panicked on macOS 14.6.1. The pins are in `engine-versions.env` and `sources.json`.

### M5 Licences

The licence list of the whole engine tree has at least one entry, no empty entry, and no GPL or AGPL entry.

Evidence: golden file `tests/conformance/golden/M5.json`. The rule comes from the licence decision of the engine items. The check runs offline.

### M6 Stop

SIGTERM ends the engine within 4 seconds, also with a request in flight. Removal leaves no process.

Evidence: row 8. The serial server stayed busy. The code is in `src/lilbee/providers/fleet/engine_reaper.py` and `src/lilbee/providers/fleet/child_guard.py`.

## Failure rows and requirements

Every spike row maps to a requirement or has a reason. Rows 7 and 49 are an llama-server extra. Row 18 breaks nothing. Rows 27 and 35 work.

| Row | Requirement | Note |
|---|---|---|
| 1 | E1 | works |
| 2 | L1 | |
| 3 | L2 | |
| 4 | M3 | |
| 5 | M2, M1 | |
| 6 | E4, E6, L3 | |
| 7 | none | prompt-progress frames are an llama-server extra; the client works without them |
| 8 | E12, M6 | |
| 9 | E7 | works |
| 10 | E8, E9 | not probed; owed |
| 11 | M4 | |
| 12 | E1 | works |
| 13 | E3, L7 | |
| 14 | L1 | |
| 15 | L1 | |
| 16 | M1, M2, M3, L2 | |
| 17 | E8 | not a defect for embeddings |
| 18 | none | an engine that truncates an oversize embedding input itself breaks nothing |
| 19 | E9, L6 | |
| 20 | E9, L6 | works, other field names |
| 21 | E9 | route absent |
| 22 | E1 | |
| 23 | L1 | |
| 24 | L2 | |
| 25 | M1, M2, M3 | |
| 26 | E4, E6 | |
| 27 | none | works |
| 28 | E12 | works |
| 29 | E7 | works |
| 30 | E8 | works |
| 31 | L1 | |
| 32 | E8, E9, E13, L6 | |
| 33 | E9, E13, L6 | |
| 34 | E3 | |
| 35 | none | works |
| 36 | E2, E4, E5, L5 | |
| 37 | E2, M3, L8 | |
| 38 | E1, E3 | |
| 39 | L1 | |
| 40 | L2 | |
| 41 | M3 | |
| 42 | M2 | |
| 43 | E12, L4 | |
| 44 | E8 | works |
| 45 | L1 | |
| 46 | E8, E9, L6 | |
| 47 | E9, L6 | |
| 48 | E6, E7 | closed by rows 50, 51 and 55 |
| 49 | none | prompt-progress frames are an llama-server extra; the client works without them |
| 50 | E6 | works |
| 51 | E6 | |
| 52 | E12, L4 | |
| 53 | L1 | |
| 54 | E8 | works |
| 55 | E7 | real run, see E7 |

## Recordings of a real engine

A recording of a real engine is a file under `tests/conformance/fixtures/engines/`. Take it with `python -m tests.conformance.record_presets` and the `--strict` flag. The flag adds the steps `json_object`, `json_schema`, `reasoning` and `named`, and a two-image `vision` step. The file stores `"strict": true`, and a test fails for each recording that does not. The replay tests run each recording through the real provider with no server, in the `integration` job.

The step `named` sends one request through the routing provider. It passes when the log line `served by engine=<name> version=<version> role=<role> model=<id>` and the `serving` entry of the status for that role both name the engine. A server preset has the preset name as its engine name and `preset` as its version. A managed engine has its name and its pin.

The 7 preset recordings in `tests/conformance/fixtures/presets/` are not strict and have no `reasoning` step. The `reasoning` step of a strict recording reads the answer text and the finish reason through the provider. It does not read `reasoning_content` or `reasoning`. A pass of the preset `reasoning` step does not prove that the engine splits reasoning from the answer. Row E6 of the wire harness proves the split.


## Add an engine

Add a profile to `tests/conformance/fake_engine.py` that replays the measured behaviour, and add a golden file only for a new requirement. The harness in `tests/conformance/harness.py` does not change for a new engine.
