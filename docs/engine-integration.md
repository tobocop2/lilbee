# Engine integration

This document says what an inference engine must do before lilbee can use it. Each requirement has an id, a failing case that proves it, and a golden file in `tests/conformance/golden/`. The suite in `tests/conformance/` checks every requirement.

## How to read it

An E requirement is for the engine. An M requirement is for a managed engine, which lilbee starts and stops. An L requirement is for lilbee itself: lilbee must work around an engine that lacks an optional route.

A verdict is `pass`, `fallback`, `fail` or `skip`. `fallback` means the engine lacks an optional route and the stated rule still works. `skip` means the engine does not claim the role. An engine is supported when no claimed role is `fail`.

Llama-server extras are not requirements. These are `/tokenize`, `/detokenize`, `/apply-template`, `/memory`, `/props`, prompt-progress frames and `embd_normalize`. If lilbee needs one, the need is an L requirement.

A "row" is a numbered observation from the engine spike, which measured five engines: mlx_lm.server, TEI, Lemonade, mistral.rs and oMLX. Rows 56 to 91 come from a macOS 15 runner with Metal. Each Evidence line states what the rows showed, so no other file is needed.

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

Evidence: row 1, row 12, row 22, row 38, row 57, row 70, row 80. Lemonade answers 404 on `/health` and 200 on `/live` (row 22). mistral.rs, mlx_lm.server and oMLX answer 200 on Metal (rows 57, 70, 80). The caller is `src/lilbee/providers/fleet/swap_manager.py`.

### E2 Ready means served

For each claimed role, one request that asks for one token gets 200 with the shape of that role. A health answer alone is not ready.

Evidence: row 36, row 37, row 91. The CUDA mistral.rs answered health 200 and chat 503. oMLX answered 507 on rerank when its memory guard refused the model while chat and embed models were loaded (row 91). The same call passed with the guard off.

### E3 Model list

`GET /v1/models` lists every claimed model id. A 404 gives `fallback`: the id comes from the preset, and every surface says the list is unavailable.

Evidence: row 13, row 34, row 38, row 79. The callers are `src/lilbee/providers/backends/liter_llm/backend.py` and `src/lilbee/providers/local_servers/registry.py`.

### E4 Chat

`POST /v1/chat/completions` without streaming returns `choices[0].message.content` as a non-empty string and `finish_reason` `stop` for a system, user, assistant, user conversation. The request allows 1024 tokens, so a reasoning model has room to think and answer. An engine that sends reasoning and no `content` fails, because lilbee shows `content` to the user.

Evidence: row 6, row 26, row 36. The caller is `src/lilbee/providers/fleet/client.py`.

### E5 Stream

A stream sends `data:` frames with `choices[0].delta` and ends with `data: [DONE]`. It sends no `error` frame. At least one frame carries text in `content`. Text in `reasoning_content` or `reasoning` alone counts only when the stream ends with `finish_reason` `length`: the request allows 32 tokens, and a reasoning model can spend them all on thinking.

Evidence: row 36, row 71. The CUDA mistral.rs sent an error frame and then `[DONE]`. A Metal mlx_lm.server stream spent all 200 tokens on reasoning and sent no text, only the finish and usage frames (row 71).

### E6 Reasoning

An engine that claims reasoning puts the reasoning text in `message.reasoning_content` or `message.reasoning`, or inline in `content` between `<think>` tags.

Evidence: row 6, row 26, row 48, row 50, row 51, row 59, row 61, row 71, row 81, row 84. mlx_lm.server names the field `reasoning` (rows 6, 71). mistral.rs on CPU and Metal and oMLX use `reasoning_content` (rows 50, 61, 84). mistral.rs and oMLX on Metal also send the reasoning inline between `<think>` tags (rows 59, 81). Rows 50 and 51 close row 48.

### E7 Tool calls

The engine returns native `message.tool_calls` with a function name and JSON arguments, also in the stream form. Bare JSON in `content` gives `fallback`, because the client recovers it.

Evidence: row 9, row 29, row 48, row 55, row 73, row 77, row 83. Row 55 is the real mistral.rs CPU run of the golden request. Rows 73, 77 and 83 are the same call on Metal for mlx_lm.server, mistral.rs and oMLX. Row 77 gives the result that an earlier Metal probe cut off. The fake checks the shape. Only a real engine proves that a model emits a tool call.

### E8 Embeddings

`POST /v1/embeddings` returns N vectors in input order, as float and as base64. The suite checks the order at batches of 3, 9 and 33 texts: vector i of a batch must match the vector of text i embedded alone. All vectors have one dimension, hold finite numbers, are not all zero, and differ for different inputs. A normalized or a raw vector both pass.

Evidence: row 10, row 17, row 30, row 32, row 44, row 46, row 54, row 67, row 87. Row 32 and row 46 answered 200 with zeros or a vector that is not a score. Row 17 is not a defect for embeddings. Row 10 was not probed.

### E9 Rerank

`POST /v1/rerank` with `{model, query, documents}` returns `{"results": [{"index": int, "relevance_score": float}]}`. Each `index` is an integer, and the indexes are exactly 0 to N-1 for N documents, because lilbee reads the score of document i at `index` i. The scores are finite and not all equal. The relevant document scores higher.

Evidence: row 10, row 19, row 20, row 21, row 32, row 33, row 46, row 47, row 76, row 78, row 89. Lemonade serves the route (row 33). oMLX serves `/v1/rerank` with a reranker model and no `/rerank` (row 89). TEI serves `/rerank` with other field names (row 20) and not `/v1/rerank` (row 21). mistral.rs serves no route (rows 47, 78). mlx_lm.server serves no embed or rerank route (row 76).

### E10 Vision

An engine that claims vision answers 200 to one chat with an `image_url` part that holds a PNG data URL. The answer names the colour of a generated solid red image.

Evidence: no spike row. The callers are `src/lilbee/vision.py` and `src/lilbee/providers/backends/liter_llm/config.py`.

### E11 Deterministic output

Two chats at temperature 0 with a fixed seed return the same text. The text is the answer and the reasoning. The answer is not empty. Reasoning alone counts only when `finish_reason` is `length`, because the request allows 32 tokens. The check also runs against the fake engine, to show that it can fail. A pass counts as a result for a real engine only.

Evidence: no spike row. The golden comparison of the next engine items needs it.

### E12 Prompt past the context

A prompt longer than the context gets a 4xx answer within the timeout (default 60 s). The engine does not hang and does not answer 5xx.

Evidence: row 8, row 28, row 43, row 52, row 66, row 75, row 86. mlx_lm.server hung until the client timeout (rows 8, 75). The wording of the 400 differs by engine (rows 43, 52, 66, 86). The caller is `src/lilbee/providers/fleet/client.py`.

mlx_lm.server also keeps working on a request that is not streamed after its client goes, so every later request waits. Its profile sets `works_on_after_disconnect`. When a request to such a server times out, lilbee restarts that role's server and reports the timeout, with no retry. New requests wait for the new server. The old server stops once every other request on it has ended, or after 45 s. That wait runs before the restart takes the build lock, so other lilbee processes can still take the lock. A request that is still running at the end of the wait ends with "The chat model server stopped answering and was restarted". A request with no answer yet then takes one retry on the new server; a stream that has sent text does not. A timeout from a server that a restart already replaced restarts nothing and takes the one retry.

### E13 Route not served

A role the engine does not claim answers 400, 404, 405 or 501, or answers with a valid body of that role. A 200 with a degenerate body fails. Every other status fails, including 401, 403, 429 and 5xx. An embedding server that answers a chat call with 400 "model not found" passes.

Evidence: row 32, row 33, row 46, row 47, row 76, row 78. Row 33 is a chat model that answers 501 on rerank. Rows 76 and 78 are 404 answers on Metal.

### E14 Response format and JSON schema

A chat request with `response_format` of type `json_object` gets content that parses whole to an object, with no code fence and no trailing text. A request of type `json_schema` gets content that also holds every required key with the declared type and no other key. An engine that ignores the field fails. The row runs only when the engine claims `json`. The `/v1/chat/completions` route refuses a value that is not an object with a 400 that names the field, and forwards an object to the engine as sent. An engine that rejects a malformed object answers with its own error. An engine whose profile has `structured_output` off (mlx-lm) gets no `json_object` field, and a `json_schema` request fails with a 400 before any request is sent. A `text` request goes through. Any other `response_format` value fails with a 400 that names the value, because that engine ignores it. Callers in `src/lilbee` send the field (`memory.py`, `searcher.py` and `extractor.py`).

Evidence: no spike row. Issue 1190 lists the field.

### L1 No tokenize route

Without a working `/tokenize` or `/detokenize` the client estimates the token count. A route that refuses the request body counts as not working. Embed and rerank do not fail.

Evidence: row 2, row 14, row 15, row 23, row 31, row 39, row 45, row 53, row 64, row 68, row 69, row 74, row 85, row 88, row 90. oMLX has the route and answers 422 to the body with `content` (row 85). That 422 broke the oversize embed and the rank-pooling rerank (rows 88, 90). The mistral.rs profile answers on Metal (rows 64, 68). The code is in `src/lilbee/providers/fleet/client.py`. The tests are strict expected failures until the client item lands.

### L2 No apply-template route

A 404 with a JSON error envelope on `/apply-template` falls back to an estimate.

Evidence: row 3, row 16, row 24, row 40, row 65, row 74, row 85. Lemonade answers with an envelope that the client does not treat as an absent route (row 24).

### L3 Reasoning field name

The client reads `reasoning` as well as `reasoning_content`.

Evidence: row 6, row 71. Row 71 is mlx_lm.server on Metal.

### L4 Overflow wording

The overflow classifier knows the code `context_length_exceeded` and the text "sequence length is greater than".

Evidence: row 43, row 52, row 66, row 86. oMLX sends the code in its 400 (row 86), and the client does not yet classify that answer as an overflow.

### L5 Error-only stream

A stream whose only frame is an `error` object raises.

Evidence: row 36.

### L6 Rerank route

An engine that claims rerank gets `/v1/rerank`. Rank pooling over embeddings stays for the managed llama engine only.

Evidence: row 19, row 20, row 32, row 33, row 46, row 47, row 78, row 89, row 90. TEI refuses rank pooling (row 19). oMLX serves `/v1/rerank` and refuses rank pooling (rows 89, 90).

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

Evidence: row 5, row 16, row 25, row 57, row 70. The help text of mlx_lm.server, TEI, Lemonade and mistral.rs on Metal lacks `--memory`. The code is in `src/lilbee/providers/fleet/adapters.py` and `src/lilbee/providers/fleet/swap_config.py`. The test runs against the fake binary.

### M2 Device list

`--list-devices` prints the llama-server line format, or the pin declares `backend` and the verdict is `fallback`. Devices then come from host detection. An engine with neither fails.

Evidence: row 5, row 16, row 25, row 42, row 58, row 70, row 80. The CUDA device was not found and the backend was unknown (row 42). On Metal, mistral.rs, mlx_lm.server and oMLX reject `--list-devices` with exit 2 and the backend is unknown (rows 58, 70, 80). The code is in `src/lilbee/providers/fleet/devices.py`.

### M3 Memory estimate

The engine has a memory estimate before launch: a sizing command that prints bytes, or a declared rule (`file_size`, verdict `fallback`). A readback after launch is optional.

Evidence: row 4, row 16, row 25, row 37, row 41, row 57, row 80. No engine in the spike served a readback (rows 41, 57, 80). The CUDA engine sized its KV cache and then had no room (row 37). The code is in `src/lilbee/providers/fleet/vram.py` and `src/lilbee/providers/fleet/readback.py`.

### M4 Pin

A pin has a version. Each platform has an https URL, a 64-character sha256, a size and `min_os`. The OS floor `min_os` comes from the platform tag of the wheel file name: a Windows wheel has none. `src/lilbee/providers/fleet/engine_packs.json` is the pack record. It ships in the wheel as package data, and lilbee reads it once at import. The record has one entry for each published wheel: `file`, `url`, `sha256` and `size`. The loader builds the table from the record. It stops with an error that names the row if a field is missing, if `size` is not a positive whole number, if `sha256` is not 64 lowercase hex digits, or if `url` is not an `https://` url that ends with the file name. Nobody edits the record by hand. Run `uv run python tools/gen_engine_packs.py TAG [TAG ...] > src/lilbee/providers/fleet/engine_packs.json` to write it. The tool gives no row to a mistral.rs CUDA wheel, whatever its version or build, because mistral.rs has no CUDA pack on purpose. The tool reads the `name`, `digest` and `size` fields of the release assets from the GitHub API and downloads nothing. For each engine version and cell, the tool keeps the wheel with the highest build number, whatever the TAG order. If two different files have the same cell, version and build, the tool stops with an error. A later TAG replaces a wheel of the same name in an earlier TAG. The tool warns when a TAG is a draft release, because its urls answer 404 until the release is published. The loader also refuses an empty record. The table has one row for each engine, version and cell. The adapter copies the row of its pinned version and of the host cell into the `platforms` field of the pin record. A version that the table does not list has no pack. The release base `PACKAGES_URL` is `https://github.com/tobocop2/lilbee-engine-packs/releases/download/`. If a pack pin has no local wheel, lilbee downloads the wheel from the url of its table row. The download refuses a url that does not start with `PACKAGES_URL`.

Evidence: row 11, row 56. The Metal archive of mistral.rs panicked on macOS 14.6.1 (row 11) and started on macOS 15.8.2 (row 56), so the real floor of that archive is macOS 15. The Metal libraries inside the 0.9.5 binary target `air64-apple-macosx15.0.0`, because the upstream build runs on macOS 15 and sets no deployment target for them. The load command of the binary itself says `minos 11.0`, and the published wheel carries the tag `macosx_11_0_arm64` from that value. A published wheel keeps its tag, so `min_os` of the 0.9.5 Metal row is 11.0 and understates the floor. The Metal pack of a later mistral.rs build must carry the tag `macosx_15_0_arm64`. The pins are in `engine-versions.env` and `sources.json`.

### M5 Licences

The licence list of the whole engine tree has at least one entry, no empty entry, and no GPL or AGPL entry.

A pin has no archive, so it carries no notice file. The pin check screens the licence terms only: a term that needs a notice (MPL-2.0, OpenSSL, the NVIDIA CUDA EULA, the Visual C++ runtime) passes it. The install then checks the archive: the wheel or the unpacked tree must hold the notice file of each such term, and a build without it is refused. An MPL-2.0 notice, `SOURCE-MPL-2.0`, has one line for each crate: the crates.io download URL, then the sha256 of the crate. Every line must match a crate in `engine_crates.py`, which a pack build fills from its `Cargo.lock`. The check never reads the network. A wheel or tree that holds a GPL, AGPL or LGPL licence file, or a bundled libquadmath, libgmp, libgmpxx, libmpfr, libreadline or libhistory (shared library or static `.a` archive), is refused whatever it declares.

The scan is a net, not an audit. It has these limits:

- The crate table in `engine_crates.py` is a literal. Each pack build fills it from its `Cargo.lock`. A pack with an MPL-2.0 crate that has no row is refused until its row is added.
- The scan reads a file only when its name holds `licence`, `license`, `copying`, `notice` or `copyright` (any case), when its name holds one of the copyleft words below, or when the file is in a `licenses` folder or in a folder named with a copyleft word. A GPL text in a file with any other name, outside such a folder, installs.
- A file is refused when its name holds the word `gpl`, `lgpl`, `agpl`, `lesser` or `affero`, or when its text holds a GNU licence title followed by the word "version". A word ends at a letter boundary, so `gpl-3.0.txt` and `LICENSE.GPLv3` match, and `gplus.py` and `LICENSE-gplus` do not. Only the file name is refused on a word; a folder name makes the scan read the file. The title match ignores case, spacing and no-break spaces. It reads every decoding that fits and refuses the file when one of them holds the title. The first decoding is UTF-8, or Latin-1 when the bytes are not UTF-8. The second is UTF-16 or UTF-32 when the file starts with a byte order mark. It is also UTF-16 without a mark when one byte of each pair is never NUL and most of the other bytes are NUL, as in ASCII text. UTF-32 without a mark installs. A text that only names the licence passes: the NVIDIA CUDA notice names the GPL and the LGPL and is not a GPL text.
- A file that carries the name of a bundled library, such as `gmp-COPYING` or `readline-LICENSE`, is read like any other licence file. Only a shared or static library file of such a name is refused as a library.
- A GPL text that has no title line, or a title with no "version" after it, installs when its name has no copyleft word.
- A static import library such as `gmp.lib` installs.

Evidence: golden file `tests/conformance/golden/M5.json`. The rule comes from the licence decision of the engine items. The check runs offline.

### M6 Stop

SIGTERM ends the engine within 4 seconds, also with a request in flight. Removal leaves no process.

Evidence: row 8. The serial server stayed busy. The code is in `src/lilbee/providers/fleet/engine_reaper.py` and `src/lilbee/providers/fleet/child_guard.py`.

## Failure rows and requirements

Every spike row maps to a requirement or has a reason. Rows 7, 49, 60 and 72 are an llama-server extra. Row 18 breaks nothing. Rows 27, 35, 63 and 82 work.

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
| 56 | M4 | macOS 15 floor for the Metal archive |
| 57 | E1, M1, M3 | works |
| 58 | M2 | |
| 59 | E6 | works, inline `<think>` |
| 60 | none | prompt-progress frames are an llama-server extra; the client works without them |
| 61 | E6 | works |
| 62 | E7 | not determined; closed by row 77 |
| 63 | none | works |
| 64 | L1 | works |
| 65 | L2 | handled fallback |
| 66 | E12, L4 | works |
| 67 | E8 | works |
| 68 | L1 | works |
| 69 | L1 | the estimate covers it |
| 70 | E1, M1, M2 | |
| 71 | E5, E6, L3 | |
| 72 | none | prompt-progress frames are an llama-server extra; the client works without them |
| 73 | E7 | works |
| 74 | L1, L2 | |
| 75 | E12 | as row 8 |
| 76 | E9, E13 | route absent |
| 77 | E7 | works |
| 78 | E9, E13, L6 | route absent |
| 79 | E3 | works |
| 80 | E1, M2, M3 | |
| 81 | E6 | works |
| 82 | none | works |
| 83 | E7 | works |
| 84 | E6 | works |
| 85 | L1, L2 | |
| 86 | E12, L4 | |
| 87 | E8 | works |
| 88 | L1 | |
| 89 | E9, L6 | works, native route |
| 90 | L1, L6 | |
| 91 | E2 | memory guard; passes with the guard off |

## Recordings of a real engine

A recording of a real engine is a file under `tests/conformance/fixtures/engines/`. Take it with `python -m tests.conformance.record_presets` and the `--strict` flag. The flag adds the steps `json_object`, `json_schema`, `reasoning` and `named`, and a two-image `vision` step. The file stores `"strict": true`, and a test fails for each recording that does not. For a managed engine, pass `--engine` with its name: the JSON steps then use that engine's request rules, so on mlx-lm the `json_schema` step expects the 400 and the `json_object` step expects the request without the field and a whole answer. The step asks as lilbee's own JSON calls do, with a small cap and thinking off. The log line says whether lilbee's scan found an object, but the step does not judge the model's JSON, because the engine promises none. For an engine whose server has no length check, pass `--window` with the window the launch serves, so the `overflow` step expects the refusal before any request. The replay tests run each recording through the real provider with no server, in the `integration` job. The recorder reads the device labels from the `/memory` reply of the engine, and the log lists them as `devices` and `reported-backend`. Only a device that holds bytes counts, and a pinned host row such as `CUDA_Host` is not a GPU. If the reported backend differs from `--backend`, or the engine lists no device for a GPU backend, the recorder writes nothing and exits. An engine that has no `/memory` route can take a GPU recording on the evidence of its own startup output. Pass that output with `--engine-log`. The engine must listen on a local port, and the process that listens must hold the log file open, as it does when its output goes to that file. The recorder reads this from the process table, asking each process for its own sockets where only root may list them all (macOS), and refuses any other file, so a file that you write or copy is no evidence. The recorder reads the lines that name the device, such as `Starting FlashBert model on Cuda(CudaDevice(DeviceId(1)))`, and the log lists them as `devices` with `devices-from: engine-log`. The engine reply wins when it lists devices. If the log and the reply name different backends, the recorder writes nothing and exits. A log with no device line is no evidence. Use `--device-url` when `--url` is a proxy without the `/memory` route. The recorder replaces each home directory, such as `/home/alice`, with `<home>`, and the hostname of the recording machine with `<host>`, before it writes the file and its log. A test fails for a committed recording that holds a home path or a hostname. Scrub text that you add by hand to a log, such as the server log, the same way.

The step `named` sends one request through the routing provider. It passes when the log line `served by engine=<name> version=<version> build=<build> role=<role> model=<id>` and the `serving` entry of the status for that role both name the engine. A server preset has the preset name as its engine name and `preset` as its version. A managed engine has its name and its pin. The `build` field is the `upstream_ref` of the pin record, the upstream source ref of the pack, and it is `none` when the pin has none or an empty one. The oracle compares `version` only.

The 7 preset recordings in `tests/conformance/fixtures/presets/` are not strict and have no `reasoning` step. The `reasoning` step of a strict recording reads the answer text and the finish reason through the provider. It does not read `reasoning_content` or `reasoning`. A pass of the preset `reasoning` step does not prove that the engine splits reasoning from the answer. Row E6 of the wire harness proves the split.

## CI cost of a cell

These costs come from `.github/workflows/`. No time per fixture is measured: the replay needs the provider extra, and the run that wrote this section did not have it.

A new recording adds no job and no runner. The file is one more parameter of the replay tests, so it runs inside cells that already exist:

| Job | Cells | Runs the recordings | Limit for each cell |
|---|---|---|---|
| `integration` in `ci.yml` | 9 (ubuntu, macos, windows, each with Python 3.11, 3.12, 3.13) | yes, through `make test-integration` | 120 minutes |
| `test` in `ci.yml` | 7 (the same 9, less macos and windows on 3.11) | the recording-strict test only; the replay skips without the extra | 120 minutes |

A recording is a file of captured replies. Replay starts no engine, loads no model and needs no GPU. The cost of a new recording is one more replay in each of the 9 `integration` cells.

A new engine build is a different cost. The matrix in `build-multigpu.yml` has 11 cells (5 ubuntu, 4 windows, 2 macos), each limited to 360 minutes. A new cell there runs on a pull request that changes `engine-versions.env` or the engine build scripts, on a push to `main`, and on the weekly schedule. The workflow files hold no run time for it.

Hardware with no CI runner (Intel NPU, Arc, Apple MLX) has no cell. A run on the project's own hardware can give a recording only if the engine answers `/memory` or its startup output names the device (`--engine-log`), because the recorder refuses a GPU backend without that evidence. Keep such a recording with its `.log` file next to the `.json` file. CI replays that file like any other.

The engine pin sync in `engine-pin-sync.yml` is one job with a 30 minute limit that runs weekly. It bumps `ENGINE_LLAMA_CPP_REF` only. A pin file for another engine is not part of this workflow.

## Add an engine

Add a profile to `tests/conformance/fake_engine.py` that replays the measured behaviour, and add a golden file only for a new requirement. The harness in `tests/conformance/harness.py` does not change for a new engine.

## Related

- [Server presets](presets.md) shows how to connect a server that you run yourself.
- [Engine results](engine-results.md) lists what each managed engine did in its recorded run.
