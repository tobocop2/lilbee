# llama-server native tool-call coverage: findings

Spike to decide whether lilbee can rely on `llama-server --jinja`'s native tool-call
parsing (llama.cpp `common/chat.cpp`) instead of lilbee's hand-rolled `response_parser`
+ per-family schemas, which exist only because llama-cpp-python can't reach that C++
code. Run on a single H200, llama-server built from llama-cpp-python 0.3.23's vendored
llama.cpp (version 9102), `sm_90`. Three passes per model: `tool_choice=auto`, then
`required` on the auto-FAILs, then re-run with the real HF tool-aware template via
`--chat-template-file`.

## Result

| Family | Native verdict | Cause |
|--------|----------------|-------|
| qwen3 | PASS | native parses it |
| qwen3-coder | PASS | native parses it |
| llama3 | PASS | native parses it |
| mistral-nemo | PASS | native parses it |
| gemma4 | PASS | native parses it |
| functionary | PASS | native parses it (was only "partial" in-process) |
| gpt-oss | PASS | native parses Harmony (hand-fixed in lilbee last session; redundant here) |
| glm-air | PASS | native parses glm46 (hand-fixed in lilbee last session; redundant here) |
| granite | FAIL (real parser gap) | emits a valid `[{"name":...}]` array; native drops it. lilbee `bare_json` fallback recovers it. |
| smollm | FAIL (narrates) | reasoning model explains + puts the call in a ```json fence instead of calling |
| phi4-mini | FAIL (model declines) | "as an AI I can't use external tools". Loads on llama-server (recovered from in-process SIGABRT). |
| internlm2 | FAIL (model declines) | refuses, hallucinates being "developed by OpenAI" |
| glm-4-9b | FAIL (model declines) | refuses. Loads on llama-server (recovered from in-process SIGABRT). |
| lfm2 | FAIL | too small to tool-call reliably |
| ernie | FAIL | too small to tool-call reliably |
| hermes | INCONCLUSIVE | real template gated (no HF token here); ran on a stripped embedded template and narrated. Re-test with a token. |
| cohere | INCONCLUSIVE | template gated (no HF token); `--jinja` 500'd with no template. Re-test with a token. |
| olmo3 | LOAD_FAIL | llama-server's jinja (minja) engine can't parse its chat template |

## What this means

- **Native covers every model that actually tool-calls well**: 8 PASS, including gpt-oss
  Harmony and glm46, the exact two formats hand-fixed in lilbee's parser last session.
  On the llama-server path those fixes are redundant.
- **Most "FAILs" are model behavior, not parser deficiency.** phi4-mini, internlm2,
  glm-4-9b, lfm2, ernie either decline or are too small. No parser, lilbee's or native,
  changes that. Several were already exceptions in-process; phi4-mini + glm-4-9b were
  never tool-verified in-process at all (they SIGABRT'd) and now at least *load* on
  llama-server.
- **One genuine native parser gap: granite** (bare-JSON-array format). The model emits a
  valid call; native doesn't structure it; lilbee's `bare_json` fallback does. This is
  the concrete case for keeping lilbee's `response_parser` as a thin native-miss fallback.
- **Two cells are blocked, not failed: hermes and cohere** (gated templates, no HF token
  on this box). hermes is the one mainstream format whose native verdict is still open;
  worth re-testing with a token before any final call.
- **Version caveat**: this is 0.3.23's vendored llama.cpp. Its PEG-based parser
  (`PEG_NATIVE`/`PEG_SIMPLE`/`PEG_GEMMA4`) is a narrower/newer design; a bumped llama.cpp
  may widen native coverage. Re-checking on a newer build is cheap and worthwhile.

## Recommendation (feeds bb-n1bj)

The evidence supports making `llama-server` lilbee's primary chat runtime (not gating it
behind multi-GPU): **native-first parsing, with lilbee's `response_parser` kept only as a
thin fallback** for emitted-but-unparsed formats (granite-class). The decline/too-small
models are model-capability limits under either runtime. Before committing, (1) re-test
hermes + cohere with an HF token, and (2) consider bumping the bundled llama.cpp to widen
native coverage.

Harness: `probe.py` + `models.py` in this directory. Per-model raw evidence (request,
response, reasoning) is under `results/` (gitignored).
