# Retrieval tuning and model setup

Open this file only when the user asks you to tune retrieval or set up models. For a thin
answer, first widen the single search with `top_k` (see SKILL.md). The steps below save
settings, so do them only when the user asks.

Settings are changed with `lilbee_settings_list`, `lilbee_settings_get`,
`lilbee_settings_set` and `lilbee_settings_reset`. The CLI has no settings command. Without
MCP, ask the user to use the `/settings` screen in `lilbee chat`, or to edit `config.toml`.

## Set up for the user's hardware and files

Move only the retrieval surface: `embedding_model`, `reranker_model`, `vision_model` and
the retrieval and ingest settings. Leave `chat_model` alone unless the user asks.

1. `lilbee_status` to see what is set.
2. `lilbee_settings_list(group="Retrieval")` for the baseline.
3. `lilbee_catalog_browse(task="embedding")`, then `"rerank"` and `"vision"`, to find
   candidates. Check the `compat` field of each entry before you pull.
4. Pull the picked model (a long operation; see SKILL.md). CLI: `lilbee --json model pull <ref>`.
5. Set the settings in one `lilbee_settings_set` call. Install a model before you assign it.
6. If the result says `reindex_required: true`, run a full re-index
   (`lilbee_sync(force_rebuild=true)`, CLI `lilbee --json rebuild`) as a long operation.
7. Tell the user which settings you moved and why. `lilbee_settings_reset` undoes them.

## Which settings to move

| Question style | Tune |
|---|---|
| Code questions, call graphs | Set `reranker_model`. Raise `rerank_candidates` to 24-48. Enable `concept_graph`. Lower `chunk_size` to 256-384. |
| Long-document walkthroughs | Raise `top_k` to 12-16 and `max_context_sources`. Lower `diversity_max_per_source` to 1. |
| Fact lookup across many sources | Raise `top_k` and `candidate_multiplier`. Keep `reranker_model` set. |
| English question, code answer | Enable `hyde`. Raise `max_distance` to about 0.8. |

`chunk_size`, `chunk_overlap` and a new `embedding_model` need a full re-index.
