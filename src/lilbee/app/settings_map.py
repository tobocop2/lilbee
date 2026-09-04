"""Shared settings map for interactive configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from pydantic_core import PydanticUndefined

from lilbee.app.themes import DARK_THEMES
from lilbee.core.config import cfg
from lilbee.core.config.enums import (
    ChatMode,
    ClustererBackend,
    CrawlRenderMode,
    KvCacheType,
    LlmProvider,
    ReasoningMode,
    RerankerType,
    TableModel,
    WikiEntityMode,
)
from lilbee.core.config.model import FTS_LANGUAGES


class RenderStyle(StrEnum):
    """How a setting is displayed in /settings."""

    COMPACT = "compact"
    FULL = "full"
    LIST_COLLAPSED = "list_collapsed"
    MULTILINE = "multiline"


class SettingGroup(StrEnum):
    """Logical bucket names rendered by ``/settings`` and ``settings_list``."""

    MODELS = "Models"
    GENERATION = "Generation"
    RETRIEVAL = "Retrieval"
    INGEST = "Ingest"
    WIKI = "Wiki"
    MEMORY = "Memory"
    CRAWLING = "Crawling"
    LOCAL_SERVERS = "Local-Servers"
    API_KEYS = "API-Keys"
    SYSTEM = "System"
    DISPLAY = "Display"
    GENERAL = "General"


@dataclass(frozen=True)
class SettingDef:
    """Metadata for an interactive setting.

    ``writable`` is a TUI rendering hint: fields marked ``writable=False``
    (the model role slots) get a dedicated picker rather than an inline
    editor, and the ``/set`` slash command refuses them. The actual
    write contract for HTTP / MCP / programmatic surfaces lives in
    ``config_meta.WRITABLE_CONFIG_FIELDS`` + ``MODEL_ROLE_FIELDS`` and
    is enforced by ``app.settings.apply_settings_update``.

    ``hidden`` keeps the setting out of the TUI settings screen while
    leaving it reachable via ``lilbee set`` and the ``LILBEE_*`` env
    var: use it for transport/server knobs that aren't relevant to a
    typical TUI session.
    """

    type: type
    nullable: bool
    writable: bool = True
    render: RenderStyle = field(default=RenderStyle.COMPACT)
    group: SettingGroup = SettingGroup.GENERAL
    help_text: str = ""
    choices: tuple[str, ...] | None = None
    hidden: bool = False
    # List editors validate each line as a regex only when this is set; flag-style
    # lists (e.g. crawl_browser_extra_args) would be wrongly rejected otherwise.
    validate_regex: bool = False
    # Credentials: the TUI masks the editor so the value is never on screen in
    # plain text, including while it is being pasted.
    secret: bool = False


def get_default(key: str) -> object:
    """Return the cfg default for a setting key."""
    field_info = type(cfg).model_fields[key]
    if field_info.default_factory is not None:
        return field_info.default_factory()  # type: ignore[call-arg]
    if field_info.default is PydanticUndefined:
        return None
    return field_info.default


SETTINGS_MAP: dict[str, SettingDef] = {
    "chat_model": SettingDef(
        str,
        nullable=False,
        writable=False,
        group=SettingGroup.MODELS,
        help_text="LLM used for chat generation (vision and reranking are separate slots)",
    ),
    "vision_model": SettingDef(
        str,
        nullable=True,
        writable=False,
        group=SettingGroup.MODELS,
        help_text="Vision model for scanned PDF OCR (empty = disabled; Tesseract only)",
    ),
    "enable_ocr": SettingDef(
        bool,
        nullable=True,
        group=SettingGroup.INGEST,
        help_text="Vision OCR for scanned PDFs (empty = auto-detect from vision_model)",
    ),
    "ocr_timeout": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Per-page timeout in seconds for vision OCR (0 = no limit)",
    ),
    "vision_load_budget_s": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text=(
            "Wall-clock seconds reserved for the vision worker to load the"
            " model. Total PDF-OCR budget = load_budget + ocr_timeout * pages."
        ),
    ),
    "vision_ocr_max_tokens": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text=(
            "Hard cap on tokens generated per OCR page (bounds runaway repetition"
            " loops); raising it lengthens page generation, so give ocr_timeout headroom"
        ),
    ),
    "vision_ocr_concurrency": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Pages OCR'd concurrently per vision server; each slot adds KV cache memory",
    ),
    "ingest_workers": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Workers for discovering and hashing files (0 = auto, all available cores)",
    ),
    "ingest_processes": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text=(
            "Ingest worker processes, one GPU each (0 = auto, one per card). Used"
            " once the corpus is big enough to pay for them; 1 keeps ingest in this"
            " process"
        ),
    ),
    "mcp_tool_threads": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.LOCAL_SERVERS,
        help_text=(
            "Threads for synchronous MCP tool handlers; the ceiling on how many agents"
            " one daemon serves before retrieval calls queue"
        ),
    ),
    "crawl_convert_workers": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text=(
            "Crawled pages converted to markdown on worker threads at once, so a crawl"
            " does not block request handling; 0 converts on the event loop"
        ),
    ),
    "auto_sync": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Run a sync before `lilbee ask` (disable on large static corpora)",
    ),
    "entity_extraction": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Extract typed entities automatically at sync (schema induced on first run)",
    ),
    "semantic_chunking": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Opt-in topic-aware chunker (default off; may fragment numbered procedures)",
    ),
    "topic_threshold": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Topic-boundary similarity threshold, 0.0-1.0, used when semantic chunking is on",
    ),
    "token_sizing": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Size chunks by real embedder tokens, not chars (changes invalidate the index)",
    ),
    "table_extraction": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Index each extracted table as its own chunk (changes invalidate the index)",
    ),
    "layout_detection": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text=(
            "Layout-aware PDF extraction: reading order plus header/footer "
            "stripping (changes invalidate the index)"
        ),
    ),
    "table_model": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.INGEST,
        choices=tuple(m.value for m in TableModel),
        help_text=(
            "Table structure model used when layout detection is on: slanet_auto "
            "(docling-parity default), other slanet variants, tatr, or disabled "
            "(changes invalidate the index)"
        ),
    ),
    "batch_extraction": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Coalesce concurrent extractions into one xberg batch call",
    ),
    "batch_extraction_size": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Max files per extract_batch call when batch extraction is on",
    ),
    "embedding_model": SettingDef(
        str,
        nullable=False,
        writable=False,
        group=SettingGroup.MODELS,
        help_text="Model used to embed document chunks",
    ),
    "reranker_model": SettingDef(
        str,
        nullable=True,
        writable=False,
        group=SettingGroup.MODELS,
        help_text="Cross-encoder model for result reranking",
    ),
    "reranker_type": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.MODELS,
        choices=tuple(t.value for t in RerankerType),
        help_text=(
            "Reranker serving mode: auto (detect cross-encoder vs LLM by model), "
            "cross_encoder, or llm"
        ),
    ),
    "reranker_prompt": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.MODELS,
        help_text="Relevance prompt for LLM rerankers (blank uses the built-in template)",
    ),
    "temperature": SettingDef(
        float,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text="Sampling temperature (higher = more creative)",
    ),
    "top_p": SettingDef(
        float,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text="Nucleus sampling cutoff probability",
    ),
    "top_k_sampling": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text="Top-K sampling: number of tokens to consider",
    ),
    "repeat_penalty": SettingDef(
        float,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text="Penalty for repeating tokens",
    ),
    "num_ctx": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text=(
            "Context window size in tokens. Leave empty to size automatically "
            "(aims for chat_n_ctx_target, ceiling at num_ctx_max or training_ctx)."
        ),
    ),
    "num_ctx_max": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text=(
            "Explicit ceiling for the dynamic context picker. Leave empty to "
            "use the model's training_ctx from GGUF metadata as the only "
            "ceiling. Set to cap below training_ctx (saves KV memory)."
        ),
    ),
    "chat_n_ctx_target": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "Working context the dynamic picker aims for. Fits a RAG turn "
            "with reasoning headroom; raise for long-document chat."
        ),
    ),
    "flash_attention": SettingDef(
        bool,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text=(
            "Flash attention. Empty (auto) enables it; disable for backends or "
            "models where it misbehaves. Resolves the V-cache padding warning "
            "on models with uneven per-layer V dims."
        ),
    ),
    "kv_cache_type": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "KV cache element type. q8_0 / q4_0 halve or quarter cache memory "
            "but require flash attention to be enabled."
        ),
        choices=tuple(t.value for t in KvCacheType),
    ),
    "n_gpu_layers": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text=(
            "Layers to offload to GPU. Empty = all (recommended), 0 = CPU only, "
            "positive int = partial offload for tight VRAM."
        ),
    ),
    "cpu_moe": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "Keep a mixture-of-experts model's expert weights in system memory so "
            "it fits a smaller GPU. No effect on dense models."
        ),
    ),
    "n_cpu_moe": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text=(
            "Offload only the first N layers' experts to system memory. Takes "
            "precedence over the offload-everything setting; smaller N stays faster."
        ),
    ),
    "fast_model_downloads": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "Warning: Hugging Face states this mode uses all available bandwidth "
            "and CPU cores, and buffers far more of the download in memory. "
            "Faster on a fast connection, at the cost of everything else running "
            "on the machine. Leave it off unless the machine can spare that. "
            "Requires a restart to take effect."
        ),
    ),
    "gpu_devices": SettingDef(
        str,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text=(
            "Restrict llama.cpp to specific GPU indexes on dual-GPU machines "
            "(e.g. NVIDIA dGPU + integrated). Comma-separated, like '0' or '0,1'. "
            "Applies to Vulkan, CUDA, and ROCm. Requires a restart to take effect."
        ),
    ),
    "main_gpu": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text=(
            "Primary GPU index for llama.cpp when multiple devices are visible. "
            "Empty = let llama.cpp pick (index 0). Set this together with "
            "gpu_devices to pin inference to a specific card. Requires a restart "
            "to take effect."
        ),
    ),
    "seed": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text="Random seed for reproducible output",
    ),
    "rag_system_prompt": SettingDef(
        str,
        nullable=False,
        render=RenderStyle.MULTILINE,
        group=SettingGroup.GENERATION,
        help_text="System prompt sent when answering with retrieved context",
    ),
    "general_system_prompt": SettingDef(
        str,
        nullable=False,
        render=RenderStyle.MULTILINE,
        group=SettingGroup.GENERATION,
        help_text="System prompt sent when there are no documents to ground the answer",
    ),
    "chat_compaction": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "Off (default): when a chat outgrows the model's context window the oldest "
            "turns are dropped. They stay on screen but the model stops seeing them, and "
            "the context chip by the prompt shows the window filling. Costs nothing. "
            "On: those turns are condensed into a short summary the model keeps reading, "
            "so it still knows roughly what was said. That costs one extra model call each "
            "time it fires, pausing the reply for a few seconds on a GPU and considerably "
            "longer on a CPU-only machine. Worth turning on if your hardware is quick."
        ),
    ),
    "sessions_enabled": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "On (default): conversations are saved automatically, and you can list, "
            "resume, rename, and delete them from the Sessions drawer (ctrl+o), the "
            "Sessions tab, and the /sessions command. Off: nothing is written to disk, "
            "the ctrl+o binding leaves the footer, and opening the Sessions view shows a "
            "notice that sessions are turned off. Turn it off if you would rather your "
            "chats not persist. Covers the TUI, the HTTP server, and the CLI; agent "
            "sessions have their own setting."
        ),
    ),
    "mcp_sessions_enabled": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "Off (default): the session tools are not offered over MCP, and a connected "
            "agent cannot create or read agent sessions. On: an agent can keep its own "
            "saved conversations, separate from yours. Most agent hosts already track "
            "their own history, and the tools cost context on every request, so this "
            "stays off unless you want an agent owning conversations."
        ),
    ),
    "chat_mode": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.GENERATION,
        choices=tuple(m.value for m in ChatMode),
        help_text="search runs every chat turn through document retrieval; chat skips it",
    ),
    "top_k": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Number of chunks returned by search",
    ),
    "rerank_candidates": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Candidate pool size for reranking",
    ),
    "rerank_blend": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Blend reranker scores with retrieval fusion (off = pure reranker order)",
    ),
    "rerank_min_score": SettingDef(
        float,
        nullable=True,
        group=SettingGroup.RETRIEVAL,
        help_text="Drop candidates whose raw reranker score is below this (unset = off)",
    ),
    "show_reasoning": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.DISPLAY,
        help_text="Show model reasoning/thinking tokens in output",
    ),
    "completions_reasoning": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "How /v1/chat/completions presents thinking: separate "
            "reasoning_content field, inline thinking as plain content text, "
            "or off (ask the model not to think)"
        ),
        choices=tuple(m.value for m in ReasoningMode),
    ),
    "messages_reasoning": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "How /v1/messages presents thinking: separate thinking block, "
            "inline thinking as plain answer text, or off (ask the model not "
            "to think)"
        ),
        choices=tuple(m.value for m in ReasoningMode),
    ),
    "lilbee_name": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.DISPLAY,
        help_text=(
            "Human-readable label for this lilbee, shown in the status bar. "
            "Empty falls back to 'global' for the platform default dir or "
            "to the project path (~-substituted and left-truncated)."
        ),
    ),
    "show_lilbee_path": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.DISPLAY,
        help_text=(
            "Show the full absolute path in the status bar: expands 'global' "
            "to its on-disk path and skips ~ substitution / truncation."
        ),
    ),
    "theme": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.DISPLAY,
        help_text="TUI color theme. Cycle with Ctrl+T; the active theme persists across sessions.",
        choices=tuple(DARK_THEMES),
    ),
    "wiki": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text=(
            "Enable the wiki layer (cited concept and entity pages). "
            "GPU-heavy: a build spends one LLM call per source document, "
            "so a large library takes hours. Enabling this generates nothing "
            "on its own: you wikify explicitly, or turn on wiki_auto_update"
        ),
    ),
    "wiki_auto_update": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Regenerate touched wiki pages after each sync (off: wikify explicitly)",
    ),
    "wiki_dir": SettingDef(
        str,
        nullable=False,
        writable=False,
        group=SettingGroup.WIKI,
        help_text=(
            "Directory under data_root where wiki pages live (set via env / config.toml only)"
        ),
    ),
    "wiki_prune_raw": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Delete raw chunks after summarizing into the wiki",
    ),
    "wiki_embedding_faithfulness_threshold": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text=(
            "Minimum cosine similarity (0-1) between a generated page and "
            "the mean of its source chunk vectors before publishing. "
            "Pages below the threshold route to drafts/."
        ),
    ),
    "wiki_stale_citation_threshold": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Fraction of stale citations before a page is flagged by wiki prune",
    ),
    "wiki_drift_threshold": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Max fraction of changed lines before regeneration requires review",
    ),
    "wiki_clusterer": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Synthesis clusterer backend (embedding or concepts)",
        choices=tuple(b.value for b in ClustererBackend),
    ),
    "wiki_entity_mode": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text=(
            "Entity extraction strategy. ner_entities (typed spaCy NER) is the "
            "only implemented mode; the other values fall back to it with a warning"
        ),
        choices=tuple(m.value for m in WikiEntityMode),
    ),
    "wiki_entity_min_mentions": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Minimum chunk mentions before an entity or concept gets its own page",
    ),
    "wiki_stub_max_chunk_refs": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text=(
            "How many source chunks a page kept for lazy generation draws on. "
            "Caps the browse index's size; already more than one page's context "
            "budget admits, so raising it rarely changes what a page says"
        ),
    ),
    "wiki_ingest_update_cap": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text=(
            "Touched-page cap for auto-update after sync. "
            "Beyond this count, run `lilbee wiki update` manually."
        ),
    ),
    "wiki_synthesis_prompt": SettingDef(
        str,
        nullable=False,
        render=RenderStyle.FULL,
        group=SettingGroup.WIKI,
        help_text=(
            "Prompt for cross-source synthesis pages. "
            "Must keep {topic}, {source_list}, and {chunks_text}."
        ),
    ),
    "wiki_entity_page_prompt": SettingDef(
        str,
        nullable=False,
        render=RenderStyle.FULL,
        group=SettingGroup.WIKI,
        help_text=(
            "Prompt for a single page generated on demand from one subject's "
            "chunks across every source naming it. "
            "Must keep {topic}, {source_list}, and {chunks_text}."
        ),
    ),
    "wiki_entity_batch_prompt": SettingDef(
        str,
        nullable=False,
        render=RenderStyle.FULL,
        group=SettingGroup.WIKI,
        help_text=(
            "Prompt for the per-source batched call. "
            "Must keep {source}, {entity_list}, {chunks_text}, and {concept_instruction}."
        ),
    ),
    "wiki_extract_concepts": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text=(
            "Whether the per-source batched call asks the LLM to curate concept pages "
            "alongside the pre-extracted entity list."
        ),
    ),
    "wiki_batch_min_chunks": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text=(
            "Minimum chunks a source must contribute before its batched call includes "
            "concept curation. Sources below the floor skip the concept-curation "
            "instruction; sources with zero entities AND below the floor are skipped entirely."
        ),
    ),
    "wiki_clusterer_k": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Mutual-kNN neighborhood size for the clusterer (0 = auto)",
    ),
    "memory_enabled": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.MEMORY,
        help_text="Master switch for long-term chat memory (off by default)",
    ),
    "memory_auto_extract": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.MEMORY,
        help_text="Auto-save durable facts and preferences from each TUI turn (needs memory on)",
    ),
    "memory_top_k": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.MEMORY,
        help_text="Maximum facts recalled into context per turn",
    ),
    "memory_max_distance": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.MEMORY,
        help_text="Recall cutoff distance, 0.0-1.0 (lower is stricter)",
    ),
    "memory_token_budget": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.MEMORY,
        help_text="Token cap on the recalled-memory block added to the prompt",
    ),
    "memory_max_per_owner": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.MEMORY,
        help_text="Soft cap before the oldest memories are evicted",
        hidden=True,
    ),
    "memory_dedup_distance": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.MEMORY,
        help_text="Near-duplicate distance below which a new memory updates the old",
        hidden=True,
    ),
    "crawl_max_depth": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.CRAWLING,
        help_text="Optional recursion-depth cap (blank = no cap; per-crawl values win)",
    ),
    "crawl_render_mode": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text=(
            "How crawls fetch pages. http = lightweight, no browser (default, best "
            "for static and server-rendered sites). browser = Chromium with "
            "JavaScript enabled for client-rendered sites, at much higher memory cost."
        ),
        choices=tuple(m.value for m in CrawlRenderMode),
    ),
    "crawl_browser_recycle_pages": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text=(
            "Browser mode: recycle the Chromium process every N pages to cap memory "
            "growth on long crawls (0 = never recycle)."
        ),
    ),
    "crawl_browser_extra_args": SettingDef(
        list,
        nullable=False,
        group=SettingGroup.CRAWLING,
        render=RenderStyle.LIST_COLLAPSED,
        help_text=(
            "Browser mode: extra Chromium launch flags, one per line. "
            "Defaults trim shared-memory and GPU use."
        ),
    ),
    "crawl_max_pages": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.CRAWLING,
        help_text="Optional global cap on total pages per crawl (blank = no cap).",
    ),
    "crawl_safety_max_pages": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Default page bound for an unbounded crawl, so a hostile site cannot "
        "exhaust the disk. An explicit max-pages overrides it; raise this to crawl "
        "larger sites unbounded.",
    ),
    "crawl_timeout": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Per-page fetch timeout in seconds",
    ),
    "crawl_sync_interval": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Seconds between periodic re-syncs during a crawl (0 = sync only at end)",
    ),
    "crawl_mean_delay": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Seconds between in-flight requests within a single crawl",
    ),
    "crawl_max_delay_range": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Random jitter (seconds) added on top of mean delay",
    ),
    "crawl_concurrent_requests": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Concurrent in-flight URLs within one crawl",
    ),
    "crawl_retry_on_rate_limit": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Enable per-domain backoff and retries on HTTP 429/503",
    ),
    "crawl_retry_base_delay_min": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Minimum base-delay (seconds) on rate-limit responses",
    ),
    "crawl_retry_base_delay_max": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Maximum base-delay (seconds) on rate-limit responses",
    ),
    "crawl_retry_max_backoff": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Upper bound on any single backoff wait (seconds)",
    ),
    "crawl_retry_max_attempts": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.CRAWLING,
        help_text="Retry count per URL when a rate-limit code comes back",
    ),
    "crawl_exclude_patterns": SettingDef(
        list,
        nullable=False,
        group=SettingGroup.CRAWLING,
        render=RenderStyle.LIST_COLLAPSED,
        validate_regex=True,
        help_text=(
            "Regex patterns that skip URLs at link-discovery time during "
            "recursive crawls. One per line."
        ),
    ),
    "openrouter_api_key": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        secret=True,
        help_text="OpenRouter API key (enables frontier models in chat picker)",
    ),
    "gemini_api_key": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        secret=True,
        help_text="Google Gemini API key (enables frontier models in chat picker)",
    ),
    "anthropic_api_key": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        secret=True,
        help_text="Anthropic API key (enables frontier models in chat picker)",
    ),
    "openai_api_key": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        secret=True,
        help_text="OpenAI API key (enables frontier models in chat picker)",
    ),
    "mistral_api_key": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        secret=True,
        help_text="Mistral API key (enables frontier models in chat picker)",
    ),
    "deepseek_api_key": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        secret=True,
        help_text="DeepSeek API key (enables frontier models in chat picker)",
    ),
    "llm_api_key": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        secret=True,
        help_text="API key for the remote OpenAI-compatible endpoint (llm_provider = remote)",
    ),
    "hf_token": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.SYSTEM,
        secret=True,
        help_text=(
            "HuggingFace access token. Avoids the unauthenticated download "
            "rate limit and unlocks gated repos. Stored in plain text in "
            "config.toml. Env vars (LILBEE_HF_TOKEN, HF_TOKEN) override."
        ),
    ),
    "chunk_size": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Document chunk size in tokens (changes invalidate the index)",
    ),
    "chunk_overlap": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Tokens of overlap between adjacent chunks (preserves context across boundaries)",
    ),
    "tesseract_timeout": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Per-page Tesseract timeout in seconds (used when no vision model is set)",
    ),
    "ocr_language": SettingDef(
        list,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text="Tesseract OCR languages when no vision model is set; '+'-join, e.g. eng+deu",
    ),
    "worker_pool_eager_start": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.INGEST,
        help_text=(
            "Spawn every configured role server at TUI startup instead of on first use. "
            "Trades cold-start time per role for first-call latency"
        ),
    ),
    "keep_engine_warm": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.SYSTEM,
        help_text=(
            "Let the engine outlive lilbee for warm launches; off stops it on last "
            "exit unless another lilbee sharing the engine asked to keep it"
        ),
    ),
    "engine_idle_ttl_minutes": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.SYSTEM,
        help_text="Idle minutes before the engine unloads its weights; 0 keeps them loaded",
    ),
    "agent_mcp_enabled": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.SYSTEM,
        help_text=(
            "Register lilbee's MCP search tool into agent launchers (opencode, hermes). "
            "Disable to bring your own MCP servers; lilbee stays the model provider"
        ),
    ),
    "max_tokens": SettingDef(
        int,
        nullable=True,
        group=SettingGroup.GENERATION,
        help_text="Hard cap on generated tokens per response (blank = no cap)",
    ),
    "max_reasoning_chars": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "Maximum reasoning characters before lilbee forces the model to answer "
            "(0 = unlimited; per-model overrides apply on top)"
        ),
    ),
    "model_keep_alive": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text="Seconds the loaded model stays warm between calls (0 = unload immediately)",
    ),
    "gpu_memory_fraction": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text="Fraction of GPU memory the model is allowed to claim (0.1-1.0)",
    ),
    "usable_vram_fraction": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "Share of a GPU placement may fill, leaving room for fragmentation and driver "
            "overhead (0.5-1.0). Raise it if a model that should fit is being refused; "
            "lower it if loads fail near the top of the card."
        ),
    ),
    "system_memory_reserve_gb": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text=(
            "RAM held back for the OS in GiB when serving from system memory (no discrete "
            "GPU). Capped at a quarter of total RAM either way."
        ),
    ),
    "embed_replicas": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text="Embedding servers in parallel (0 = auto, one per GPU; positive pins the count)",
    ),
    "vision_replicas": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.GENERATION,
        help_text="Vision OCR servers in parallel (0 = auto, one per GPU; positive pins the count)",
    ),
    "candidate_multiplier": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Candidate-pool multiplier over top_k before reranking",
    ),
    "title_search": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Match queries against document titles as a third hybrid-search arm",
    ),
    "title_search_weight": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Title arm weight in rank fusion (1.0 = equal voice with the other arms)",
    ),
    "lexical_fusion_weight": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="BM25 arm weight in fusion (1.0 = equal to vector; lower to favor dense)",
    ),
    "adaptive_fusion": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Scale the BM25 weight per query by vector-arm confidence, not a fixed value",
    ),
    "adaptive_fusion_margin": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Vector-similarity margin at which adaptive fusion fully silences the BM25 arm",
    ),
    "filter_structural_chunks": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Drop tables-of-contents and classification-banner cover pages from results",
    ),
    "fts_language": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        choices=tuple(sorted(FTS_LANGUAGES)),
        help_text="Stemmer/stop-word language for BM25 indexes (rebuild to apply)",
    ),
    "embed_titles": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Prefix document titles to chunk embeddings (rebuild to apply)",
    ),
    "contextual_enrichment": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="LLM context sentence per chunk embedding (slow ingest; rebuild to apply)",
    ),
    "history_rewrite": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Rewrite follow-ups into standalone retrieval queries using chat history",
    ),
    "intent_routing": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Route document-name lookups to exact retrieval, count questions to a scan",
    ),
    "intent_llm": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text=(
            "Classify count questions with the chat model when the fast patterns "
            "miss (covers phrasing variants and other languages; adds one short "
            "LLM call to those turns)"
        ),
    ),
    "ann_index_threshold": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Chunk count to start building an ANN vector index (0 = always flat search)",
    ),
    "max_distance": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Maximum vector distance for retrieval matches (lower = stricter)",
    ),
    "min_relevance_score": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Minimum RRF relevance score for hybrid search results (0.0 = no filter)",
    ),
    "max_context_sources": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Maximum unique sources contributing chunks to a single answer",
    ),
    "neighbor_expansion": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Adjacent chunks merged into each retrieved passage per side (0 = off)",
    ),
    "diversity_max_per_source": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Maximum chunks accepted from any one source (caps source dominance)",
    ),
    "mmr_lambda": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text=(
            "MMR lambda balancing relevance vs diversity (0 = max diversity, 1 = max relevance)"
        ),
    ),
    "temporal_filtering": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Detect temporal queries and bias retrieval toward recent chunks",
    ),
    "hyde": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Use HyDE (hypothetical answer expansion) to broaden retrieval",
    ),
    "hyde_weight": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Weight on the HyDE-generated query vector when blending with the original",
    ),
    "query_expansion_count": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Number of paraphrase expansions per query (0 disables expansion)",
    ),
    "expansion_similarity_threshold": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Minimum cosine similarity an expansion must keep with the original query",
    ),
    "expansion_short_query_tokens": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Queries at or below this token count skip expansion (saves a model call)",
    ),
    "expansion_guardrails": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Drop expansions that diverge from the original intent",
    ),
    "adaptive_threshold": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Widen the distance cutoff when too few results pass (vector-only fallback path)",
    ),
    "adaptive_threshold_step": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Step size for adaptive relevance-score relaxation when initial recall is empty",
    ),
    "concept_graph": SettingDef(
        bool,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Boost retrieval scores for chunks that share concepts with the query",
    ),
    "concept_boost_weight": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Maximum boost (0-1) the concept graph can add to a chunk's relevance",
    ),
    "concept_max_per_chunk": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.RETRIEVAL,
        help_text="Maximum concept tags stored per chunk (caps graph density)",
    ),
    "documents_dir": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.SYSTEM,
        help_text="Local documents root that lilbee sync ingests (blank = data_root/documents)",
    ),
    "vault_base": SettingDef(
        str,
        nullable=True,
        group=SettingGroup.SYSTEM,
        help_text="Markdown vault root; results carry a vault-relative path (blank = none)",
    ),
    "sse_heartbeat_interval": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.SYSTEM,
        help_text="Seconds between SSE keep-alive frames sent to idle HTTP stream clients",
        hidden=True,
    ),
    "llm_provider": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        choices=tuple(p.value for p in LlmProvider),
        help_text=(
            "Inference provider: auto (default, runs models locally on llama-server) "
            "or remote (external OpenAI-compatible endpoint)"
        ),
    ),
    "ollama_base_url": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.LOCAL_SERVERS,
        help_text="Ollama server URL (blank uses http://localhost:11434)",
    ),
    "lm_studio_base_url": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.LOCAL_SERVERS,
        help_text="LM Studio server URL (blank uses http://localhost:1234/v1)",
    ),
    "llama_server_path": SettingDef(
        str,
        nullable=False,
        group=SettingGroup.API_KEYS,
        help_text="Path to a llama-server binary (empty: bundled wheel or PATH)",
    ),
    "wiki_summary_max_tokens": SettingDef(
        int,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Maximum tokens generated per wiki page",
    ),
    "wiki_temperature": SettingDef(
        float,
        nullable=False,
        group=SettingGroup.WIKI,
        help_text="Temperature used for wiki page synthesis (low = stay close to sources)",
    ),
}
