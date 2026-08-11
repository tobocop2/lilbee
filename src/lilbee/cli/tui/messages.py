"""Centralized user-facing messages for the TUI.

ALL user-facing text MUST be defined here. Inline strings in
screens and widgets are forbidden -- this enables future i18n
and ensures consistent messaging.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from importlib.util import find_spec

from lilbee.core.config import cfg
from lilbee.providers.fleet.gpu_backends import IntelHintKind, IntelUtilHint
from lilbee.wiki.shared import WIKI_TYPE_HEADINGS as _WIKI_TYPE_HEADINGS

log = logging.getLogger(__name__)


def app_title(model: str) -> str:
    """The window title showing the active chat model. Single source so every
    code path that sets the title uses the same format."""
    return f"lilbee: {model}" if model else "lilbee"


CMD_UNKNOWN = "Unknown command: {cmd}"
CMD_ADD_NOT_FOUND = "Not found: {path}"
CMD_ADD_SUCCESS = "Added {count} file(s), syncing..."
CMD_ADD_RELOCATED = "{count} already indexed, location changed: relinked (no re-embed)."
CMD_ADD_DUPLICATE_TITLE = "File already in knowledge base"
CMD_ADD_DUPLICATE_MESSAGE = "{name} is already in the knowledge base. Overwrite and re-sync?"
CMD_ADD_SKIPPED_DUPLICATE = "Kept existing copy of {name}."
CMD_ADD_NAME_TAKEN = "The name {name} is taken by another source (use --force to overwrite)."
CMD_ADD_TRACKED = "Already tracked: {names}. Syncing."
CMD_ADD_ERROR = "Error: {error}"
CMD_CRAWL_USAGE = "Usage: /crawl <url> [--depth N] [--max-pages N]"
CMD_CRAWL_STARTED = "Crawling {url}..."
CMD_CRAWL_PAGE = "Crawling [{current}/{total}]: {url}"
CMD_CRAWL_PAGE_INDETERMINATE = "Crawling... ({current} pages so far): {url}"
MODEL_REASON_DEFAULT = "it could not be resolved"
MODEL_FALLBACK_NOTICE = (
    "{label} model {original!r} is unavailable ({reason}); using {effective!r} for this session. "
    "Pick a different model or restore the original to clear this notice."
)
MODEL_FALLBACK_FAILED = (
    "{label} model {original!r} is unavailable ({reason}) and the fallback {effective!r} was "
    "rejected; keeping {original!r}. Pick a working {label} model in settings."
)
MODEL_UNUSABLE_NO_FALLBACK = (
    "{label} model {original!r} is unavailable ({reason}) and nothing is installed to fall back "
    "to. Pick one from the catalog."
)
MODEL_ADOPTED_LOG = "{label} model: using installed {effective!r}."
CMD_CRAWL_SUCCESS = "Crawled {count} page(s) from {url}"
CMD_CRAWL_FAILED = "Crawl failed: {error}"
CMD_CRAWL_SYNCING = "Syncing crawled pages..."
SETUP_CHROMIUM_NAME = "Install Chromium browser"
SETUP_CHROMIUM_FAILED = "Chromium install failed: {error}"
SETUP_CHROMIUM_DETAIL = "chromium: {done}/{total} MB"
SETUP_CHROMIUM_DETAIL_UNKNOWN = "chromium: {done} MB"
SETUP_CHROMIUM_CLI_PROGRESS = "  chromium: {pct}%"
SYNC_FAILED_FILES = "Sync failed for {files}"
SYNC_SKIPPED_NO_VISION = (
    "Skipped (no text extracted): {files}. "
    "Configure a vision_model in Settings to OCR scanned PDFs."
)
SYNC_SKIPPED_VISION_FAILED = (
    "Skipped (vision OCR returned no text): {files}. See {log_path} for the underlying error."
)
CMD_RETRY_SKIPPED_NONE = "No skipped files to retry; running a normal sync."
CMD_RETRY_SKIPPED_SOME = "Cleared {count} skip marker(s); retrying those files."


def sync_skipped_message(files: str) -> str:
    """Pick the right skipped-files message based on whether vision_model is set.

    When the user has no vision_model configured the actionable advice is
    'go set one'; when one IS configured the OCR failed at runtime, so the
    message points the user at the worker log instead of telling them to
    do something they have already done.
    """
    if cfg.vision_model:
        log_path = cfg.data_root / "logs" / "server.log"
        return SYNC_SKIPPED_VISION_FAILED.format(files=files, log_path=log_path)
    return SYNC_SKIPPED_NO_VISION.format(files=files)


def retry_skipped_message(count: int) -> str:
    """Toast for the 'Retry skipped documents' command."""
    return CMD_RETRY_SKIPPED_NONE if count == 0 else CMD_RETRY_SKIPPED_SOME.format(count=count)


CMD_DELETE_NO_DOCS = "No documents indexed"
CMD_DELETE_READ_FAILED = "Could not read the document list"
CMD_DELETE_USAGE = "Documents: {names}\nUsage: /delete <filename>"
CMD_DELETE_NOT_FOUND = "Not found: {name}"
CMD_DELETE_SUGGESTION = "Did you mean {name}?"
CMD_DELETE_SUCCESS = "Deleted {name}"
CMD_REMEMBER_USAGE = "Usage: /remember <text>  (prefix with 'pref:' for a preference)"
CMD_REMEMBER_SUCCESS = "Remembered ({kind})."
CMD_REMEMBER_NO_EMBED = "Set an embedding model before saving memories."
MEMORY_AUTO_EXTRACTED = "Noted {count} memory(s) to review in /memories"
CMD_EXPORT_USAGE = "Usage: /export <path.parquet|path.jsonl>"
CMD_EXPORT_SUCCESS = "Exported {pages} page(s) to {output}"
CMD_IMPORT_USAGE = "Usage: /import <path.parquet|path.jsonl>"
CMD_IMPORT_SUCCESS = "Imported {sources} source(s) ({pages} pages, {chunks} chunks)"
CMD_RESET_SUCCESS = "Knowledge base reset"
CMD_RESET_PARTIAL = "Knowledge base reset ({skipped} item(s) could not be deleted)"
CMD_RESET_FAILED = "Reset failed: {error}"
CMD_RESET_CONFIRM_TITLE = "Reset the knowledge base?"
CMD_RESET_CONFIRM_MESSAGE = "This permanently deletes all indexed data."
CMD_REBUILD_CONFIRM_TITLE = "Rebuild the index?"
CMD_REBUILD_CONFIRM_MESSAGE = (
    "Drops every chunk and re-embeds the documents directory from "
    "scratch. Takes minutes on large libraries. Source files on disk "
    "are left alone."
)
TASK_NAME_SYNC = "Sync documents"
TASK_NAME_WIKI = "Wikify"
TASK_NAME_WIKI_WIPE = "Delete wiki"
TASK_NAME_REBUILD = "Rebuild index"
TASK_NAME_IMPORT = "Import {file}"
TASK_NAME_EXPORT = "Export {file}"
IMPORT_STATUS_LOADING = "Loading dataset..."
EXPORT_STATUS_RUNNING = "Exporting..."
CMD_SET_UNKNOWN = "Unknown setting: {key}"
CMD_SET_SUCCESS = "{key} = {value}"
# Stands in for a credential's value wherever one would otherwise be echoed.
MASKED_VALUE = "************"
CMD_SET_INVALID = "Invalid value for {key}: {error}"
CMD_SET_TYPE_HINT = "{key} needs {kind}"
CMD_SET_CHOICES = "{key} must be one of {choices}"
CMD_SET_READONLY = "{key} is read-only; use the Models screen"
CMD_MODEL_SET = "Model set to {name}"
# Shown while a model swap's fleet reload runs off the event loop, so the swap
# reads as in-progress instead of a frozen TUI. Role-neutral: shared by the chat
# swap and the embed/vision/rerank swaps in model_pick, which name the model in
# their own surfaces (the chat input placeholder and warm footer for chat).
MODEL_SWAP_APPLYING = "Switching model, loading…"
MODEL_SWAP_QUEUED = "Switching to {name} when this answer finishes"
MODEL_SWAP_DONE = "Now using {name}"
MODEL_SWAP_FAILED = "Could not switch model: {error}"
# Chat-input placeholder while a swap holds the input disabled: names the target
# model and says the input is waiting on it, so the disabled box is never a
# silent dead end. The warm progress itself shows in the task-bar footer.
CHAT_INPUT_SWITCHING = "Switching to {name} · chat unlocks when the model is ready…"
# Chat-input placeholder while a placement change reloads the whole fleet.
CHAT_INPUT_RELOADING = "Reloading the engine · one moment…"
# Shown when the user tries to send a prompt while the new chat model is still loading.
CHAT_MODEL_SWITCHING = "Still switching model. One moment, then send your prompt."
FLEET_RELOADING = "Applying placement, reloading the fleet. One moment, then send your prompt."
# Startup gate: shown from the moment the TUI paints until the app can serve.
STARTUP_PREPARING = "Preparing lilbee"
STARTUP_FAILED = "lilbee could not start: {error}"
CHAT_STACK_FAILED = "lilbee could not load its chat screen: {error}"
# Engine-load status painted into the pending answer while a prompt waits on a
# cold engine, and the failure that wait can end in.
ENGINE_READING_WEIGHTS = "Reading {name} weights"
ENGINE_WARMING = "Warming up the model"
# Names the phase, like its siblings above; "almost ready" promised a finish
# time the engine had not committed to (same rename as TASKBAR_WARM_LOADING).
ENGINE_ALMOST_READY = "Loading the engine"
ENGINE_LOAD_FAILED = "The engine failed to load: {error}"
ENGINE_FAILED_HINT = "Open the Catalog to install a model, or pick a different one in Settings."
ENGINE_NOT_READY = "The engine is not ready yet. Send your prompt again in a moment."
# Shown once when a prompt first waits on a cold engine and keep_engine_warm is off.
ENGINE_WARM_TIP = "Tip: Settings > Keep engine warm makes the next launch fast"
CMD_REMOVE_USAGE = "Usage: /remove <model_name>"
CMD_REMOVE_NOT_FOUND = "{name} is not installed"
CMD_REMOVE_SUCCESS = "Removed {name}"
CMD_REMOVE_FAILED = "Failed to remove {name}"
CMD_CANCEL = "Cancelled active operations"
CMD_CLEAR = "Conversation cleared"
SESSIONS_DISABLED_TITLE = "Sessions are turned off"
SESSIONS_DISABLED_MESSAGE = (
    "Conversations are not being saved. Turn sessions on in Settings to list, "
    "resume, and manage past chats."
)
SESSIONS_COUNT = "{count} saved"
SESSIONS_EMPTY = "No saved conversations yet."
SESSIONS_FILTER_PLACEHOLDER = "Filter conversations…"
SESSIONS_RENAME_PLACEHOLDER = "New name…  enter saves, esc cancels"
SESSIONS_ROW_META = "{count} msgs · {model}"
SESSIONS_RESUMED = "Resumed · {title}"
SESSIONS_MODEL_UNAVAILABLE = (
    "This conversation used {model}, which isn't installed. Keeping {current}."
)
SESSIONS_NEW = "Started a new chat"
SESSIONS_DELETED = "Deleted · {title}"
SESSIONS_DELETE_CONFIRM_TITLE = "Delete session"
SESSIONS_DELETE_CONFIRM = "Delete “{title}”? This cannot be undone."
SESSIONS_HINT = "↵ resume   ^n new   ^r rename   ^d delete   esc close"
# The context chip: how much of this chat the model can still see.
#
# "context", not "memory": lilbee already has a Memory feature (/memories, the
# Memory settings group), so "memory 85%" reads as though those were 85% full.
# The jargon was never the problem; the collision was.
CONTEXT_CHIP_USAGE = "context {percent}%"
# Only when compaction is off and the window is nearly full: say what is about to
# happen, not what to do. The user can still act (turn compaction on, or ask the
# thing they care about now), which is what makes it worth saying at all. With
# compaction on there is nothing to decide, so the plain percentage stands.
CONTEXT_CHIP_USAGE_DROPPING = "context {percent}% · dropping soon"
CONTEXT_CHIP_COMPACTING = "condensing…"
CONTEXT_CHIP_TOOLTIP = (
    "How much of this chat still fits in the model's context. Older turns drop out "
    "when it fills; they stay on screen but the model stops seeing them. Turn on "
    "chat_compaction in Settings to condense them into a summary instead."
)

# Rules drawn in the log where the model's view of the chat changes. The
# transcript above them stays whole and scrollable, so every one of these is
# about what the model is *sent*, never about deleting anything.
#
# One shape for the family -- "N earlier messages <what happened>" -- so a reader
# learns it once instead of parsing three sentences about one idea. These are
# titles for a rich Rule, which draws the line out to the full width itself: no
# dashes belong in the strings.
CHAT_COMPACTED = "{count} earlier messages condensed to a summary"
CHAT_TRIMMED = "{count} earlier messages dropped from context"
# Turns that fell out with no summary standing in for them: this model's context
# is too small to carry that much conversation, however it is condensed. Say so,
# or the model just looks like it forgot for no reason.
CHAT_COMPACTION_STRANDED = "{count} more dropped · too much for this context"

CHAT_COMPACTED_TOAST = "The context filled up, so earlier turns were condensed to keep them."
CHAT_TRIMMED_TOAST = (
    "The chat outgrew this model's context, so earlier turns dropped out of what it "
    "can see. Turn on chat_compaction in Settings to condense them instead."
)
CHAT_COMPACTED_STRANDED_TOAST = (
    "This model's context is too small for the whole conversation. Recent turns were "
    "condensed; older ones were dropped."
)
CMD_THEME_UNKNOWN = "No theme called {name}. Themes: {names}"
CMD_WIKI_DISABLED = "Wiki is disabled (set wiki = true in settings)"
CMD_WIKI_GENERATE_NO_EVIDENCE = (
    "Nothing left in the index for {slug}; its sources are gone. "
    "Run `lilbee wiki index` to refresh."
)
CMD_WIKI_WIPE_NEEDS_YES = "Use --yes to confirm wiping the wiki in JSON mode"
CMD_WIKI_WIPE_WARNING = (
    "This deletes every generated wiki page and its indexed rows.\n  Pages: {path}"
)
TASK_NAME_CRAWL = "Crawl {url}"
STREAM_ERROR = "\n\n*Error: {error}*"
STREAM_CANCELLED = "\n\n*Response cancelled.*"
SYNC_STATUS_SYNCING = "Syncing..."
SYNC_STATUS_DONE = "Synced ({count} docs)"
SYNC_STATUS_FAILED = "Sync failed"
SYNC_FILE_PROGRESS = "Syncing [{current}/{total}]: {file}"
SYNC_ALREADY_ACTIVE = "Sync in progress, please wait"
EMBEDDING_SET = "Embedding model: {name}"
CMD_CRAWL_UNAVAILABLE = "Web crawling is not available. Run 'uv sync --extra crawler' to enable it."
CRAWL_DIALOG_TITLE = "Crawl a URL"
CRAWL_DIALOG_URL_PLACEHOLDER = "example.com (https:// added automatically)"
CRAWL_DIALOG_DEPTH_PLACEHOLDER = "blank = no limit"
CRAWL_DIALOG_MAX_PAGES_PLACEHOLDER = "clear for unlimited"
CRAWL_DIALOG_URL_LABEL = "URL"
CRAWL_DIALOG_RECURSIVE_LABEL = "Recursive (crawl whole site)"
CRAWL_DIALOG_BROWSER_LABEL = "Use browser (enables JavaScript, uses more memory)"
CRAWL_DIALOG_DEPTH_LABEL = "Depth cap"
CRAWL_DIALOG_MAX_PAGES_LABEL = "Max pages (clear for unlimited)"
CRAWL_DIALOG_SUBMIT = "Crawl"
CRAWL_DIALOG_CANCEL = "Cancel"
CRAWL_DIALOG_URL_REQUIRED = "URL is required"
CRAWL_DIALOG_INVALID_URL = "Invalid URL: {error}"
CRAWL_DIALOG_INVALID_NUMBER = "{field} must be a positive integer or blank"
EMBEDDING_MISSING = (
    "No embedding model, search disabled. "
    "Run /pull to install one, or: lilbee model pull nomic-ai/nomic-embed-text-v1.5-GGUF"
)
THEME_SET = "Theme: {name}"
HEADING_INSTALLED = "Installed"
HEADING_MATCHES = "Matches"
CATALOG_TAB_LOCAL = "Local"
CATALOG_TAB_FRONTIER = "Frontier"
CATALOG_TAB_DISCOVER = "Discover"
CATALOG_TAB_CHAT = "Chat"
CATALOG_TAB_EMBED = "Embed"
CATALOG_TAB_VISION = "Vision"
CATALOG_TAB_RERANK = "Rerank"
CATALOG_TAB_LIBRARY = "Library"
CATALOG_FRONTIER_SUMMARY = "{count} cloud models across {providers} providers"
CATALOG_GRID_OVERFLOW = "+{count} more on HF. Press v for the full list view"
CATALOG_GRID_LOAD_MORE = "{count} loaded · keep scrolling for more"
CATALOG_GRID_ALL_LOADED = "All {count} models loaded"
CATALOG_GRID_LOADING_MORE = "{frame} loading more models…"
CATALOG_USING_FRONTIER = "Using {name} via the {provider} API"
CATALOG_NEEDS_KEY = "{provider} needs an API key. Set {key_field} in Settings to enable this model."
CATALOG_USING_REMOTE = "Using {name} (remote)"
CATALOG_ALREADY_INSTALLED = "{name} is already installed"
CATALOG_ALREADY_DOWNLOADING = "{name} is already downloading, press t to watch it"
CATALOG_QUEUED_DOWNLOAD = "Queued download: {name}"
CATALOG_WELCOME = "Pick a model to start chatting — the fit chip shows what runs on this machine."
CHAT_READY_TOAST = "Chat is ready — press c."
CATALOG_INSTALLED_OK = "{name} installed"
CATALOG_GATED_REPO = "{name} requires login, run /login or lilbee login"
CATALOG_DOWNLOAD_FAILED = "{name}: download failed"
CATALOG_SELECT_FOR_INFO = "Select a model to view info"
CATALOG_FRONTIER_NO_INFO = "Info modal is for downloadable models only"
MODEL_INFO_HINT = "Esc / i / q to close"
MODEL_INFO_HF_LINK = "View on HuggingFace: https://huggingface.co/{repo}"
CATALOG_SELECT_TO_DELETE = "Select a model to delete"
CATALOG_NOT_INSTALLED = "{name} is not installed"
CATALOG_CONFIRM_DELETE = "Delete {name}? Press d again to confirm"
CATALOG_DELETED = "Deleted {name}"
CATALOG_DELETE_FAILED = "Delete failed: {error}"
CATALOG_NO_MATCH = "No models match your filters."
CATALOG_FILTER_PLACEHOLDER = "Filter models..."
CATALOG_VIEW_TOGGLE_GRID = "Press v for full list view · / to search"
CATALOG_VIEW_TOGGLE_LIST = "Press v for card view · s to sort"
CATALOG_VIEW_GRID = "Grid"
CATALOG_VIEW_LIST = "List"
CATALOG_SORT_LIST_ONLY = "Sort is available in list view (press v)"
CATALOG_SEARCHING_HF = "Searching HuggingFace…"
CATALOG_SEARCH_HF_CTA = '→ Search HuggingFace for "{query}"'
CHAT_INPUT_PLACEHOLDER_DEFAULT = "Ask…   /  commands   ?  keys   F2  all commands"
# Replaces Textual's default magnifying-glass emoji, which the system emoji font
# draws in its own colors and at double width. Single cell, and distinct from the
# task list's ▶ so the two never read as the same mark.
COMMAND_PALETTE_ICON = "✦"
# Box-drawing, not block elements: shade blocks are dither patterns that draw as
# sparse dashes and full blocks seam per cell wherever the font is not cell-exact.
PROGRESS_BAR_FILL = "━"
PROGRESS_BAR_TRACK = "─"
SLASH_CATALOG_TITLE = "Slash Commands"
SLASH_CATALOG_FILTER_PLACEHOLDER = "Filter commands..."
SLASH_CATALOG_FOOTER_HINT = "↑↓ select   Enter run   Esc close"
SLASH_CATALOG_NO_MATCH = "No commands match"
HELP_HINT_COMMANDS = "type / for commands"
HELP_HINT_KEYS = "? for keys"
HELP_HINT_SEPARATOR = "  ·  "
SCOPE_PILL_BOTH = "Both"
SCOPE_PILL_WIKI = "Wiki"
SCOPE_PILL_RAW = "Raw"
CHAT_BUSY = "Already answering. Press Ctrl+C to cancel, then submit your next prompt."
CHAT_MODEL_DOWNLOADING = "{name} is still downloading. Wait for it to finish, then submit."
MODEL_BEING_DOWNLOADED = (
    "{name} is still downloading. Wait for it to finish before setting it active."
)
CHAT_WELCOME_TITLE = "lilbee"
CHAT_WELCOME_TAGLINE = "your local AI stack and personal encyclopedia."
CHAT_WELCOME_HINT = "Press / for commands, or just ask."
CHAT_WELCOME_NO_MODEL_HINT = "No chat model installed yet. Press m to pick one from the catalog."
CHAT_INPUT_NO_MODEL = "No chat model — press m to pick one"
CHAT_LOGIN_PROMPT = "Paste your token with /login <token>"
CHAT_LOGGED_IN = "Logged in to HuggingFace"
CHAT_LOGIN_FAILED = "Login failed: {error}"
CHAT_VERSION = "lilbee {version}"
CHAT_RENDERING = "Rendering: {label}"
SETTINGS_READ_ONLY = "read-only"
SETTINGS_INVALID_VALUE = "Invalid value: {error}"
SETTINGS_RESET_TO_DEFAULT_TOOLTIP = "Reset to default"

EMBED_SWAP_CONFIRM_TITLE = "Switch embedding model?"
EMBED_SWAP_CONFIRM_MESSAGE = (
    "The vector store was built under a different embedder. "
    "Switching invalidates it: search and ingest are disabled until you rebuild. "
    "Run `lilbee rebuild` afterward (or press S to sync) to re-embed every document. "
    "Continue?"
)
EMBED_SWAP_CANCELLED = "Embedding model swap cancelled"
MODEL_ASSIGN_REJECTED = "Model not set: {error}"

EMBED_ADOPT_CONFIRM_TITLE = "Use this index's embedder?"
EMBED_ADOPT_CONFIRM_MESSAGE = (
    "This index was built with embedding model '{model}'. Use it for this vault? "
    "lilbee will download it if needed and switch to it. No rebuild is required."
)
EMBED_ADOPT_NOTICE = "This index was built with a different embedder ('{model}')."
EMBED_ADOPT_REBUILD_NOTICE = (
    "This index needs a {dim}-dim embedder. Rebuild it (press S to sync, or run "
    "`lilbee rebuild`) to use your current model."
)
EMBED_ADOPTING = "Switching to embedder '{model}'..."
EMBED_ADOPTED = "Now embedding with '{model}'."
EMBED_ADOPT_FAILED = "Could not adopt embedder: {error}"
EMBED_ADOPT_CANCELLED = "Kept the current embedder."

SETTINGS_RESET_ALL_LABEL = "Reset all defaults"
SETTINGS_RESET_ALL_CONFIRM_TITLE = "Reset all settings?"
SETTINGS_RESET_ALL_CONFIRM_MESSAGE = (
    "Every writable setting will be restored to its built-in default. "
    "Readonly fields (like installed models) are not affected."
)
SETTINGS_RESET_ALL_SUCCESS = "All settings reset to defaults"
SETTINGS_LIST_EDITOR_TITLE = "{key}  ({count} lines)"
SETTINGS_LIST_EDITOR_INVALID_REGEX = "Invalid regex on line {n}: {error}"
SETTINGS_LIST_EDITOR_RESTORE_DEFAULTS = "Restore defaults"
WIKI_EMPTY_STATE = "No wiki pages found"
# spaCy installs its NER model as an importable top-level package.
_SPACY_MODEL_PACKAGE = "en_core_web_sm"
WIKI_EMPTY_NEEDS_SPACY_LEAF = "spaCy not installed (see right pane)"
WIKI_EMPTY_NEEDS_SPACY_DETAIL = (
    "## Wiki entity extraction needs spaCy\n\n"
    "Install it then re-ingest documents:\n\n"
    "```sh\n"
    "uv pip install spacy\n"
    "python -m spacy download en_core_web_sm\n"
    "```"
)


def wiki_empty_state_leaf() -> str:
    """Single-line sidebar tree leaf for the empty-wiki state."""
    if not _spacy_available():
        return WIKI_EMPTY_NEEDS_SPACY_LEAF
    return WIKI_EMPTY_STATE


def wiki_empty_state_detail() -> str:
    """Right-pane markdown body for the empty-wiki state."""
    if not _spacy_available():
        return WIKI_EMPTY_NEEDS_SPACY_DETAIL
    return WIKI_NO_CONTENT


@lru_cache(maxsize=1)
def _spacy_available() -> bool:
    """True when spaCy and its NER model are both importable.

    Presence check only: the empty state repaints on every view switch and
    every filter keystroke, so loading the pipeline here would stall the UI
    thread for a second per paint. Cached because the answer cannot change
    within a session.
    """
    return all(find_spec(name) is not None for name in ("spacy", _SPACY_MODEL_PACKAGE))


WIKI_SEARCH_PLACEHOLDER = "Filter pages..."
WIKI_NO_CONTENT = "Select a page to view"
WIKI_NO_MATCHES = "No pages match '{filter}'"
WIKI_LOAD_FAILED_LEAF = "Failed to load pages (see right pane)"
WIKI_LOAD_FAILED = "## Failed to load wiki pages\n\n{error}"
WIKI_BUILD_STARTING = "Starting wiki run..."
WIKI_BUILD_PHASE = "{phase}..."
WIKI_BUILD_PAGE = "{label} ({current}/{total})"
WIKI_BUILD_DONE = "Wiki build finished: {count} pages"
WIKI_ALREADY_ACTIVE = "Wiki build in progress, please wait"
WIKI_STUBS_HEADING = "Not written yet"
WIKI_STUB_LABEL = "[dim]{title}[/] [dim italic](not written)[/]"
WIKI_STUB_DETAIL = (
    "# {title}\n\n"
    "*This page has not been written yet.*\n\n"
    "{label} appears in {sources}. Opening it offers to write the page, which "
    "spends one LLM call and is GPU-heavy.\n\n"
    "If you would rather not be asked about pages like this, turn the wiki off "
    "in settings."
)
WIKI_STUB_CONFIRM_TITLE = "Write this page?"
WIKI_STUB_CONFIRM_MESSAGE = (
    "Writing {label} spends one LLM call and is GPU-heavy. It draws on {sources}.\n\n"
    "You can turn the wiki off in settings if you would rather not be asked."
)
WIKI_STUB_TASK = "Write {label}"
WIKI_STUB_DONE = "Wrote {label}"
WIKI_STUB_FAILED = "Could not write {label}: {error}"
WIKI_STUB_STALE = "Nothing left to write {label} from; its sources are gone"
WIKI_WIPE_CONFIRM_TITLE = "Delete the wiki?"
WIKI_WIPE_CONFIRM_MESSAGE = (
    "This deletes every generated page and its indexed rows. "
    "Your documents are not touched. This cannot be undone."
)
WIKI_WIPE_DISABLED_TITLE = "Wiki turned off. Delete what it generated?"
WIKI_WIPE_DISABLED_MESSAGE = (
    "Turning the wiki off stops new pages being written, but the pages already "
    "generated stay on disk and in search. Delete them now?"
)
WIKI_WIPE_RUNNING = "Deleting wiki pages..."
WIKI_WIPE_DONE = "Wiki deleted: {count} pages removed"
WIKI_WIPE_NOTHING = "No wiki pages to delete"
WIKI_INDEX_LABEL = "Index"
WIKI_LOG_LABEL = "Log"
WIKI_DRAFTS_TITLE = "Wiki Drafts"
WIKI_DRAFTS_EMPTY = "No drafts pending review"
WIKI_DRAFTS_LOAD_FAILED = "Failed to load drafts: {error}"
WIKI_DRAFTS_COLUMN_SLUG = "Slug"
WIKI_DRAFTS_COLUMN_KIND = "Kind"
WIKI_DRAFTS_COLUMN_DRIFT = "Drift"
WIKI_DRAFTS_COLUMN_FAITHFULNESS = "Faithfulness"
WIKI_DRAFTS_COLUMN_PUBLISHED = "Published?"
WIKI_DRAFTS_KIND_DRIFT = "drift"
WIKI_DRAFTS_DIFF_EMPTY = "Select a draft to view its diff"
WIKI_DRAFTS_DIFF_NONE = "(no differences)"
WIKI_DRAFTS_DIFF_FAILED = "Failed to load diff: {error}"
WIKI_DRAFTS_ACCEPT_CONFIRM_TITLE = "Accept draft?"
WIKI_DRAFTS_ACCEPT_CONFIRM_MESSAGE = (
    "Overwrite the published page with {slug} and re-index? This cannot be undone."
)
WIKI_DRAFTS_REJECT_CONFIRM_TITLE = "Reject draft?"
WIKI_DRAFTS_REJECT_CONFIRM_MESSAGE = "Delete draft {slug}? The published page will not change."
WIKI_DRAFTS_ACCEPTED = "Accepted {slug}"
WIKI_DRAFTS_REJECTED = "Rejected {slug}"
WIKI_DRAFTS_ACCEPT_FAILED = "Accept failed: {error}"
WIKI_DRAFTS_REJECT_FAILED = "Reject failed: {error}"
WIKI_DRAFTS_ACCEPT_TASK = "Accept draft {slug}"
WIKI_DRAFTS_REJECT_TASK = "Reject draft {slug}"
WIKI_DRAFTS_MISSING = "missing: {slug}"
WIKI_DRAFTS_NO_MATCHES = "No drafts match '{filter}'"
WIKI_DRAFTS_PUBLISHED_YES = "yes"
WIKI_DRAFTS_PUBLISHED_NO = "no"
WIKI_DRAFTS_SEARCH_PLACEHOLDER = "Filter drafts..."
MEMORIES_EMPTY = "No memories stored. Use /remember to add one."
MEMORIES_DISABLED = "Memory is off. Enable it with /set memory_enabled true."
MEMORIES_LOAD_FAILED = "Failed to load memories: {error}"
MEMORIES_COLUMN_KIND = "Kind"
MEMORIES_COLUMN_SHARED = "Shared"
MEMORIES_COLUMN_TEXT = "Memory"
MEMORIES_FLAG_YES = "yes"
MEMORIES_FLAG_NO = "no"
MEMORIES_SEARCH_PLACEHOLDER = "Filter memories..."
MEMORIES_EMPTY_STATE = (
    "Memories are notes lilbee keeps about you between chats. Use /remember to save one."
)
MEMORIES_NO_MATCHES = "No memories match this filter."
MEMORIES_DELETE_CONFIRM_TITLE = "Delete memory?"
MEMORIES_DELETE_CONFIRM_MESSAGE = "Delete this memory? This cannot be undone."
MEMORIES_DELETED = "Deleted memory"
MEMORIES_DELETE_FAILED = "Delete failed: {error}"
MEMORIES_DELETE_NOT_FOUND = "Memory not found; it may already be gone."
MEMORIES_FLAG_NOT_FOUND = "Memory not found; it may already be gone."
MEMORIES_SHARED_ON = "Shared with agents"
MEMORIES_SHARED_OFF = "No longer shared with agents"
MEMORIES_FLAG_FAILED = "Update failed: {error}"
# Re-export the shared heading map with string keys so callers can
# look up by raw ``page_type`` string without coercion.
WIKI_TYPE_HEADINGS: dict[str, str] = {
    kind.value: label for kind, label in _WIKI_TYPE_HEADINGS.items()
}
APP_QUIT_AGAIN_HINT = "Answer cancelled. Press Ctrl+C again to quit."
SETUP_WELCOME = "Welcome to lilbee"
SETUP_SUBTITLE = "Pick a chat model and an embedding model to get started."
SETUP_INTRO = (
    "lilbee needs two models to work: one for chat and one for search. "
    "Pick one of each below: highlight a card and press [b]Enter[/b] to install. "
    "Downloads continue in the background, so you can keep picking or press [b]Esc[/b] when done."
)
SETUP_LOADING = "Finding popular models…"
SETUP_PICKS_UNAVAILABLE = (
    "Couldn't reach HuggingFace. Check your connection, or press Esc and use "
    "'lilbee model pull <ref>'."
)
SETUP_HEADING_CHAT = "Chat Models"
SETUP_HEADING_EMBED = "Embedding Models"
SETUP_ENTER_HINT = "Enter on a card to install  ·  Esc when done"
SETUP_RETURN_HINT = "Your existing models are ready  ·  Esc to return"
SETUP_CARD_HINT = "↵ Enter to install"
INSTALLED_CARD_HINT = "D / ⌫ to delete"

# Architecture compatibility pill labels (catalog row).
# SUPPORTED renders nothing to keep the row visually quiet for the common case.
COMPAT_PILL_UNSUPPORTED = "unsupported"
COMPAT_PILL_UNKNOWN = "untested"

# Architecture compatibility copy for the catalog detail view + confirm modal.
COMPAT_DETAIL_SENTENCE_SUPPORTED = "Supported by your llama.cpp build."
COMPAT_DETAIL_SENTENCE_UNSUPPORTED = (
    "Architecture {arch} is not in the supported set. Pull may fail at load."
)
COMPAT_DETAIL_SENTENCE_UNKNOWN = (
    "Architecture unknown until download. Pull will probe the header first."
)
COMPAT_MODAL_TITLE = "Architecture not supported"
COMPAT_MODAL_BODY = (
    "This model uses architecture {arch}. Your lilbee build doesn't support it, "
    "so loading after download will probably fail. Pull anyway?"
)
DEFAULT_VIEW = "Chat"
CATALOG_VIEW = "Catalog"
WIKI_VIEW = "Wiki"
FLEET_VIEW = "Fleet"
SESSIONS_VIEW = "Sessions"
FLEET_TITLE = "Placement"
FLEET_STATE_AUTO = "auto"
FLEET_STATE_MANUAL = "manual"
FLEET_STATE_EDITED = "edited · ctrl+s to apply"
FLEET_STATE_REBUILDING = "rebuilding fleet…"
FLEET_SINGLE_GPU_NOTE = "One graphics card: everything runs here."
FLEET_GPU_PROBING = "probing GPUs…"
FLEET_NO_GPUS = "(no GPUs detected)"
# Shown when the device probe failed outright (e.g. a wedged GPU driver), so the
# panel names the problem instead of sitting on the probing placeholder forever.
FLEET_GPU_PROBE_FAILED = "GPU probe failed: {reason}"
# Shown for a role that has no placement because its model isn't downloaded, so the
# empty slot reads as a fixable state instead of "GPU placement is broken".
FLEET_MODEL_NOT_DOWNLOADED = "{role}: {model} not downloaded, pull it to place it"
FLEET_SAVED_PLACEMENT_IGNORED = (
    "A saved placement does not fit this hardware and is being ignored. "
    "Press the auto command to clear it, or set a new one."
)
# Shown when an Intel GPU's utilization is unreadable only because intel_gpu_top
# lacks the CAP_PERFMON grant, so the muted "--" reads as a fixable state.
FLEET_INTEL_UTIL_GRANT = (
    "Intel GPU utilization needs a one-time grant: "
    "sudo setcap cap_perfmon+ep {binary}  (or Linux 6.5+ reads it with no setup)"
)
# Shown when intel_gpu_top is not installed at all: the i915 PMU covers kernels
# too old to publish fdinfo engine counters, so installing igt-gpu-tools is the
# only path to utilization there.
FLEET_INTEL_UTIL_INSTALL = (
    "Intel GPU utilization needs the igt-gpu-tools package "
    "(then: sudo setcap cap_perfmon+ep $(command -v intel_gpu_top))"
)


def intel_util_hint_text(hint: IntelUtilHint) -> str:
    """Localized fix instruction for an unreadable Intel util reading."""
    if hint.kind is IntelHintKind.GRANT and hint.binary is not None:
        return FLEET_INTEL_UTIL_GRANT.format(binary=hint.binary)
    return FLEET_INTEL_UTIL_INSTALL


FLEET_CMD_PREVIEW = "Preview"
FLEET_CMD_APPLY = "Apply"
FLEET_CMD_AUTO = "Auto"
FLEET_TAG_SPLIT = "split"
FLEET_TAG_SINGLE = "one card"
FLEET_HELP_ICON = "?"
# Single hover explanation for the whole drawer (kept friendly, no jargon).
FLEET_HELP_TOOLTIP = (
    "Top: how busy each GPU is right now.\n"
    "Grid: what runs on which GPU.\n"
    "  • chat is one model split across the highlighted cards (they work as one).\n"
    "  • embed/vision run as a full copy on each highlighted card.\n"
    "  • rerank runs on one card; pick a single GPU.\n"
    "Click a cell to change it, then Apply. Auto lets lilbee choose."
)
# The full nav-view universe in order. Single source for the view set: the
# settings bar pre-creates a tab per entry (toggling Wiki visibility at
# runtime), get_nav_views() gates Wiki, and app.get_views() derives its
# factory map from get_nav_views().
ALL_NAV_VIEWS: tuple[str, ...] = (
    DEFAULT_VIEW,
    CATALOG_VIEW,
    "Status",
    "Settings",
    "Tasks",
    WIKI_VIEW,
    FLEET_VIEW,
    SESSIONS_VIEW,
)


def get_nav_views() -> list[str]:
    """Return the active nav view names, including Wiki when enabled."""
    return [v for v in ALL_NAV_VIEWS if v != WIKI_VIEW or cfg.wiki]


MODE_NORMAL = "NORMAL"
MODE_INSERT = "INSERT"
TASKBAR_HINT = "Press t for Tasks"
TASKBAR_HINT_INPUT = "Esc then t for Tasks"
CHAT_REASONING_FINISHED = "reasoning · {tokens} tokens"

STATUS_DOCS_LOAD_FAILED = "(unable to read store)"
STATUS_DOCS_EMPTY = "(no documents yet)"
STATUS_DOCS_TITLE = "Documents"
TASKBAR_STARTING_WORKER = "Starting {labels} worker..."
TASKBAR_STARTING_WORKERS = "Starting {labels} workers..."
# Cold-start chat warm line: a spinner, the model being loaded, and the phase
# (with byte % while paging weights) so the held input reads as "loading {model}",
# not "stuck". The name is the model's display label, or this fallback before the
# warm has stamped which model it is loading.
TASKBAR_WARM_LINE = "warming up {name} · {detail}"
TASKBAR_WARM_FALLBACK_NAME = "chat"
TASKBAR_WARM_STARTING = "starting engine"
TASKBAR_WARM_READING = "reading weights {pct}%"
# Names the phase, like its two siblings above. "almost ready" predicted a
# finish time the engine has not promised, and was the one phase saying nothing
# about what is happening.
TASKBAR_WARM_LOADING = "loading into VRAM"

TASK_CENTER_TITLE = "Background Tasks"
TASK_CENTER_COUNTS = "{active} running  ·  {queued} queued  ·  {done} done"
TASK_CENTER_HINT = "r refresh   c cancel   C clear done   q back   j/k navigate"
TASK_CENTER_EMPTY_HEADLINE = "✓ all caught up"
TASK_CENTER_EMPTY_DETAIL = "no background tasks"
TASKBAR_SINGLE = "{name}  [b]{pct:.1f}%[/b]"
TASKBAR_MULTIPLE = "[b]{count} tasks running[/b]"
TASKBAR_ONE = "[b]1 task running[/b]"
TASKBAR_QUEUED_COUNT = "{count} queued"
TASKBAR_ALL_DONE = "[b]Done[/b]"
TASKBAR_FAILED = "[b]{count} task failed[/b]"
TASKBAR_FAILED_PLURAL = "[b]{count} tasks failed[/b]"
TASKBAR_SYNC_PENDING_ONE = "[b]1 doc to sync[/b] · S to sync"
TASKBAR_SYNC_PENDING_PLURAL = "[b]{count} docs to sync[/b] · S to sync"
TASKBAR_SYNC_PENDING_ONE_INPUT = "[b]1 doc to sync[/b] · Esc then S to sync"
TASKBAR_SYNC_PENDING_PLURAL_INPUT = "[b]{count} docs to sync[/b] · Esc then S to sync"
SYNC_CANCELLED_RESUME = "Sync cancelled. Press S to resume."
SYNC_EMBEDDING = "Embedding {file}"
SYNC_FILE_DONE = "Done: {file}"
ADD_SYNCING_FILE = "Syncing {file}..."
ADD_PAGE_PROGRESS = "{status} page {current} of {total}"
ADD_FILE_DONE = "Done {file}"

SETTINGS_API_KEYS_WARNING = (
    "These keys are stored in plain text at {path}. "
    "Anything you send to these providers leaves your machine. "
    "Do not route sensitive documents from lilbee through them."
)
MODEL_BAR_CLOUD_PROVIDER_WARNING = (
    "Chat prompts are being sent to {provider}. Do not share sensitive data."
)
CHAT_MODE_SEARCH_LABEL = "Search"
CHAT_MODE_CHAT_LABEL = "Chat"
CHAT_MODE_TOGGLE_TOOLTIP = (
    "Search runs your question through document retrieval. "
    "Chat skips retrieval and answers directly. Click or press F3 to flip."
)
CHAT_MODE_TOGGLE_DISABLED_TOOLTIP = (
    "Search needs an embedding model. Install one to enable Search mode."
)
CHAT_MODE_SEARCH_NO_RESULTS = "Search returned 0 results, falling back to chat for this turn."
CHAT_MODE_SET = "Mode: {label}"
MODEL_PICKER_TITLE_CHAT = "Pick a chat model"
MODEL_PICKER_TITLE_EMBED = "Pick an embedding model"
MODEL_PICKER_TITLE_VISION = "Pick a vision model"
MODEL_PICKER_TITLE_RERANK = "Pick a reranker model"
MODEL_VALUE_NONE = "(none)"
MODEL_PICKER_DISABLE_LABEL = "(disabled, no model)"
MODEL_PICKER_CHAT_TOOLTIP = "Model used to answer your questions. Click to pick a different one."
MODEL_PICKER_EMBED_TOOLTIP = (
    "Model used to vectorize search queries (Search mode). Click to pick a different one."
)
MODEL_PICKER_VISION_TOOLTIP = (
    "Optional. Model used to read scanned PDFs and images. Click to pick one or browse the catalog."
)
MODEL_PICKER_RERANK_TOOLTIP = (
    "Optional. Model used to sharpen search results. Click to pick one or browse the catalog."
)
MODEL_PICKER_BROWSE_CATALOG = "Browse catalog to download..."
MODEL_PICKER_SEARCH_PLACEHOLDER = "Search models..."

# Model bar (chat-screen, below the input)
MODEL_BAR_CHAT_LABEL = "Chat"
MODEL_BAR_EMBED_LABEL = "Embed"
MODEL_BAR_VISION_LABEL = "Vision"
MODEL_BAR_RERANK_LABEL = "Rerank"
MODEL_BAR_DISABLED = "disabled"
MODEL_BAR_NOT_INSTALLED = "{name} (not installed)"
MODEL_BAR_NOT_INSTALLED_TOOLTIP = (
    "This model is not installed. Click to pick another, "
    "or press m to open the Catalog and install it."
)
MODEL_PICKER_TURN_OFF = "Turn off this model"
MODEL_PICKER_HINT = "Enter to pick · Esc to cancel · / to search"
