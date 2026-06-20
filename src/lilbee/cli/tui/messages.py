"""Centralized user-facing messages for the TUI.

ALL user-facing text MUST be defined here. Inline strings in
screens and widgets are forbidden -- this enables future i18n
and ensures consistent messaging.
"""

from __future__ import annotations

from lilbee.core.config import cfg
from lilbee.wiki.shared import WIKI_TYPE_HEADINGS as _WIKI_TYPE_HEADINGS

CMD_UNKNOWN = "Unknown command: {cmd}"
CMD_ADD_NOT_FOUND = "Not found: {path}"
CMD_ADD_SUCCESS = "Added {count} file(s), syncing..."
CMD_ADD_DUPLICATE_TITLE = "File already in knowledge base"
CMD_ADD_DUPLICATE_MESSAGE = "{name} is already in the knowledge base. Overwrite and re-sync?"
CMD_ADD_SKIPPED_DUPLICATE = "Kept existing copy of {name}."
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
MODEL_UNUSABLE_OPENING_SETUP = (
    "{label} model {original!r} is unavailable ({reason}) and nothing is installed to fall back "
    "to. Opening setup so you can pick one."
)
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
    "Skipped (vision OCR returned no text): {files}. "
    "See ~/Library/Application Support/lilbee/logs/worker-vision.log "
    "for the underlying error."
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
        return SYNC_SKIPPED_VISION_FAILED.format(files=files)
    return SYNC_SKIPPED_NO_VISION.format(files=files)


def retry_skipped_message(count: int) -> str:
    """Toast for the 'Retry skipped documents' command."""
    return CMD_RETRY_SKIPPED_NONE if count == 0 else CMD_RETRY_SKIPPED_SOME.format(count=count)


CMD_DELETE_NO_DOCS = "No documents indexed"
CMD_DELETE_USAGE = "Documents: {names}\nUsage: /delete <filename>"
CMD_DELETE_NOT_FOUND = "Not found: {name}"
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
TASK_NAME_REBUILD = "Rebuild index"
CMD_SET_UNKNOWN = "Unknown setting: {key}"
CMD_SET_SUCCESS = "{key} = {value}"
CMD_SET_INVALID = "Invalid value for {key}: {error}"
CMD_SET_READONLY = "{key} is read-only; use the Models screen"
CMD_MODEL_SET = "Model set to {name}"
CMD_REMOVE_USAGE = "Usage: /remove <model_name>"
CMD_REMOVE_NOT_FOUND = "{name} is not installed"
CMD_REMOVE_SUCCESS = "Removed {name}"
CMD_REMOVE_FAILED = "Failed to remove {name}"
CMD_CANCEL = "Cancelled active operations"
CMD_CLEAR = "Conversation cleared"
CMD_THEME_LIST = "Themes: {names}"
CMD_WIKI_DISABLED = "Wiki is disabled (set wiki = true in settings)"
TASK_NAME_CRAWL = "Crawl {url}"
STREAM_ERROR = "\n\n*Error: {error}*"
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
CATALOG_QUEUED_DOWNLOAD = "Queued download: {name}"
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
CHAT_INPUT_PLACEHOLDER_DEFAULT = "Ask…   /  commands   F1  keys   F2  all commands"
SLASH_CATALOG_TITLE = "Slash Commands"
SLASH_CATALOG_FILTER_PLACEHOLDER = "Filter commands..."
SLASH_CATALOG_FOOTER_HINT = "↑↓ select   Enter run   Esc close"
SLASH_CATALOG_NO_MATCH = "No commands match"
HELP_HINT_COMMANDS = "type / for commands"
HELP_HINT_KEYS = "F1 for keys"
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
CHAT_WELCOME_TAGLINE = "your local search engine and personal encyclopedia."
CHAT_WELCOME_HINT = "Press / for commands, or just ask."
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


def _spacy_available() -> bool:
    try:
        from lilbee.retrieval.concepts.nlp import load_spacy_pipeline

        load_spacy_pipeline()
    except (ImportError, OSError):
        return False
    except Exception:
        return True
    return True


WIKI_SEARCH_PLACEHOLDER = "Filter pages..."
WIKI_NO_CONTENT = "Select a page to view"
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
APP_CANCELLED = "Cancelled"
SETUP_WELCOME = "Welcome to lilbee"
SETUP_SUBTITLE = "Pick a chat model and an embedding model to get started."
SETUP_INTRO = (
    "lilbee needs two models to work: one for chat and one for search. "
    "Pick one of each below: highlight a card and press [b]Enter[/b] to install. "
    "Downloads continue in the background, so you can keep picking or press [b]Esc[/b] when done."
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
COMPAT_PILL_UNKNOWN = "?"

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
_BASE_NAV_VIEWS: tuple[str, ...] = (DEFAULT_VIEW, "Catalog", "Status", "Settings", "Tasks")


def get_nav_views() -> list[str]:
    """Return the active nav view names, including Wiki when enabled."""
    views = list(_BASE_NAV_VIEWS)
    if cfg.wiki:
        views.append("Wiki")
    return views


MODE_NORMAL = "NORMAL"
MODE_INSERT = "INSERT"
TASKBAR_HINT = "Press t for Tasks"
TASKBAR_HINT_INPUT = "Esc then t for Tasks"
CHAT_REASONING_FINISHED = "reasoning · {tokens} tokens"
CHAT_SOURCES_LABEL = "sources"

STATUS_DOCS_LOAD_FAILED = "(unable to read store)"
STATUS_DOCS_EMPTY = "(no documents yet)"
STATUS_DOCS_TITLE = "Documents"
TASKBAR_STARTING_WORKER = "Starting {labels} worker..."
TASKBAR_STARTING_WORKERS = "Starting {labels} workers..."

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
MODEL_PICKER_TURN_OFF = "Turn off this model"
MODEL_PICKER_HINT = "Enter to pick · Esc to cancel · / to search"
