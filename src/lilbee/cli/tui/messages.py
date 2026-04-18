"""Centralized user-facing messages for the TUI.

ALL user-facing text MUST be defined here. Inline strings in
screens and widgets are forbidden -- this enables future i18n
and ensures consistent messaging.
"""

from __future__ import annotations

from lilbee.config import cfg
from lilbee.wiki.shared import WikiPageType

CMD_UNKNOWN = "Unknown command: {cmd}"
CMD_ADD_NOT_FOUND = "Not found: {path}"
CMD_ADD_SUCCESS = "Added {count} file(s), syncing..."
CMD_ADD_ERROR = "Error: {error}"
CMD_CRAWL_USAGE = "Usage: /crawl <url> [--depth N] [--max-pages N]"
CMD_CRAWL_STARTED = "Crawling {url}..."
CMD_CRAWL_PAGE = "Crawling [{current}/{total}]: {url}"
CMD_CRAWL_SUCCESS = "Crawled {count} page(s) from {url}"
CMD_CRAWL_FAILED = "Crawl failed: {error}"
CMD_CRAWL_SYNCING = "Syncing crawled pages..."
CMD_DELETE_NO_DOCS = "No documents indexed"
CMD_DELETE_USAGE = "Documents: {names}\nUsage: /delete <filename>"
CMD_DELETE_NOT_FOUND = "Not found: {name}"
CMD_DELETE_SUCCESS = "Deleted {name}"
CMD_RESET_CONFIRM = "Type '/reset confirm' to delete all data"
CMD_RESET_SUCCESS = "Knowledge base reset"
CMD_RESET_PARTIAL = "Knowledge base reset ({skipped} item(s) could not be deleted)"
CMD_RESET_FAILED = "Reset failed: {error}"
CMD_SET_UNKNOWN = "Unknown setting: {key}"
CMD_SET_SUCCESS = "{key} = {value}"
CMD_SET_INVALID = "Invalid value for {key}: {error}"
CMD_MODEL_SET = "Model set to {name}"
CMD_REMOVE_USAGE = "Usage: /remove <model_name>"
CMD_REMOVE_NOT_FOUND = "{name} is not installed"
CMD_REMOVE_SUCCESS = "Removed {name}"
CMD_REMOVE_FAILED = "Failed to remove {name}"
CMD_CANCEL = "Cancelled active operations"
CMD_CLEAR = "Conversation cleared"
CMD_THEME_LIST = "Themes: {names}"
CMD_WIKI_DISABLED = "Wiki is disabled (set wiki = true in settings)"
CMD_WIKI_NO_SOURCES = "No indexed documents. Use /add first."
CMD_WIKI_NOT_FOUND = "Source not found: {name}"
CMD_WIKI_STARTED = "Generating wiki for {count} source(s)..."
CMD_WIKI_PROGRESS = "{name}: {stage}"
CMD_WIKI_SUCCESS = "Generated {generated}/{total} wiki page(s)"
CMD_WIKI_FAILED = "Wiki generation failed: {error}"
TASK_NAME_CRAWL = "Crawl {url}"
TASK_NAME_WIKI = "Wiki ({count})"
CMD_WIKI_NO_PAGES = "No pages generated"
CMD_WIKI_UNKNOWN_ERROR = "Unknown error"
CMD_WIKI_NONE_GENERATED = (
    "Wiki generation produced no pages for {total} source(s). "
    "The model may not support structured output."
)
STREAM_ERROR = "\n\n*Error: {error}*"
SYNC_STATUS_SYNCING = "Syncing..."
SYNC_STATUS_DONE = "Synced ({count} docs)"
SYNC_STATUS_FAILED = "Sync failed"
SYNC_FILE_PROGRESS = "Syncing [{current}/{total}]: {file}"
SYNC_ALREADY_ACTIVE = "Sync in progress, please wait"
EMBEDDING_SET = "Embedding model: {name}"
CMD_CRAWL_UNAVAILABLE = "Install crawl4ai: pip install 'lilbee[crawler]'"
CRAWL_DIALOG_TITLE = "Crawl a URL"
CRAWL_DIALOG_URL_PLACEHOLDER = "example.com (https:// added automatically)"
CRAWL_DIALOG_DEPTH_PLACEHOLDER = "default from config"
CRAWL_DIALOG_MAX_PAGES_PLACEHOLDER = "default from config"
CRAWL_DIALOG_URL_LABEL = "URL"
CRAWL_DIALOG_DEPTH_LABEL = "Depth"
CRAWL_DIALOG_MAX_PAGES_LABEL = "Max pages"
CRAWL_DIALOG_SUBMIT = "Crawl"
CRAWL_DIALOG_CANCEL = "Cancel"
CRAWL_DIALOG_URL_REQUIRED = "URL is required"
CRAWL_DIALOG_INVALID_URL = "Invalid URL: {error}"
CRAWL_DIALOG_INVALID_NUMBER = "{field} must be a non-negative integer"
EMBEDDING_MISSING = (
    "No embedding model, search disabled. "
    "Run /models to install one, or: lilbee models install nomic-embed-text"
)
THEME_SET = "Theme: {name}"
HEADING_OUR_PICKS = "Our picks"
HEADING_INSTALLED = "Installed"
CATALOG_USING_REMOTE = "Using {name} (remote)"
CATALOG_ALREADY_INSTALLED = "{name} is already installed"
CATALOG_NO_TASK_BAR = "Cannot download: task bar not found"
CATALOG_QUEUED_DOWNLOAD = "Queued download: {name}"
CATALOG_INSTALLED_OK = "{name} installed"
CATALOG_GATED_REPO = "{name} requires login, run /login or lilbee login"
CATALOG_DOWNLOAD_FAILED = "{name}: download failed"
CATALOG_SELECT_TO_DELETE = "Select a model to delete"
CATALOG_NOT_INSTALLED = "{name} is not installed"
CATALOG_CONFIRM_DELETE = "Delete {name}? Press d again to confirm"
CATALOG_DELETED = "Deleted {name}"
CATALOG_DELETE_FAILED = "Delete failed: {error}"
CATALOG_NO_MATCH = "No models match your filters."
CATALOG_FILTER_PLACEHOLDER = "Filter models..."
CATALOG_VIEW_TOGGLE_GRID = "Press v for full list view · / to search"
CATALOG_VIEW_TOGGLE_TABLE = "Press v for card view"
CATALOG_BROWSE_MORE = "Browse more models →"
CHAT_INPUT_PLACEHOLDER = "Ask a question or type / for commands"
CHAT_ONLY_BANNER = "Chat only, no document search. Press F5 to set up embedding model."
CHAT_LOGIN_PROMPT = "Paste your token with /login <token>"
CHAT_LOGGED_IN = "Logged in to HuggingFace"
CHAT_LOGIN_FAILED = "Login failed: {error}"
CHAT_VERSION = "lilbee {version}"
CHAT_RENDERING = "Rendering: {label}"
SETTINGS_READ_ONLY = "read-only"
SETTINGS_INVALID_VALUE = "Invalid value: {error}"
WIKI_EMPTY_STATE = "No wiki pages found"
WIKI_SEARCH_PLACEHOLDER = "Filter pages..."
WIKI_NO_CONTENT = "Select a page to view"
# Keyed by the WikiPageType value (a ``str`` via StrEnum) so callers can
# look up a heading from a raw ``page_type`` string without coercion.
WIKI_TYPE_HEADINGS: dict[str, str] = {
    WikiPageType.SUMMARY: "Summaries",
    WikiPageType.SYNTHESIS: "Synthesis",
}
APP_CANCELLED = "Cancelled"
SETUP_WELCOME = "Welcome to lilbee"
SETUP_SUBTITLE = "Pick a chat model and an embedding model to get started."
SETUP_HEADING_CHAT = "Chat Models"
SETUP_HEADING_EMBED = "Embedding Models"
SETUP_CHAT_SLOT = "Chat: {name}"
SETUP_EMBED_SLOT = "Embed: {name}"
SETUP_SLOT_EMPTY = "not selected"
SETUP_TOTAL_DOWNLOAD = "Download: {size}"
SETUP_INSTALL_BUTTON = "Install & Go"
SETUP_BROWSE_CATALOG = "Browse full catalog"
SETUP_SKIP_BUTTON = "Skip setup"
SETUP_CONTINUE_NO_SEARCH = "Continue without search"
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
TASKBAR_HINT = "Esc then t for Task Center"
SYNC_EMBEDDING = "Embedding {file}"
SYNC_FILE_DONE = "Done: {file}"
