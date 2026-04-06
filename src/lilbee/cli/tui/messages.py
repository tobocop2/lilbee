"""Centralized user-facing messages for the TUI.

ALL user-facing text MUST be defined here. Inline strings in
screens and widgets are forbidden -- this enables future i18n
and ensures consistent messaging.
"""

from __future__ import annotations

# -- Slash command notifications -----------------------------------------------

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
CMD_RESET_FAILED = "Reset failed: {error}"
CMD_SET_UNKNOWN = "Unknown setting: {key}"
CMD_SET_SUCCESS = "{key} = {value}"
CMD_SET_INVALID = "Invalid value for {key}: {error}"
CMD_MODEL_SET = "Model set to {name}"
CMD_VISION_DISABLED = "Vision OCR disabled"
CMD_VISION_SET = "Vision model: {name}"
CMD_VISION_STATUS = (
    "Vision: {current}\n"
    "Recommended: maternion/LightOnOCR-2 (fastest, best quality)\n"
    "Usage: /vision maternion/LightOnOCR-2:latest  or  /vision off"
)
CMD_REMOVE_USAGE = "Usage: /remove <model_name>"
CMD_REMOVE_NOT_FOUND = "{name} is not installed"
CMD_REMOVE_SUCCESS = "Removed {name}"
CMD_REMOVE_FAILED = "Failed to remove {name}"
CMD_CANCEL = "Cancelled active operations"
CMD_THEME_LIST = "Themes: {names}"
STREAM_ERROR = "\n\n*Error: {error}*"
SYNC_STATUS_SYNCING = "Syncing..."
SYNC_STATUS_DONE = "Synced ({count} docs)"
SYNC_STATUS_FAILED = "Sync failed"
SYNC_FILE_PROGRESS = "Syncing [{current}/{total}]: {file}"
SYNC_ALREADY_ACTIVE = "Sync in progress, please wait"
EMBEDDING_SET = "Embedding model: {name}"
CMD_CRAWL_UNAVAILABLE = "Install crawl4ai: pip install 'lilbee[crawler]'"
EMBEDDING_MISSING = (
    "No embedding model \u2014 search disabled. "
    "Run /models to install one, or: lilbee models install nomic-embed-text"
)

# -- Shared / cross-screen ----------------------------------------------------

THEME_SET = "Theme: {name}"
HEADING_OUR_PICKS = "Our picks"
HEADING_INSTALLED = "Installed"

# -- Catalog screen ------------------------------------------------------------

CATALOG_USING_REMOTE = "Using {name} (remote)"
CATALOG_ALREADY_INSTALLED = "{name} is already installed"
CATALOG_NO_TASK_BAR = "Cannot download: task bar not found"
CATALOG_QUEUED_DOWNLOAD = "Queued download: {name}"
CATALOG_INSTALLED_OK = "{name} installed"
CATALOG_GATED_REPO = "{name} requires login \u2014 run /login or lilbee login"
CATALOG_DOWNLOAD_FAILED = "{name}: download failed"
CATALOG_SELECT_TO_DELETE = "Select a model to delete"
CATALOG_NOT_INSTALLED = "{name} is not installed"
CATALOG_CONFIRM_DELETE = "Delete {name}? Press d again to confirm"
CATALOG_DELETED = "Deleted {name}"
CATALOG_DELETE_FAILED = "Delete failed: {error}"
CATALOG_NO_MATCH = "No models match your filters."
CATALOG_FILTER_PLACEHOLDER = "Filter models... ( Esc to close)"
CATALOG_VIEW_TOGGLE_GRID = "Press v for full list view \u00b7 / to search"
CATALOG_VIEW_TOGGLE_TABLE = "Press v for card view"
CATALOG_BROWSE_MORE = "Browse more models \u2192"

# -- Chat screen ---------------------------------------------------------------

CHAT_INPUT_PLACEHOLDER = "Ask a question or type / for commands"
CHAT_ONLY_BANNER = "Chat only \u2014 no document search. Press F5 to set up embedding model."
CHAT_LOGIN_PROMPT = "Paste your token with /login <token>"
CHAT_LOGGED_IN = "Logged in to HuggingFace"
CHAT_LOGIN_FAILED = "Login failed: {error}"
CHAT_VERSION = "lilbee {version}"
CHAT_RENDERING = "Rendering: {label}"

# -- Settings screen -----------------------------------------------------------

SETTINGS_READ_ONLY = "read-only"
SETTINGS_INVALID_VALUE = "Invalid value: {error}"

# -- App -----------------------------------------------------------------------

APP_CANCELLED = "Cancelled"

# -- Setup wizard --------------------------------------------------------------

SETUP_TITLE = "Setup Wizard"
SETUP_STEP_CHAT = "Step 1/2: Choose a chat model"
SETUP_STEP_EMBED = "Step 2/2: Choose an embedding model"
SETUP_SKIP_BUTTON = "Skip \u2014 chat only (no document search)"
SETUP_CONFIRM_BUTTON = "Confirm"
SETUP_INSTALLED_LABEL = "Installed locally:"
SETUP_FEATURED_LABEL = "Featured models (download):"
SETUP_BROWSE_CATALOG = "Browse full catalog"
SETUP_CONNECTING = "Connecting to HuggingFace..."
SETUP_INSTALLED_STATUS = "{name} installed!"
SETUP_LOGIN_REQUIRED = "{name} requires login (run: lilbee login)"

# -- Status bar ----------------------------------------------------------------

DEFAULT_VIEW = "Chat"
NAV_VIEWS: list[str] = [DEFAULT_VIEW, "Catalog", "Status", "Settings", "Tasks"]
STATUS_BAR_HINTS = "  [dim]h/l[/] nav  [dim]?[/] help  [dim]^c[/] quit"
MODE_NORMAL = "NORMAL"
MODE_INSERT = "INSERT"

# -- Help modal ----------------------------------------------------------------

HELP_TEXT_TEMPLATE = """\
[bold]Navigation[/bold]

  h / Left       Previous view
  l / Right      Next view
  ? / F1 / ^h    Help (this screen)
  ^t             Cycle theme
  ^c             Quit

  [bold]Chat[/bold]
  Enter          Send message
  Escape         Cancel stream / normal mode
  /              Focus command input
  Tab            Autocomplete
  ^n / ^p        Cycle completions fwd / back
  Up / Down      Input history (when input focused)
  j / k          Scroll line (normal mode)
  g / G          Scroll to top / bottom
  ^d / ^u        Half-page down / up
  PgUp / PgDn    Full page scroll
  ^r             Toggle markdown rendering

  [bold]Catalog[/bold]
  j / k          Navigate list
  g / G          Jump to top / bottom
  /              Focus search
  s              Cycle sort order
  d / x          Delete model
  Space          Page down
  ^d / ^u        Half-page down / up
  Enter          Install / select model
  q / Escape     Back

  [bold]Settings / Status / Tasks[/bold]
  j / k          Navigate rows
  g / G          Jump to top / bottom
  d              Cancel task (Tasks only)
  q / Escape     Back

  [bold]Commands[/bold]  (type / for suggestions)
{commands_block}

  Press Escape or q to close.
"""
