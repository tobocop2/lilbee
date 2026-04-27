"""Default values and constants for :mod:`lilbee.config`.

Holds frozen literal data: directory ignore lists, NER label allow-list,
LanceDB table names, the crawl URL exclusion patterns (grouped per
category), and the default system / CORS prompts.
"""

DEFAULT_IGNORE_DIRS = frozenset(
    {
        "node_modules",
        "__pycache__",
        "venv",
        "build",
        "dist",
        "target",
        "vendor",
        "_build",
        "coverage",
        "htmlcov",
    }
)

# spaCy NER labels that map onto something wiki-shaped. Excludes
# QUANTITY / ORDINAL / CARDINAL / DATE / TIME / MONEY / PERCENT /
# LANGUAGE / LAW because pages for "42" or "2021" are never useful.
# FAC (buildings / airports) and NORP (nationalities / political /
# religious groups) are included because corpora routinely surface
# them as wiki-worthy topics.
DEFAULT_ALLOWED_NER_LABELS = frozenset(
    {"PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART", "PRODUCT", "FAC", "NORP"}
)

# Timeout for backend catalog / management HTTP calls.
DEFAULT_HTTP_TIMEOUT = 30.0

# Safe default + cap for chat-mode n_ctx; full 128K+ training contexts OOM laptops.
DEFAULT_NUM_CTX = 8192

CHUNKS_TABLE = "chunks"
SOURCES_TABLE = "_sources"
CITATIONS_TABLE = "_citations"
META_TABLE = "_meta"
CONCEPT_NODES_TABLE = "concept_nodes"
CONCEPT_EDGES_TABLE = "concept_edges"
CHUNK_CONCEPTS_TABLE = "chunk_concepts"

# Default URL-exclusion regexes for recursive crawls. Grouped by source
# CMS / category. User overrides come from LILBEE_CRAWL_EXCLUDE_PATTERNS
# (newline-separated) or config.toml.

# WordPress scaffolding: admin UIs, APIs, RPC, numeric permalinks, Elementor.
_WP_EXCLUDE: tuple[str, ...] = (
    r"/wp-admin/",
    r"/wp-login(\.php)?",
    r"/wp-json/",
    r"/xmlrpc\.php",
    r"/wp-cron\.php",
    r"/wp-includes/",
    r"/wp-content/uploads/",
    r"\?p=\d+",
    r"\?page_id=\d+",
    r"\?cat=\d+",
    r"/elementor-\d+",
    r"\?elementor_library",
)

# Pagination and archive permalinks (WP + other CMSes share this shape).
_ARCHIVE_EXCLUDE: tuple[str, ...] = (
    r"/page/\d+/?$",
    r"\?paged?=\d+",
    r"/20\d{2}(/\d{2}(/\d{2})?)?/?$",
    r"/tag/",
    r"/category/",
    r"/author/",
    r"/archives?/?$",
    r"/comment-page-\d+",
)

# Syndication feeds (content-duplicated in HTML pages).
_FEED_EXCLUDE: tuple[str, ...] = (
    r"/feed/?$",
    r"/feed/atom/?$",
    r"/feed/rdf/?$",
    r"/comments/feed/?$",
    r"/rss/?$",
)

# Duplicate views of the same canonical page (AMP, print, preview).
_DUPLICATE_VIEW_EXCLUDE: tuple[str, ...] = (
    r"/amp/?$",
    r"\?amp=",
    r"\?print=",
    r"/print/?$",
    r"\?preview=",
)

# WP attachment URLs (point at media, not content pages).
_ATTACHMENT_EXCLUDE: tuple[str, ...] = (
    r"/attachment/",
    r"\?attachment_id=",
)

# Auth and account flows (generic across CMSes and e-commerce platforms).
_AUTH_EXCLUDE: tuple[str, ...] = (
    r"/login",
    r"/logout",
    r"/register",
    r"/signup",
    r"/signin",
    r"/account",
    r"/my-account/",
    r"/profile",
    r"/password-reset",
    r"/forgot-password",
)

# E-commerce transactional flows (cart / checkout / compare / etc.).
_ECOMMERCE_EXCLUDE: tuple[str, ...] = (
    r"/cart",
    r"/checkout",
    r"/wishlist",
    r"/orders?",
    r"/compare",
    r"/products\.json",
    r"/collections/.+/products/.+\?page=",
)

# Marketing / tracking query parameters (utm_*, fbclid, gclid, etc.).
_TRACKING_EXCLUDE: tuple[str, ...] = (
    (
        r"[?&]("
        r"utm_[a-z_]+"
        r"|fbclid|gclid|msclkid|yclid"
        r"|mc_cid|mc_eid"
        r"|_hsenc|_hsmi|hsCtaTracking"
        r"|mkt_tok|mkt_[a-z_]+"
        r"|trk|trkInfo"
        r"|dm_i"
        r"|vero_id|vero_conv"
        r"|oly_anon_id|oly_enc_id"
        r"|igshid"
        r"|pk_campaign|pk_source|pk_medium|pk_[a-z_]+"
        r"|_ga"
        r"|ref|referrer"
        r"|affiliate|aff_id|aff_ref|aff|partner"
        r"|srsltid"
        r"|share|replytocom"
        r")="
    ),
)

# Site-meta URLs and non-HTML resources; skipped before fetch.
_META_EXCLUDE: tuple[str, ...] = (
    r"/sitemap[^/]*\.xml",
    r"/robots\.txt",
    r"/humans\.txt",
    r"/favicon\.ico",
    r"/\.well-known/",
    r"\.(jpe?g|png|gif|webp|avif|svg|ico|pdf|docx?|xlsx?|pptx?|zip|tar|gz|mp3|mp4|webm|ogg|ttf|woff2?|css|js|map|json|xml)(\?.*)?$",
)

# Mediawiki/Wikipedia navlinks that dominate BFS before the article body.
_MEDIAWIKI_EXCLUDE: tuple[str, ...] = (
    r"/wiki/Main_Page$",
    r"/wiki/Wikipedia:",
    r"/wiki/Portal:",
    r"/wiki/Help:",
    r"/wiki/Special:",
    r"/wiki/Category:",
    r"/wiki/Template:",
    r"/wiki/Template_talk:",
    r"/wiki/Talk:",
    r"/wiki/File:",
    r"/wiki/File_talk:",
    r"/wiki/User:",
    r"/wiki/User_talk:",
    r"/w/index\.php",
)

DEFAULT_CRAWL_EXCLUDE_PATTERNS: tuple[str, ...] = (
    *_WP_EXCLUDE,
    *_ARCHIVE_EXCLUDE,
    *_FEED_EXCLUDE,
    *_DUPLICATE_VIEW_EXCLUDE,
    *_ATTACHMENT_EXCLUDE,
    *_AUTH_EXCLUDE,
    *_ECOMMERCE_EXCLUDE,
    *_TRACKING_EXCLUDE,
    *_META_EXCLUDE,
    *_MEDIAWIKI_EXCLUDE,
)


_DEFAULT_SYSTEM_PROMPT = (
    "You are a precise, direct assistant grounded in the provided context. "
    "Answer using only the context: if it doesn't contain enough information, "
    "say so rather than guessing. Be specific: quote relevant passages and "
    "reference context by number (e.g. [1], [2]) inline. Prefer exact values "
    "over approximations. For code, prefer working examples over abstract "
    "explanations. Keep responses concise unless asked to elaborate."
)

# CORS allow-origin regex: Obsidian (desktop + iOS) and localhost loopback.
# Mutating endpoints still require auth regardless of origin.
_DEFAULT_CORS_ORIGIN_REGEX = (
    r"^(app://obsidian\.md"
    r"|capacitor://localhost"
    r"|https?://localhost(:\d+)?"
    r"|https?://127\.0\.0\.1(:\d+)?"
    r"|https?://\[::1\](:\d+)?)$"
)
