"""Default values and constants for :mod:`lilbee.core.config`.

Holds frozen literal data: directory ignore lists, the NER label allow-list,
LanceDB table names, the default HTTP timeout and context size, the crawl URL
exclusion patterns (grouped per category), the default RAG and general system
prompts, and the CORS allow-origin regex.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

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
# LANGUAGE / LAW because pages for "42" or "2021" are never useful, and
# NORP (nationalities / political / religious groups) because its surfaces
# are adjectival (Saturnian, American) and make poor page subjects; opt
# back in via the config override. FAC (buildings / airports) stays:
# corpora routinely surface them as wiki-worthy topics.
DEFAULT_ALLOWED_NER_LABELS = frozenset(
    {"PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART", "PRODUCT", "FAC"}
)

# Timeout for backend catalog / management HTTP calls.
DEFAULT_HTTP_TIMEOUT = 30.0

# Safe default + cap for chat-mode n_ctx; full 128K+ training contexts OOM laptops.
DEFAULT_NUM_CTX = 8192

CHUNKS_TABLE = "chunks"
SOURCES_TABLE = "_sources"
CITATIONS_TABLE = "_citations"
MEMORIES_TABLE = "_memories"
META_TABLE = "_meta"
PAGE_TEXTS_TABLE = "_page_texts"
CONCEPT_NODES_TABLE = "concept_nodes"
CONCEPT_EDGES_TABLE = "concept_edges"
CHUNK_CONCEPTS_TABLE = "chunk_concepts"
ENTITIES_TABLE = "entities"
ENTITY_SCHEMA_TABLE = "_entity_schema"
# Per-(subject, source) wiki mention evidence. The wiki stub index is a
# corpus-wide aggregate over this table, so a subject named below the floor in
# each separately-synced source still crosses it once its rows are all present.
WIKI_MENTIONS_TABLE = "_wiki_mentions"

# Tables an ingest writes per source, and the column holding the source key.
INGEST_SOURCE_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        CHUNKS_TABLE: "source",
        PAGE_TEXTS_TABLE: "source",
        CHUNK_CONCEPTS_TABLE: "chunk_source",
        ENTITIES_TABLE: "source",
        CITATIONS_TABLE: "source_filename",
        SOURCES_TABLE: "filename",
    }
)

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

# Regexes against the whole URL, not globs, so a bare prefix also matches
# longer words: /cart excluded /cartography. Require a segment boundary.
_PATH_BOUNDARY = r"(?:/|\?|#|$)"


def _whole_segments(*paths: str) -> tuple[str, ...]:
    """Anchor each path prefix so it matches a whole segment, not a word."""
    return tuple(path + _PATH_BOUNDARY for path in paths)


# Auth and account flows (generic across CMSes and e-commerce platforms).
_AUTH_EXCLUDE: tuple[str, ...] = _whole_segments(
    r"/login",
    r"/logout",
    r"/register",
    r"/signup",
    r"/signin",
    r"/account",
    r"/profile",
    r"/password-reset",
    r"/forgot-password",
)
_AUTH_EXCLUDE = (*_AUTH_EXCLUDE, r"/my-account/")

# E-commerce transactional flows (cart / checkout / compare / etc.).
_ECOMMERCE_EXCLUDE: tuple[str, ...] = _whole_segments(
    r"/cart",
    r"/checkout",
    r"/wishlist",
    r"/orders?",
    r"/compare",
)
_ECOMMERCE_EXCLUDE = (
    *_ECOMMERCE_EXCLUDE,
    r"/products\.json",
    r"/collections/.+/products/.+\?page=",
)

# Marketing / tracking query parameters (utm_*, fbclid, gclid, etc.).
# Vendor campaign tokens only. Dropping ?utm_source= is free (the canonical
# URL is in the frontier too), but ?ref= and ?share= are ordinary content
# links on docs and forum platforms.
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
        r"|affiliate|aff_id|aff_ref|aff|partner"
        r"|srsltid"
        r"|replytocom"
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


DEFAULT_RAG_SYSTEM_PROMPT = (
    "You are a precise assistant answering from the user's own documents. "
    "Ground every claim in the numbered context passages and nothing else; if "
    "they don't cover the question, say so plainly instead of guessing or "
    "answering from general knowledge. Synthesize across passages rather than "
    "leaning on one, and if they disagree, note the conflict. Cite inline by "
    "placing the passage number in brackets right after the claim it supports "
    "(e.g. [1] or [2][5]), and cite only passages you actually used. Do not "
    "write a Sources, References, or Bibliography list at the end; the app adds "
    "the real source list for you. Prefer exact values, names, and short quotes "
    "from the context over paraphrase. Handle any material: prose, notes, "
    "tables, transcripts, or code; for code, prefer a working example. When "
    "asked how to do something, lay the answer out as ordered steps; if the "
    "context covers the procedure only partially, give the steps it contains "
    "and name what's missing rather than glossing over the gap. Match the "
    "answer's length to the question: exhaustive requests deserve every "
    "relevant detail the context offers."
)

DEFAULT_GENERAL_SYSTEM_PROMPT = (
    "You are a helpful, direct assistant. Answer the user's question from "
    "general knowledge. Keep responses concise unless asked to elaborate. "
    "For code, prefer working examples over abstract explanations."
)

# CORS allow-origin regex: Obsidian (desktop + iOS) and localhost loopback.
# Mutating endpoints still require auth regardless of origin.
DEFAULT_CORS_ORIGIN_REGEX = (
    r"^(app://obsidian\.md"
    r"|capacitor://localhost"
    r"|https?://localhost(:\d+)?"
    r"|https?://127\.0\.0\.1(:\d+)?"
    r"|https?://\[::1\](:\d+)?)$"
)
