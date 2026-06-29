"""SEO invariants for the marketing site.

Encodes the on-page SEO rules as assertions so a stale canonical, a missing
Open Graph tag, a duplicate title, or a page that drops out of the sitemap is
caught in CI instead of silently costing rankings.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

SITE = Path(__file__).resolve().parents[1] / "site"
BASE = "https://lilbee.sh/"

# Each marketing page mapped to the canonical URL it must declare.
MARKETING_PAGES: dict[str, str] = {
    "index.html": BASE,
    "model-manager/index.html": BASE + "model-manager/",
    "local-rag/index.html": BASE + "local-rag/",
    "code-search/index.html": BASE + "code-search/",
    "mcp/index.html": BASE + "mcp/",
    "gpu/index.html": BASE + "gpu/",
}

_PAGE_ITEMS = list(MARKETING_PAGES.items())

# Satellites the home page must link to so they are crawlable and inherit authority.
SATELLITES = ("model-manager/", "local-rag/", "code-search/", "mcp/", "gpu/")


def _html(rel: str) -> str:
    return (SITE / rel).read_text(encoding="utf-8")


def _meta(html: str, key: str, attr: str = "property") -> str | None:
    m = re.search(rf'<meta {attr}="{re.escape(key)}" content="([^"]*)"', html)
    return m.group(1) if m else None


def _title(html: str) -> str | None:
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    return m.group(1).strip() if m else None


def _canonical(html: str) -> str | None:
    m = re.search(r'<link rel="canonical" href="([^"]+)"', html)
    return m.group(1) if m else None


def _jsonld_blocks(html: str) -> list[str]:
    return re.findall(r'<script type="application/ld\+json">(.*?)</script>', html, re.S)


@pytest.mark.parametrize("rel,url", _PAGE_ITEMS)
def test_title_present_and_bounded(rel: str, url: str) -> None:
    title = _title(_html(rel))
    assert title, f"{rel} missing <title>"
    assert 20 <= len(title) <= 70, f"{rel} title length {len(title)}: {title!r}"


@pytest.mark.parametrize("rel,url", _PAGE_ITEMS)
def test_description_present_and_bounded(rel: str, url: str) -> None:
    desc = _meta(_html(rel), "description", attr="name")
    assert desc, f"{rel} missing meta description"
    assert 50 <= len(desc) <= 320, f"{rel} description length {len(desc)}"


@pytest.mark.parametrize("rel,url", _PAGE_ITEMS)
def test_canonical_self_referential(rel: str, url: str) -> None:
    assert _canonical(_html(rel)) == url, f"{rel} canonical is not self-referential"


@pytest.mark.parametrize("rel,url", _PAGE_ITEMS)
def test_open_graph_and_twitter_complete(rel: str, url: str) -> None:
    html = _html(rel)
    for prop in ("og:type", "og:title", "og:description", "og:url", "og:image"):
        assert _meta(html, prop), f"{rel} missing {prop}"
    assert _meta(html, "og:url") == url, f"{rel} og:url does not match canonical"
    assert _meta(html, "twitter:card", attr="name"), f"{rel} missing twitter:card"


@pytest.mark.parametrize("rel,url", _PAGE_ITEMS)
def test_exactly_one_h1(rel: str, url: str) -> None:
    count = len(re.findall(r"<h1[\s>]", _html(rel)))
    assert count == 1, f"{rel} has {count} <h1> elements, expected 1"


@pytest.mark.parametrize("rel,url", _PAGE_ITEMS)
def test_images_have_alt(rel: str, url: str) -> None:
    for tag in re.findall(r"<img\b[^>]*>", _html(rel)):
        m = re.search(r'alt="([^"]*)"', tag)
        assert m and m.group(1).strip(), f"{rel} has an <img> without alt text: {tag}"


@pytest.mark.parametrize("rel,url", _PAGE_ITEMS)
def test_jsonld_parses_with_required_fields(rel: str, url: str) -> None:
    blocks = _jsonld_blocks(_html(rel))
    assert blocks, f"{rel} has no JSON-LD"
    for block in blocks:
        data = json.loads(block)
        assert data.get("@context"), f"{rel} JSON-LD missing @context"
        assert data.get("@type"), f"{rel} JSON-LD missing @type"


def test_titles_unique() -> None:
    titles = [_title(_html(rel)) for rel in MARKETING_PAGES]
    assert len(set(titles)) == len(titles), "duplicate <title> across pages"


def test_descriptions_unique() -> None:
    descs = [_meta(_html(rel), "description", attr="name") for rel in MARKETING_PAGES]
    assert len(set(descs)) == len(descs), "duplicate meta description across pages"


def test_sitemap_bidirectional() -> None:
    sitemap = (SITE / "sitemap.xml").read_text(encoding="utf-8")
    locs = set(re.findall(r"<loc>([^<]+)</loc>", sitemap))
    for url in MARKETING_PAGES.values():
        assert url in locs, f"{url} missing from sitemap.xml"
    for loc in locs:
        rel = loc.removeprefix(BASE)
        path = SITE / (f"{rel}index.html" if rel.endswith("/") or rel == "" else rel)
        assert path.exists(), f"sitemap URL {loc} has no file on disk ({path})"


def test_satellites_linked_from_home() -> None:
    home = _html("index.html")
    for slug in SATELLITES:
        assert f'href="{slug}"' in home, f"home does not link to satellite {slug}"
