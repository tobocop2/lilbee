"""End-to-end crawls through ``lilbee add --crawl`` against the local fixture site."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from lilbee.core.config.enums import CrawlRenderMode
from tests.integration import _crawl_site as site_mod
from tests.integration._crawl_site import (
    LOOPBACK,
    NAMED_HOST,
    SUB_HOST,
    CrawlRun,
    CrawlSite,
    all_saved_text,
    cli_json,
    crawl_sandbox,
    full_crawl_only,
    needs_crawler,
    needs_named_hosts,
    require_chromium,
    run_add,
    run_cli,
    saved_pages,
    windows_proactor_loop,
)

pytestmark = [pytest.mark.slow, needs_crawler]

MAIN_CRAWL_DEPTH = 2
TREE_CRAWL_DEPTH = 2
SUBDOMAIN_CRAWL_DEPTH = 1
MAX_PAGES_CAP = 5
MIN_PAGES_BEYOND_SEED = 2

# Pages a depth-2 crawl of the home page saves, by name under the host's ``_web`` dir.
HOME_DEPTH_2_PAGES = {
    "index.md",
    "a/index.md",
    "a/b/index.md",
    "rel/child.md",
    "rel/sibling.md",
    "docs/base/page.md",
    "other/leaf.md",
    "redirect-me/index.md",
    "landing/next.md",
}
BEYOND_DEPTH_2_PREFIX = "a/b/c/"
TREE_DEPTH_2_PAGES = {"a/index.md", "a/b/index.md", "a/b/c/index.md"}

KEY_TEXT = [
    ("index.md", site_mod.HOME_TEXT),
    ("index.md", site_mod.NAV_TEXT),
    ("a/index.md", site_mod.TREE_TEXT.format(path="/a/")),
    ("a/b/index.md", site_mod.TREE_TEXT.format(path="/a/b/")),
    ("rel/child.md", site_mod.CHILD_TEXT),
    ("rel/sibling.md", site_mod.SIBLING_TEXT),
    ("docs/base/page.md", site_mod.BASE_TEXT),
    ("other/leaf.md", site_mod.LEAF_TEXT),
    ("redirect-me/index.md", site_mod.TARGET_TEXT),
    ("landing/next.md", site_mod.NEXT_TEXT),
]

NOT_SAVED_TEXT = [
    site_mod.LEAF_WITHOUT_BASE_TEXT,
    site_mod.EXCLUDED_TEXT,
    site_mod.PDF_TEXT,
    site_mod.SUB_TEXT,
    "404 Not Found",
]


@dataclass(frozen=True)
class MainCrawl:
    """The depth-2 home crawl, its recrawl, and a search over the result."""

    root: Path
    first: CrawlRun
    mtimes: dict[str, float]
    recrawl: CrawlRun
    recrawl_mtimes: dict[str, float]
    search_sources: list[str]


def _mtimes(root: Path) -> dict[str, float]:
    web = root / "documents" / "_web"
    return {p.relative_to(web).as_posix(): p.stat().st_mtime for p in web.rglob("*.md")}


def _search_sources(query: str) -> list[str]:
    results = cli_json(run_cli("search", query))["results"]
    assert isinstance(results, list)
    return [str(hit["source"]) for hit in results]


def _main_crawl(tmp: Path, site: CrawlSite, render_mode: CrawlRenderMode) -> MainCrawl:
    """Crawl the home page twice and search it, then close the sandbox."""
    with crawl_sandbox(tmp, render_mode) as root:
        flags = ("--crawl", "--depth", str(MAIN_CRAWL_DEPTH))
        first = run_add(root, site.url("/"), *flags)
        mtimes = _mtimes(root)
        recrawl = run_add(root, site.url("/"), *flags)
        return MainCrawl(
            root, first, mtimes, recrawl, _mtimes(root), _search_sources(site_mod.CHILD_TEXT)
        )


@pytest.fixture(scope="module")
def http_crawl(tmp_path_factory: pytest.TempPathFactory, crawl_site: CrawlSite) -> MainCrawl:
    return _main_crawl(tmp_path_factory.mktemp("http-crawl"), crawl_site, CrawlRenderMode.HTTP)


@pytest.fixture(scope="module")
def browser_crawl(tmp_path_factory: pytest.TempPathFactory, crawl_site: CrawlSite) -> MainCrawl:
    require_chromium()
    tmp = tmp_path_factory.mktemp("browser-crawl")
    with windows_proactor_loop():
        return _main_crawl(tmp, crawl_site, CrawlRenderMode.BROWSER)


class TestRecursiveCrawl:
    def test_saves_the_pages_within_the_depth_cap(self, http_crawl: MainCrawl) -> None:
        assert set(http_crawl.first.pages) == HOME_DEPTH_2_PAGES
        assert http_crawl.first.report["crawled"] == len(HOME_DEPTH_2_PAGES)

    @pytest.mark.parametrize(("name", "text"), KEY_TEXT)
    def test_saved_page_carries_its_text(self, http_crawl: MainCrawl, name: str, text: str) -> None:
        assert text in http_crawl.first.pages[name]

    @pytest.mark.parametrize("text", NOT_SAVED_TEXT)
    def test_unwanted_pages_are_not_saved(self, http_crawl: MainCrawl, text: str) -> None:
        assert text not in all_saved_text(http_crawl.root)

    def test_table_cells_stay_on_one_row(self, http_crawl: MainCrawl) -> None:
        row = rf"{site_mod.TABLE_CELL_NAME}\s*\|\s*{site_mod.TABLE_CELL_VALUE}"
        assert re.search(row, http_crawl.first.pages["index.md"])

    def test_code_block_is_fenced(self, http_crawl: MainCrawl) -> None:
        fenced = re.findall(r"```[^\n]*\n(.*?)```", http_crawl.first.pages["index.md"], re.S)
        assert any(site_mod.CODE_LINE in block for block in fenced)

    def test_base_href_page_links_resolve_against_the_base(
        self, http_crawl: MainCrawl, crawl_site: CrawlSite
    ) -> None:
        page = http_crawl.first.pages["docs/base/page.md"]
        assert crawl_site.url("/other/leaf.html") in page
        assert crawl_site.url("/docs/base/leaf.html") not in page

    def test_redirected_page_links_resolve_against_the_final_url(
        self, http_crawl: MainCrawl, crawl_site: CrawlSite
    ) -> None:
        page = http_crawl.first.pages["redirect-me/index.md"]
        assert crawl_site.url("/landing/next.html") in page
        assert crawl_site.url("/next.html") not in page

    def test_relative_links_resolve_against_the_page(
        self, http_crawl: MainCrawl, crawl_site: CrawlSite
    ) -> None:
        assert crawl_site.url("/rel/sibling.html") in http_crawl.first.pages["rel/child.md"]

    def test_recrawl_writes_nothing_new(self, http_crawl: MainCrawl) -> None:
        assert http_crawl.recrawl.report["crawled"] == 0
        assert http_crawl.recrawl_mtimes == http_crawl.mtimes

    def test_search_returns_the_crawled_page(self, http_crawl: MainCrawl) -> None:
        top_hit, *_rest = http_crawl.search_sources
        assert top_hit.endswith(f"_web/{LOOPBACK}/rel/child.md"), http_crawl.search_sources


class TestCrawlLimits:
    def test_depth_cap_stops_link_following(self, tmp_path: Path, crawl_site: CrawlSite) -> None:
        with crawl_sandbox(tmp_path) as root:
            run = run_add(root, crawl_site.url("/a/"), "--crawl", "--depth", str(TREE_CRAWL_DEPTH))
        assert set(run.pages) == TREE_DEPTH_2_PAGES

    def test_max_pages_caps_the_crawl_well_below_the_site(
        self, tmp_path: Path, crawl_site: CrawlSite
    ) -> None:
        with crawl_sandbox(tmp_path) as root:
            url = crawl_site.url("/wide/")
            run = run_add(root, url, "--crawl", "--max-pages", str(MAX_PAGES_CAP))
        assert MIN_PAGES_BEYOND_SEED <= len(run.pages) <= MAX_PAGES_CAP, sorted(run.pages)
        assert len(run.pages) < site_mod.WIDE_PAGE_COUNT

    def test_single_url_add_saves_only_that_page(
        self, tmp_path: Path, crawl_site: CrawlSite
    ) -> None:
        with crawl_sandbox(tmp_path) as root:
            run = run_add(root, crawl_site.url("/"))
        assert set(run.pages) == {"index.md"}


class TestRedirectedSeed:
    def test_seed_redirect_saves_the_target_under_the_requested_url(
        self, tmp_path: Path, crawl_site: CrawlSite
    ) -> None:
        with crawl_sandbox(tmp_path) as root:
            run = run_add(root, crawl_site.url("/seed-redirect"))
        assert set(run.pages) == {"seed-redirect/index.md"}
        page = run.pages["seed-redirect/index.md"]
        assert site_mod.TARGET_TEXT in page
        assert crawl_site.url("/landing/next.html") in page


@needs_named_hosts
class TestSubdomains:
    @pytest.mark.parametrize(
        ("flags", "expect_sub"),
        [((), False), (("--include-subdomains",), True)],
        ids=["exact-host", "include-subdomains"],
    )
    def test_subdomain_is_followed_only_when_asked(
        self, tmp_path: Path, crawl_site: CrawlSite, flags: tuple[str, ...], expect_sub: bool
    ) -> None:
        with crawl_sandbox(tmp_path) as root:
            seed = crawl_site.url("/", host=NAMED_HOST)
            depth = ("--crawl", "--depth", str(SUBDOMAIN_CRAWL_DEPTH))
            run = run_add(root, seed, *depth, *flags, host=NAMED_HOST)
            sub_pages = saved_pages(root, SUB_HOST)
        assert "index.md" in run.pages
        assert ("index.md" in sub_pages) is expect_sub
        if expect_sub:
            assert site_mod.SUB_TEXT in sub_pages["index.md"]


@full_crawl_only
class TestBrowserModeCrawl:
    def test_saves_the_pages_within_the_depth_cap(self, browser_crawl: MainCrawl) -> None:
        pages = set(browser_crawl.first.pages)
        assert HOME_DEPTH_2_PAGES.issubset(pages)
        assert not [name for name in pages if name.startswith(BEYOND_DEPTH_2_PREFIX)]

    @pytest.mark.parametrize(("name", "text"), KEY_TEXT)
    def test_saved_page_carries_its_text(
        self, browser_crawl: MainCrawl, name: str, text: str
    ) -> None:
        assert text in browser_crawl.first.pages[name]

    def test_recrawl_writes_nothing_new(self, browser_crawl: MainCrawl) -> None:
        assert browser_crawl.recrawl.report["crawled"] == 0
        assert browser_crawl.recrawl_mtimes == browser_crawl.mtimes

    def test_page_scripts_run_before_the_page_is_saved(
        self, tmp_path: Path, crawl_site: CrawlSite
    ) -> None:
        require_chromium()
        with windows_proactor_loop(), crawl_sandbox(tmp_path, CrawlRenderMode.BROWSER) as root:
            run = run_add(root, crawl_site.url("/js/"))
        assert site_mod.SCRIPT_TEXT in run.pages["js/index.md"]
