"""Conversion of crawled HTML to markdown, and the seam that keeps it off the loop.

crawl4ai is an optional extra and is absent from the unit-test environment, so the
crawl4ai-facing paths run against a stubbed module here. The proof that a real
crawl still produces the same markdown lives in
``tests/integration/test_crawl_integration.py``, which runs with the extra installed.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from lilbee.crawler.markdown import base_url_for, html_to_markdown
from tests._sys_modules import inject_modules


class TestWhichUrlLinksResolveAgainst:
    @pytest.mark.parametrize(
        ("html", "url", "redirected", "expected"),
        [
            (
                "<html><head><base href='https://base.example/x/'></head></html>",
                "https://orig.example/",
                None,
                "https://base.example/x/",
            ),
            ("<html></html>", "https://orig.example/", None, "https://orig.example/"),
            (
                "<html></html>",
                "https://orig.example/",
                "https://redirected.example/",
                "https://redirected.example/",
            ),
        ],
        ids=["base_tag_wins", "falls_back_to_url", "prefers_redirect_over_url"],
    )
    def test_the_base_url_is_resolved(self, html, url, redirected, expected) -> None:
        assert base_url_for(html, url, redirected) == expected

    def test_a_base_tag_is_matched_regardless_of_case(self) -> None:
        html = '<HTML><BASE HREF="https://base.example/"></HTML>'

        assert base_url_for(html, "https://orig.example/", None) == "https://base.example/"


def _stub_crawl4ai(raw: str = "# converted") -> dict[str, object]:
    """A crawl4ai stub exposing only what the conversion seam touches."""

    class _Result:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class _DefaultMarkdownGenerator:
        def generate_markdown(self, html, base_url="", **_kwargs):
            return _Result(raw_markdown=raw, markdown_with_citations=raw)

    class _MarkdownGenerationStrategy:
        def __init__(self, *_a, **_k) -> None:
            pass

    strategy_mod = types.ModuleType("crawl4ai.markdown_generation_strategy")
    strategy_mod.DefaultMarkdownGenerator = _DefaultMarkdownGenerator  # type: ignore[attr-defined]
    strategy_mod.MarkdownGenerationStrategy = _MarkdownGenerationStrategy  # type: ignore[attr-defined]
    models_mod = types.ModuleType("crawl4ai.models")
    models_mod.MarkdownGenerationResult = _Result  # type: ignore[attr-defined]
    root = types.ModuleType("crawl4ai")
    return {
        "crawl4ai": root,
        "crawl4ai.markdown_generation_strategy": strategy_mod,
        "crawl4ai.models": models_mod,
    }


class TestTheConversionItself:
    def test_the_backend_result_is_returned_as_text(self) -> None:
        with inject_modules(_stub_crawl4ai("# hello")):
            assert html_to_markdown("<h1>hello</h1>", "https://example.com/") == "# hello"

    def test_a_backend_returning_nothing_yields_empty_text(self) -> None:
        with inject_modules(_stub_crawl4ai(raw="")):
            assert html_to_markdown("<h1>x</h1>", "https://example.com/") == ""


class TestWhereTheConversionRuns:
    """crawl4ai calls its generator synchronously from inside its own async crawl,
    so converting there blocks the event loop for the whole conversion. lilbee
    silences that generator and converts where it can await instead."""

    def test_no_workers_keeps_the_conversion_on_the_loop(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.crawler import crawl4ai_fetcher

        monkeypatch.setattr(cfg, "crawl_convert_workers", 0)

        assert crawl4ai_fetcher._conversion_limiter() is None

    def test_workers_bound_how_many_pages_convert_at_once(self, monkeypatch) -> None:
        from lilbee.core.config import cfg
        from lilbee.crawler import crawl4ai_fetcher

        monkeypatch.setattr(cfg, "crawl_convert_workers", 3)
        limiter = crawl4ai_fetcher._conversion_limiter()

        assert limiter is not None
        assert limiter.total_tokens == 3

    async def test_without_a_limiter_it_converts_on_the_loop(self) -> None:
        from lilbee.crawler import crawl4ai_fetcher

        page = mock.MagicMock(
            cleaned_html="<h1>x</h1>", html="<h1>x</h1>", url="https://e/", redirected_url=None
        )
        with inject_modules(_stub_crawl4ai("# inline")):
            assert (
                await crawl4ai_fetcher._markdown_for(page, silenced=True, limiter=None)
                == "# inline"
            )

    async def test_with_a_limiter_it_converts_off_the_event_loop(self) -> None:
        """The whole point: the conversion is awaited, not run on the loop."""
        from anyio import CapacityLimiter

        from lilbee.crawler import crawl4ai_fetcher

        page = mock.MagicMock(
            cleaned_html="<h1>x</h1>", html="<h1>x</h1>", url="https://e/", redirected_url=None
        )
        limiter = CapacityLimiter(1)
        with mock.patch(
            "anyio.to_thread.run_sync", new=mock.AsyncMock(return_value="# offloaded")
        ) as run_sync:
            result = await crawl4ai_fetcher._markdown_for(page, silenced=True, limiter=limiter)

        assert result == "# offloaded"
        assert run_sync.await_args.kwargs["limiter"] is limiter

    async def test_an_unsilenced_backend_keeps_its_own_markdown(self) -> None:
        """Converting again would duplicate the work this exists to move."""
        from lilbee.crawler import crawl4ai_fetcher

        page = mock.MagicMock(
            cleaned_html="<h1>x</h1>", html="<h1>x</h1>", url="https://e/", markdown="# backend"
        )

        got = await crawl4ai_fetcher._markdown_for(page, silenced=False, limiter=None)

        assert got == "# backend"

    async def test_a_page_with_no_html_keeps_whatever_the_backend_gave(self) -> None:
        from lilbee.crawler import crawl4ai_fetcher

        page = mock.MagicMock(cleaned_html="", html="", markdown="# from backend")

        assert (
            await crawl4ai_fetcher._markdown_for(page, silenced=True, limiter=None)
            == "# from backend"
        )


class TestTheSilencedBackendGenerator:
    def test_it_produces_nothing_so_the_backend_does_no_work(self) -> None:
        from lilbee.crawler import crawl4ai_fetcher

        with inject_modules(_stub_crawl4ai()):
            generator = crawl4ai_fetcher._silent_markdown_generator()
            assert generator is not None
            result = generator.generate_markdown(input_html="<h1>x</h1>", base_url="")

        assert result.raw_markdown == ""

    def test_an_absent_backend_leaves_the_crawl_to_convert_itself(self) -> None:
        """Passing no generator is valid; the crawl then converts in-process."""
        from lilbee.crawler import crawl4ai_fetcher

        with inject_modules({"crawl4ai.markdown_generation_strategy": None}):
            assert crawl4ai_fetcher._silent_markdown_generator() is None

    def test_a_stubbed_backend_is_treated_as_absent(self, monkeypatch) -> None:
        """A stubbed crawl4ai raises AttributeError rather than ImportError."""
        from lilbee.crawler import crawl4ai_fetcher

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

        def _raise(name, *args, **kwargs):
            if name == "crawl4ai.markdown_generation_strategy":
                raise AttributeError("'crawl4ai' is not a package")
            return real_import(name, *args, **kwargs)

        monkeypatch.setitem(sys.modules, "crawl4ai", types.ModuleType("crawl4ai"))
        monkeypatch.setattr("builtins.__import__", _raise)

        assert crawl4ai_fetcher._silent_markdown_generator() is None
