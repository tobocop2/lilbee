"""Tests for generating one wiki page on demand from the index."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.core.config import cfg
from lilbee.data.store import SearchChunk
from lilbee.wiki.entity_extractor import EntityKind
from lilbee.wiki.lazy import UnknownStubError, generate_stub_page
from lilbee.wiki.shared import WikiSubdir
from lilbee.wiki.stubs import WikiStub, save_stub_index


@pytest.fixture(autouse=True)
def isolated_env(wiki_isolated_env: Path):
    yield wiki_isolated_env


def make_search_chunk(source: str, chunk_index: int, chunk: str = "text") -> SearchChunk:
    return SearchChunk(
        source=source,
        content_type="text",
        chunk_type="raw",
        page_start=0,
        page_end=0,
        line_start=0,
        line_end=0,
        chunk=chunk,
        chunk_index=chunk_index,
        vector=[0.1] * cfg.embedding_dim,
    )


def _stub(slug: str, refs: tuple[tuple[str, int], ...]) -> WikiStub:
    return WikiStub(
        slug=slug,
        label=slug.replace("-", " ").title(),
        kind=EntityKind.ENTITY,
        type_hint="PERSON",
        source_mentions=tuple(
            sorted({s: sum(1 for x, _ in refs if x == s) for s, _ in refs}.items())
        ),
        chunk_refs=refs,
    )


def _store_with(chunks_by_source: dict[str, list]) -> MagicMock:
    store = MagicMock()
    store.get_chunks_by_source.side_effect = lambda name: chunks_by_source.get(name, [])
    return store


class TestGenerateStubPage:
    def test_unknown_slug_raises(self):
        save_stub_index({})
        with pytest.raises(UnknownStubError, match="no indexed page"):
            generate_stub_page("ford", MagicMock(), cfg)

    def test_gathers_evidence_across_every_source_naming_the_subject(self):
        """A build assigns an entity to the one source mentioning it most and
        never sees the rest. Generating the page directly uses all of them."""
        save_stub_index({"ford": _stub("ford", (("a.md", 0), ("b.md", 0)))})
        store = _store_with(
            {
                "a.md": [make_search_chunk(source="a.md", chunk_index=0, chunk="from a")],
                "b.md": [make_search_chunk(source="b.md", chunk_index=0, chunk="from b")],
            }
        )
        with patch("lilbee.wiki.lazy.generate_page", return_value=Path("p.md")) as gen:
            generate_stub_page("ford", store, cfg)
        kwargs = gen.call_args.kwargs
        assert kwargs["source_names"] == ["a.md", "b.md"]
        assert {c.source for c in kwargs["chunks"]} == {"a.md", "b.md"}

    def test_lands_under_the_stubs_own_page_type(self):
        save_stub_index({"ford": _stub("ford", (("a.md", 0),))})
        store = _store_with({"a.md": [make_search_chunk(source="a.md", chunk_index=0)]})
        with patch("lilbee.wiki.lazy.generate_page", return_value=Path("p.md")) as gen:
            generate_stub_page("ford", store, cfg)
        assert gen.call_args.kwargs["page_type"] == WikiSubdir.ENTITIES

    def test_a_stale_ref_is_skipped_rather_than_failing_the_page(self):
        """A re-ingested source can have fewer chunks than the index recorded."""
        save_stub_index({"ford": _stub("ford", (("a.md", 0), ("a.md", 99)))})
        store = _store_with({"a.md": [make_search_chunk(source="a.md", chunk_index=0)]})
        with patch("lilbee.wiki.lazy.generate_page", return_value=Path("p.md")) as gen:
            generate_stub_page("ford", store, cfg)
        assert len(gen.call_args.kwargs["chunks"]) == 1

    def test_an_entirely_stale_index_entry_generates_nothing(self):
        save_stub_index({"ford": _stub("ford", (("gone.md", 0),))})
        with patch("lilbee.wiki.lazy.generate_page") as gen:
            assert generate_stub_page("ford", _store_with({}), cfg) is None
        gen.assert_not_called()

    def test_the_prompt_names_the_subject_and_its_sources(self):
        save_stub_index({"ford": _stub("ford", (("a.md", 0),))})
        store = _store_with({"a.md": [make_search_chunk(source="a.md", chunk_index=0)]})
        with patch("lilbee.wiki.lazy.generate_page", return_value=Path("p.md")) as gen:
            generate_stub_page("ford", store, cfg)
        prompt = gen.call_args.kwargs["prompt"]
        assert "Ford" in prompt
        assert "a.md" in prompt

    def test_citations_resolve_against_whichever_source_holds_the_excerpt(self):
        """The page cites two documents, so its resolver has to attribute each
        excerpt to the source that actually contains it. A build could not do
        this: it only ever hands one source's chunks to a page."""
        from lilbee.wiki.citations import ParsedCitation

        save_stub_index({"ford": _stub("ford", (("a.md", 0), ("b.md", 0)))})
        store = _store_with(
            {
                "a.md": [make_search_chunk("a.md", 0, "the model T sold well")],
                "b.md": [make_search_chunk("b.md", 0, "the plant opened in 1913")],
            }
        )
        with patch("lilbee.wiki.lazy.generate_page", return_value=Path("p.md")) as gen:
            generate_stub_page("ford", store, cfg)
        resolver = gen.call_args.kwargs["citation_resolver"]

        records = resolver(
            [
                ParsedCitation(
                    citation_key="src1",
                    source_ref='a.md, excerpt: "the model T sold well"',
                    line_number=1,
                ),
                ParsedCitation(
                    citation_key="src2",
                    source_ref='b.md, excerpt: "the plant opened in 1913"',
                    line_number=2,
                ),
            ]
        )

        assert {r["source_filename"] for r in records} == {"a.md", "b.md"}

    def test_a_lazily_written_page_never_prunes_the_documents_that_mention_it(self):
        """With wiki_prune_raw on, a build deletes the one document its page
        replaces. These documents merely name the subject, so pruning them
        would delete every document that mentioned it."""
        save_stub_index({"ford": _stub("ford", (("a.md", 0), ("b.md", 0)))})
        store = _store_with(
            {
                "a.md": [make_search_chunk("a.md", 0)],
                "b.md": [make_search_chunk("b.md", 0)],
            }
        )
        with patch("lilbee.wiki.lazy.generate_page", return_value=Path("p.md")) as gen:
            generate_stub_page("ford", store, cfg)
        assert gen.call_args.kwargs["supersedes_sources"] is False

    @pytest.mark.parametrize("form", ["ford", "entities/ford"], ids=["bare", "qualified"])
    def test_either_slug_form_resolves(self, form: str):
        """Every surface shows pages as entities/ford, so that is what a user
        types; the index is keyed by the bare slug."""
        save_stub_index({"ford": _stub("ford", (("a.md", 0),))})
        store = _store_with({"a.md": [make_search_chunk("a.md", 0)]})
        with patch("lilbee.wiki.lazy.generate_page", return_value=Path("p.md")):
            assert generate_stub_page(form, store, cfg) == Path("p.md")

    def test_evidence_is_ordered_by_how_much_each_source_says(self):
        """The context budget truncates from the end, so the documents with
        least to say must be the ones that fall off."""
        stub = WikiStub(
            slug="ford",
            label="Ford",
            kind=EntityKind.ENTITY,
            type_hint="PERSON",
            # "a" sorts before "z", so the loud source must be the LATER name
            # for this to distinguish mention ordering from alphabetical.
            source_mentions=(("a-quiet.md", 1), ("z-loud.md", 9)),
            chunk_refs=(("a-quiet.md", 0), ("z-loud.md", 0)),
        )
        save_stub_index({"ford": stub})
        store = _store_with(
            {
                "a-quiet.md": [make_search_chunk("a-quiet.md", 0)],
                "z-loud.md": [make_search_chunk("z-loud.md", 0)],
            }
        )
        with patch("lilbee.wiki.lazy.generate_page", return_value=Path("p.md")) as gen:
            generate_stub_page("ford", store, cfg)
        assert gen.call_args.kwargs["chunks"][0].source == "z-loud.md"

    def test_a_subject_whose_refs_were_subtracted_still_generates(self):
        """The cap can hand every ref to a source that is later re-indexed,
        leaving an entry with sources but no refs. The sources are the truth,
        so generation reads them rather than reporting nothing to write from."""
        stub = WikiStub(
            slug="ford",
            label="Ford",
            kind=EntityKind.ENTITY,
            type_hint="PERSON",
            source_mentions=(("b.md", 5),),
            chunk_refs=(),
        )
        save_stub_index({"ford": stub})
        store = _store_with({"b.md": [make_search_chunk("b.md", 0), make_search_chunk("b.md", 1)]})
        with patch("lilbee.wiki.lazy.generate_page", return_value=Path("p.md")) as gen:
            assert generate_stub_page("ford", store, cfg) == Path("p.md")
        assert len(gen.call_args.kwargs["chunks"]) == 2

    def test_defaults_to_the_global_config(self):
        save_stub_index({"ford": _stub("ford", (("a.md", 0),))})
        store = _store_with({"a.md": [make_search_chunk(source="a.md", chunk_index=0)]})
        with patch("lilbee.wiki.lazy.generate_page", return_value=Path("p.md")):
            assert generate_stub_page("ford", store) == Path("p.md")
