"""Tests for generating one wiki page on demand from the index."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.core.config import cfg
from lilbee.data.store import SearchChunk
from lilbee.runtime.progress import EventType, WikiPageEvent, WikiPhase, WikiPhaseEvent
from lilbee.wiki.entity_extractor import EntityKind
from lilbee.wiki.lazy import UnknownStubError, generate_stub_page, resolve_stub
from lilbee.wiki.shared import WikiSubdir
from lilbee.wiki.stubs import WikiStub, save_stub_index


@pytest.fixture(autouse=True)
def isolated_env(wiki_isolated_env: Path):
    yield wiki_isolated_env


def make_search_chunk(
    source: str, chunk_index: int, chunk: str = "text", page: int = 0
) -> SearchChunk:
    return SearchChunk(
        source=source,
        content_type="text",
        chunk_type="raw",
        page_start=page,
        page_end=page,
        line_start=0,
        line_end=0,
        chunk=chunk,
        chunk_index=chunk_index,
        vector=[0.1] * cfg.embedding_dim,
    )


def _stub(
    slug: str, refs: tuple[tuple[str, int], ...], kind: EntityKind = EntityKind.ENTITY
) -> WikiStub:
    return WikiStub(
        slug=slug,
        label=slug.replace("-", " ").title(),
        kind=kind,
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

    @pytest.mark.parametrize(
        ("kind", "subdir"),
        [
            (EntityKind.ENTITY, WikiSubdir.ENTITIES),
            (EntityKind.CONCEPT, WikiSubdir.CONCEPTS),
        ],
        ids=["entity", "concept"],
    )
    def test_lands_under_the_stubs_own_page_type(self, kind: EntityKind, subdir: WikiSubdir):
        """Both kinds, because the index holds concept stubs too and every
        surface passes whatever slug the user picked. Testing only entities
        would let a hardcoded ENTITIES pass, and a concept page written under
        entities/ is invisible to a browse tree looking under concepts/."""
        save_stub_index({"ford": _stub("ford", (("a.md", 0),), kind=kind)})
        store = _store_with({"a.md": [make_search_chunk(source="a.md", chunk_index=0)]})
        with patch("lilbee.wiki.lazy.generate_page", return_value=Path("p.md")) as gen:
            generate_stub_page("ford", store, cfg)
        assert gen.call_args.kwargs["page_type"] == subdir

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
        this: it only ever hands one source's chunks to a page.

        The second footnote deliberately omits the filename. A model writes
        those, and when it does the only way to attribute the excerpt is to look
        it up in the chunk map. Naming both files instead would let the resolver
        match on the reference text alone and the map would go untested."""
        from lilbee.wiki.citations import ParsedCitation

        save_stub_index({"ford": _stub("ford", (("a.md", 0), ("b.md", 0), ("b.md", 1)))})
        store = _store_with(
            {
                "a.md": [make_search_chunk("a.md", 0, "the model T sold well")],
                # Two chunks, the non-matching one first: with a single chunk
                # "the first chunk holding the excerpt" and "the first chunk"
                # are the same object, and a lookup that skipped the excerpt
                # check entirely would still land on page 7.
                "b.md": [
                    make_search_chunk("b.md", 0, "unrelated prose", page=3),
                    make_search_chunk("b.md", 1, "the plant opened in 1913", page=7),
                ],
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
                    source_ref='excerpt: "the plant opened in 1913"',
                    line_number=2,
                ),
            ]
        )

        assert {r["source_filename"] for r in records} == {"a.md", "b.md"}
        # The page number can only have come from b.md's chunk, so the map is
        # carrying the location too, not just the attribution.
        assert next(r["page_start"] for r in records if r["source_filename"] == "b.md") == 7

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
        types; the index is keyed by the bare slug.

        The written slug matters as much as the lookup. generate_page puts it
        straight into the page path and the store row, so forwarding the
        qualified form would write entities/entities/ford.md, which browse
        cannot see, and a later generate by the bare slug would produce a
        second, divergent page for the same subject."""
        save_stub_index({"ford": _stub("ford", (("a.md", 0),))})
        store = _store_with({"a.md": [make_search_chunk("a.md", 0)]})
        with patch("lilbee.wiki.lazy.generate_page", return_value=Path("p.md")) as gen:
            assert generate_stub_page(form, store, cfg) == Path("p.md")
        assert gen.call_args.kwargs["slug"] == "ford"

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

    def test_progress_reports_the_generate_phase_and_the_written_page(self):
        """An HTTP client watching the stream needs the phase before the model
        call and the page after it; with neither it can only show 0% then 100%."""
        save_stub_index({"ford": _stub("ford", (("a.md", 0),))})
        store = _store_with({"a.md": [make_search_chunk(source="a.md", chunk_index=0)]})
        events: list[tuple[EventType, object]] = []
        with patch("lilbee.wiki.lazy.generate_page", return_value=Path("p.md")):
            generate_stub_page("ford", store, cfg, on_progress=lambda e, d: events.append((e, d)))
        assert events == [
            (EventType.WIKI_PHASE, WikiPhaseEvent(phase=WikiPhase.GENERATE, total=1)),
            (EventType.WIKI_PAGE, WikiPageEvent(label="Ford", pages=1, current=1, total=1)),
        ]

    def test_a_failed_generation_reports_no_page(self):
        save_stub_index({"ford": _stub("ford", (("a.md", 0),))})
        store = _store_with({"a.md": [make_search_chunk(source="a.md", chunk_index=0)]})
        events: list[tuple[EventType, object]] = []
        with patch("lilbee.wiki.lazy.generate_page", return_value=None):
            generate_stub_page("ford", store, cfg, on_progress=lambda e, d: events.append((e, d)))
        assert [e for e, _ in events] == [EventType.WIKI_PHASE]

    def test_cancel_before_the_model_call_generates_nothing(self):
        """A client that disconnected must not spend the LLM call."""
        save_stub_index({"ford": _stub("ford", (("a.md", 0),))})
        store = _store_with({"a.md": [make_search_chunk(source="a.md", chunk_index=0)]})
        cancelled = threading.Event()
        cancelled.set()
        with patch("lilbee.wiki.lazy.generate_page") as gen:
            assert generate_stub_page("ford", store, cfg, cancel=cancelled) is None
        gen.assert_not_called()


class TestResolveStub:
    @pytest.mark.parametrize("form", ["ford", "entities/ford"], ids=["bare", "qualified"])
    def test_either_slug_form_resolves(self, form: str):
        save_stub_index({"ford": _stub("ford", (("a.md", 0),))})
        stub = resolve_stub(form, cfg)
        assert stub is not None
        assert stub.slug == "ford"

    def test_an_unknown_slug_resolves_to_none(self):
        save_stub_index({})
        assert resolve_stub("nope", cfg) is None


class TestGeneratedPagesJoinTheWiki:
    """A page written on request used to carry no ``[[links]]`` at all, while
    every page a build wrote carried several. It landed in the vault as an
    isolated node: in the wiki, but not part of it, and alone in the graph."""

    def _entities_dir(self) -> Path:
        d = cfg.data_root / cfg.wiki_dir / "entities"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_the_new_page_links_out_to_its_neighbours(self):
        entities = self._entities_dir()
        (entities / "henry-ford.md").write_text("---\n---\nAbout the man.\n", encoding="utf-8")
        new_page = entities / "model-t.md"

        def write_it(**kwargs):
            new_page.write_text(
                "---\n---\nThe Model T was built by henry ford.\n", encoding="utf-8"
            )
            return new_page

        save_stub_index({"model-t": _stub("model-t", (("a.md", 0),))})
        store = _store_with({"a.md": [make_search_chunk(source="a.md", chunk_index=0)]})
        with patch("lilbee.wiki.lazy.generate_page", side_effect=write_it):
            generate_stub_page("model-t", store, cfg)

        assert "[[henry-ford]]" in new_page.read_text(encoding="utf-8")

    def test_its_neighbours_link_back_to_the_new_page(self):
        """Links only outward would still leave the page with no backlinks, so
        it would sit on the edge of the graph rather than inside it."""
        entities = self._entities_dir()
        neighbour = entities / "henry-ford.md"
        neighbour.write_text("---\n---\nHe built the model t.\n", encoding="utf-8")
        new_page = entities / "model-t.md"

        def write_it(**kwargs):
            new_page.write_text("---\n---\nA car.\n", encoding="utf-8")
            return new_page

        save_stub_index({"model-t": _stub("model-t", (("a.md", 0),))})
        store = _store_with({"a.md": [make_search_chunk(source="a.md", chunk_index=0)]})
        with patch("lilbee.wiki.lazy.generate_page", side_effect=write_it):
            generate_stub_page("model-t", store, cfg)

        assert "[[model-t]]" in neighbour.read_text(encoding="utf-8")

    def test_a_failed_generation_does_not_rewrite_anything(self):
        entities = self._entities_dir()
        neighbour = entities / "henry-ford.md"
        original = "---\n---\nHe built the model t.\n"
        neighbour.write_text(original, encoding="utf-8")

        save_stub_index({"model-t": _stub("model-t", (("a.md", 0),))})
        store = _store_with({"a.md": [make_search_chunk(source="a.md", chunk_index=0)]})
        with patch("lilbee.wiki.lazy.generate_page", return_value=None):
            generate_stub_page("model-t", store, cfg)

        assert neighbour.read_text(encoding="utf-8") == original
