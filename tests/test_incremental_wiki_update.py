"""Tests for the incremental wiki update hook in sync().

Focus on the three branches: no-touch (noop), normal regeneration
under the cap, and cap-exceeded short circuit with manual-update hint.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lilbee.core.config import cfg
from lilbee.wiki.entity_extractor import (
    ChunkRef,
    EntityKind,
    ExtractedEntity,
)
from lilbee.wiki.ingest import incremental_update


@pytest.fixture(autouse=True)
def _isolated_wiki(wiki_isolated_env: Path) -> Path:
    cfg.wiki = True
    cfg.wiki_ingest_update_cap = 3
    return wiki_isolated_env


def _entity(slug: str, kind: EntityKind, sources: list[str]) -> ExtractedEntity:
    return ExtractedEntity(
        slug=slug,
        kind=kind,
        label=slug.replace("-", " "),
        type_hint="PERSON" if kind is EntityKind.ENTITY else "noun_phrase",
        chunk_refs=tuple(ChunkRef(source=s, chunk_index=0) for s in sources),
    )


def _install_service_stubs(
    monkeypatch: pytest.MonkeyPatch, entities: list[ExtractedEntity]
) -> tuple[MagicMock, MagicMock]:
    extractor = MagicMock()
    extractor.extract.return_value = entities
    monkeypatch.setattr(
        "lilbee.wiki.entity_extractor.get_entity_extractor",
        lambda *a, **kw: extractor,
    )
    services = MagicMock()
    services.store.get_sources.return_value = []
    services.store.get_chunks_by_source.return_value = []
    monkeypatch.setattr("lilbee.wiki.ingest.get_services", lambda: services)
    return extractor, services


class TestIncrementalWikiUpdate:
    @pytest.mark.asyncio
    async def test_noop_when_wiki_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg.wiki = False
        with patch("lilbee.wiki.build_wiki") as build:
            await incremental_update({"a.txt"})
        build.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_when_changed_sources_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with patch("lilbee.wiki.build_wiki") as build:
            await incremental_update(set())
        build.assert_not_called()

    @pytest.mark.asyncio
    async def test_regenerates_only_touched_pages(
        self, monkeypatch: pytest.MonkeyPatch, _isolated_wiki: Path
    ) -> None:
        touched = _entity("braking", EntityKind.CONCEPT, ["changed.txt"])
        untouched = _entity("unrelated", EntityKind.CONCEPT, ["other.txt"])
        # "unrelated" has an existing page, is not in changed_sources -> skipped.
        wiki_root = _isolated_wiki / "wiki"
        (wiki_root / "concepts").mkdir(parents=True)
        (wiki_root / "concepts" / "unrelated.md").write_text("existing\n")
        _install_service_stubs(monkeypatch, [touched, untouched])
        with patch("lilbee.wiki.build_wiki", return_value=[]) as build:
            await incremental_update({"changed.txt"})
        build.assert_called_once()
        args = build.call_args.args
        assert args[0] == [touched]

    @pytest.mark.asyncio
    async def test_new_entity_without_existing_page_is_touched(
        self, monkeypatch: pytest.MonkeyPatch, _isolated_wiki: Path
    ) -> None:
        brand_new = _entity("fresh", EntityKind.CONCEPT, ["untouched.txt"])
        _install_service_stubs(monkeypatch, [brand_new])
        with patch("lilbee.wiki.build_wiki", return_value=[]) as build:
            await incremental_update({"untouched.txt"})
        build.assert_called_once()
        assert build.call_args.args[0] == [brand_new]

    @pytest.mark.asyncio
    async def test_existing_page_with_changed_source_is_touched(
        self, monkeypatch: pytest.MonkeyPatch, _isolated_wiki: Path
    ) -> None:
        """Existing-page + cited-source-changed is the separate branch from new-entity."""
        existing = _entity("braking", EntityKind.CONCEPT, ["changed.txt"])
        wiki_root = _isolated_wiki / "wiki"
        (wiki_root / "concepts").mkdir(parents=True)
        (wiki_root / "concepts" / "braking.md").write_text("stale\n")
        _install_service_stubs(monkeypatch, [existing])
        with patch("lilbee.wiki.build_wiki", return_value=[]) as build:
            await incremental_update({"changed.txt"})
        build.assert_called_once()
        assert build.call_args.args[0] == [existing]

    @pytest.mark.asyncio
    async def test_cap_exceeded_skips_regeneration_and_logs_hint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _isolated_wiki: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        cfg.wiki_ingest_update_cap = 1
        touched_a = _entity("a", EntityKind.CONCEPT, ["changed.txt"])
        touched_b = _entity("b", EntityKind.CONCEPT, ["changed.txt"])
        touched_c = _entity("c", EntityKind.CONCEPT, ["changed.txt"])
        _install_service_stubs(monkeypatch, [touched_a, touched_b, touched_c])
        with patch("lilbee.wiki.build_wiki") as build, caplog.at_level("WARNING"):
            await incremental_update({"changed.txt"})
        build.assert_not_called()
        log_path = _isolated_wiki / "wiki" / "log.md"
        assert log_path.exists()
        content = log_path.read_text()
        assert "skipped" in content
        assert "exceeds cap" in content
        # The manual-update hint must land at WARNING level so the default
        # LILBEE_LOG_LEVEL surfaces it during `lilbee sync`.
        hint_records = [r for r in caplog.records if "lilbee wiki update" in r.getMessage()]
        assert hint_records, "expected manual-update hint in logs"
        assert all(r.levelname == "WARNING" for r in hint_records)

    @pytest.mark.asyncio
    async def test_writes_ingest_log_entry_on_success(
        self, monkeypatch: pytest.MonkeyPatch, _isolated_wiki: Path
    ) -> None:
        cfg.wiki_ingest_update_cap = 10
        touched = _entity("braking", EntityKind.ENTITY, ["changed.txt"])
        _install_service_stubs(monkeypatch, [touched])
        page = _isolated_wiki / "wiki" / "concepts" / "braking.md"
        with patch("lilbee.wiki.build_wiki", return_value=[page]):
            await incremental_update({"changed.txt"})
        log_path = _isolated_wiki / "wiki" / "log.md"
        assert log_path.exists()
        assert "ingest" in log_path.read_text()
        assert "changed.txt" in log_path.read_text()

    @pytest.mark.asyncio
    async def test_returns_when_no_entities_touched(
        self, monkeypatch: pytest.MonkeyPatch, _isolated_wiki: Path
    ) -> None:
        """Sources exist but every extracted entity has an existing page and is
        unrelated to changed_sources -> touched stays empty -> early return.

        Exercises both the source-iteration branch (chunks.extend on a non-empty
        get_sources()) and the empty-touched-list early return.
        """
        unrelated = _entity("unrelated", EntityKind.ENTITY, ["other.txt"])
        wiki_root = _isolated_wiki / "wiki"
        (wiki_root / "entities").mkdir(parents=True)
        (wiki_root / "entities" / "unrelated.md").write_text("existing\n")

        _extractor, services = _install_service_stubs(monkeypatch, [unrelated])
        services.store.get_sources.return_value = [{"filename": "other.txt"}]
        services.store.get_chunks_by_source.return_value = []

        with patch("lilbee.wiki.build_wiki") as build:
            await incremental_update({"changed.txt"})
        build.assert_not_called()
        services.store.get_chunks_by_source.assert_called_once_with("other.txt")

    @pytest.mark.asyncio
    async def test_build_wiki_extract_concepts_false_on_incremental(
        self, monkeypatch: pytest.MonkeyPatch, _isolated_wiki: Path
    ) -> None:
        """The incremental hook passes ``extract_concepts=False`` so a
        sync does not churn LLM-curated concept slugs per source touch.
        """
        cfg.wiki_ingest_update_cap = 10
        touched = _entity("braking", EntityKind.ENTITY, ["changed.txt"])
        _install_service_stubs(monkeypatch, [touched])
        with patch("lilbee.wiki.build_wiki", return_value=[]) as build:
            await incremental_update({"changed.txt"})
        build.assert_called_once()
        assert build.call_args.kwargs.get("extract_concepts") is False
