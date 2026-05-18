"""Tests for the entity-extractor package scaffolding.

Covers the protocol, record shapes, factory dispatch, stub NotImplementedError
behaviour, and the config + settings_map plumbing. The per-strategy logic
(NER, concepts, LLM) is exercised in the strategy-specific test modules.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lilbee.app.settings_map import SETTINGS_MAP
from lilbee.core.config import WikiEntityMode, cfg
from lilbee.wiki.entity_extractor import (
    ChunkRef,
    EntityExtractor,
    EntityKind,
    ExtractedEntity,
    get_entity_extractor,
)
from lilbee.wiki.entity_extractor.llm_tagged import LlmTaggedExtractor
from lilbee.wiki.entity_extractor.ner_concepts import NerConceptsExtractor
from lilbee.wiki.entity_extractor.ner_concepts_plus_llm_types import (
    NerConceptsPlusLlmTypesExtractor,
)


class TestExtractedEntityRecord:
    def test_has_expected_fields_and_is_frozen(self) -> None:
        entity = ExtractedEntity(
            slug="tire-pressure",
            kind=EntityKind.CONCEPT,
            label="Tire pressure",
            type_hint="noun_phrase",
            chunk_refs=(ChunkRef(source="manual.pdf", chunk_index=12),),
        )
        assert entity.slug == "tire-pressure"
        assert entity.kind is EntityKind.CONCEPT
        assert entity.label == "Tire pressure"
        assert entity.type_hint == "noun_phrase"
        assert entity.chunk_refs == (ChunkRef(source="manual.pdf", chunk_index=12),)
        # frozen=True: assignment must fail to keep records hashable.
        with pytest.raises(AttributeError):
            entity.slug = "other"  # type: ignore[misc]

    def test_kind_enum_covers_both_variants(self) -> None:
        assert EntityKind.CONCEPT.value == "concept"
        assert EntityKind.ENTITY.value == "entity"


class TestFactoryDispatch:
    """Each mode routes to the matching extractor class."""

    def test_implemented_mode_returns_its_class(self) -> None:
        provider = MagicMock()
        extractor = get_entity_extractor(WikiEntityMode.NER_ENTITIES, provider, cfg)
        assert isinstance(extractor, NerConceptsExtractor)
        # Implementations must satisfy the runtime-checkable protocol.
        assert isinstance(extractor, EntityExtractor)


class TestUnimplementedModesFallBack:
    """Unimplemented strategies fall back to NER_ENTITIES via the factory.

    The stub classes themselves still raise on direct use, so future
    implementation work has a clear TODO site, but routing through
    ``get_entity_extractor`` never hands a caller something that crashes
    mid-build.
    """

    @pytest.mark.parametrize(
        "mode",
        [WikiEntityMode.NER_CONCEPTS_PLUS_LLM_TYPES, WikiEntityMode.LLM_TAGGED],
    )
    def test_unimplemented_mode_returns_ner_entities(
        self, mode: WikiEntityMode, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = MagicMock()
        with caplog.at_level("WARNING"):
            extractor = get_entity_extractor(mode, provider, cfg)
        assert isinstance(extractor, NerConceptsExtractor)
        assert any("not yet implemented" in r.message for r in caplog.records)

    @pytest.mark.parametrize(
        "cls",
        [NerConceptsPlusLlmTypesExtractor, LlmTaggedExtractor],
    )
    def test_stub_extract_still_raises_when_used_directly(self, cls: type) -> None:
        extractor = cls(MagicMock(), cfg)
        with pytest.raises(NotImplementedError):
            extractor.extract([])


class TestConfigPlumbing:
    def test_default_mode_is_ner_entities(self) -> None:
        # Late-bound: read from cfg so a test that mutates the field restores
        # visibility via the fixture snapshot.
        assert cfg.wiki_entity_mode is WikiEntityMode.NER_ENTITIES

    def test_settings_map_entry_lists_all_modes(self) -> None:
        entry = SETTINGS_MAP["wiki_entity_mode"]
        assert entry.type is str
        assert entry.group == "Wiki"
        assert entry.choices is not None
        assert set(entry.choices) == {m.value for m in WikiEntityMode}

    def test_wiki_entity_mode_renamed_to_ner_entities(self) -> None:
        """The default mode is NER_ENTITIES; NER_CONCEPTS is not accepted.

        No alias shim: an old env var value `ner_concepts` raises a
        ValidationError at assignment. Acceptable because the enum
        value never shipped on main.
        """
        assert "NER_ENTITIES" in WikiEntityMode.__members__
        assert "NER_CONCEPTS" not in WikiEntityMode.__members__
