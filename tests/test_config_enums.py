"""Tests for config enum definitions."""

from __future__ import annotations

from lilbee.core.config.enums import RerankerType


def test_reranker_type_values() -> None:
    assert RerankerType.AUTO == "auto"
    assert RerankerType.CROSS_ENCODER == "cross_encoder"
    assert RerankerType.LLM == "llm"
    assert {t.value for t in RerankerType} == {"auto", "cross_encoder", "llm"}
