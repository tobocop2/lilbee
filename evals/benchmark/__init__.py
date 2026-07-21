"""Preregistered, reproducible A/B benchmark: lilbee vs RAGFlow retrieval.

Two arms answer the same public, human-labeled datasets with the same served
model, so retrieval is the only variable. Tier 1 scores ranked results against
the datasets' own relevance labels with ir_measures (no model opinion); Tier 2
layers RAGAS answer-quality metrics on top, corroborated by the blind judge
from ``evals.retrieval``. Lives outside ``src/`` on purpose: it never ships in
the package.
"""
