"""Preregistered, reproducible A/B benchmark for lilbee retrieval.

Two or more lilbee arms answer the same public, human-labeled datasets with the
same served model, so a configuration feature is the only variable between them.
Tier 1 scores ranked results against the datasets' own relevance labels with
ir_measures (no model opinion); Tier 2 layers RAGAS answer-quality metrics on
top, corroborated by the blind judge from ``evals.retrieval``. Lives outside
``src/`` on purpose: it never ships in the package.
"""
