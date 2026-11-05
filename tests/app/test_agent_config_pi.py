"""Tests for the pi model definitions."""

from __future__ import annotations

from lilbee.app.agent_configs.pi import pi_models
from lilbee.catalog import agent_model_id, display_label_for_ref

_REF_A = "Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
_REF_B = "Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"


def test_every_field_pi_reads_is_set():
    assert pi_models([_REF_A], 65536) == [
        {
            "id": agent_model_id(_REF_A),
            "name": display_label_for_ref(_REF_A),
            "reasoning": False,
            "input": ["text"],
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 65536,
            "maxTokens": 8192,
        }
    ]


def test_one_model_per_ref_sorted_by_ref():
    models = pi_models([_REF_B, _REF_A], 65536)
    assert [m["id"] for m in models] == [agent_model_id(_REF_A), agent_model_id(_REF_B)]


def test_output_cap_is_half_a_small_window():
    assert pi_models([_REF_A], 4096)[0]["maxTokens"] == 2048


def test_output_cap_stays_positive_on_a_one_token_window():
    assert pi_models([_REF_A], 1)[0]["maxTokens"] == 1


def test_models_do_not_share_a_cost_dict():
    first, second = pi_models([_REF_A, _REF_B], 65536)
    first["cost"]["input"] = 5
    assert second["cost"]["input"] == 0
