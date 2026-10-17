"""Integration runs talk to the real Hugging Face hub."""

import os

from huggingface_hub import constants as hf_constants


def test_an_integration_run_has_the_hub_switched_on():
    assert hf_constants.HF_HUB_OFFLINE is False
    assert "HF_HUB_OFFLINE" not in os.environ
