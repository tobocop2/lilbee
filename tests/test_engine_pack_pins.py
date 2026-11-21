"""The committed pack record agrees with the sha256 and size that the release assets state.

The table below is copied from the release assets API (the digest and size of each asset), not
read from the record, so a hand edit of the record fails here.
"""

from __future__ import annotations

import json
from importlib import resources

import pytest

from lilbee.providers.fleet.engine_record import RECORD_FILE

PUBLISHED = {
    "lilbee_engine_ik_llama_cpp-0+git13cffc006b82-2.cuda-py3-none-manylinux_2_34_x86_64.whl": (
        "4407cb7986ceb42ce2c1c2c743be5f074317af37ab88f0f33ab17ab589142d1d",
        961177649,
    ),
    "lilbee_engine_ik_llama_cpp-0+git13cffc006b82-3.cpu-py3-none-manylinux_2_34_x86_64.whl": (
        "96d7a448b522bde5c970767094cb4d8c0e6add596e9ed5f9bb681dfde97e734c",
        14951773,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-2.cpu-py3-none-manylinux_2_34_x86_64.whl": (
        "cfa58ff9f48b238d187b8bf06c6914cec4ff30378a047cf4aa7707e222194d06",
        45691140,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-2.cuda-py3-none-manylinux_2_34_x86_64.whl": (
        "4ff9e1a52a6d50762458a497307f7b0e775a9346a6b5a5eb8e9ff88c39ff6bcd",
        671178889,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-2.metal-py3-none-macosx_14_0_arm64.whl": (
        "ae859d55e57d58a1f59de4084adbf41a9d5e84a7cb993562abd838ec2d17c5ab",
        42202565,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-2.rocm-py3-none-manylinux_2_34_x86_64.whl": (
        "6616907f7c5c49065767ecd59a1912a8b7df454d6e459593faff08e0f59bd674",
        507199053,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-2.vulkan-py3-none-manylinux_2_34_x86_64.whl": (
        "03b16d8054726674c63a61b924f470c4d9ffe0cf86b4d4c23c34f2cbf7f1e094",
        56088704,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-3.cpu-py3-none-win_amd64.whl": (
        "14a0039d4348f19017aaa23549b7fac2c4b1515071214e9ec98b225e5fdae18d",
        51838997,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-3.cuda-py3-none-win_amd64.whl": (
        "aa34661441f26d8e964d33ddf9c1f68c69c45d314dc8f42fa3f7d3fb69679107",
        709727703,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-3.vulkan-py3-none-win_amd64.whl": (
        "a01a032c23e41f51d696d0ce6bb80bfeb87994cb5d56afc2819f30bca4296ce9",
        64798338,
    ),
    "lilbee_engine_mistral_rs-0.9.5-1.cpu-py3-none-manylinux_2_35_x86_64.whl": (
        "b55ace7732b20c64a0f91e46325d68370388620b02e5deeee4317d7e2e9fef4d",
        36675724,
    ),
    "lilbee_engine_mistral_rs-0.9.5-1.metal-py3-none-macosx_11_0_arm64.whl": (
        "08c30b536f6b6795587902fbe050ae2e53b27cc6873da6ceff55faca21179c2a",
        50923671,
    ),
    "lilbee_engine_tei-1.9.4-1.cpu-py3-none-manylinux_2_34_x86_64.whl": (
        "a92f666c0cdc825f173c3e29486c409a4b5435d037f57d3a60f970844eaf09b8",
        13045675,
    ),
    "lilbee_engine_tei-1.9.4-1.cuda_sm100-py3-none-manylinux_2_34_x86_64.whl": (
        "9fc0a9bff5b4a69b290858f6d761c629e4050727e76de22a29d216d20e920880",
        708463736,
    ),
    "lilbee_engine_tei-1.9.4-1.cuda_sm120-py3-none-manylinux_2_34_x86_64.whl": (
        "0482fe2efecfaed8cea2e90f6dd6f0978aa7b73878d93fd972b4bf86741ac656",
        707695248,
    ),
    "lilbee_engine_tei-1.9.4-1.cuda_sm75-py3-none-manylinux_2_34_x86_64.whl": (
        "1245143a609c9669950602affa9133e1ec5730f334c4f0f585015e84e1f518f4",
        605572892,
    ),
    "lilbee_engine_tei-1.9.4-1.cuda_sm80-py3-none-manylinux_2_34_x86_64.whl": (
        "1fd3fd7a9b50db3f0b5bae956f6fb8c2865dd23c0c454ef7e4d6a54740343cfd",
        694539355,
    ),
    "lilbee_engine_tei-1.9.4-1.cuda_sm86-py3-none-manylinux_2_34_x86_64.whl": (
        "538c78b6280c25df33acd6bf9fb0c02291a1056ff0091aad20c529b2b824fba1",
        694146182,
    ),
    "lilbee_engine_tei-1.9.4-1.cuda_sm89-py3-none-manylinux_2_34_x86_64.whl": (
        "e697869e269c936a99e39244a2098d2c0c5ee7e8ae7798de48d3714ad4a4a19b",
        693616480,
    ),
    "lilbee_engine_tei-1.9.4-1.cuda_sm90-py3-none-manylinux_2_34_x86_64.whl": (
        "61258fdcccd2322665ab3523c309204a986c2f461e843449cf1e53aeac63dca7",
        696284765,
    ),
    "lilbee_engine_tei-1.9.4-1.metal-py3-none-macosx_14_0_arm64.whl": (
        "adef1ba024f7fd18e65f335820cd9d6169eee5a3ca823108650bf67a45afed04",
        9494777,
    ),
}

RECORD = json.loads(
    resources.files("lilbee.providers.fleet").joinpath(RECORD_FILE).read_text(encoding="utf-8")
)


def test_the_record_lists_each_published_wheel_and_no_other() -> None:
    assert PUBLISHED
    assert sorted(row["file"] for row in RECORD) == sorted(PUBLISHED)


@pytest.mark.parametrize("row", RECORD, ids=[row["file"] for row in RECORD])
def test_each_row_has_the_published_sha256_and_size(row: dict[str, object]) -> None:
    assert (row["sha256"], row["size"]) == PUBLISHED[str(row["file"])]
