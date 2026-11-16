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
    "lilbee_engine_ik_llama_cpp-0+git13cffc006b82-1.cpu-py3-none-manylinux_2_34_x86_64.whl": (
        "8bba9afef14daa860692e6011f5cee892c544e9e0f43bb774dbe39ec83e575b5",
        15073566,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-1.cpu-py3-none-manylinux_2_34_x86_64.whl": (
        "7955bb3f571422fbe4789e7f50a4b56b6a05f723b65babad8c3e1d4426b8be80",
        45687918,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-1.cuda-py3-none-manylinux_2_34_x86_64.whl": (
        "12ea17e67f6931d3bb95ee529d729baadcbe0a0726ebd86501a3d1477fcab785",
        671180426,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-1.metal-py3-none-macosx_14_0_arm64.whl": (
        "07d954f73523abfe72c10ed61fa21d3e1877610be0991b7901e400841db7d8ce",
        42199089,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-1.rocm-py3-none-manylinux_2_34_x86_64.whl": (
        "22b8e8e0df62382db4bdb624df6a569ee4ba147c4f2fce80a42b4e2ea24853e3",
        507195826,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-1.vulkan-py3-none-manylinux_2_34_x86_64.whl": (
        "4346a6bbbee15dbfa1e0e736f174fa86d568d23b52b69ab64959a5369e727662",
        56085481,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-2.cpu-py3-none-win_amd64.whl": (
        "da9133af67820251f669e53d44e8adf0f2d59ce9a11107127348cbf5a68e15f9",
        51835763,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-2.cuda-py3-none-win_amd64.whl": (
        "f535f6d7a796c957df2bc9799c9a82c86457202d229985bd633a0cd32cc84d69",
        709724466,
    ),
    "lilbee_engine_llama_cpp-1.12032.1-2.vulkan-py3-none-win_amd64.whl": (
        "87e5a2e2735aedb7446273d0989fa9ec69843828fa57cc8e5ca12cee0e3e992d",
        64795105,
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
