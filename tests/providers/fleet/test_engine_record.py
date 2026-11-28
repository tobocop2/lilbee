"""The pack record is read at run time and each row is checked when it loads."""

from __future__ import annotations

import json
import re
from importlib import resources
from typing import Any

import pytest

from lilbee.providers.fleet import engine_manifest, engine_record
from lilbee.providers.fleet.engine_record import PackRecordError, build_manifests, load_manifests

PREFIX = "https://index.test/packs/"
TEI = "lilbee_engine_tei-1.9.4-1.cpu-py3-none-manylinux_2_34_x86_64.whl"
SHA = "a" * 64


def entry(name: str, sha: str = SHA, size: Any = 1, url: str | None = None) -> dict[str, Any]:
    return {"file": name, "url": url or PREFIX + name, "sha256": sha, "size": size}


def _shipped_rows() -> list[dict[str, Any]]:
    text = (
        resources.files("lilbee.providers.fleet")
        .joinpath("engine_packs.json")
        .read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = json.loads(text)
    return rows


def test_the_table_is_built_from_the_shipped_record_once() -> None:
    rows = _shipped_rows()
    assert rows
    assert load_manifests() == engine_manifest.MANIFESTS
    by_url = {
        row["url"]: row
        for versions in engine_manifest.MANIFESTS.values()
        for cells in versions.values()
        for row in cells.values()
    }
    assert set(by_url) == {row["url"] for row in rows}
    for row in rows:
        assert (by_url[row["url"]]["sha256"], by_url[row["url"]]["size"]) == (
            row["sha256"],
            row["size"],
        )
    assert {row["file"].split("-")[0] for row in rows} == {
        "lilbee_engine_llama_cpp",
        "lilbee_engine_ik_llama_cpp",
        "lilbee_engine_mistral_rs",
        "lilbee_engine_mlx",
        "lilbee_engine_tei",
    }


def test_a_wheel_becomes_a_cell_of_its_engine() -> None:
    assert build_manifests([entry(TEI, size=99)]) == {
        "tei": {
            "1.9.4": {
                "linux-x86_64-cpu": {
                    "sha256": SHA,
                    "url": PREFIX + TEI,
                    "size": 99,
                    "min_os": "2.34",
                }
            }
        }
    }


@pytest.mark.parametrize(
    ("tag", "cell", "floor"),
    [
        ("manylinux_2_28_x86_64", "linux-x86_64-cpu", "2.28"),
        ("manylinux_2_39_aarch64", "linux-aarch64-cpu", "2.39"),
        ("macosx_14_0_arm64", "macos-arm64-cpu", "14.0"),
        ("macosx_13_3_arm64", "macos-arm64-cpu", "13.3"),
        ("macosx_15_0_arm64", "macos-arm64-cpu", "15.0"),
        ("win_amd64", "windows-x86_64-cpu", None),
    ],
)
def test_the_platform_tag_gives_the_cell_and_the_floor(
    tag: str, cell: str, floor: str | None
) -> None:
    name = f"lilbee_engine_tei-1.0-1.cpu-py3-none-{tag}.whl"
    got = build_manifests([entry(name)])["tei"]["1.0"]
    assert list(got) == [cell]
    assert got[cell]["min_os"] == floor


@pytest.mark.parametrize(
    ("slug", "engine"),
    [
        ("mistral_rs", "mistral.rs"),
        ("tei", "tei"),
        ("ik_llama_cpp", "ik_llama.cpp"),
        ("llama_cpp", "llama.cpp"),
        ("mlx", "mlx"),
    ],
)
def test_the_project_slug_gives_the_engine_name(slug: str, engine: str) -> None:
    name = f"lilbee_engine_{slug}-1.0-1.cpu-py3-none-win_amd64.whl"
    assert list(build_manifests([entry(name)])) == [engine]


def test_the_build_tag_gives_the_backend_with_dashes() -> None:
    name = "lilbee_engine_llama_cpp-b1-1.cuda128_sm89-py3-none-win_amd64.whl"
    cells = build_manifests([entry(name)])["llama.cpp"]["b1"]
    assert list(cells) == ["windows-x86_64-cuda128-sm89"]


def test_each_version_of_a_cell_keeps_its_own_digest_whatever_the_row_order() -> None:
    newer = TEI.replace("1.9.4", "1.10.0")
    versions = build_manifests([entry(newer, "b" * 64, 2), entry(TEI)])["tei"]
    assert versions["1.9.4"]["linux-x86_64-cpu"]["sha256"] == SHA
    assert versions["1.10.0"]["linux-x86_64-cpu"]["sha256"] == "b" * 64


def test_two_wheels_for_one_version_and_cell_are_refused() -> None:
    rebuilt = TEI.replace("-1.cpu", "-2.cpu")
    with pytest.raises(PackRecordError, match=r"row 2 .*already has a wheel"):
        build_manifests([entry(TEI), entry(rebuilt, "b" * 64)])


def _without(field: str) -> dict[str, Any]:
    return {key: value for key, value in entry(TEI).items() if key != field}


@pytest.mark.parametrize(
    ("row", "match"),
    [
        (entry("lilbee_engine_nope-1-1.cpu-py3-none-win_amd64.whl"), "not an engine project"),
        (entry("lilbee_engine_tei-1-1.cpu-py3-none-plan9_mips.whl"), "platform tag"),
        (entry("lilbee_engine_tei-1-cpu-py3-none-win_amd64.whl"), "build tag"),
        (entry("lilbee_engine_tei-1.tar.gz"), "not the file name of an engine wheel"),
        (entry(TEI, "abc"), "sha256"),
        (entry(TEI, f"sha256:{SHA}"), "sha256"),
        (entry(TEI, SHA.upper()), "sha256"),
        (entry(TEI, size=0), "size 0"),
        (entry(TEI, size=-5), "size -5"),
        (entry(TEI, size=1.5), "size 1.5"),
        (entry(TEI, size=True), "size True"),
        (entry(TEI, size="5"), "size '5'"),
        (entry(TEI, size=None), "size None"),
        (entry(TEI, sha=None), "'sha256' is not a string"),
        (entry(TEI, url="http://index.test/packs/" + TEI), "https"),
        (entry(TEI, url="https://index.test/packs/other.whl"), "ends with the file"),
        (entry(TEI, url=PREFIX + "..%2f..%2fevil/" + TEI), "ends with the file"),
        (entry(TEI, url=PREFIX + "..%5cevil/" + TEI), "ends with the file"),
        (entry(TEI, url=PREFIX + "..\\evil/" + TEI), "ends with the file"),
        (entry(TEI, url=PREFIX + "/" + TEI), "ends with the file"),
        (entry(TEI, url=PREFIX + "x//" + TEI), "ends with the file"),
        (entry(TEI, url=PREFIX + "%252e%252e/" + TEI), "ends with the file"),
        (entry(TEI, url=PREFIX + "t%3F/" + TEI), "ends with the file"),
        (entry(TEI, url=PREFIX + "t%00/" + TEI), "ends with the file"),
        (entry(TEI, url=PREFIX + "a b/" + TEI), "ends with the file"),
        (entry(TEI, url=PREFIX + "..;/..;/" + TEI), "ends with the file"),
        (entry(TEI, url=PREFIX + "t\n/" + TEI), "ends with the file"),
        (entry(TEI, url=PREFIX + "../" + TEI), "ends with the file"),
        (entry(TEI, url="https://index.test/" + TEI), "ends with the file"),
        (entry(TEI.replace("1.9.4", "1\n9")), "not the file name of an engine wheel"),
        (entry(TEI.replace("1.9.4", "1%2f9")), "not the file name of an engine wheel"),
        (_without("size"), "'size' is missing"),
        (_without("file"), "'file' is missing"),
        (_without("url"), "'url' is missing"),
        (_without("sha256"), "'sha256' is missing"),
    ],
)
def test_a_row_the_table_cannot_state_is_refused_with_its_name(
    row: dict[str, Any], match: str
) -> None:
    name = row.get("file", "no file")
    with pytest.raises(PackRecordError, match=rf"^row 1 \({re.escape(name)}\): .*{match}"):
        build_manifests([row])


def test_a_record_with_no_rows_is_refused() -> None:
    with pytest.raises(PackRecordError, match=r"^the pack record has no rows"):
        build_manifests([])


def test_a_wheel_file_name_gives_its_engine_version_cell_build_and_floor() -> None:
    name = "lilbee_engine_llama_cpp-1.12032.1-2.cuda128_sm89-py3-none-manylinux_2_34_x86_64.whl"
    assert engine_record.wheel_cell("x", name) == (
        "llama.cpp",
        "1.12032.1",
        "linux-x86_64-cuda128-sm89",
        2,
        "2.34",
    )


def test_a_row_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(PackRecordError, match="row 2: not an object"):
        build_manifests([entry(TEI), "x"])


def test_a_record_that_is_not_a_list_is_refused() -> None:
    with pytest.raises(PackRecordError, match="not a list of rows"):
        build_manifests({"file": TEI})


def test_a_record_that_is_not_json_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Text:
        def joinpath(self, _name: str) -> _Text:
            return self

        def read_text(self, encoding: str) -> str:
            return "[{"

    monkeypatch.setattr(engine_record.resources, "files", lambda _package: _Text())
    with pytest.raises(PackRecordError, match="not JSON"):
        load_manifests()


def test_the_pack_base_is_the_release_downloads_of_the_packs_repository() -> None:
    assert (
        engine_record.PACKAGES_URL
        == "https://github.com/tobocop2/lilbee-engine-packs/releases/download/"
    )
