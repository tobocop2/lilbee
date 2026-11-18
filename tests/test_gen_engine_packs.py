"""The pack record is written from the release assets API, not by hand."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "gen_engine_packs.py"
TEI = "lilbee_engine_tei-1.9.4-1.cpu-py3-none-manylinux_2_34_x86_64.whl"
LLAMA = "lilbee_engine_llama_cpp-1.0-1.cpu-py3-none-win_amd64.whl"
BASE = "https://github.com/tobocop2/lilbee-engine-packs/releases/download/"


@pytest.fixture(scope="module")
def tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_engine_packs", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def asset(name: str, digest: str | None, size: int) -> dict[str, Any]:
    return {"name": name, "digest": digest, "size": size}


def release(tag: str, *assets: dict[str, Any], draft: bool = False) -> dict[str, Any]:
    return {"tag_name": tag, "draft": draft, "assets": list(assets)}


def test_a_row_takes_its_digest_and_size_from_the_asset_and_its_url_from_the_tag(
    tool: ModuleType,
) -> None:
    releases = [release("packs-1", asset(TEI, "sha256:" + "a" * 64, 7))]
    assert tool.build_record(releases, ["packs-1"]) == [
        {"file": TEI, "url": f"{BASE}packs-1/{TEI}", "sha256": "a" * 64, "size": 7}
    ]


def test_only_engine_wheels_become_rows_and_rows_are_sorted_by_file(tool: ModuleType) -> None:
    releases = [
        release(
            "packs-1",
            asset(TEI, "sha256:" + "a" * 64, 7),
            asset("notes.txt", None, 1),
            asset("lilbee_engine_tei-1.tar.gz", None, 1),
            asset(LLAMA, "sha256:" + "b" * 64, 8),
        )
    ]
    assert [row["file"] for row in tool.build_record(releases, ["packs-1"])] == [LLAMA, TEI]


def test_a_later_tag_replaces_a_wheel_of_the_same_name(tool: ModuleType) -> None:
    releases = [
        release("old", asset(TEI, "sha256:" + "a" * 64, 7)),
        release("new", asset(TEI, "sha256:" + "b" * 64, 9)),
    ]
    (row,) = tool.build_record(releases, ["old", "new"])
    assert (row["sha256"], row["size"], row["url"]) == ("b" * 64, 9, f"{BASE}new/{TEI}")


LLAMA_B2 = LLAMA.replace("-1.cpu", "-2.cpu")
LLAMA_ABI = LLAMA.replace("py3-none", "py3-abi3")


@pytest.mark.parametrize("tags", [["old", "new"], ["new", "old"]])
def test_the_highest_build_of_a_cell_wins_whatever_the_tag_order(
    tool: ModuleType, tags: list[str]
) -> None:
    releases = [
        release("old", asset(LLAMA, "sha256:" + "a" * 64, 7), asset(TEI, "sha256:" + "c" * 64, 5)),
        release("new", asset(LLAMA_B2, "sha256:" + "b" * 64, 9)),
    ]
    record = tool.build_record(releases, tags)
    assert [(row["file"], row["sha256"]) for row in record] == [
        (LLAMA_B2, "b" * 64),
        (TEI, "c" * 64),
    ]


def test_two_different_wheels_of_one_cell_version_and_build_are_refused(tool: ModuleType) -> None:
    releases = [
        release("one", asset(LLAMA, "sha256:" + "a" * 64, 7)),
        release("two", asset(LLAMA_ABI, "sha256:" + "b" * 64, 9)),
    ]
    with pytest.raises(tool.ReleaseError, match=rf"two: {LLAMA_ABI} and {LLAMA} are both build 1"):
        tool.build_record(releases, ["one", "two"])


MISTRAL_CPU = "lilbee_engine_mistral_rs-0.9.5-1.cpu-py3-none-manylinux_2_35_x86_64.whl"
MISTRAL_CUDA = [
    "lilbee_engine_mistral_rs-0.9.5-1.cuda128_sm89-py3-none-manylinux_2_35_x86_64.whl",
    "lilbee_engine_mistral_rs-0.9.5-2.cuda128_sm89-py3-none-manylinux_2_35_x86_64.whl",
    "lilbee_engine_mistral_rs-0.9.6-1.cuda128_sm89-py3-none-manylinux_2_35_x86_64.whl",
    "lilbee_engine_mistral_rs-0.9.5-1.cuda-py3-none-manylinux_2_35_x86_64.whl",
    "lilbee_engine_mistral_rs-0.9.5-1.cuda_sm90-py3-none-win_amd64.whl",
]
OTHER_CUDA = [
    "lilbee_engine_llama_cpp-1.0-1.cuda-py3-none-win_amd64.whl",
    "lilbee_engine_tei-1.9.4-1.cuda_sm89-py3-none-manylinux_2_34_x86_64.whl",
]


@pytest.mark.parametrize("cuda", MISTRAL_CUDA)
def test_a_mistral_rs_cuda_wheel_of_any_version_or_build_gets_no_row(
    tool: ModuleType, cuda: str
) -> None:
    releases = [
        release(
            "t",
            asset(MISTRAL_CPU, "sha256:" + "a" * 64, 7),
            asset(cuda, "sha256:" + "b" * 64, 9),
        )
    ]
    assert [row["file"] for row in tool.build_record(releases, ["t"])] == [MISTRAL_CPU]


@pytest.mark.parametrize("cuda", OTHER_CUDA)
def test_a_cuda_wheel_of_another_engine_keeps_its_row(tool: ModuleType, cuda: str) -> None:
    releases = [release("t", asset(cuda, "sha256:" + "b" * 64, 9))]
    assert [row["file"] for row in tool.build_record(releases, ["t"])] == [cuda]


def test_a_draft_release_is_named_so_the_command_can_warn(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    releases = [
        release("live", asset(TEI, "sha256:" + "a" * 64, 7)),
        release("wip", asset(LLAMA, "sha256:" + "b" * 64, 8), draft=True),
        release("unused", draft=True),
    ]
    assert tool.draft_tags(releases, ["live", "wip"]) == ["wip"]
    monkeypatch.setattr(tool, "fetch_releases", lambda: releases)
    assert tool.main(["live", "wip"]) == 0
    assert "warning: wip is a draft; its urls answer 404" in capsys.readouterr().err


def test_a_tag_with_no_release_is_refused(tool: ModuleType) -> None:
    with pytest.raises(tool.ReleaseError, match="gone: no such release"):
        tool.build_record([release("here")], ["gone"])


def test_an_asset_without_a_sha256_digest_is_refused(tool: ModuleType) -> None:
    for digest in (None, "md5:abc"):
        with pytest.raises(tool.ReleaseError, match="has no sha256 digest"):
            tool.build_record([release("t", asset(TEI, digest, 7))], ["t"])


def test_a_row_the_loader_refuses_is_refused_by_name(tool: ModuleType) -> None:
    releases = [release("t", asset(TEI, "sha256:" + "a" * 64, 0))]
    with pytest.raises(tool.PackRecordError, match=rf"row 1 \({TEI}\)"):
        tool.build_record(releases, ["t"])


def test_the_releases_come_from_the_github_api_one_object_per_line(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        text = json.dumps(release("a")) + "\n" + json.dumps(release("b")) + "\n"
        return subprocess.CompletedProcess(command, 0, stdout=text, stderr="")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)
    assert [r["tag_name"] for r in tool.fetch_releases()] == ["a", "b"]
    assert seen[0][:3] == ["gh", "api", "--paginate"]
    assert "repos/tobocop2/lilbee-engine-packs/releases" in seen[0]


def test_the_tool_reads_the_repository_that_the_pack_url_names(tool: ModuleType) -> None:
    from lilbee.providers.fleet.engine_record import PACKAGES_URL

    assert f"https://github.com/{tool.REPOSITORY}/releases/download/" == PACKAGES_URL


def test_the_command_writes_the_record_to_standard_output(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        tool, "fetch_releases", lambda: [release("t", asset(TEI, "sha256:" + "a" * 64, 7))]
    )
    assert tool.main(["t"]) == 0
    assert json.loads(capsys.readouterr().out) == [
        {"file": TEI, "url": f"{BASE}t/{TEI}", "sha256": "a" * 64, "size": 7}
    ]


def test_the_command_reports_a_refusal_and_exits_non_zero(
    tool: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(tool, "fetch_releases", lambda: [])
    assert tool.main(["gone"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "gone: no such release" in captured.err


def test_the_command_needs_a_tag(tool: ModuleType) -> None:
    with pytest.raises(SystemExit) as done:
        tool.main([])
    assert done.value.code == 2


def test_the_module_runs_as_a_command(monkeypatch: pytest.MonkeyPatch) -> None:
    import runpy

    monkeypatch.setattr(sys, "argv", ["gen_engine_packs.py"])
    with pytest.raises(SystemExit) as done:
        runpy.run_path(str(TOOL), run_name="__main__")
    assert done.value.code == 2
