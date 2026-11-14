"""The NPU model rows that the Lemonade engine publishes: which entries list, and their fields."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from lilbee.catalog.models import ListsModels, estimate_min_ram_gb
from lilbee.catalog.refs import parse_model_ref
from lilbee.catalog.types import ModelFormat, ModelTask
from lilbee.providers.fleet.engines import LLAMA
from lilbee.providers.fleet.engines.lemonade import LEMONADE
from lilbee.providers.fleet.engines.lemonade_models import parse_models

PINNED = json.loads(
    (Path(__file__).parent / "fixtures" / "engine_model_lists" / "lemonade-pinned.json").read_text(
        encoding="utf-8"
    )
)
GIB = 1024**3


def _entry(**over: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "name": "Zed-4B-NPU2",
        "size": 4 * GIB,
        "default_context_length": 8192,
        "details": {"parameter_size": "4B"},
        "label": [],
        "footprint": 3.14,
    }
    entry.update(over)
    return entry


def _payload(*entries: tuple[str, dict[str, Any] | Any]) -> dict[str, Any]:
    """A list with one family per ``family:tag`` key and its entry."""
    models: dict[str, dict[str, Any]] = {}
    for ref, entry in entries:
        family, tag = ref.split(":")
        models.setdefault(family, {})[tag] = entry
    return {"model_path": "models", "models": models}


def _refs(payload: Any) -> list[str]:
    return [m.hf_repo for m in parse_models(payload)]


def test_pinned_list_gives_41_chat_rows_and_1_embedding_row() -> None:
    rows = parse_models(PINNED)
    by_task = {t: [m for m in rows if m.task is t] for t in (ModelTask.CHAT, ModelTask.EMBEDDING)}
    assert (len(by_task[ModelTask.CHAT]), len(by_task[ModelTask.EMBEDDING])) == (41, 1)
    assert by_task[ModelTask.EMBEDDING][0].hf_repo == "embed-gemma:300m"
    assert {m.format for m in rows} == {ModelFormat.XDNA}


def test_rows_keep_the_order_of_the_file() -> None:
    families = [f"{fam}:{tag}" for fam, tags in PINNED["models"].items() for tag in tags]
    listed = _refs(PINNED)
    assert listed == [ref for ref in families if ref in set(listed)]
    assert listed[0] == families[0]


def test_a_row_carries_the_published_fields() -> None:
    row = {m.hf_repo: m for m in parse_models(PINNED)}["qwen3:4b"]
    assert (row.format, row.formats, row.task) == (
        ModelFormat.XDNA,
        frozenset({ModelFormat.XDNA}),
        ModelTask.CHAT,
    )
    assert (row.size_gb, row.min_ram_gb, row.params) == (3.7, 3.1, 4_000_000_000)
    assert (row.downloads, row.featured, row.safety_stripped) == (0, False, False)
    assert row.pull_ref == "qwen3:4b#xdna"
    assert parse_model_ref(row.pull_ref).hf_repo == "qwen3:4b"
    assert row.description == "Qwen3-4B-NPU2, qwen3 4b npu2, AMD NPU, 32k context"


def test_no_footprint_falls_back_to_the_catalog_estimate() -> None:
    entry = _entry(size=GIB)
    del entry["footprint"]
    (row,) = parse_models(_payload(("zed:1b", entry)))
    assert row.size_gb == 1.0
    assert row.min_ram_gb == estimate_min_ram_gb(1.0)


@pytest.mark.parametrize(
    ("size_text", "params"),
    [("4B", 4 * 10**9), ("300M", 300 * 10**6), ("35B-A3B", 35 * 10**9), ("0.6B", 600_000_000)],
)
def test_parameter_size_reads_the_leading_count(size_text: str, params: int) -> None:
    entry = _entry(details={"parameter_size": size_text})
    assert [m.params for m in parse_models(_payload(("zed:1b", entry)))] == [params]


@pytest.mark.parametrize(
    ("context", "window"), [(32768, "32k"), (4096, "4k"), (1024, "1k"), (512, "512")]
)
def test_description_shows_the_context_window(context: int, window: str) -> None:
    entry = _entry(default_context_length=context)
    (row,) = parse_models(_payload(("zed:1b", entry)))
    assert row.description == f"Zed-4B-NPU2, zed 4b npu2, AMD NPU, {window} context"


def test_description_holds_at_most_120_characters() -> None:
    entry = _entry(name="Long-" * 30 + "NPU2")
    (row,) = parse_models(_payload(("zed:1b", entry)))
    assert len(row.description) == 120


def test_description_reads_underscores_as_spaces() -> None:
    (row,) = parse_models(_payload(("zed:1b", _entry(name="Zed_4B-NPU2"))))
    assert ", zed 4b npu2," in row.description


@pytest.mark.parametrize(
    ("entry", "listed"),
    [
        pytest.param(_entry(supported_platforms=["aie2p"], name="Zed"), True, id="aie2p"),
        pytest.param(_entry(supported_platforms=["dev04"]), False, id="dev04"),
        pytest.param(_entry(supported_platforms=["dev04", "aie2p"]), True, id="both"),
        pytest.param(_entry(supported_platforms=[]), False, id="empty-list"),
        pytest.param(_entry(), True, id="no-field-npu2-name"),
        pytest.param(_entry(name="Zed-4B-core"), False, id="no-field-other-name"),
        pytest.param(_entry(supported_platforms=None), True, id="null-field-npu2-name"),
    ],
)
def test_an_entry_lists_for_xdna2_only(entry: dict[str, Any], listed: bool) -> None:
    assert (_refs(_payload(("zed:1b", entry))) == ["zed:1b"]) is listed


@pytest.mark.parametrize("label", ["decision", "transcription", "realtime-transcription"])
def test_an_entry_that_serves_no_text_is_not_listed(label: str) -> None:
    kept = _entry(label=["reasoning"])
    payload = _payload(("zed:1b", _entry(label=[label, "reasoning"])), ("zed:2b", kept))
    assert _refs(payload) == ["zed:2b"]


@pytest.mark.parametrize("label", ["vision", "speculative", "memory", "tool-calling"])
def test_an_entry_that_generates_text_lists_as_chat(label: str) -> None:
    (row,) = parse_models(_payload(("zed:1b", _entry(label=[label]))))
    assert row.task is ModelTask.CHAT


def test_the_embeddings_label_makes_an_embedding_row() -> None:
    (row,) = parse_models(_payload(("zed:1b", _entry(label=["embeddings"]))))
    assert row.task is ModelTask.EMBEDDING


def test_a_null_label_list_reads_as_none() -> None:
    (row,) = parse_models(_payload(("zed:1b", _entry(label=None))))
    assert row.task is ModelTask.CHAT


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"name": None}, id="name-none"),
        pytest.param({"name": ""}, id="name-empty"),
        pytest.param({"name": 7}, id="name-number"),
        pytest.param({"size": "big"}, id="size-text"),
        pytest.param({"size": True}, id="size-bool"),
        pytest.param({"size": 0}, id="size-zero"),
        pytest.param({"default_context_length": "long"}, id="context-text"),
        pytest.param({"details": {"parameter_size": "many"}}, id="params-no-count"),
        pytest.param({"details": {"parameter_size": 4}}, id="params-number"),
        pytest.param({"details": {}}, id="params-missing"),
        pytest.param({"footprint": "3.1"}, id="footprint-text"),
        pytest.param({"label": 5}, id="label-number"),
    ],
)
def test_an_invalid_entry_is_skipped_and_its_neighbour_stays(bad: dict[str, Any]) -> None:
    payload = _payload(("zed:1b", _entry(**bad)), ("zed:2b", _entry()))
    assert _refs(payload) == ["zed:2b"]


@pytest.mark.parametrize("field", ["name", "size", "default_context_length", "details"])
def test_an_entry_without_a_required_field_is_skipped(field: str) -> None:
    entry = _entry()
    del entry[field]
    assert _refs(_payload(("zed:1b", entry), ("zed:2b", _entry()))) == ["zed:2b"]


@pytest.mark.parametrize("ref", ["Bad:1b", "zed:1B", "zed:", "zed:-x"])
def test_a_ref_that_is_not_a_model_tag_is_skipped(ref: str) -> None:
    family, tag = ref.split(":")
    payload = {"models": {family: {tag: _entry()}, "zed": {"2b": _entry()}}}
    assert _refs(payload) == ["zed:2b"]


def test_an_entry_that_is_not_an_object_is_skipped() -> None:
    assert _refs(_payload(("zed:1b", "text"), ("zed:2b", _entry()))) == ["zed:2b"]


@pytest.mark.parametrize(
    "payload",
    [[], "text", {}, {"models": []}, {"models": {"zed": []}}, {"models": {"zed": 1}}],
    ids=["list", "text", "no-models", "models-list", "family-list", "family-number"],
)
def test_a_payload_without_families_is_refused(payload: Any) -> None:
    with pytest.raises(ValueError, match="no families of models"):
        parse_models(payload)


def test_parsing_leaves_the_payload_unchanged() -> None:
    before = copy.deepcopy(PINNED)
    parse_models(PINNED)
    assert before == PINNED


def test_the_adapter_publishes_the_list_of_its_pinned_build() -> None:
    assert isinstance(LEMONADE, ListsModels)
    assert LEMONADE.models_url == (
        "https://raw.githubusercontent.com/ROCm/FastFlowLM/v1.0.7/src/model_list.json"
    )
    assert [m.hf_repo for m in LEMONADE.parse_models(PINNED)] == _refs(PINNED)


def test_an_engine_without_its_own_list_is_not_a_source() -> None:
    assert not isinstance(LLAMA, ListsModels)
