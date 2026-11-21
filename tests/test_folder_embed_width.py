"""The output width an embedding model folder declares in its config files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lilbee.providers.folder_embed_width import folder_embedding_width


def _write(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_hidden_size_is_the_width(tmp_path: Path) -> None:
    _write(tmp_path / "config.json", {"hidden_size": 1024})
    assert folder_embedding_width(tmp_path) == 1024


def test_text_section_wins_over_top_level(tmp_path: Path) -> None:
    _write(tmp_path / "config.json", {"hidden_size": 9, "text_config": {"hidden_size": 640}})
    assert folder_embedding_width(tmp_path) == 640


def test_pooling_width_wins_over_hidden_size(tmp_path: Path) -> None:
    _write(tmp_path / "config.json", {"hidden_size": 1024})
    _write(
        tmp_path / "modules.json",
        [{"type": "pipeline.step.Pooling", "path": "1_Pooling"}],
    )
    _write(tmp_path / "1_Pooling" / "config.json", {"word_embedding_dimension": 768})
    assert folder_embedding_width(tmp_path) == 768


def test_last_dense_layer_gives_the_width(tmp_path: Path) -> None:
    _write(tmp_path / "config.json", {"hidden_size": 1024})
    _write(
        tmp_path / "modules.json",
        [
            {"type": "pipeline.step.Transformer", "path": ""},
            {"type": "pipeline.step.Pooling", "path": "1_Pooling"},
            {"type": "pipeline.step.Dense", "path": "2_Dense"},
            {"type": "pipeline.step.Dense", "path": "3_Dense"},
        ],
    )
    _write(tmp_path / "1_Pooling" / "config.json", {"word_embedding_dimension": 1024})
    _write(tmp_path / "2_Dense" / "config.json", {"in_features": 1024, "out_features": 3072})
    _write(tmp_path / "3_Dense" / "config.json", {"in_features": 3072, "out_features": 512})
    assert folder_embedding_width(tmp_path) == 512


def test_unreadable_module_config_keeps_the_earlier_width(tmp_path: Path) -> None:
    _write(tmp_path / "config.json", {"hidden_size": 1024})
    _write(
        tmp_path / "modules.json",
        [
            {"type": "pipeline.step.Dense", "path": "2_Dense"},
            {"type": "pipeline.step.Dense"},
            {"type": "pipeline.step.Dense", "path": 7},
            "not an object",
        ],
    )
    assert folder_embedding_width(tmp_path) == 1024


@pytest.mark.parametrize("hidden", [0, -4, "1024", True, None, 10.5])
def test_a_junk_hidden_size_is_no_width(tmp_path: Path, hidden: object) -> None:
    _write(tmp_path / "config.json", {"hidden_size": hidden})
    assert folder_embedding_width(tmp_path) is None


def test_a_hidden_size_of_one_is_a_width(tmp_path: Path) -> None:
    _write(tmp_path / "config.json", {"hidden_size": 1})
    assert folder_embedding_width(tmp_path) == 1


def test_no_files_is_no_width(tmp_path: Path) -> None:
    assert folder_embedding_width(tmp_path) is None


def test_config_that_is_not_an_object_is_no_width(tmp_path: Path) -> None:
    _write(tmp_path / "config.json", [1024])
    _write(tmp_path / "modules.json", {"type": "Dense"})
    assert folder_embedding_width(tmp_path) is None


def test_unparseable_config_is_no_width(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
    assert folder_embedding_width(tmp_path) is None


def _pooled(tmp_path: Path, **modes: object) -> Path:
    _write(tmp_path / "config.json", {"hidden_size": 384})
    _write(tmp_path / "modules.json", [{"type": "pipeline.step.Pooling", "path": "1_Pooling"}])
    _write(tmp_path / "1_Pooling" / "config.json", {"word_embedding_dimension": 384, **modes})
    return tmp_path


def test_each_enabled_pooling_mode_adds_one_width(tmp_path: Path) -> None:
    folder = _pooled(tmp_path, pooling_mode_cls_token=True, pooling_mode_mean_tokens=True)
    assert folder_embedding_width(folder, concatenates_pooling=True) == 768


def test_an_engine_that_serves_one_mode_gets_one_block(tmp_path: Path) -> None:
    folder = _pooled(tmp_path, pooling_mode_cls_token=True, pooling_mode_mean_tokens=True)
    assert folder_embedding_width(folder, concatenates_pooling=False) == 384
    assert folder_embedding_width(folder) == 384


@pytest.mark.parametrize(
    "mode",
    [
        "pooling_mode_max_tokens",
        "pooling_mode_mean_tokens",
        "pooling_mode_mean_sqrt_len_tokens",
        "pooling_mode_weightedmean_tokens",
        "pooling_mode_lasttoken",
    ],
)
def test_every_pooling_mode_counts(tmp_path: Path, mode: str) -> None:
    folder = _pooled(tmp_path, pooling_mode_cls_token=True, **{mode: True})
    assert folder_embedding_width(folder, concatenates_pooling=True) == 768


def test_a_pooling_step_without_a_width_keeps_the_earlier_width(tmp_path: Path) -> None:
    folder = _pooled(tmp_path, pooling_mode_mean_tokens=True)
    _write(folder / "1_Pooling" / "config.json", {"pooling_mode_mean_tokens": True})
    assert folder_embedding_width(folder, concatenates_pooling=True) == 384


def test_disabled_modes_do_not_count(tmp_path: Path) -> None:
    folder = _pooled(tmp_path, pooling_mode_cls_token=False, pooling_mode_mean_tokens=True)
    assert folder_embedding_width(folder, concatenates_pooling=True) == 384


def test_a_non_boolean_mode_flag_does_not_count(tmp_path: Path) -> None:
    folder = _pooled(tmp_path, pooling_mode_cls_token="yes", pooling_mode_mean_tokens=1)
    assert folder_embedding_width(folder, concatenates_pooling=True) == 384
