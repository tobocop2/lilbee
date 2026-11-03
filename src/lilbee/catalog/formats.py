"""Model formats on the Hugging Face Hub: tags, file selection, listing parameters."""

import posixpath
import re
from collections.abc import Iterable

from lilbee.catalog.refs import rank_gguf_candidates, split_shard_filenames
from lilbee.catalog.types import ModelFormat

_GGUF_SEARCH_TERM = "GGUF"
_SAFETENSORS_SUFFIX = ".safetensors"
_SAFETENSORS_INDEX = "model.safetensors.index.json"
_SHARD_RE = re.compile(r"-\d+-of-\d+\.safetensors$")
_SHARD_PARTS_RE = re.compile(r"(.*)-(\d+)-of-(\d+)\.safetensors")
_IR_MODEL = "openvino_model.xml"
_IR_COMPANIONS = (
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
)
_TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)
_POOLING_FILES = ("1_Pooling/config.json", "modules.json")
_MODULES_FILE = "modules.json"
_MODULE_FILE_RE = re.compile(r"\d+_[^/]+/(?:config\.json|model\.safetensors)")


def formats_from_tags(tags: Iterable[str]) -> frozenset[ModelFormat]:
    """The formats a repo's Hub tags name; an MLX repo is not plain safetensors."""
    found = {fmt for fmt in ModelFormat if fmt.value in set(tags)}
    if ModelFormat.MLX in found:
        found.discard(ModelFormat.SAFETENSORS)
    return frozenset(found)


def hf_listing_params(fmt: ModelFormat, search: str) -> dict[str, str]:
    """The Hub ``filter=`` and ``search=`` values for a catalog listing of *fmt*.

    The Hub reads one space-joined ``search`` value; the GGUF listing keeps the
    word GGUF in front of the user's tokens.
    """
    tokens = [p for p in re.split(r"[\s/\-]+", search.strip()) if p]
    if fmt == ModelFormat.GGUF:
        tokens.insert(0, _GGUF_SEARCH_TERM)
    return {"filter": ModelFormat(fmt).value, "search": " ".join(tokens)}


def select_files(fmt: ModelFormat, filenames: Iterable[str]) -> list[str]:
    """The repo files a pull of *fmt* fetches, entry file first; empty when the repo has none.

    Selection reads the listing, never the Hub tags: a tag only says the format
    exists somewhere in the repo. Only exact names and weight suffixes match, so
    a hash-named blob or a pickled ``pytorch_model.bin`` is never chosen.
    """
    names = sorted(set(filenames))
    if fmt == ModelFormat.GGUF:
        selected = _select_gguf(names)
    elif fmt == ModelFormat.OPENVINO:
        selected = _select_openvino(names)
    else:
        selected = _select_safetensors(fmt, names)
    _refuse_unsafe_names(selected)
    return selected


def _refuse_unsafe_names(selected: Iterable[str]) -> None:
    """Raise ValueError for a selected name that is absolute or has a ``..`` part."""
    for name in selected:
        parts = name.replace("\\", "/").split("/")
        if name.startswith(("/", "\\")) or ".." in parts:
            raise ValueError(f"Refusing model file {name!r}: its path leaves the model folder")


def _require_whole_shard_sets(names: list[str], shards: list[str]) -> None:
    """Raise ValueError naming the first shard that a ``-NN-of-MM`` set lacks."""
    for stem, index, total in [m.groups() for m in map(_SHARD_PARTS_RE.fullmatch, shards) if m]:
        for number in range(1, int(total) + 1):
            expected = f"{stem}-{number:0{len(index)}d}-of-{total}{_SAFETENSORS_SUFFIX}"
            if expected not in names:
                raise ValueError(f"Incomplete model: {expected} is missing from the file list")


def _present(names: list[str], wanted: Iterable[str]) -> list[str]:
    return [name for name in wanted if name in names]


def _select_gguf(names: list[str]) -> list[str]:
    ranked = rank_gguf_candidates(names)
    return split_shard_filenames(ranked[0]) if ranked else []


def _select_safetensors(fmt: ModelFormat, names: list[str]) -> list[str]:
    root = [name for name in names if "/" not in name and name.endswith(_SAFETENSORS_SUFFIX)]
    shards = [name for name in root if _SHARD_RE.search(name)]
    weights = shards or [name for name in root if name == "model" + _SAFETENSORS_SUFFIX] or root
    if not weights:
        return []
    _require_whole_shard_sets(names, shards)
    index = _present(names, [_SAFETENSORS_INDEX])
    extras = _present(names, ["config.json", *_TOKENIZER_FILES])
    if fmt == ModelFormat.SAFETENSORS:
        extras += _present(names, ["generation_config.json"]) + _module_files(names)
    entry = index + weights if shards else weights + index
    return [*entry, *sorted(extras)]


def _module_files(names: list[str]) -> list[str]:
    """``modules.json`` plus the config and weight file of each numbered module folder."""
    if _MODULES_FILE not in names:
        return []
    return [_MODULES_FILE, *(name for name in names if _MODULE_FILE_RE.fullmatch(name))]


def _select_openvino(names: list[str]) -> list[str]:
    folders = sorted(
        (posixpath.dirname(name) for name in names if posixpath.basename(name) == _IR_MODEL),
        key=lambda folder: (folder.count("/"), folder),
    )
    if not folders:
        return []
    folder = folders[0]
    ir = [posixpath.join(folder, name) for name in (_IR_MODEL, *_IR_COMPANIONS)]
    extras = ["config.json", "generation_config.json", *_TOKENIZER_FILES, *_POOLING_FILES]
    return [*_present(names, ir), *_present(names, extras)]
