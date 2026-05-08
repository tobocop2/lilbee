"""Catalog dataclasses and pydantic types: leaf module, no sibling imports."""

from dataclasses import dataclass

from pydantic import BaseModel

from lilbee.modelhub.models import ModelTask


class HfGgufMeta(BaseModel):
    """GGUF metadata returned by the HF API when expand=gguf is requested.

    ModelInfo.gguf is typed as ``dict | None`` upstream, so we validate it ourselves.
    """

    total: int = 0
    architecture: str = ""
    context_length: int = 0


@dataclass
class DownloadProgress:
    """Human-readable snapshot of download progress.

    ``percent`` is a float (0.0 to 100.0) so the ProgressBar renders smooth
    fractional movement during multi-GB downloads. Call sites that need
    an integer for display format it themselves.
    """

    percent: float
    detail: str
    is_cache_hit: bool


@dataclass(frozen=True)
class CatalogModel:
    """One catalog entry, keyed by HuggingFace repo. ``gguf_filename`` may be a glob."""

    hf_repo: str
    gguf_filename: str
    size_gb: float
    min_ram_gb: float
    description: str
    featured: bool
    downloads: int
    task: ModelTask
    recommended: bool = False

    @property
    def ref(self) -> str:
        """Browse-time ref (the HF repo); concrete filename is resolved at install."""
        return self.hf_repo

    @property
    def display_name(self) -> str:
        """Human-readable label derived from the HuggingFace repo id."""
        # Local import keeps models.py a leaf (no sibling imports at module top).
        from lilbee.catalog.formatting import clean_display_name

        return clean_display_name(self.hf_repo)


@dataclass(frozen=True)
class CatalogResult:
    """Paginated catalog result."""

    total: int
    limit: int
    offset: int
    models: list[CatalogModel]
    has_more: bool = False


@dataclass(frozen=True)
class HfPage:
    """One page of HuggingFace API results."""

    models: list[CatalogModel]
    has_more: bool


@dataclass(frozen=True)
class ModelVariant:
    """One quantization within a model family. ``filename`` may be a glob."""

    hf_repo: str
    filename: str
    param_count: str
    quant: str
    size_mb: int
    recommended: bool
    mmproj_filename: str = ""


@dataclass(frozen=True)
class ModelFamily:
    """A group of related model variants (e.g. Qwen3 in multiple sizes)."""

    slug: str  # family slug for building refs: "qwen3"
    name: str  # display name: "Qwen3"
    task: ModelTask
    description: str
    variants: tuple[ModelVariant, ...]
