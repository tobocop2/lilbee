"""Model status bar: Select dropdowns for chat and embedding models."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import ClassVar, NamedTuple

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Select, Static
from textual.widgets._select import SelectCurrent

from lilbee.catalog import clean_display_name, display_label_for_ref, extract_quant
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import apply_active_model
from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.core.config import cfg
from lilbee.core.services import get_services, reset_services
from lilbee.data.store import SearchScope
from lilbee.modelhub.models import ModelTask
from lilbee.providers.model_ref import OLLAMA_PREFIX, parse_model_ref
from lilbee.providers.sdk_backend import (
    OLLAMA_BACKEND_NAME,
    PROVIDER_KEYS,
    detect_backend_name,
)

log = logging.getLogger(__name__)

_DISABLED = Select.NULL

_MMPROJ_MARKER = "mmproj"

_CLOUD_WARNING_ID = "cloud-provider-warning"
_CLOUD_WARNING_CLASS = "cloud-warning"
_CLOUD_WARNING_VISIBLE_CLASS = "-visible"

# Routing-name -> display-label map derived from PROVIDER_KEYS. Any new
# entry added there lights up the warning without further changes here.
_CLOUD_PROVIDER_LABELS: dict[str, str] = {name: label for name, _, _, label in PROVIDER_KEYS}


def _cloud_provider_label(chat_model: str) -> str | None:
    """Return the provider display label for cloud-routed models, else None."""
    if not chat_model:
        return None
    ref = parse_model_ref(chat_model)
    if not ref.is_api:
        return None
    return _CLOUD_PROVIDER_LABELS.get(ref.provider)


class ModelOption(NamedTuple):
    """A selectable model with display label and config ref."""

    label: str  # human-readable name for the dropdown
    ref: str  # canonical ref persisted to config


def _is_mmproj(name: str) -> bool:
    """Return True if a model name refers to an mmproj projection file."""
    return _MMPROJ_MARKER in name.lower()


def _classify_installed_models() -> tuple[list[ModelOption], list[ModelOption]]:
    """Classify installed models into (chat, embedding) lists, dropping mmproj."""
    buckets: dict[ModelTask, list[ModelOption]] = {task: [] for task in ModelTask}
    seen: set[str] = set()

    _collect_native_models(buckets, seen)
    _collect_remote_models(buckets, seen)
    _collect_api_models(buckets, seen)

    # ModelBar exposes only chat + embedding Selects. Vision and rerank
    # models are collected to claim their refs in ``seen`` so later
    # buckets don't duplicate them, but aren't returned to the UI.
    return (
        sorted(buckets[ModelTask.CHAT], key=lambda o: o.ref),
        sorted(buckets[ModelTask.EMBEDDING], key=lambda o: o.ref),
    )


def _lookup_bucket(
    buckets: dict[ModelTask, list[ModelOption]], task: str, ref: str
) -> list[ModelOption] | None:
    """Return the bucket for *task*, or None if it is not a known ModelTask."""
    try:
        key = ModelTask(task)
    except ValueError:
        log.debug("dropping %r with unknown task %r", ref, task)
        return None
    return buckets.get(key)


def _native_label(hf_repo: str, gguf_filename: str, repo_count: int) -> str:
    """Build the picker label, appending the quant suffix only on collision."""
    base = clean_display_name(hf_repo)
    if repo_count <= 1:
        return base
    quant = extract_quant(gguf_filename)
    return f"{base} ({quant})" if quant else base


def _collect_native_models(buckets: dict[ModelTask, list[ModelOption]], seen: set[str]) -> None:
    """Add native registry models to buckets."""
    try:
        from lilbee.modelhub.registry import ModelRegistry

        registry = ModelRegistry(cfg.models_dir)
        manifests = registry.list_installed()
        repo_counts: dict[str, int] = {}
        for m in manifests:
            repo_counts[m.hf_repo] = repo_counts.get(m.hf_repo, 0) + 1

        for manifest in manifests:
            ref = manifest.ref
            if _is_mmproj(manifest.gguf_filename) or ref in seen:
                continue
            bucket = _lookup_bucket(buckets, manifest.task, ref)
            if bucket is None:
                continue
            seen.add(ref)
            label = _native_label(
                manifest.hf_repo, manifest.gguf_filename, repo_counts[manifest.hf_repo]
            )
            bucket.append(ModelOption(label=label, ref=ref))
    except Exception:
        log.debug("Could not read native model registry", exc_info=True)


def _collect_remote_models(buckets: dict[ModelTask, list[ModelOption]], seen: set[str]) -> None:
    """Add remote (Ollama / OpenAI-compatible) models, prefixed for routing.

    Skipped when the litellm extra is not installed -- surfacing a model
    the SDK cannot route is a guaranteed runtime error.
    """
    from lilbee.providers.litellm_sdk import litellm_available

    if not litellm_available():
        return
    try:
        from lilbee.modelhub.model_manager import classify_remote_models

        base_url = cfg.remote_base_url
        is_ollama = detect_backend_name(base_url) == OLLAMA_BACKEND_NAME
        for model in classify_remote_models(base_url):
            # Skip backend rows with a blank model name so the picker
            # doesn't render an empty " (Ollama)" row.
            if not model.name.strip():
                continue
            ref = f"{OLLAMA_PREFIX}{model.name}" if is_ollama else model.name
            if ref in seen or _is_mmproj(model.name):
                continue
            bucket = _lookup_bucket(buckets, model.task, ref)
            if bucket is None:
                continue
            seen.add(ref)
            label = f"{model.name} ({model.provider})"
            bucket.append(ModelOption(label=label, ref=ref))
    except Exception:
        log.debug("Could not classify remote models", exc_info=True)


def _collect_api_models(buckets: dict[ModelTask, list[ModelOption]], seen: set[str]) -> None:
    """Add frontier API chat models. Skipped without litellm (cannot route)."""
    from lilbee.providers.litellm_sdk import litellm_available

    if not litellm_available():
        return
    try:
        from lilbee.modelhub.model_manager import discover_api_models

        # API discovery returns only chat-capable refs; revisit if providers
        # expose embedding/vision/rerank.
        for display_name, models in discover_api_models().items():
            for model in models:
                # model.provider is the display name ("Anthropic"), but the ref
                # needs the backend-qualified prefix ("anthropic/model-name") for routing.
                prefix = display_name.lower()
                qualified = f"{prefix}/{model.name}"
                if qualified in seen:
                    continue
                seen.add(qualified)
                label = f"{model.name} ({display_name})"
                buckets[ModelTask.CHAT].append(ModelOption(label=label, ref=qualified))
    except Exception:
        log.debug("Could not discover API models", exc_info=True)


def _options_fingerprint(opts: list[ModelOption], default: str) -> tuple[tuple[str, str], ...]:
    """Hashable fingerprint of (options, active default) for cache hits."""
    return ((default, default), *((o.label, o.ref) for o in opts))


def _sync_select(sel: Select, opts: list[ModelOption], default: str = "") -> None:
    """Populate a Select; show ``(not installed)`` for a default missing from opts."""
    if default:
        try:
            ref = parse_model_ref(default)
        except ValueError:
            ref = None
        if ref is not None:
            default = ref.for_openai_prefix()
    if default and not any(o.ref == default for o in opts):
        shown = display_label_for_ref(default) or default
        opts.insert(0, ModelOption(f"{shown} (not installed)", default))
    sel.set_options(opts)
    if default:
        sel.value = default
    _refresh_select_label(sel, opts, default)


def _refresh_select_label(sel: Select, opts: list[ModelOption], value: str) -> None:
    """Push the matching option's label into ``SelectCurrent``.

    Textual's ``Select.set_options`` updates the option list but doesn't
    re-render ``SelectCurrent`` if the existing ``value`` still matches
    an option: the reactive watcher short-circuits on ``old == new``.
    That meant a freshly-labelled option (e.g. ``"<ref> (not installed)"``)
    kept the compose-time bare-ref label on screen. Poke the inner
    widget directly so the visible label matches what tests assert.
    """
    if not value:
        return
    with contextlib.suppress(Exception):
        current = sel.query_one(SelectCurrent)
        for label, ref_value in opts:
            if ref_value == value:
                current.update(label)
                return


_SELECT_IDS = ("#chat-model-select", "#embed-model-select")

_CSS_FILE = Path(__file__).parent / "model_bar.tcss"

# Presentation labels for the scope toggle. Values match ``SearchScope``
# so the widget's ``.value`` feeds directly into ``scope_to_chunk_type``.
_SCOPE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Both", SearchScope.BOTH.value),
    ("Wiki", SearchScope.WIKI.value),
    ("Raw", SearchScope.RAW.value),
)


class ModelBar(Widget, can_focus=False):
    """Compact bar with Select dropdowns for active model assignments."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._populating = True  # Guard against change events during init
        self._scope: SearchScope = SearchScope.BOTH
        # Cached option fingerprints for set_options skip. Each entry is
        # the ``(label, ref)`` tuple list that is currently mounted on
        # the Select; ``_scan_models`` runs on every chat ``on_show``,
        # but the install set rarely changes between visits and Textual
        # rebuilds the dropdown unconditionally on every set_options.
        self._chat_options_cache: tuple[tuple[str, str], ...] = ()
        self._embed_options_cache: tuple[tuple[str, str], ...] = ()

    @property
    def scope(self) -> SearchScope:
        """Current scope selection; consumed by ChatScreen when building RAG context."""
        return self._scope

    def compose(self) -> ComposeResult:
        chat_label = display_label_for_ref(cfg.chat_model) or cfg.chat_model
        embed_label = display_label_for_ref(cfg.embedding_model) or cfg.embedding_model
        chat_opts = [(chat_label, cfg.chat_model)] if cfg.chat_model else []
        embed_opts = [(embed_label, cfg.embedding_model)] if cfg.embedding_model else []
        with Horizontal():
            yield Static(pill("Chat", "$primary", "$text"), classes="model-bar-pill")
            yield Select[str](
                options=chat_opts,
                prompt="Chat model",
                id="chat-model-select",
                allow_blank=False,
            )
            yield Static(pill("Embed", "$secondary", "$text"), classes="model-bar-pill")
            yield Select[str](
                options=embed_opts,
                prompt="Embed model",
                id="embed-model-select",
                allow_blank=False,
            )
            # Scope picker only appears when the wiki layer is on. With wiki
            # off, ``CHUNKS_TABLE`` contains only raw rows so a wiki/raw/both
            # toggle has nothing to pick between; hiding it keeps the choice
            # from implying a capability the user hasn't opted into.
            if cfg.wiki:
                yield Static(pill("Scope", "$accent", "$text"), classes="model-bar-pill")
                yield Select[str](
                    options=list(_SCOPE_OPTIONS),
                    value=SearchScope.BOTH.value,
                    id="scope-select",
                    allow_blank=False,
                )
        yield Static("", id=_CLOUD_WARNING_ID, classes=_CLOUD_WARNING_CLASS)

    def on_mount(self) -> None:
        chat_sel = self.query_one("#chat-model-select", Select)
        embed_sel = self.query_one("#embed-model-select", Select)

        if cfg.chat_model:
            chat_sel.value = cfg.chat_model
        if cfg.embedding_model:
            embed_sel.value = cfg.embedding_model

        self._watch_overlay_collapse(chat_sel)
        self._watch_overlay_collapse(embed_sel)

        self._refresh_cloud_warning()
        self._scan_models()

    def _watch_overlay_collapse(self, sel: Select) -> None:
        """Force a full screen refresh when a Select overlay collapses."""

        def _on_expanded_change(expanded: bool) -> None:
            if not expanded and self.is_mounted:
                with contextlib.suppress(Exception):
                    self.screen.refresh()

        self.watch(sel, "expanded", _on_expanded_change, init=False)

    def on_unmount(self) -> None:
        """Collapse any open dropdown before tear-down so the SelectOverlay
        does not leak its border cells into the next screen's render."""
        for sel in self.query(Select):
            if sel.expanded:
                sel.expanded = False

    @work(thread=True)
    def _scan_models(self) -> None:
        """Scan installed models in background, then populate dropdowns."""
        chat, embed = _classify_installed_models()
        call_from_thread(self, self._populate, chat, embed)

    def _populate(
        self,
        chat_models: list[ModelOption],
        embed_models: list[ModelOption],
    ) -> None:
        """Populate Select widgets from scanned models (main thread)."""
        self._populating = True

        chat_sel = self.query_one("#chat-model-select", Select)
        embed_sel = self.query_one("#embed-model-select", Select)

        chat_opts = list(chat_models) if chat_models else [ModelOption("(none)", "")]
        embed_opts = list(embed_models) if embed_models else [ModelOption("(none)", "")]

        chat_fingerprint = _options_fingerprint(chat_opts, cfg.chat_model)
        if chat_fingerprint != self._chat_options_cache:
            _sync_select(chat_sel, chat_opts, cfg.chat_model)
            self._chat_options_cache = chat_fingerprint

        embed_fingerprint = _options_fingerprint(embed_opts, cfg.embedding_model)
        if embed_fingerprint != self._embed_options_cache:
            _sync_select(embed_sel, embed_opts, cfg.embedding_model)
            self._embed_options_cache = embed_fingerprint

        self._populating = False

    @on(Select.Changed, "#chat-model-select")
    def _on_chat_model_changed(self, event: Select.Changed) -> None:
        """Write the new chat model to cfg and settings."""
        chat_sel = self.query_one("#chat-model-select", Select)
        value = self._extract_value(event, chat_sel)
        if value is None or value == cfg.chat_model:
            return
        apply_active_model(self.app, "chat_model", value)
        self._refresh_cloud_warning()
        self._after_model_change()

    def _refresh_cloud_warning(self) -> None:
        """Show a warning if the active chat model routes to a cloud API."""
        warning = self.query_one(f"#{_CLOUD_WARNING_ID}", Static)
        label = _cloud_provider_label(cfg.chat_model)
        if label is None:
            warning.remove_class(_CLOUD_WARNING_VISIBLE_CLASS)
            return
        warning.update(msg.MODEL_BAR_CLOUD_PROVIDER_WARNING.format(provider=label))
        warning.add_class(_CLOUD_WARNING_VISIBLE_CLASS)

    @on(Select.Changed, "#embed-model-select")
    def _on_embed_model_changed(self, event: Select.Changed) -> None:
        """Write the new embedding model to cfg and settings."""
        embed_sel = self.query_one("#embed-model-select", Select)
        value = self._extract_value(event, embed_sel)
        if value is None or value == cfg.embedding_model:
            return
        # Pin a legacy store's identity to the OLD model BEFORE the cfg mutation
        # so the gate in store.search/add_chunks correctly detects drift on the
        # next op. See bb-x1qa.
        get_services().store.initialize_meta_if_legacy()
        apply_active_model(self.app, "embedding_model", value)
        self._after_model_change()

    @on(Select.Changed, "#scope-select")
    def _on_scope_changed(self, event: Select.Changed) -> None:
        """Track scope selection for the next ask_stream call.

        Session-scoped on purpose; not written to settings so each new
        session starts at "both" and the user opts into a narrower pool
        explicitly each time.
        """
        scope_sel = self.query_one("#scope-select", Select)
        value = self._extract_value(event, scope_sel)
        if value is None:
            return
        self._scope = SearchScope(value)

    def _extract_value(self, event: Select.Changed, sel: Select) -> str | None:
        """Extract a non-empty value from a Select.Changed event, or None to skip.

        Drops events whose payload no longer matches the widget's current
        value. Textual posts Select.Changed asynchronously, so a prior
        event carrying an intermediate auto-picked option can still be in
        the queue after ``_populate`` reassigns ``sel.value`` to the
        configured model. The stale-event check is deterministic across
        platforms because it compares two synchronously-set values rather
        than relying on event-loop ordering.
        """
        if self._populating:
            return None
        if event.value is _DISABLED or event.value is None or str(event.value) == "":
            return None
        if str(event.value) != str(sel.value):
            return None
        return str(event.value)

    def _after_model_change(self) -> None:
        """Shared post-change logic: cancel active stream and reset services safely."""
        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = self.app.screen
        if isinstance(screen, ChatScreen):
            screen._apply_model_change()
        else:
            reset_services()

    def refresh_models(self) -> None:
        """Re-scan models (called after downloads complete)."""
        self._scan_models()
