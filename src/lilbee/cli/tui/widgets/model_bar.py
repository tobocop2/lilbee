"""Model status bar — Select dropdowns for chat and embedding models."""

from __future__ import annotations

import contextlib
import logging
from typing import NamedTuple

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Select, Static
from textual.widgets._select import SelectCurrent

from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.config import cfg
from lilbee.models import ModelTask
from lilbee.providers.model_ref import OLLAMA_PREFIX, parse_model_ref
from lilbee.providers.sdk_backend import (
    OLLAMA_BACKEND_NAME,
    PROVIDER_KEYS,
    detect_backend_name,
)
from lilbee.services import reset_services
from lilbee.store import SearchScope

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
    """Return the provider display label for cloud-routed models, else None.

    Local models and Ollama (remote but user-local) return None. Anything
    classified as an API provider by ``parse_model_ref`` returns the
    matching label from PROVIDER_KEYS.
    """
    if not chat_model:
        return None
    ref = parse_model_ref(chat_model)
    if not ref.is_api:
        return None
    return _CLOUD_PROVIDER_LABELS.get(ref.provider)


class ModelOption(NamedTuple):
    """A selectable model with display label and config ref."""

    label: str  # human-readable name for the dropdown
    ref: str  # name:tag identity for config persistence


def _is_mmproj(name: str) -> bool:
    """Return True if a model name refers to an mmproj projection file."""
    return _MMPROJ_MARKER in name.lower()


def _classify_installed_models() -> tuple[list[ModelOption], list[ModelOption]]:
    """Classify installed models into (chat, embedding) lists.
    Uses registry manifests for native models and the SDK backend's
    backend metadata for remote models. Filters out mmproj files.
    """
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
    """Return the bucket for *task*, or None if *task* is not a known ModelTask.

    Unknown tasks from forward-compat manifests are dropped rather than
    silently misclassified into chat.
    """
    try:
        key = ModelTask(task)
    except ValueError:
        log.debug("dropping %r with unknown task %r", ref, task)
        return None
    return buckets.get(key)


def _collect_native_models(buckets: dict[ModelTask, list[ModelOption]], seen: set[str]) -> None:
    """Add native registry models to buckets."""
    try:
        from lilbee.registry import ModelRegistry

        registry = ModelRegistry(cfg.models_dir)
        for manifest in registry.list_installed():
            ref = f"{manifest.name}:{manifest.tag}"
            if _is_mmproj(ref) or ref in seen:
                continue
            bucket = _lookup_bucket(buckets, manifest.task, ref)
            if bucket is None:
                continue
            seen.add(ref)
            label = manifest.display_name or ref
            bucket.append(ModelOption(label=label, ref=ref))
    except Exception:
        log.debug("Could not read native model registry", exc_info=True)


def _collect_remote_models(buckets: dict[ModelTask, list[ModelOption]], seen: set[str]) -> None:
    """Add remote (Ollama / OpenAI-compatible) models to buckets, prefixed for routing."""
    try:
        from lilbee.model_manager import classify_remote_models

        base_url = cfg.remote_base_url
        is_ollama = detect_backend_name(base_url) == OLLAMA_BACKEND_NAME
        for model in classify_remote_models(base_url):
            if not model.name:
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
    """Add frontier API models (OpenAI, Anthropic, Gemini) to chat bucket."""
    try:
        from lilbee.model_manager import discover_api_models

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


def _sync_select(sel: Select, opts: list[ModelOption], default: str = "") -> None:
    """Populate a model Select and set it to *default* (from cfg).

    Normalizes *default* with :latest when no tag is present so that a
    bare name like ``qwen3`` matches the installed ``qwen3:latest`` option
    instead of creating a broken fallback entry.

    If *default* is set but not actually installed, surfaces it with a
    ``(not installed)`` label so the user doesn't mistake the config
    default for a working model. Select still allows picking it for
    backward compatibility, but the UI makes the real state obvious.
    """
    ref = parse_model_ref(default) if default else None
    default = ref.for_openai_prefix() if ref else default
    if default and not any(o.ref == default for o in opts):
        opts.insert(0, ModelOption(f"{default} (not installed)", default))
    sel.set_options(opts)
    if default:
        sel.value = default
    _refresh_select_label(sel, opts, default)


def _refresh_select_label(sel: Select, opts: list[ModelOption], value: str) -> None:
    """Push the matching option's label into ``SelectCurrent``.

    Textual's ``Select.set_options`` updates the option list but doesn't
    re-render ``SelectCurrent`` if the existing ``value`` still matches
    an option — the reactive watcher short-circuits on ``old == new``.
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

# Presentation labels for the scope toggle. Values match ``SearchScope``
# so the widget's ``.value`` feeds directly into ``scope_to_chunk_type``.
_SCOPE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Both", SearchScope.BOTH.value),
    ("Wiki", SearchScope.WIKI.value),
    ("Raw", SearchScope.RAW.value),
)


class ModelBar(Widget, can_focus=False):
    """Compact bar with Select dropdowns for active model assignments."""

    # Textual's SelectOverlay floats with overlay: screen and can leak
    # border cells into terminal scrollback on collapse. Capping the
    # height and constraining inside the screen keeps the overlay from
    # crossing the viewport; the refresh in _watch_overlay_collapse
    # forces the compositor to re-paint the covered region.
    DEFAULT_CSS = """
    ModelBar {
        height: auto;
        padding: 0 1;
    }
    ModelBar Horizontal {
        height: 3;
        width: 100%;
    }
    ModelBar .model-bar-pill {
        width: auto;
        padding: 1 1 0 0;
    }
    ModelBar Select {
        width: 1fr;
        margin: 0 1 0 0;
    }
    ModelBar Select > SelectOverlay {
        max-height: 8;
        constrain: inside inside;
    }
    ModelBar .cloud-warning {
        display: none;
        color: $warning;
        text-style: bold;
        padding: 0 1;
    }
    ModelBar .cloud-warning.-visible {
        display: block;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._populating = True  # Guard against change events during init
        self._scope: SearchScope = SearchScope.BOTH

    @property
    def scope(self) -> SearchScope:
        """Current scope selection; consumed by ChatScreen when building RAG context."""
        return self._scope

    def compose(self) -> ComposeResult:
        chat_opts = [(cfg.chat_model, cfg.chat_model)] if cfg.chat_model else []
        embed_opts = [(cfg.embedding_model, cfg.embedding_model)] if cfg.embedding_model else []
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

        _sync_select(chat_sel, chat_opts, cfg.chat_model)
        _sync_select(embed_sel, embed_opts, cfg.embedding_model)

        self._populating = False

    @on(Select.Changed, "#chat-model-select")
    def _on_chat_model_changed(self, event: Select.Changed) -> None:
        """Write the new chat model to cfg and settings."""
        from lilbee.cli.tui.app import apply_active_model

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
        from lilbee.cli.tui.app import apply_active_model

        embed_sel = self.query_one("#embed-model-select", Select)
        value = self._extract_value(event, embed_sel)
        if value is None or value == cfg.embedding_model:
            return
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
            screen._refresh_status_line()
        else:
            reset_services()

    def refresh_models(self) -> None:
        """Re-scan models (called after downloads complete)."""
        self._scan_models()
