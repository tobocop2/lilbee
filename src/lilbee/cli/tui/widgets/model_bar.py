"""Model status bar — Select dropdowns for chat, embedding, and vision models."""

from __future__ import annotations

import logging
from typing import NamedTuple

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Label, Select

from lilbee import settings
from lilbee.config import cfg
from lilbee.models import ModelTask
from lilbee.services import reset_services

log = logging.getLogger(__name__)

_DISABLED = Select.NULL

_MMPROJ_MARKER = "mmproj"


class ModelOption(NamedTuple):
    """A selectable model with display label and config ref."""

    label: str  # human-readable name for the dropdown
    ref: str  # name:tag identity for config persistence


def _is_mmproj(name: str) -> bool:
    """Return True if a model name refers to an mmproj projection file."""
    return _MMPROJ_MARKER in name.lower()


def _classify_installed_models() -> tuple[list[ModelOption], list[ModelOption], list[ModelOption]]:
    """Classify installed models into (chat, embedding, vision) lists.
    Uses registry manifests for native models and the litellm backend's
    backend metadata for remote models. Filters out mmproj files.
    """
    buckets: dict[str, list[ModelOption]] = {
        ModelTask.CHAT: [],
        ModelTask.EMBEDDING: [],
        ModelTask.VISION: [],
    }
    seen: set[str] = set()

    _collect_native_models(buckets, seen)
    _collect_remote_models(buckets, seen)

    return (
        sorted(buckets[ModelTask.CHAT], key=lambda o: o.ref),
        sorted(buckets[ModelTask.EMBEDDING], key=lambda o: o.ref),
        sorted(buckets[ModelTask.VISION], key=lambda o: o.ref),
    )


def _collect_native_models(buckets: dict[str, list[ModelOption]], seen: set[str]) -> None:
    """Add native registry models to buckets."""
    try:
        from lilbee.registry import ModelRegistry

        registry = ModelRegistry(cfg.models_dir)
        for manifest in registry.list_installed():
            ref = f"{manifest.name}:{manifest.tag}"
            if _is_mmproj(ref) or ref in seen:
                continue
            seen.add(ref)
            label = manifest.display_name or ref
            buckets.get(manifest.task, buckets[ModelTask.CHAT]).append(
                ModelOption(label=label, ref=ref)
            )
    except Exception:
        log.debug("Could not read native model registry", exc_info=True)


def _collect_remote_models(buckets: dict[str, list[ModelOption]], seen: set[str]) -> None:
    """Add remote (litellm) models to buckets."""
    try:
        from lilbee.model_manager import classify_remote_models

        for model in classify_remote_models(cfg.litellm_base_url):
            if model.name in seen or _is_mmproj(model.name):
                continue
            seen.add(model.name)
            buckets.get(model.task, buckets[ModelTask.CHAT]).append(
                ModelOption(label=model.name, ref=model.name)
            )
    except Exception:
        log.debug("Could not classify remote models", exc_info=True)


def _sync_select(sel: Select, opts: list[ModelOption], default: str = "") -> None:
    """Set options and value for a model Select widget.
    Preserves the current value if it's in the options. Falls back to
    *default* (typically the configured model from ``cfg``). If the
    resolved value isn't in *opts*, prepends it so it remains selectable.

    Note: may mutate *opts* by inserting the resolved value at index 0.
    """
    sel.set_options(opts)
    current = str(sel.value) if sel.value != _DISABLED else ""
    target = current or default
    if target:
        if not any(o.ref == target for o in opts):
            opts.insert(0, ModelOption(target, target))
            sel.set_options(opts)
        sel.value = target


_SELECT_IDS = ("#chat-model-select", "#embed-model-select", "#vision-model-select")


class ModelBar(Widget, can_focus=False):
    """Compact bar with Select dropdowns for active model assignments."""

    DEFAULT_CSS = """
    ModelBar {
        dock: top;
        height: 3;
        padding: 0 1;
    }
    ModelBar Horizontal {
        height: 3;
        width: 100%;
    }
    ModelBar Label {
        width: auto;
        padding: 1 1 0 0;
        text-style: bold;
    }
    ModelBar Select {
        width: 1fr;
        margin: 0 1 0 0;
    }
    """

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        self._populating = True  # Guard against change events during init

    def compose(self) -> ComposeResult:
        chat_opts = [(cfg.chat_model, cfg.chat_model)] if cfg.chat_model else []
        embed_opts = [(cfg.embedding_model, cfg.embedding_model)] if cfg.embedding_model else []
        vision_opts = [(cfg.vision_model, cfg.vision_model)] if cfg.vision_model else []
        with Horizontal():
            yield Label("Chat:")
            yield Select[str](
                options=chat_opts,
                prompt="Chat model",
                id="chat-model-select",
                allow_blank=False,
            )
            yield Label("Embed:")
            yield Select[str](
                options=embed_opts,
                prompt="Embed model",
                id="embed-model-select",
                allow_blank=False,
            )
            yield Label("Vision:")
            yield Select[str](
                options=vision_opts,
                prompt="Vision (optional)",
                id="vision-model-select",
                allow_blank=True,
            )

    def on_mount(self) -> None:
        chat_sel = self.query_one("#chat-model-select", Select)
        embed_sel = self.query_one("#embed-model-select", Select)
        vision_sel = self.query_one("#vision-model-select", Select)

        if cfg.chat_model:
            chat_sel.value = cfg.chat_model
        if cfg.embedding_model:
            embed_sel.value = cfg.embedding_model
        if cfg.vision_model:
            vision_sel.value = cfg.vision_model

        self._scan_models()

    @work(thread=True)
    def _scan_models(self) -> None:
        """Scan installed models in background, then populate dropdowns."""
        chat, embed, vision = _classify_installed_models()
        self.app.call_from_thread(self._populate, chat, embed, vision)

    def _populate(
        self,
        chat_models: list[ModelOption],
        embed_models: list[ModelOption],
        vision_models: list[ModelOption],
    ) -> None:
        """Populate Select widgets from scanned models (main thread)."""
        self._populating = True

        chat_sel = self.query_one("#chat-model-select", Select)
        embed_sel = self.query_one("#embed-model-select", Select)
        vision_sel = self.query_one("#vision-model-select", Select)

        chat_opts = list(chat_models) if chat_models else [ModelOption("(none)", "")]
        embed_opts = list(embed_models) if embed_models else [ModelOption("(none)", "")]

        _sync_select(chat_sel, chat_opts, cfg.chat_model)
        _sync_select(embed_sel, embed_opts, cfg.embedding_model)
        self._sync_vision_select(vision_sel, vision_models)

        self._populating = False

    def _sync_vision_select(self, sel: Select, models: list[ModelOption]) -> None:
        """Sync vision Select with extra fallback to cfg.vision_model."""
        opts = list(models)
        current = str(sel.value) if sel.value != _DISABLED else ""
        target = current or cfg.vision_model
        if target:
            if not any(o.ref == target for o in opts):
                opts.insert(0, ModelOption(target, target))
            sel.set_options(opts)
            sel.value = target
        else:
            sel.set_options(opts)
            sel.value = _DISABLED

    @on(Select.Changed, "#chat-model-select")
    def _on_chat_model_changed(self, event: Select.Changed) -> None:
        """Handle chat model selection change."""
        value = self._extract_value(event)
        if value is None:
            return
        cfg.chat_model = value
        settings.set_value(cfg.data_root, "chat_model", value)
        self._after_model_change()

    @on(Select.Changed, "#embed-model-select")
    def _on_embed_model_changed(self, event: Select.Changed) -> None:
        """Handle embedding model selection change."""
        value = self._extract_value(event)
        if value is None:
            return
        cfg.embedding_model = value
        settings.set_value(cfg.data_root, "embedding_model", value)
        self._after_model_change()

    @on(Select.Changed, "#vision-model-select")
    def _on_vision_model_changed(self, event: Select.Changed) -> None:
        """Handle vision model selection change."""
        if self._populating:
            return
        if event.value is _DISABLED or event.value is None or str(event.value) == "":
            cfg.vision_model = ""
            settings.set_value(cfg.data_root, "vision_model", "")
            return
        cfg.vision_model = str(event.value)
        settings.set_value(cfg.data_root, "vision_model", cfg.vision_model)
        self._after_model_change()

    def _extract_value(self, event: Select.Changed) -> str | None:
        """Extract a non-empty value from a Select.Changed event, or None to skip."""
        if self._populating:
            return None
        if event.value is _DISABLED or event.value is None or str(event.value) == "":
            return None
        return str(event.value)

    def _after_model_change(self) -> None:
        """Shared post-change logic: cancel active stream and reset services safely."""
        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = self.app.screen
        if isinstance(screen, ChatScreen) and screen.streaming:
            screen.action_cancel_stream()
            self.app.call_later(self._deferred_reset)
        else:
            self._reset_services()

    def _deferred_reset(self) -> None:
        """Reset services after workers have finished (avoids freeing in-use models)."""
        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = self.app.screen
        if isinstance(screen, ChatScreen) and screen.workers:
            self.app.call_later(self._deferred_reset)
            return
        self._reset_services()

    @staticmethod
    def _reset_services() -> None:
        reset_services()

    def refresh_models(self) -> None:
        """Re-scan models (called after downloads complete)."""
        self._scan_models()
