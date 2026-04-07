"""Model status bar — Select dropdowns for chat, embedding, and vision models."""

from __future__ import annotations

import logging

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Label, Select

from lilbee import settings
from lilbee.config import cfg

log = logging.getLogger(__name__)

_DISABLED = Select.NULL

_MMPROJ_MARKER = "mmproj"


def _is_mmproj(name: str) -> bool:
    """Return True if a model name refers to an mmproj projection file."""
    return _MMPROJ_MARKER in name.lower()


def _classify_installed_models() -> tuple[list[str], list[str], list[str]]:
    """Classify installed models into (chat, embedding, vision) lists.

    Uses registry manifests for native models and the litellm backend's
    /api/tags metadata for remote models. Filters out mmproj files.
    """
    buckets: dict[str, list[str]] = {"chat": [], "embedding": [], "vision": []}
    seen: set[str] = set()

    _collect_native_models(buckets, seen)
    _collect_remote_models(buckets, seen)

    return sorted(buckets["chat"]), sorted(buckets["embedding"]), sorted(buckets["vision"])


def _collect_native_models(buckets: dict[str, list[str]], seen: set[str]) -> None:
    """Add native registry models to buckets."""
    try:
        from lilbee.registry import ModelRegistry

        registry = ModelRegistry(cfg.models_dir)
        for manifest in registry.list_installed():
            name = f"{manifest.name}:{manifest.tag}"
            if _is_mmproj(name) or name in seen:
                continue
            seen.add(name)
            buckets.get(manifest.task, buckets["chat"]).append(name)
    except Exception:
        log.debug("Could not read native model registry", exc_info=True)


def _collect_remote_models(buckets: dict[str, list[str]], seen: set[str]) -> None:
    """Add remote (litellm) models to buckets."""
    try:
        from lilbee.model_manager import classify_remote_models

        for model in classify_remote_models(cfg.litellm_base_url):
            if model.name in seen or _is_mmproj(model.name):
                continue
            seen.add(model.name)
            buckets.get(model.task, buckets["chat"]).append(model.name)
    except Exception:
        log.debug("Could not classify remote models", exc_info=True)


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
        chat_models: list[str],
        embed_models: list[str],
        vision_models: list[str],
    ) -> None:
        """Populate Select widgets from scanned models (main thread)."""
        self._populating = True

        chat_sel = self.query_one("#chat-model-select", Select)
        embed_sel = self.query_one("#embed-model-select", Select)
        vision_sel = self.query_one("#vision-model-select", Select)

        chat_opts = [(m, m) for m in chat_models] if chat_models else [("(none)", "")]
        embed_opts = [(m, m) for m in embed_models] if embed_models else [("(none)", "")]
        vision_opts = [(m, m) for m in vision_models]

        chat_sel.set_options(chat_opts)
        embed_sel.set_options(embed_opts)
        vision_sel.set_options(vision_opts)

        current_chat = str(chat_sel.value) if chat_sel.value != Select.NULL else ""
        current_embed = str(embed_sel.value) if embed_sel.value != Select.NULL else ""
        current_vision = str(vision_sel.value) if vision_sel.value != Select.NULL else ""

        has_chat_model = bool(current_chat)
        has_embed_model = bool(current_embed)
        has_vision_model = bool(current_vision)

        if has_chat_model:
            if not any(v == current_chat for _, v in chat_opts):
                chat_opts.insert(0, (current_chat, current_chat))
            chat_sel.set_options(chat_opts)
            chat_sel.value = current_chat
        elif chat_models:
            chat_sel.value = chat_models[0]
        elif chat_opts:
            chat_sel.value = chat_opts[0][1]

        if has_embed_model:
            if not any(v == current_embed for _, v in embed_opts):
                embed_opts.insert(0, (current_embed, current_embed))
            embed_sel.set_options(embed_opts)
            embed_sel.value = current_embed
        elif embed_models:
            embed_sel.value = embed_models[0]
        elif embed_opts:
            embed_sel.value = embed_opts[0][1]

        if has_vision_model:
            if not any(v == current_vision for _, v in vision_opts):
                vision_opts.insert(0, (current_vision, current_vision))
            vision_sel.set_options(vision_opts)
            vision_sel.value = current_vision
        elif cfg.vision_model:
            if not any(v == cfg.vision_model for _, v in vision_opts):
                vision_opts.insert(0, (cfg.vision_model, cfg.vision_model))
            vision_sel.set_options(vision_opts)
            vision_sel.value = cfg.vision_model
        else:
            vision_sel.value = _DISABLED

        self._populating = False

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
        """Shared post-change logic: cancel active stream and reset services."""
        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = self.app.screen
        if isinstance(screen, ChatScreen) and screen._streaming:
            screen.action_cancel_stream()

        from lilbee.services import reset_services

        reset_services()

    def refresh_models(self) -> None:
        """Re-scan models (called after downloads complete)."""
        self._scan_models()
