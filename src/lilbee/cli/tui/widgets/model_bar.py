"""Model status bar: pill buttons for chat / embedding plus mode + scope."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple

if TYPE_CHECKING:
    from lilbee.modelhub.registry import ModelRegistry

from textual import events, work
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal
from textual.widget import Widget
from textual.widgets import Static

from lilbee.catalog import clean_display_name, display_label_for_ref, extract_quant
from lilbee.cli.tui import messages as msg
from lilbee.cli.tui.app import apply_active_model, apply_setting
from lilbee.cli.tui.pill import pill
from lilbee.cli.tui.thread_safe import call_from_thread
from lilbee.core.config import cfg
from lilbee.core.config.enums import ChatMode
from lilbee.core.services import get_services, reset_services
from lilbee.modelhub.models import ModelTask
from lilbee.providers.model_ref import format_remote_ref, parse_model_ref
from lilbee.providers.sdk_backend import PROVIDER_KEYS
from lilbee.retrieval.embedder import is_model_available

log = logging.getLogger(__name__)

_MMPROJ_MARKER = "mmproj"

_CLOUD_WARNING_ID = "cloud-provider-warning"
_CLOUD_WARNING_CLASS = "cloud-warning"
_CLOUD_WARNING_VISIBLE_CLASS = "-visible"

_CHAT_MODEL_BUTTON_ID = "chat-model-button"
_EMBED_MODEL_BUTTON_ID = "embed-model-button"

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
    """Classify installed models into (chat, embedding) lists, dropping mmproj.

    The chat-bar surfaces only chat + embedding pickers; vision and rerank
    use ``classify_installed_models_full`` directly. Vision/rerank entries
    are still discovered here so their refs are claimed in ``seen`` and
    later buckets don't duplicate them.
    """
    buckets = classify_installed_models_full()
    return (buckets[ModelTask.CHAT], buckets[ModelTask.EMBEDDING])


def classify_installed_models_full() -> dict[ModelTask, list[ModelOption]]:
    """Classify installed models into per-task lists, dropping mmproj entries."""
    buckets: dict[ModelTask, list[ModelOption]] = {task: [] for task in ModelTask}
    seen: set[str] = set()

    _collect_native_models(buckets, seen)
    _collect_remote_models(buckets, seen)
    _collect_api_models(buckets, seen)

    return {task: sorted(opts, key=lambda o: o.ref) for task, opts in buckets.items()}


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


def _has_vision_sidecar(registry: ModelRegistry, ref: str) -> bool:
    """Return True if *ref* resolves to a model with an adjacent ``*mmproj*.gguf`` file.

    Models like ``google/gemma-3-12b-it`` carry their vision capability in
    a sibling ``mmproj`` GGUF; without checking the file system, the
    ref's name alone gives no signal that the model is multimodal, so the
    vision picker would silently miss it.
    """
    try:
        path = registry.resolve(ref)
    except (KeyError, ValueError):
        return False
    return any(path.parent.glob("*mmproj*.gguf"))


def _collect_native_models(buckets: dict[ModelTask, list[ModelOption]], seen: set[str]) -> None:
    """Add native registry models to buckets."""
    try:
        from lilbee.modelhub.registry import ModelRegistry

        registry = ModelRegistry(cfg.models_dir)
        manifests = registry.list_installed()
        repo_counts: dict[str, int] = {}
        for m in manifests:
            repo_counts[m.hf_repo] = repo_counts.get(m.hf_repo, 0) + 1

        from lilbee.modelhub.model_manager.discovery import reclassify_by_name

        for manifest in manifests:
            ref = manifest.ref
            if _is_mmproj(manifest.gguf_filename) or ref in seen:
                continue
            task = reclassify_by_name(ref, manifest.task)
            label = _native_label(
                manifest.hf_repo, manifest.gguf_filename, repo_counts[manifest.hf_repo]
            )
            primary_bucket = _lookup_bucket(buckets, task, ref)
            if primary_bucket is None:
                continue
            seen.add(ref)
            primary_bucket.append(ModelOption(label=label, ref=ref))
            # If the model has an mmproj sidecar it is also vision-capable.
            # Surface it under the vision picker too without dropping its
            # primary classification, so a chat model with vision (e.g.
            # gemma-3 with mmproj) shows up in both pickers.
            if task != ModelTask.VISION and _has_vision_sidecar(registry, ref):
                buckets[ModelTask.VISION].append(ModelOption(label=label, ref=ref))
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

        for model in classify_remote_models(cfg.remote_base_url):
            # Skip backend rows with a blank model name so the picker
            # doesn't render an empty " (Ollama)" row.
            if not model.name.strip():
                continue
            ref = format_remote_ref(model)
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
                qualified = format_remote_ref(model)
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


_CSS_FILE = Path(__file__).parent / "model_bar.tcss"

_CHAT_MODE_TOGGLE_ID = "chat-mode-toggle"
_CHAT_MODE_SEARCH_PILL_ID = "chat-mode-search"
_CHAT_MODE_CHAT_PILL_ID = "chat-mode-chat"
_CHAT_MODE_PILL_CLASS = "chat-mode-pill"
_CHAT_MODE_DISABLED_CLASS = "-disabled"
_CHAT_MODE_ACTIVE_CLASS = "-active"


class ModelPickerButton(Static, can_focus=True):
    """Pill button that opens a ModelPickerModal scoped to chat or embed."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "open_picker", "Pick model", show=False),
        Binding("space", "open_picker", "Pick model", show=False),
    ]

    def __init__(self, *, scope: Literal["chat", "embed"], button_id: str) -> None:
        super().__init__(id=button_id)
        self._scope: Literal["chat", "embed"] = scope
        self._options: list[ModelOption] = []
        self.tooltip = (
            msg.MODEL_PICKER_CHAT_TOOLTIP if scope == "chat" else msg.MODEL_PICKER_EMBED_TOOLTIP
        )

    def on_mount(self) -> None:
        self._refresh()

    def set_options(self, options: list[ModelOption]) -> None:
        """Update the options pool. Repaints the label from cfg."""
        self._options = options
        if self.is_mounted:
            self._refresh()

    def _refresh(self) -> None:
        ref = cfg.chat_model if self._scope == "chat" else cfg.embedding_model
        label = display_label_for_ref(ref) or ref or msg.MODEL_VALUE_NONE
        self.update(label)

    def on_click(self, event: events.Click) -> None:
        event.stop()
        self.open_picker()

    def action_open_picker(self) -> None:
        self.open_picker()

    def open_picker(self) -> None:
        # Lazy import: model_picker imports ModelOption from this module.
        from lilbee.cli.tui.screens.model_picker import ModelPickerModal

        modal = ModelPickerModal(scope=self._scope, options=self._options)
        self.app.push_screen(modal, self._on_picker_dismissed)

    def _on_picker_dismissed(self, ref: str | None) -> None:
        if not ref:
            return
        if self._scope == "chat":
            if ref == cfg.chat_model:
                return
            apply_active_model(self.app, "chat_model", ref)
        else:
            if ref == cfg.embedding_model:
                return
            get_services().store.initialize_meta_if_legacy()
            apply_active_model(self.app, "embedding_model", ref)
        self._refresh()
        bar = self.screen.query(ModelBar)
        for b in bar:
            b._after_model_change()


class ChatModeToggle(Widget, can_focus=True):
    """Two-pill control toggling cfg.chat_mode between Search and Chat."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "flip_mode", "Toggle mode", show=False),
        Binding("space", "flip_mode", "Toggle mode", show=False),
        Binding("left", "select_search", "Search mode", show=False),
        Binding("right", "select_chat", "Chat mode", show=False),
    ]

    def __init__(self) -> None:
        super().__init__(id=_CHAT_MODE_TOGGLE_ID)

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static(
                msg.CHAT_MODE_SEARCH_LABEL,
                id=_CHAT_MODE_SEARCH_PILL_ID,
                classes=_CHAT_MODE_PILL_CLASS,
            )
            yield Static(
                msg.CHAT_MODE_CHAT_LABEL,
                id=_CHAT_MODE_CHAT_PILL_ID,
                classes=_CHAT_MODE_PILL_CLASS,
            )

    def on_mount(self) -> None:
        self._refresh()

    def refresh_state(self) -> None:
        """Repaint label/state. Call after settings or embedding-model changes."""
        if self.is_mounted:
            self._refresh()

    def _embedding_ready(self) -> bool:
        return is_model_available(cfg.embedding_model, get_services().provider)

    def _refresh(self) -> None:
        ready = self._embedding_ready()
        mode = cfg.chat_mode if ready else ChatMode.CHAT.value
        active_search = mode == ChatMode.SEARCH.value
        search_pill = self.query_one(f"#{_CHAT_MODE_SEARCH_PILL_ID}", Static)
        chat_pill = self.query_one(f"#{_CHAT_MODE_CHAT_PILL_ID}", Static)
        # Search half is disabled whenever embedding isn't ready; Chat is
        # always reachable so it never carries the disabled class.
        search_pill.set_class(active_search, _CHAT_MODE_ACTIVE_CLASS)
        search_pill.set_class(not ready, _CHAT_MODE_DISABLED_CLASS)
        chat_pill.set_class(not active_search, _CHAT_MODE_ACTIVE_CLASS)
        chat_pill.set_class(False, _CHAT_MODE_DISABLED_CLASS)
        # Parent carries the disabled class so external selectors can
        # disable interaction on the whole toggle when search is gated.
        self.set_class(not ready, _CHAT_MODE_DISABLED_CLASS)
        self.tooltip = (
            msg.CHAT_MODE_TOGGLE_DISABLED_TOOLTIP if not ready else msg.CHAT_MODE_TOGGLE_TOOLTIP
        )

    def _set_mode(self, target: str) -> bool:
        """Apply *target* if it differs from the current mode and Search is allowed."""
        if cfg.chat_mode == target:
            return False
        if target == ChatMode.SEARCH.value and not self._embedding_ready():
            return False
        apply_setting(self.app, "chat_mode", target)
        self._refresh()
        return True

    def toggle(self) -> bool:
        """Flip mode if embedding is ready. Returns True when the mode changed."""
        target = (
            ChatMode.CHAT.value if cfg.chat_mode == ChatMode.SEARCH.value else ChatMode.SEARCH.value
        )
        return self._set_mode(target)

    def on_click(self, event: events.Click) -> None:
        event.stop()
        # Click on a specific pill picks that side; click on the container
        # frame falls through to a toggle.
        widget = event.widget
        if widget is not None:
            wid = widget.id
            if wid == _CHAT_MODE_SEARCH_PILL_ID:
                self._set_mode(ChatMode.SEARCH.value)
                return
            if wid == _CHAT_MODE_CHAT_PILL_ID:
                self._set_mode(ChatMode.CHAT.value)
                return
        self.toggle()

    def action_flip_mode(self) -> None:
        self.toggle()

    def action_select_search(self) -> None:
        self._set_mode(ChatMode.SEARCH.value)

    def action_select_chat(self) -> None:
        self._set_mode(ChatMode.CHAT.value)


class ModelBar(Widget, can_focus=False):
    """Compact bar with picker buttons for active model assignments + mode toggle."""

    DEFAULT_CSS: ClassVar[str] = _CSS_FILE.read_text(encoding="utf-8")

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id=id)
        # _scan_models runs on every chat on_show but the install set rarely
        # changes between visits; fingerprint to skip redundant set_options.
        self._chat_options_cache: tuple[tuple[str, str], ...] = ()
        self._embed_options_cache: tuple[tuple[str, str], ...] = ()

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Static(pill("Chat", "$primary", "$text"), classes="model-bar-pill")
            yield ModelPickerButton(scope="chat", button_id=_CHAT_MODEL_BUTTON_ID)
            yield Static(pill("Embed", "$secondary", "$text"), classes="model-bar-pill")
            yield ModelPickerButton(scope="embed", button_id=_EMBED_MODEL_BUTTON_ID)
            yield ChatModeToggle()
        yield Static("", id=_CLOUD_WARNING_ID, classes=_CLOUD_WARNING_CLASS)

    def on_mount(self) -> None:
        self._refresh_cloud_warning()
        self._scan_models()

    @work(thread=True)
    def _scan_models(self) -> None:
        """Scan installed models in background, then populate buttons."""
        chat, embed = _classify_installed_models()
        call_from_thread(self, self._populate, chat, embed)

    def _populate(
        self,
        chat_models: list[ModelOption],
        embed_models: list[ModelOption],
    ) -> None:
        chat_opts = list(chat_models) if chat_models else [ModelOption(msg.MODEL_VALUE_NONE, "")]
        embed_opts = list(embed_models) if embed_models else [ModelOption(msg.MODEL_VALUE_NONE, "")]
        chat_fingerprint = _options_fingerprint(chat_opts, cfg.chat_model)
        if chat_fingerprint != self._chat_options_cache:
            self.query_one(f"#{_CHAT_MODEL_BUTTON_ID}", ModelPickerButton).set_options(chat_opts)
            self._chat_options_cache = chat_fingerprint
        embed_fingerprint = _options_fingerprint(embed_opts, cfg.embedding_model)
        if embed_fingerprint != self._embed_options_cache:
            self.query_one(f"#{_EMBED_MODEL_BUTTON_ID}", ModelPickerButton).set_options(embed_opts)
            self._embed_options_cache = embed_fingerprint
        self._refresh_cloud_warning()
        self._refresh_chat_mode_toggle()

    def _refresh_cloud_warning(self) -> None:
        """Show a warning if the active chat model routes to a cloud API."""
        warning = self.query_one(f"#{_CLOUD_WARNING_ID}", Static)
        label = _cloud_provider_label(cfg.chat_model)
        if label is None:
            warning.remove_class(_CLOUD_WARNING_VISIBLE_CLASS)
            return
        warning.update(msg.MODEL_BAR_CLOUD_PROVIDER_WARNING.format(provider=label))
        warning.add_class(_CLOUD_WARNING_VISIBLE_CLASS)

    def _refresh_chat_mode_toggle(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one(ChatModeToggle).refresh_state()

    def _after_model_change(self) -> None:
        """Shared post-change logic: cancel active stream and reset services safely."""
        from lilbee.cli.tui.screens.chat import ChatScreen

        screen = self.app.screen
        if isinstance(screen, ChatScreen):
            screen.apply_model_change()
        else:
            reset_services()

    def refresh_models(self) -> None:
        """Re-scan models (called after downloads complete)."""
        self._scan_models()
